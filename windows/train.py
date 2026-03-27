# train.py — Optimized for Dell Precision 5680 (i7-13800H, RTX A2000, 32GB, CUDA)
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from collections import deque
import random
import time
import numpy as np
from tqdm import tqdm
from HeXO import HeXOGame
from model import GNNModel
from mcts import MCTS
from torch_geometric.data import Data, Batch

# ─── Device Setup ───────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── Hyperparameters tuned for Precision 5680 (32GB RAM, A2000 GPU) ─────────────
REPLAY_CAPACITY = 100000
SELF_PLAY_GAMES = 10
SELF_PLAY_SIMS = 200
TRAIN_BATCH_SIZE = 128
EVAL_GAMES = 30
EVAL_SIMS = 100
TEMP_THRESHOLD = 12
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_ITERATIONS = 100
MAX_MOVES = 300
USE_AMP = True


class ReplayBuffer:
    def __init__(self, capacity=REPLAY_CAPACITY):
        self.buffer = deque(maxlen=capacity)

    def push(self, data):
        self.buffer.append(data)

    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def __len__(self):
        return len(self.buffer)


def self_play(model, num_games=SELF_PLAY_GAMES, num_sims=SELF_PLAY_SIMS,
              temp_threshold=TEMP_THRESHOLD):
    model.eval()
    experiences = []

    pbar = tqdm(range(num_games), desc="  Self-play", unit="game",
                bar_format="  {desc}: {bar:30} {n_fmt}/{total_fmt} games [{elapsed}<{remaining}]")
    for game_idx in pbar:
        game = HeXOGame()
        mcts_engine = MCTS(model)
        game_history = []
        move_count = 0

        while not game.done and move_count < MAX_MOVES:
            root = mcts_engine.search(game, num_sims)
            temp = 1.0 if move_count < temp_threshold else 0.0
            best_move, policy = mcts_engine.get_policy(root, temperature=temp)

            x, edge_index, sorted_nodes = game.get_graph()
            game_history.append((
                Data(x=x, edge_index=edge_index),
                policy,
                game.current_player,
                sorted_nodes,
            ))

            game.step(*best_move)
            move_count += 1
            pbar.set_postfix_str(f"moves={move_count}")

        winner = game.winner
        for graph_data, policy, player, sorted_nodes in game_history:
            if winner is None:
                z = 0.0
            elif winner == player:
                z = 1.0
            else:
                z = -1.0

            num_nodes = graph_data.x.size(0)
            pi = torch.zeros(num_nodes)
            node_to_idx = {n: i for i, n in enumerate(sorted_nodes)}
            for action, prob in policy.items():
                idx = node_to_idx.get(action)
                if idx is not None:
                    pi[idx] = prob

            graph_data.pi = pi
            graph_data.z = torch.tensor([z], dtype=torch.float)
            experiences.append(graph_data)

    return experiences


def train_step(model, optimizer, replay_buffer, scaler, batch_size=TRAIN_BATCH_SIZE):
    if len(replay_buffer) < batch_size:
        return None

    model.train()
    batch_list = replay_buffer.sample(batch_size)
    data_batch = Batch.from_data_list(batch_list).to(DEVICE)

    optimizer.zero_grad()

    with autocast('cuda', enabled=USE_AMP):
        policy_logits, value = model(data_batch.x, data_batch.edge_index, data_batch.batch)

        pi_target = data_batch.pi
        batch_idx = data_batch.batch
        num_graphs = data_batch.num_graphs

        from torch_geometric.utils import scatter
        max_per_graph = scatter(policy_logits.float(), batch_idx, reduce='max')
        logits_shifted = policy_logits.float() - max_per_graph[batch_idx]
        exp_logits = torch.exp(logits_shifted)
        sum_exp = scatter(exp_logits, batch_idx, reduce='sum')
        log_softmax = logits_shifted - torch.log(sum_exp[batch_idx])

        policy_loss = -(pi_target * log_softmax).sum() / num_graphs
        z_target = data_batch.z.squeeze(-1)
        value_loss = F.mse_loss(value.float(), z_target)

        loss = policy_loss + value_loss

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return {
        'loss': loss.item(),
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
    }


def evaluate_models(challenger, best_model, num_games=EVAL_GAMES, num_sims=EVAL_SIMS):
    challenger.eval()
    best_model.eval()
    challenger_wins = 0

    pbar = tqdm(range(num_games), desc="  Eval", unit="game",
                bar_format="  {desc}: {bar:30} {n_fmt}/{total_fmt} games [{elapsed}<{remaining}]")
    for i in pbar:
        game = HeXOGame()
        challenger_player = 1 if i % 2 == 0 else 2

        while not game.done:
            if game.current_player == challenger_player:
                engine = MCTS(challenger)
            else:
                engine = MCTS(best_model)
            root = engine.search(game, num_sims)
            best_move, _ = engine.get_policy(root, temperature=0)
            game.step(*best_move)

        if game.winner == challenger_player:
            challenger_wins += 1

        pbar.set_postfix_str(f"wins={challenger_wins}/{i+1}")

    return challenger_wins / num_games


def main():
    gpu_info = ""
    if DEVICE.type == 'cuda':
        gpu_info = f" ({torch.cuda.get_device_name(0)})"
    print(f"HeXO Training — Device: {DEVICE}{gpu_info}")
    print(f"Config: {SELF_PLAY_GAMES} games × {SELF_PLAY_SIMS} sims, "
          f"batch={TRAIN_BATCH_SIZE}, buffer={REPLAY_CAPACITY}, AMP={USE_AMP}")
    print()

    best_model = GNNModel().to(DEVICE)
    challenger = GNNModel().to(DEVICE)
    challenger.load_state_dict(best_model.state_dict())

    optimizer = optim.Adam(challenger.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(enabled=USE_AMP)
    replay_buffer = ReplayBuffer()
    best_iteration = -1

    for iteration in range(NUM_ITERATIONS):
        iter_start = time.time()
        print(f"── Iteration {iteration}/{NUM_ITERATIONS} ──")

        # ── Self-play ──
        new_data = self_play(best_model)
        for d in new_data:
            replay_buffer.push(d)

        # ── Training ──
        num_steps = max(1, len(new_data) // 32)
        losses = []
        pbar = tqdm(range(num_steps), desc="  Training", unit="step",
                    bar_format="  {desc}: {bar:30} {n_fmt}/{total_fmt} steps [{elapsed}<{remaining}]")
        for step in pbar:
            metrics = train_step(challenger, optimizer, replay_buffer, scaler)
            if metrics:
                losses.append(metrics['loss'])
                pbar.set_postfix_str(f"loss={metrics['loss']:.3f}")

        avg_loss = sum(losses) / len(losses) if losses else 0

        # ── Evaluation ──
        win_rate = evaluate_models(challenger, best_model)

        # ── Summary line ──
        elapsed = time.time() - iter_start
        status = ""
        if win_rate > 0.55:
            best_iteration = iteration
            best_model.load_state_dict(challenger.state_dict())
            torch.save(best_model.state_dict(), "model_best.pth")
            optimizer = optim.Adam(challenger.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scaler = GradScaler(enabled=USE_AMP)
            status = "★ NEW BEST"
        else:
            challenger.load_state_dict(best_model.state_dict())
            optimizer = optim.Adam(challenger.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scaler = GradScaler(enabled=USE_AMP)
            status = "  no change"

        vram = ""
        if DEVICE.type == 'cuda':
            alloc = torch.cuda.memory_allocated() / 1e6
            vram = f"  vram={alloc:.0f}MB"

        print(f"  ↳ loss={avg_loss:.3f}  win={win_rate:.0%}  "
              f"buf={len(replay_buffer)}  best@iter={best_iteration}{vram}  "
              f"{elapsed:.0f}s  {status}")
        print()


if __name__ == "__main__":
    main()
