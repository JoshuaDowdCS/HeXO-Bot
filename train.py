# train.py
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import numpy as np
from HeXO import HeXOGame
from model import GNNModel
from mcts import MCTS
from torch_geometric.data import Data, Batch


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, data):
        self.buffer.append(data)

    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def __len__(self):
        return len(self.buffer)


def self_play(model, num_games=10, num_sims=100, temp_threshold=10):
    """
    Play games with MCTS, collecting (graph, policy_target, value_target) tuples.
    Uses temperature=1 for the first `temp_threshold` moves, then temperature=0.
    """
    model.eval()
    experiences = []

    for game_idx in range(num_games):
        game = HeXOGame()
        mcts_engine = MCTS(model)
        game_history = []  # (Data, policy_dict, current_player)
        move_count = 0

        while not game.done:
            # Run MCTS
            root = mcts_engine.search(game, num_sims)

            # Temperature schedule
            temp = 1.0 if move_count < temp_threshold else 0.0
            best_move, policy = mcts_engine.get_policy(root, temperature=temp)

            # Record state before move
            x, edge_index, sorted_nodes = game.get_graph()
            game_history.append((
                Data(x=x, edge_index=edge_index),
                policy,
                game.current_player,
                sorted_nodes,
            ))

            # Apply move
            game.step(*best_move)
            move_count += 1

        # Label all experiences with outcome
        winner = game.winner
        for graph_data, policy, player, sorted_nodes in game_history:
            if winner is None:
                z = 0.0
            elif winner == player:
                z = 1.0
            else:
                z = -1.0

            # Build dense policy target aligned to graph nodes
            num_nodes = graph_data.x.size(0)
            pi = torch.zeros(num_nodes)
            node_to_idx = {n: i for i, n in enumerate(sorted_nodes)}
            for action, prob in policy.items():
                idx = node_to_idx.get(action)
                if idx is not None:
                    pi[idx] = prob

            # Store the policy target inside the Data object for proper batching
            graph_data.pi = pi
            graph_data.z = torch.tensor([z], dtype=torch.float)
            experiences.append(graph_data)

    return experiences


def train_step(model, optimizer, replay_buffer, batch_size=64):
    """One gradient step on a mini-batch from the replay buffer."""
    if len(replay_buffer) < batch_size:
        return None

    model.train()
    batch_list = replay_buffer.sample(batch_size)
    data_batch = Batch.from_data_list(batch_list)

    optimizer.zero_grad()

    policy_logits, value = model(data_batch.x, data_batch.edge_index, data_batch.batch)

    # --- Policy loss (per-graph cross-entropy) ---
    # We need to apply log_softmax per graph, not globally.
    # Scatter approach: for each graph in the batch, compute cross-entropy.
    pi_target = data_batch.pi  # [total_nodes]
    batch_idx = data_batch.batch  # [total_nodes]
    num_graphs = data_batch.num_graphs

    # Per-graph log_softmax: subtract per-graph max for numerical stability
    # then compute log_softmax manually
    from torch_geometric.utils import scatter
    max_per_graph = scatter(policy_logits, batch_idx, reduce='max')  # [num_graphs]
    logits_shifted = policy_logits - max_per_graph[batch_idx]
    exp_logits = torch.exp(logits_shifted)
    sum_exp = scatter(exp_logits, batch_idx, reduce='sum')  # [num_graphs]
    log_softmax = logits_shifted - torch.log(sum_exp[batch_idx])

    # Cross-entropy: -sum(pi * log_softmax) per graph, averaged
    policy_loss = -(pi_target * log_softmax).sum() / num_graphs

    # --- Value loss ---
    z_target = data_batch.z.squeeze(-1)  # [num_graphs]
    value_loss = F.mse_loss(value, z_target)

    loss = policy_loss + value_loss
    loss.backward()
    optimizer.step()

    return {
        'loss': loss.item(),
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
    }


def evaluate_models(challenger, best_model, num_games=40, num_sims=50):
    """Play challenger against best model. Returns challenger win rate."""
    challenger.eval()
    best_model.eval()
    challenger_wins = 0

    for i in range(num_games):
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

    return challenger_wins / num_games


def main():
    best_model = GNNModel()
    challenger = GNNModel()
    challenger.load_state_dict(best_model.state_dict())

    optimizer = optim.Adam(challenger.parameters(), lr=1e-3, weight_decay=1e-4)
    replay_buffer = ReplayBuffer(capacity=50000)

    for iteration in range(100):
        print(f"\n{'='*50}")
        print(f"Iteration {iteration}")
        print(f"{'='*50}")

        # --- Self-play with best model ---
        print("Self-play...")
        new_data = self_play(best_model, num_games=10, num_sims=100)
        for d in new_data:
            replay_buffer.push(d)
        print(f"  Generated {len(new_data)} positions, buffer size: {len(replay_buffer)}")

        # --- Train challenger ---
        print("Training...")
        train_steps = max(1, len(new_data) // 32)
        for step in range(train_steps):
            metrics = train_step(challenger, optimizer, replay_buffer, batch_size=64)
            if metrics and step % 10 == 0:
                print(f"  Step {step}: loss={metrics['loss']:.4f} "
                      f"(policy={metrics['policy_loss']:.4f}, value={metrics['value_loss']:.4f})")

        # --- Evaluate ---
        print("Evaluating...")
        win_rate = evaluate_models(challenger, best_model, num_games=20, num_sims=50)
        print(f"  Challenger win rate: {win_rate:.1%}")

        if win_rate > 0.55:
            print("  -> New best model!")
            best_model.load_state_dict(challenger.state_dict())
            torch.save(best_model.state_dict(), "model_best.pth")
            # Reset optimizer for fresh start with new best
            optimizer = optim.Adam(challenger.parameters(), lr=1e-3, weight_decay=1e-4)
        else:
            print("  -> Best model unchanged, resetting challenger.")
            challenger.load_state_dict(best_model.state_dict())
            optimizer = optim.Adam(challenger.parameters(), lr=1e-3, weight_decay=1e-4)


if __name__ == "__main__":
    main()
