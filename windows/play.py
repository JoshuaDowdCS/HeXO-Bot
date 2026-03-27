# play.py — Play against the trained HeXO model
import torch
import os
from HeXO import HeXOGame
from model import GNNModel
from mcts import MCTS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def render_board(game):
    """Render the hex board as ASCII art in the terminal."""
    if not game.board:
        print("\n  (empty board — place your first piece at any coordinate)")
        return

    all_cells = set(game.board.keys())
    for pos in list(game.board.keys()):
        for dq, dr in [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]:
            all_cells.add((pos[0]+dq, pos[1]+dr))

    min_q = min(c[0] for c in all_cells)
    max_q = max(c[0] for c in all_cells)
    min_r = min(c[1] for c in all_cells)
    max_r = max(c[1] for c in all_cells)

    print()
    # Header
    header = "     "
    for q in range(min_q, max_q + 1):
        header += f" {q:>2} "
    print(header)
    print("     " + "----" * (max_q - min_q + 1))

    for r in range(min_r, max_r + 1):
        indent = " " * (r - min_r) * 2
        row = f"{indent} {r:>2} |"
        for q in range(min_q, max_q + 1):
            owner = game.board.get((q, r))
            if owner == 1:
                row += " X "
            elif owner == 2:
                row += " O "
            else:
                row += " · "
            row += "|" if q < max_q else "|"
        print(row)

    print()
    pieces_p1 = sum(1 for v in game.board.values() if v == 1)
    pieces_p2 = sum(1 for v in game.board.values() if v == 2)
    print(f"  X (You): {pieces_p1} pieces    O (AI): {pieces_p2} pieces")
    print(f"  Turn {game.turn_number}, placements this turn: {game.placements_this_turn}")
    print()


def load_model():
    """Load the best model if it exists, otherwise use a fresh random model."""
    model = GNNModel().to(DEVICE)
    model_path = "model_best.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        print(f"Loaded trained model from {model_path}")
    else:
        print("No trained model found — using random model (train first for a challenge!)")
    model.eval()
    return model


def get_human_move(game):
    """Prompt the human for a move."""
    legal = game.get_legal_moves()

    while True:
        try:
            raw = input("  Your move (q r): ").strip()
            if raw.lower() in ('quit', 'exit', 'q'):
                return None
            parts = raw.replace(',', ' ').split()
            if len(parts) != 2:
                print("  Enter two numbers: q r")
                continue
            q, r = int(parts[0]), int(parts[1])
            if (q, r) not in legal:
                if (q, r) in game.board:
                    print(f"  ({q}, {r}) is already occupied!")
                else:
                    print(f"  ({q}, {r}) is not adjacent to any piece. Legal moves:")
                    sorted_legal = sorted(legal)
                    for i in range(0, len(sorted_legal), 8):
                        chunk = sorted_legal[i:i+8]
                        print("    " + "  ".join(f"({q},{r})" for q, r in chunk))
                continue
            return (q, r)
        except ValueError:
            print("  Enter two integers: q r")
        except (EOFError, KeyboardInterrupt):
            return None


def get_ai_move(model, game, num_sims=200):
    """Use MCTS to pick the AI's move."""
    print("  AI is thinking...")
    mcts_engine = MCTS(model)
    root = mcts_engine.search(game, num_sims)
    best_move, _ = mcts_engine.get_policy(root, temperature=0)

    # Show top moves considered
    top_children = sorted(root.children.items(), key=lambda x: x[1].visits, reverse=True)[:5]
    print("  AI considered:")
    for action, child in top_children:
        pct = child.visits / root.visits * 100 if root.visits > 0 else 0
        print(f"    ({action[0]}, {action[1]}): {child.visits} visits ({pct:.0f}%), value={child.value:.2f}")

    print(f"  AI plays: ({best_move[0]}, {best_move[1]})")
    return best_move


def main():
    print("=" * 50)
    print("  HeXO — Play Against the AI")
    print("  You are X (Player 1), AI is O (Player 2)")
    print("  Get 6 in a row on a hex grid to win!")
    print("  Type 'quit' to exit at any time.")
    print("=" * 50)

    model = load_model()
    game = HeXOGame()
    human_player = 1
    ai_player = 2

    while not game.done:
        if game.placements_this_turn == 0:
            render_board(game)

        if game.current_player == human_player:
            expected = 1 if (game.turn_number == 1 and game.current_player == 1) else 2
            remaining = expected - game.placements_this_turn
            print(f"  You need to place {remaining} piece{'s' if remaining > 1 else ''} this turn.")

            move = get_human_move(game)
            if move is None:
                print("\n  Thanks for playing!")
                return
            game.step(*move)
        else:
            expected = 2
            remaining = expected - game.placements_this_turn
            if remaining > 0:
                move = get_ai_move(model, game, num_sims=200)
                game.step(*move)

    # Game over
    render_board(game)
    if game.winner == human_player:
        print("  🎉 You win! Congratulations!")
    elif game.winner == ai_player:
        print("  🤖 AI wins! Better luck next time.")
    else:
        print("  It's a draw!")

    print()


if __name__ == "__main__":
    main()
