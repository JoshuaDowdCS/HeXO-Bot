# HeXO.py
import torch
from utils import get_neighbors, get_cells_within_distance


class HeXOGame:
    def __init__(self):
        self.board = {}  # (q, r) -> player (1 or 2)
        self.current_player = 1
        self.turn_number = 1
        self.placements_this_turn = 0
        self.done = False
        self.winner = None

    def copy(self):
        g = HeXOGame.__new__(HeXOGame)
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.turn_number = self.turn_number
        g.placements_this_turn = self.placements_this_turn
        g.done = self.done
        g.winner = self.winner
        return g

    def step(self, q, r):
        """Apply a single placement. Returns (done, winner)."""
        if self.done:
            return True, self.winner

        if (q, r) in self.board:
            raise ValueError(f"Cell ({q}, {r}) is already occupied.")

        self.board[(q, r)] = self.current_player
        self.placements_this_turn += 1

        if self.check_win(q, r):
            self.done = True
            self.winner = self.current_player
            return True, self.winner

        # Advance turn when all placements for this turn are done
        expected = 1 if (self.turn_number == 1 and self.current_player == 1) else 2
        if self.placements_this_turn >= expected:
            self.current_player = 3 - self.current_player  # flip 1<->2
            self.turn_number += 1
            self.placements_this_turn = 0

        return self.done, self.winner

    def check_win(self, q, r):
        """Check for 6 consecutive same-player pieces along any axis through (q, r)."""
        player = self.board[(q, r)]
        axes = [(1, 0), (0, 1), (1, -1)]

        for dq, dr in axes:
            count = 1
            # Positive direction
            nq, nr = q + dq, r + dr
            while self.board.get((nq, nr)) == player:
                count += 1
                nq += dq
                nr += dr
            # Negative direction
            nq, nr = q - dq, r - dr
            while self.board.get((nq, nr)) == player:
                count += 1
                nq -= dq
                nr -= dr

            if count >= 6:
                return True
        return False

    def get_legal_moves(self):
        """Return set of (q, r) cells that are empty and within the playable zone."""
        if not self.board:
            return {(0, 0)}  # First move

        zone = set()
        for pos in self.board:
            zone.update(get_cells_within_distance(pos, 8))

        return zone - set(self.board.keys())

    def get_graph(self):
        """
        Build a graph for the GNN.
        Nodes: all occupied cells + empty cells within distance 1 of any occupied cell.
        Features: [is_current_player, is_opponent, is_empty, q, r]
        """
        node_set = set()
        for pos in self.board:
            node_set.add(pos)
            node_set.update(get_cells_within_distance(pos, 8))

        if not node_set:
            node_set.add((0, 0))

        sorted_nodes = sorted(node_set)
        node_to_idx = {n: i for i, n in enumerate(sorted_nodes)}

        # Features are relative to the current player
        features = []
        for q, r in sorted_nodes:
            owner = self.board.get((q, r))
            is_me = 1.0 if owner == self.current_player else 0.0
            is_opp = 1.0 if (owner is not None and owner != self.current_player) else 0.0
            is_empty = 1.0 if owner is None else 0.0
            features.append([is_me, is_opp, is_empty, float(q), float(r)])

        x = torch.tensor(features, dtype=torch.float)

        # Edges: hex neighbors
        edges = []
        for i, (q, r) in enumerate(sorted_nodes):
            for nb in get_neighbors(q, r):
                j = node_to_idx.get(nb)
                if j is not None:
                    edges.append([i, j])

        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        return x, edge_index, sorted_nodes
