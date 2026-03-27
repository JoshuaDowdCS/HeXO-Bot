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
        self._graph_cache = None
        self._cached_prior_pieces = None
        self._base_x_cache = None

    def copy(self):
        g = HeXOGame.__new__(HeXOGame)
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.turn_number = self.turn_number
        g.placements_this_turn = self.placements_this_turn
        g.done = self.done
        g.winner = self.winner
        # Share the cache securely
        if hasattr(self, '_graph_cache'):
            g._graph_cache = self._graph_cache
            g._cached_prior_pieces = self._cached_prior_pieces
            g._base_x_cache = self._base_x_cache
        else:
            g._graph_cache = None
            g._cached_prior_pieces = None
            g._base_x_cache = None
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
        if self.placements_this_turn > 0:
            prior_pieces = list(self.board.keys())[:-self.placements_this_turn]
        else:
            prior_pieces = list(self.board.keys())

        for pos in prior_pieces:
            zone.update(get_cells_within_distance(pos, 8))

        return zone - set(self.board.keys())

    def get_graph(self):
        """
        Build a graph for the GNN.
        Nodes: all occupied cells + empty cells within defined radius.
        Features: [is_current_player, is_opponent, is_empty, q, r]
        """
        if self.placements_this_turn > 0:
            prior_pieces = list(self.board.keys())[:-self.placements_this_turn]
        else:
            prior_pieces = list(self.board.keys())

        prior_pieces_tuple = tuple(prior_pieces)

        # Use cache if graph boundary pieces haven't changed (which is true mid-turn)
        if hasattr(self, '_cached_prior_pieces') and self._cached_prior_pieces == prior_pieces_tuple and self._graph_cache is not None:
            sorted_nodes, node_to_idx, edge_index = self._graph_cache
            base_x = self._base_x_cache
        else:
            node_set = set()
            for pos in prior_pieces:
                node_set.update(get_cells_within_distance(pos, 8))
                
            for pos in self.board:
                node_set.add(pos)

            if not node_set:
                node_set.add((0, 0))

            sorted_nodes = sorted(node_set)
            node_to_idx = {n: i for i, n in enumerate(sorted_nodes)}

            # Cache the base features tensor (coordinates)
            base_x = torch.zeros((len(sorted_nodes), 5), dtype=torch.float)
            for i, (q, r) in enumerate(sorted_nodes):
                base_x[i, 3] = float(q)
                base_x[i, 4] = float(r)

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
                
            self._graph_cache = (sorted_nodes, node_to_idx, edge_index)
            self._cached_prior_pieces = prior_pieces_tuple
            self._base_x_cache = base_x

        # Deep copy the pre-allocated tensor
        x = base_x.clone()
        # Initialize all nodes as empty
        x[:, 2] = 1.0

        # Only loop over occupied pieces (~40 items) instead of all nodes (~8000 items)
        for (q, r), owner in self.board.items():
            idx = node_to_idx.get((q, r))
            if idx is not None:
                x[idx, 2] = 0.0  # Mark not empty
                if owner == self.current_player:
                    x[idx, 0] = 1.0
                else:
                    x[idx, 1] = 1.0

        return x, edge_index, sorted_nodes
