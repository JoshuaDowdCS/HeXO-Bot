# mcts.py
import math
import numpy as np
import torch
import torch.nn.functional as F


class MCTSNode:
    def __init__(self, state=None, parent=None, prior=0.0):
        self.state = state  # HeXOGame instance
        self.parent = parent
        self.children = {}  # (q, r) -> MCTSNode
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior
        self.is_expanded = False

    @property
    def value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def select_child(self, cpuct):
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        # Pre-compute to avoid math operations inside the tight 2000+ iteration loop
        sqrt_visits = math.sqrt(self.visits)

        for action, child in self.children.items():
            # Inlining property lookup removes thousands of Python function call frame overheads
            val = child.value_sum / child.visits if child.visits > 0 else 0.0
            score = val + cpuct * child.prior * sqrt_visits / (1 + child.visits)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, action_probs):
        """
        action_probs: dict mapping (q, r) -> prior probability
        """
        self.is_expanded = True
        for action, prob in action_probs.items():
            if action not in self.children:
                # Lazy instantiation: don't copy the game state 2000 times!
                self.children[action] = MCTSNode(state=None, parent=self, prior=prob)


class MCTS:
    def __init__(self, model, cpuct=1.5, dirichlet_alpha=0.3, dirichlet_frac=0.25):
        self.model = model
        self.device = next(model.parameters()).device
        self.cpuct = cpuct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_frac = dirichlet_frac

    def search(self, game_state, num_simulations):
        root = MCTSNode(game_state.copy())

        # Expand root immediately so we can add noise
        self._expand_node(root)

        # Add Dirichlet noise to root priors for exploration
        if root.children:
            actions = list(root.children.keys())
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
            frac = self.dirichlet_frac
            for action, n in zip(actions, noise):
                child = root.children[action]
                child.prior = (1 - frac) * child.prior + frac * n

        for _ in range(num_simulations):
            current = root

            # Selection
            while current.is_expanded and not current.state.done:
                action, child = current.select_child(self.cpuct)
                if child.state is None:
                    # Instantiate state only when actually explored, saving 99% of copying overhead
                    child.state = current.state.copy()
                    child.state.step(*action)
                current = child

            # Expansion & Evaluation
            if not current.state.done:
                v = self._expand_node(current)
            else:
                # Terminal node value from perspective of current_player at that node
                if current.state.winner is None:
                    v = 0.0
                elif current.state.winner == current.state.current_player:
                    v = 1.0
                else:
                    v = -1.0

            # Backup
            self._backup(current, v)

        return root

    def _expand_node(self, node):
        """Run the GNN on the node's state and create children. Returns the value estimate."""
        node_features, edge_index, sorted_nodes = node.state.get_graph()

        with torch.inference_mode():
            x = node_features.to(self.device)
            edge = edge_index.to(self.device)
            
            if self.device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    logits, value = self.model(x, edge)
                    # Convert back to float32 for softmax stability
                    logits = logits.float()
            else:
                logits, value = self.model(x, edge)

            # Filter for empty (legal) cells entirely on device
            is_empty_dev = x[:, 2].bool()
            legal_logits = logits[is_empty_dev]

            if legal_logits.numel() == 0:
                node.is_expanded = True
                return value.item()

            probs = F.softmax(legal_logits, dim=0).cpu().numpy()
        
        # Calculate exactly which cells were empty using lightning-fast zip iteration
        is_empty_cpu = node_features[:, 2].bool().tolist()
        candidate_cells = [n for n, is_emp in zip(sorted_nodes, is_empty_cpu) if is_emp]

        action_probs = dict(zip(candidate_cells, probs))
        node.expand(action_probs)

        return value.item()

    def _backup(self, node, v):
        """Walk back to root, updating visit counts and value sums."""
        current = node
        while current is not None:
            current.visits += 1
            current.value_sum += v

            parent = current.parent
            if parent is not None and parent.state.current_player != current.state.current_player:
                v = -v
            current = parent

    def get_policy(self, root, temperature=1.0):
        """
        Extract the improved policy from the root node.
        Returns (best_move, policy_dict).
        """
        actions = list(root.children.keys())
        visits = np.array([root.children[a].visits for a in actions], dtype=np.float64)

        if temperature == 0:
            # Greedy
            best_idx = np.argmax(visits)
            best_move = actions[best_idx]
            policy = {a: (1.0 if a == best_move else 0.0) for a in actions}
        else:
            # Temperature-weighted sampling
            visits_temp = visits ** (1.0 / temperature)
            total = visits_temp.sum()
            probs = visits_temp / total
            chosen_idx = np.random.choice(len(actions), p=probs)
            best_move = actions[chosen_idx]
            policy = {a: p for a, p in zip(actions, probs)}

        return best_move, policy
