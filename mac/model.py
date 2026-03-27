# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class ResGCNBlock(nn.Module):
    """GCN layer with residual connection and batch normalization."""
    def __init__(self, dim):
        super().__init__()
        self.conv = GCNConv(dim, dim)
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x, edge_index):
        return F.relu(self.bn(self.conv(x, edge_index))) + x


class GNNModel(nn.Module):
    def __init__(self, in_channels=5, hidden_dim=128, num_layers=6):
        super().__init__()
        # Project input features to hidden dim
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)

        # Residual GCN blocks
        self.blocks = nn.ModuleList([
            ResGCNBlock(hidden_dim) for _ in range(num_layers)
        ])

        # Policy Head — per-node score
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Value Head — global board evaluation
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Input projection
        x = F.relu(self.input_bn(self.input_proj(x)))

        # Message passing with residual connections
        for block in self.blocks:
            x = block(x, edge_index)

        # Policy: per-node logit
        policy_logits = self.policy_head(x).squeeze(-1)

        # Value: global state -> tanh
        global_state = global_mean_pool(x, batch)
        value = torch.tanh(self.value_head(global_state)).squeeze(-1)

        return policy_logits, value
