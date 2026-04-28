# """
# GNN backbone architectures for PPO policy.
# """
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool, global_max_pool


# class GraphSAGEBackbone(nn.Module):
#     """GraphSAGE-based GNN backbone."""
    
#     def __init__(self, input_dim, hidden_dim, num_layers=2, activation='relu', dropout=0.0):
#         super().__init__()
        
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.dropout = dropout
        
#         # Activation function
#         if activation == 'relu':
#             self.activation = nn.ReLU()
#         elif activation == 'elu':
#             self.activation = nn.ELU()
#         elif activation == 'tanh':
#             self.activation = nn.Tanh()
#         else:
#             raise ValueError(f"Unknown activation: {activation}")
        
#         # Build layers
#         self.convs = nn.ModuleList()
#         self.batch_norms = nn.ModuleList()
        
#         # First layer
#         self.convs.append(SAGEConv(input_dim, hidden_dim))
#         self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
#         # Hidden layers
#         for _ in range(num_layers - 1):
#             self.convs.append(SAGEConv(hidden_dim, hidden_dim))
#             self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
#     def forward(self, x, edge_index, batch=None):
#         """
#         Args:
#             x: Node features [num_nodes, input_dim]
#             edge_index: Edge indices [2, num_edges]
#             batch: Batch vector [num_nodes] (optional)
        
#         Returns:
#             node_embeddings: [num_nodes, hidden_dim]
#         """
#         for i, conv in enumerate(self.convs):
#             x = conv(x, edge_index)
#             x = self.batch_norms[i](x)
#             x = self.activation(x)
#             if self.dropout > 0:
#                 x = F.dropout(x, p=self.dropout, training=self.training)
        
#         return x


# class GCNBackbone(nn.Module):
#     """GCN-based GNN backbone."""
    
#     def __init__(self, input_dim, hidden_dim, num_layers=2, activation='relu', dropout=0.0):
#         super().__init__()
        
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.dropout = dropout
        
#         if activation == 'relu':
#             self.activation = nn.ReLU()
#         elif activation == 'elu':
#             self.activation = nn.ELU()
#         else:
#             raise ValueError(f"Unknown activation: {activation}")
        
#         self.convs = nn.ModuleList()
#         self.batch_norms = nn.ModuleList()
        
#         self.convs.append(GCNConv(input_dim, hidden_dim))
#         self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
#         for _ in range(num_layers - 1):
#             self.convs.append(GCNConv(hidden_dim, hidden_dim))
#             self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
#     def forward(self, x, edge_index, batch=None):
#         for i, conv in enumerate(self.convs):
#             x = conv(x, edge_index)
#             x = self.batch_norms[i](x)
#             x = self.activation(x)
#             if self.dropout > 0:
#                 x = F.dropout(x, p=self.dropout, training=self.training)
        
#         return x


# def build_gnn_backbone(gnn_type, input_dim, hidden_dim, num_layers, activation, dropout):
#     """Factory function to build GNN backbone."""
#     if gnn_type == 'graphsage':
#         return GraphSAGEBackbone(input_dim, hidden_dim, num_layers, activation, dropout)
#     elif gnn_type == 'gcn':
#         return GCNBackbone(input_dim, hidden_dim, num_layers, activation, dropout)
#     else:
#         raise ValueError(f"Unknown GNN type: {gnn_type}")

# src/models/gnn_backbone.py
"""
GNN backbone architectures for PPO policy.

Changes vs your current version (to help logit separation/stability):
- Optional self-loops (recommended True)
- Replace BatchNorm with LayerNorm (default), because BN is often unstable in RL
- Optional residual connections (helps preserve node identity)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import add_self_loops


def _make_norm(norm_type: str, hidden_dim: int) -> nn.Module:
    norm_type = (norm_type or "layer").lower()
    if norm_type in ("layer", "layernorm", "ln"):
        return nn.LayerNorm(hidden_dim)
    if norm_type in ("batch", "batchnorm", "bn"):
        return nn.BatchNorm1d(hidden_dim)
    if norm_type in ("none", "identity", ""):
        return nn.Identity()
    raise ValueError(f"Unknown norm_type: {norm_type}")


class GraphSAGEBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        activation: str = "relu",
        dropout: float = 0.0,
        norm_type: str = "layer",
        add_self_loops_flag: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.add_self_loops_flag = bool(add_self_loops_flag)
        self.residual = bool(residual)

        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "elu":
            self.act = nn.ELU()
        elif activation == "tanh":
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # first layer
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.norms.append(_make_norm(norm_type, hidden_dim))

        # hidden layers
        for _ in range(self.num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(_make_norm(norm_type, hidden_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # edge_index: [2, E]
        if self.add_self_loops_flag:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        h = x
        for i, conv in enumerate(self.convs):
            h_in = h
            h = conv(h, edge_index)  # [N, H]

            # LayerNorm expects [N,H] (ok). BatchNorm expects [N,H] too.
            h = self.norms[i](h)

            h = self.act(h)

            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)

            if self.residual and h.shape == h_in.shape:
                h = h + h_in

        return h


class GCNBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        activation: str = "relu",
        dropout: float = 0.0,
        norm_type: str = "layer",
        add_self_loops_flag: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.add_self_loops_flag = bool(add_self_loops_flag)
        self.residual = bool(residual)

        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "elu":
            self.act = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.norms.append(_make_norm(norm_type, hidden_dim))

        for _ in range(self.num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(_make_norm(norm_type, hidden_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.add_self_loops_flag:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        h = x
        for i, conv in enumerate(self.convs):
            h_in = h
            h = conv(h, edge_index)
            h = self.norms[i](h)
            h = self.act(h)
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)
            if self.residual and h.shape == h_in.shape:
                h = h + h_in
        return h


def build_gnn_backbone(
    gnn_type: str,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    activation: str,
    dropout: float,
    norm_type: str = "layer",
    add_self_loops_flag: bool = True,
    residual: bool = True,
) -> nn.Module:
    gnn_type = (gnn_type or "").lower()
    if gnn_type == "graphsage":
        return GraphSAGEBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
            norm_type=norm_type,
            add_self_loops_flag=add_self_loops_flag,
            residual=residual,
        )
    if gnn_type == "gcn":
        return GCNBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
            norm_type=norm_type,
            add_self_loops_flag=add_self_loops_flag,
            residual=residual,
        )
    raise ValueError(f"Unknown GNN type: {gnn_type}")