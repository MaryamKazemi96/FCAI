"""
GNN backbone architectures for PPO policy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool, global_max_pool


class GraphSAGEBackbone(nn.Module):
    """GraphSAGE-based GNN backbone."""
    
    def __init__(self, input_dim, hidden_dim, num_layers=2, activation='relu', dropout=0.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch vector [num_nodes] (optional)
        
        Returns:
            node_embeddings: [num_nodes, hidden_dim]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = self.activation(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class GCNBackbone(nn.Module):
    """GCN-based GNN backbone."""
    
    def __init__(self, input_dim, hidden_dim, num_layers=2, activation='relu', dropout=0.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = self.activation(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


def build_gnn_backbone(gnn_type, input_dim, hidden_dim, num_layers, activation, dropout):
    """Factory function to build GNN backbone."""
    if gnn_type == 'graphsage':
        return GraphSAGEBackbone(input_dim, hidden_dim, num_layers, activation, dropout)
    elif gnn_type == 'gcn':
        return GCNBackbone(input_dim, hidden_dim, num_layers, activation, dropout)
    else:
        raise ValueError(f"Unknown GNN type: {gnn_type}")