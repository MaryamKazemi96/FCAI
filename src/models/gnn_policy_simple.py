"""
Simplified Stable-Baselines3 policy wrapper for GNN-based actor-critic.
"""
import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict

from .gnn_backbone import build_gnn_backbone


class SimpleGNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    Simplified GNN-based feature extractor.
    
    Processes graph observations and outputs flat features.
    """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        gnn_type: str = 'graphsage',
        hidden_dim: int = 64,
        num_gnn_layers: int = 2,
        activation: str = 'relu',
        dropout: float = 0.0
    ):
        super().__init__(observation_space, features_dim=hidden_dim)
        
        # Get input dimension
        input_dim = observation_space['node_features'].shape[-1]
        
        self.gnn = build_gnn_backbone(
            gnn_type=gnn_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            activation=activation,
            dropout=dropout
        )
        
        self.hidden_dim = hidden_dim
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process graph observation.
        
        Args:
            observations: Dict with 'node_features', 'edge_index', 'num_nodes', 'num_edges'
        
        Returns:
            features: [batch_size, hidden_dim]
        """
        node_features = observations['node_features']  # [batch_size, max_nodes, feature_dim]
        edge_index = observations['edge_index']        # [batch_size, 2, max_edges]
        num_nodes = observations['num_nodes']          # [batch_size, 1]
        num_edges = observations['num_edges']          # [batch_size, 1]
        
        batch_size = node_features.shape[0]
        
        # Process each graph in the batch separately
        batch_features = []
        
        for i in range(batch_size):
            # Extract valid nodes and edges for this graph
            n_nodes = int(num_nodes[i, 0].item())
            n_edges = int(num_edges[i, 0].item())
            
            if n_nodes == 0 or n_edges == 0:
                # Empty graph - return zeros
                batch_features.append(torch.zeros(self.hidden_dim, device=node_features.device))
                continue
            
            # Get valid data
            x = node_features[i, :n_nodes, :]  # [n_nodes, feature_dim]
            edge_idx = edge_index[i, :, :n_edges]  # [2, n_edges]
            
            # 🔥 FIX: Ensure edge_index is long (int64) type
            edge_idx = edge_idx.long()
            
            # 🔥 FIX: Ensure node features are float
            x = x.float()
            
            # Pass through GNN
            node_embeddings = self.gnn(x, edge_idx)  # [n_nodes, hidden_dim]
            
            # Aggregate to graph-level feature (mean pooling)
            graph_feature = node_embeddings.mean(dim=0)  # [hidden_dim]
            
            batch_features.append(graph_feature)
        
        # Stack batch
        return torch.stack(batch_features, dim=0)  # [batch_size, hidden_dim]


class SimpleGNNActorCriticPolicy(ActorCriticPolicy):
    """
    Simplified ActorCriticPolicy using GNN feature extractor.
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        gnn_type: str = 'graphsage',
        hidden_dim: int = 64,
        num_gnn_layers: int = 2,
        activation: str = 'relu',
        dropout: float = 0.0,
        *args,
        **kwargs
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=SimpleGNNFeaturesExtractor,
            features_extractor_kwargs=dict(
                gnn_type=gnn_type,
                hidden_dim=hidden_dim,
                num_gnn_layers=num_gnn_layers,
                activation=activation,
                dropout=dropout
            ),
            *args,
            **kwargs
        )