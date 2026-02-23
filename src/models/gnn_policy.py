"""
Stable-Baselines3 policy wrapper for GNN-based actor-critic.
"""
import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict, List, Tuple, Type

from .gnn_backbone import build_gnn_backbone


class GNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    GNN-based feature extractor for ego-graph observations.
    
    This processes the graph observation and outputs flat features
    for the SB3 policy heads.
    """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        gnn_type: str = 'graphsage',
        hidden_dim: int = 64,
        num_gnn_layers: int = 2,
        activation: str = 'relu',
        dropout: float = 0.0,
        aggregation: str = 'mean'
    ):
        # Features dimension is the hidden dim after GNN processing
        super().__init__(observation_space, features_dim=hidden_dim)
        
        # Get input dimension from observation space
        # Assumes observation_space['node_features'] contains shape info
        input_dim = observation_space['node_features'].shape[-1]
        
        self.gnn = build_gnn_backbone(
            gnn_type=gnn_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            activation=activation,
            dropout=dropout
        )
        
        self.aggregation = aggregation
        self.hidden_dim = hidden_dim
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process graph observation and return flat features.
        
        Args:
            observations: Dict with 'node_features' and 'edge_index'
        
        Returns:
            features: [batch_size, hidden_dim]
        """
        node_features = observations['node_features']  # [batch_size * num_nodes, feature_dim]
        edge_index = observations['edge_index']  # [2, num_edges]
        batch = observations.get('batch', None)  # [batch_size * num_nodes]
        
        # Pass through GNN
        node_embeddings = self.gnn(node_features, edge_index, batch)
        
        # Aggregate to get graph-level features
        if batch is not None:
            # Batch processing: aggregate per graph
            if self.aggregation == 'mean':
                from torch_geometric.nn import global_mean_pool
                graph_features = global_mean_pool(node_embeddings, batch)
            elif self.aggregation == 'max':
                from torch_geometric.nn import global_max_pool
                graph_features = global_max_pool(node_embeddings, batch)
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")
        else:
            # Single graph: simple aggregation
            if self.aggregation == 'mean':
                graph_features = node_embeddings.mean(dim=0, keepdim=True)
            elif self.aggregation == 'max':
                graph_features = node_embeddings.max(dim=0, keepdim=True)[0]
        
        return graph_features


class GNNActorCriticPolicy(ActorCriticPolicy):
    """
    Custom ActorCriticPolicy using GNN feature extractor.
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
        aggregation: str = 'mean',
        *args,
        **kwargs
    ):
        # Store GNN config
        self.gnn_type = gnn_type
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.activation_fn = activation
        self.dropout = dropout
        self.aggregation = aggregation
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=GNNFeaturesExtractor,
            features_extractor_kwargs=dict(
                gnn_type=gnn_type,
                hidden_dim=hidden_dim,
                num_gnn_layers=num_gnn_layers,
                activation=activation,
                dropout=dropout,
                aggregation=aggregation
            ),
            *args,
            **kwargs
        )