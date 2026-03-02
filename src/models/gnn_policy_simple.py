# """
# Simplified Stable-Baselines3 policy wrapper for GNN-based actor-critic.
# """
# import torch
# import torch.nn as nn
# import numpy as np
# from gymnasium import spaces
# from stable_baselines3.common.policies import ActorCriticPolicy
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from typing import Dict

# from .gnn_backbone import build_gnn_backbone


# class SimpleGNNFeaturesExtractor(BaseFeaturesExtractor):
#     """
#     Simplified GNN-based feature extractor.
    
#     Processes graph observations and outputs flat features.
#     """
    
#     def __init__(
#         self,
#         observation_space: spaces.Dict,
#         gnn_type: str = 'graphsage',
#         hidden_dim: int = 64,
#         num_gnn_layers: int = 2,
#         activation: str = 'relu',
#         dropout: float = 0.0
#     ):
#         super().__init__(observation_space, features_dim=hidden_dim)
        
#         # Get input dimension
#         input_dim = observation_space['node_features'].shape[-1]
        
#         self.gnn = build_gnn_backbone(
#             gnn_type=gnn_type,
#             input_dim=input_dim,
#             hidden_dim=hidden_dim,
#             num_layers=num_gnn_layers,
#             activation=activation,
#             dropout=dropout
#         )
        
#         self.hidden_dim = hidden_dim
    
#     def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
#         """
#         Process graph observation.
        
#         Args:
#             observations: Dict with 'node_features', 'edge_index', 'num_nodes', 'num_edges'
        
#         Returns:
#             features: [batch_size, hidden_dim]
#         """
#         node_features = observations['node_features']  # [batch_size, max_nodes, feature_dim]
#         edge_index = observations['edge_index']        # [batch_size, 2, max_edges]
#         num_nodes = observations['num_nodes']          # [batch_size, 1]
#         num_edges = observations['num_edges']          # [batch_size, 1]
        
#         batch_size = node_features.shape[0]
        
#         # Process each graph in the batch separately
#         batch_features = []
        
#         for i in range(batch_size):
#             # Extract valid nodes and edges for this graph
#             n_nodes = int(num_nodes[i, 0].item())
#             n_edges = int(num_edges[i, 0].item())
            
#             if n_nodes == 0 or n_edges == 0:
#                 # Empty graph - return zeros
#                 batch_features.append(torch.zeros(self.hidden_dim, device=node_features.device))
#                 continue
            
#             # Get valid data
#             x = node_features[i, :n_nodes, :]  # [n_nodes, feature_dim]
#             edge_idx = edge_index[i, :, :n_edges]  # [2, n_edges]
            
#             # 🔥 FIX: Ensure edge_index is long (int64) type
#             edge_idx = edge_idx.long()
            
#             # 🔥 FIX: Ensure node features are float
#             x = x.float()
            
#             # Pass through GNN
#             node_embeddings = self.gnn(x, edge_idx)  # [n_nodes, hidden_dim]
            
#             # Aggregate to graph-level feature (mean pooling)
#             graph_feature = node_embeddings.mean(dim=0)  # [hidden_dim]
            
#             batch_features.append(graph_feature)
        
#         # Stack batch
#         return torch.stack(batch_features, dim=0)  # [batch_size, hidden_dim]


# class SimpleGNNActorCriticPolicy(ActorCriticPolicy):
#     """
#     Simplified ActorCriticPolicy using GNN feature extractor.
#     """
    
#     def __init__(
#         self,
#         observation_space: spaces.Space,
#         action_space: spaces.Space,
#         lr_schedule,
#         gnn_type: str = 'graphsage',
#         hidden_dim: int = 64,
#         num_gnn_layers: int = 2,
#         activation: str = 'relu',
#         dropout: float = 0.0,
#         *args,
#         **kwargs
#     ):
#         super().__init__(
#             observation_space,
#             action_space,
#             lr_schedule,
#             features_extractor_class=SimpleGNNFeaturesExtractor,
#             features_extractor_kwargs=dict(
#                 gnn_type=gnn_type,
#                 hidden_dim=hidden_dim,
#                 num_gnn_layers=num_gnn_layers,
#                 activation=activation,
#                 dropout=dropout
#             ),
#             *args,
#             **kwargs
#         )

"""
GNN policy with colleague-style internal action masking for MultiDiscrete([K+1]*R).
Works with standard SB3 PPO (not MaskablePPO).
"""
import torch as th
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Tuple

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .gnn_backbone import build_gnn_backbone


class SimpleGNNFeaturesExtractor(BaseFeaturesExtractor):
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

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        node_features = observations['node_features']  # [B, max_nodes, F]
        edge_index = observations['edge_index']        # [B, 2, max_edges]
        num_nodes = observations['num_nodes']          # [B, 1]
        num_edges = observations['num_edges']          # [B, 1]

        B = node_features.shape[0]
        out = []

        for i in range(B):
            n_nodes = int(num_nodes[i, 0].item())
            n_edges = int(num_edges[i, 0].item())
            if n_nodes == 0 or n_edges == 0:
                out.append(th.zeros(self.hidden_dim, device=node_features.device))
                continue

            x = node_features[i, :n_nodes, :].float()
            e = edge_index[i, :, :n_edges].long()

            node_emb = self.gnn(x, e)          # [n_nodes, hidden]
            g = node_emb.mean(dim=0)           # [hidden]
            out.append(g)

        return th.stack(out, dim=0)            # [B, hidden]

class SimpleGNNActorCriticPolicy(ActorCriticPolicy):
    """
    ActorCriticPolicy that outputs MultiDiscrete([K+1]*R) actions with internal masking.
    Compatible with standard SB3 PPO.
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

        assert isinstance(action_space, spaces.MultiDiscrete), "This policy requires MultiDiscrete action space"
        self.R = int(len(action_space.nvec))
        self.Kp1 = int(action_space.nvec[0])
        for n in action_space.nvec:
            assert int(n) == self.Kp1, "All robots must share same (K+1) action size"

        # ✅ IMPORTANT: use SB3's mlp_extractor latent sizes (not self.features_dim)
        latent_dim_pi = int(self.mlp_extractor.latent_dim_pi)
        latent_dim_vf = int(self.mlp_extractor.latent_dim_vf)

        # Replace default action/value heads with ones that match MultiDiscrete([K+1]*R)
        self.action_net = nn.Linear(latent_dim_pi, self.R * self.Kp1)
        self.value_net = nn.Linear(latent_dim_vf, 1)

        # Rebuild optimizer after replacing heads
        self._build(lr_schedule)

    def _masked_logits(self, obs_dict: Dict[str, th.Tensor]) -> th.Tensor:
        """
        Returns masked logits [B, R, K+1], mask from obs_dict['action_mask'] [B,R,K+1].
        """
        features = self.extract_features(obs_dict)          # [B, features_dim]
        latent_pi, _latent_vf = self.mlp_extractor(features)

        logits_flat = self.action_net(latent_pi)            # [B, R*(K+1)]
        logits = logits_flat.view(-1, self.R, self.Kp1)     # [B, R, K+1]

        mask = obs_dict["action_mask"]
        if mask.dtype != th.bool:
            mask = mask.bool()

        return logits.masked_fill(~mask, -1e9)

    @staticmethod
    def _logprob_entropy_from_logits(logits: th.Tensor, actions: th.Tensor, active: th.Tensor):
        logp = th.log_softmax(logits, dim=-1)               # [B,R,K]
        a = actions.long().unsqueeze(-1)                    # [B,R,1]
        chosen_logp = logp.gather(-1, a).squeeze(-1)        # [B,R]

        p = th.softmax(logits, dim=-1)
        ent = -th.sum(p * logp, dim=-1)                     # [B,R]

        chosen_logp = chosen_logp * active.float()
        ent = ent * active.float()

        return chosen_logp.sum(dim=1), ent.sum(dim=1)

    def forward(self, obs: Any, deterministic: bool = False):
        obs_dict = obs

        logits = self._masked_logits(obs_dict)              # [B,R,K+1]

        features = self.extract_features(obs_dict)
        _latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)                  # [B,1]

        logits_flat = logits.reshape(logits.shape[0], -1)   # [B, R*(K+1)]
        dist = self.action_dist.proba_distribution(action_logits=logits_flat)
        actions = dist.get_actions(deterministic=deterministic)  # [B,R]

        mask = obs_dict["action_mask"].bool()
        active = mask[..., :-1].any(dim=-1)                 # [B,R] (any non-NOOP valid)

        log_prob, _ = self._logprob_entropy_from_logits(logits, actions, active)
        return actions, values, log_prob

    def evaluate_actions(self, obs: Any, actions: th.Tensor):
        obs_dict = obs

        logits = self._masked_logits(obs_dict)              # [B,R,K+1]

        features = self.extract_features(obs_dict)
        _latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)                  # [B,1]

        mask = obs_dict["action_mask"].bool()
        active = mask[..., :-1].any(dim=-1)

        log_prob, entropy = self._logprob_entropy_from_logits(logits, actions, active)
        return values, log_prob, entropy

    def predict_values(self, obs: Any) -> th.Tensor:
        obs_dict = obs
        features = self.extract_features(obs_dict)
        _latent_pi, latent_vf = self.mlp_extractor(features)
        return self.value_net(latent_vf)