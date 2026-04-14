"""
SB3 policy wrapper (colleague-style, fixed):
- uses RTActorCritic to produce logits per robot per candidate
- appends NOOP column with a learnable (or optionally frozen) noop_logit
- optional logit temperature scaling
- masks invalid actions using obs['action_mask']
- builds MultiDiscrete distribution by flattening logits
- computes logprob/entropy over ACTIVE robots only (robots with any valid candidate)
- IMPORTANT: explicitly adds RTActorCritic params + noop_logit to SB3 optimizer param groups

This is copy-ready to replace your current policy implementation.

Recommended defaults match your colleague's successful recipe:
- noop_logit initialized to -1.0
- optionally freeze noop_logit (so it stays -1.0)
- temperature > 1 (e.g., 5.0) to soften logits early

Usage from PPO:
  model = PPO(
      policy=RTGNNPolicy,
      env=env,
      policy_kwargs=dict(
          in_dim=...,
          hidden_dim=...,
          k_max=...,
          gnn_type=...,
          num_gnn_layers=...,
          activation=...,
          dropout=...,
          noop_init=-1.0,
          noop_frozen=True,
          logit_temperature=5.0,
      ),
      ...
  )
"""
from __future__ import annotations

from typing import Any, Dict, Tuple, cast, Optional

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .actor_critic import RTActorCritic


class IdentityFeaturesExtractor(BaseFeaturesExtractor):
    """
    SB3 requires a features extractor, but we do all modeling ourselves in the policy.
    This extractor just returns a dummy tensor; the policy ignores it.
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 1):
        super().__init__(observation_space, features_dim=features_dim)

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        any_tensor = observations["node_features"]
        B = any_tensor.shape[0]
        return th.zeros((B, self._features_dim), device=any_tensor.device, dtype=any_tensor.dtype)


class RTGNNPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        in_dim: int,
        hidden_dim: int = 128,
        k_max: int = 5,
        gnn_type: str = "graphsage",
        num_gnn_layers: int = 2,
        activation: str = "relu",
        dropout: float = 0.0,
        noop_init: float = -1.0,
        noop_frozen: bool = False, #True
        logit_temperature: float = 5.0, #5.0,
        
        *args,
        **kwargs,
    ):
        assert isinstance(action_space, spaces.MultiDiscrete), "RTGNNPolicy requires MultiDiscrete action space"

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=IdentityFeaturesExtractor,
            features_extractor_kwargs={},
            *args,
            **kwargs,
        )

        # MultiDiscrete([K+1] * R)
        self.R = int(len(action_space.nvec))
        self.Kp1 = int(action_space.nvec[0])
        self.K = self.Kp1 - 1
        assert self.K == int(k_max), f"k_max mismatch: action space implies K={self.K}, got k_max={k_max}"
        self.noop_index = self.K

        # --- core model ---
        self.model = RTActorCritic(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            gnn_type=gnn_type,
            num_gnn_layers=num_gnn_layers,
            activation=activation,
            dropout=dropout,
        )

        # --- noop logit (learnable or frozen) ---
        # shared scalar across batch and robots
        self.noop_logit = nn.Parameter(th.tensor(float(noop_init), dtype=th.float32), requires_grad=(not noop_frozen))

        # --- temperature (no grad) ---
        # if > 1: soften; if == 1: no effect
        self.logit_temperature: float = float(logit_temperature)

        # SB3 expects these to exist even if we bypass them
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()

        # Build optimizer etc.
        self._build(lr_schedule)

        # CRITICAL: ensure PPO optimizer also updates our GNN + noop_logit
        # extra_params = list(self.model.parameters()) + [self.noop_logit]
        # if hasattr(self, "optimizer") and self.optimizer is not None and len(extra_params) > 0:
        #     self.optimizer.add_param_group({"params": extra_params})
        extra_params = list(self.model.parameters()) + [self.noop_logit]

        if hasattr(self, "optimizer") and self.optimizer is not None:
            # collect existing params already in optimizer
            existing = set()
            for g in self.optimizer.param_groups:
                for p in g.get("params", []):
                    existing.add(id(p))

            new_params = [p for p in extra_params if id(p) not in existing]

            if len(new_params) > 0:
                self.optimizer.add_param_group({"params": new_params})

    # ---------------- helpers ----------------

    def _append_noop(self, logits_k: th.Tensor) -> th.Tensor:
        """
        logits_k: [B,R,K]
        returns:  [B,R,K+1] with NOOP logit as a shared scalar parameter.
        """
        B, R, _K = logits_k.shape
        noop_col = self.noop_logit.to(device=logits_k.device, dtype=logits_k.dtype).expand(B, R, 1)
        return th.cat([logits_k, noop_col], dim=-1)

    @staticmethod
    def masked_logprob_entropy(logits: th.Tensor, actions: th.Tensor, active: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        logits:  [B,R,Kp1] already masked with -1e9 for invalid
        actions: [B,R]
        active:  [B,R] bool -> include only active robots in sums

        Returns:
          log_prob_sum: [B]
          entropy_sum:  [B]
        """
        logp = th.log_softmax(logits, dim=-1)                 # [B,R,Kp1]
        a = actions.long().unsqueeze(-1)                      # [B,R,1]
        chosen_logp = logp.gather(-1, a).squeeze(-1)          # [B,R]

        p = th.softmax(logits, dim=-1)
        ent = -th.sum(p * logp, dim=-1)                       # [B,R]

        active_f = active.to(dtype=chosen_logp.dtype)
        chosen_logp = chosen_logp * active_f
        ent = ent * active_f

        return chosen_logp.sum(dim=1), ent.sum(dim=1)

    def _dist_from_logits(self, logits: th.Tensor):
        """
        MultiDiscrete distribution expects flattened logits [B, R*(K+1)].
        """
        B = logits.shape[0]
        logits_flat = logits.reshape(B, -1)
        return self.action_dist.proba_distribution(action_logits=logits_flat)

    def _masked_logits_and_value(self, obs_dict: Dict[str, th.Tensor]) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Returns:
          logits_masked: [B,R,Kp1]
          values:        [B,1]
          active:        [B,R] bool (has any valid candidate excluding NOOP)
        """
        x = obs_dict["node_features"]        # [B,N,F]
        edge_index = obs_dict["edge_index"]  # [B,2,E]
        num_nodes = obs_dict["num_nodes"]    # [B,1]
        num_edges = obs_dict["num_edges"]    # [B,1]
        cand_node_idx = obs_dict["cand_node_idx"]  # [B,R,K]
        mask_full = obs_dict["action_mask"]         # [B,R,K+1] (0/1 or bool)

        logits_k, values = self.model(x, edge_index, num_nodes, num_edges, cand_node_idx)  # [B,R,K], [B,1]
        logits = self._append_noop(logits_k)                                               # [B,R,K+1]

        # temperature scaling (soften)
        if self.logit_temperature and self.logit_temperature != 1.0:
            logits = logits / float(self.logit_temperature)

        # mask
        if mask_full.dtype != th.bool:
            mask_full = mask_full.bool()
        logits = logits.masked_fill(~mask_full, -1e9)

        # active robots = those with at least one valid candidate slot (exclude NOOP)
        active = mask_full[..., :-1].any(dim=-1)

        return logits, values, active

    # ---------------- SB3 API ----------------

    def forward(self, obs: Any, deterministic: bool = False):
        obs_dict = cast(Dict[str, th.Tensor], obs)
        logits, values, active = self._masked_logits_and_value(obs_dict)

        dist = self._dist_from_logits(logits)
        actions = dist.get_actions(deterministic=deterministic)  # [B,R]

        log_prob, _entropy = self.masked_logprob_entropy(logits, actions, active)
        return actions, values, log_prob

    def evaluate_actions(self, obs: Any, actions: th.Tensor):
        obs_dict = cast(Dict[str, th.Tensor], obs)
        logits, values, active = self._masked_logits_and_value(obs_dict)

        dist = self._dist_from_logits(logits)
        log_prob = dist.log_prob(actions)

        # Replace dist entropy with our "active-only" entropy to avoid inactive robots dominating
        _lp_sum, entropy_sum = self.masked_logprob_entropy(logits, actions, active)

        return values, log_prob, entropy_sum

    def predict_values(self, obs: Any) -> th.Tensor:
        obs_dict = cast(Dict[str, th.Tensor], obs)
        # We don't need logits here
        x = obs_dict["node_features"]
        edge_index = obs_dict["edge_index"]
        num_nodes = obs_dict["num_nodes"]
        num_edges = obs_dict["num_edges"]
        cand_node_idx = obs_dict["cand_node_idx"]

        _logits_k, values = self.model(x, edge_index, num_nodes, num_edges, cand_node_idx)
        return values