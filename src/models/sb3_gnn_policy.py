# """
# SB3 policy wrapper (colleague-style, fixed):
# - uses RTActorCritic to produce logits per robot per candidate
# - appends NOOP column with a learnable (or optionally frozen) noop_logit
# - optional logit temperature scaling
# - masks invalid actions using obs['action_mask']
# - builds MultiDiscrete distribution by flattening logits
# - computes logprob/entropy over ACTIVE robots only (robots with any valid candidate)
# - IMPORTANT: explicitly adds RTActorCritic params + noop_logit to SB3 optimizer param groups

# This is copy-ready to replace your current policy implementation.

# Recommended defaults match your colleague's successful recipe:
# - noop_logit initialized to -1.0
# - optionally freeze noop_logit (so it stays -1.0)
# - temperature > 1 (e.g., 5.0) to soften logits early

# Usage from PPO:
#   model = PPO(
#       policy=RTGNNPolicy,
#       env=env,
#       policy_kwargs=dict(
#           in_dim=...,
#           hidden_dim=...,
#           k_max=...,
#           gnn_type=...,
#           num_gnn_layers=...,
#           activation=...,
#           dropout=...,
#           noop_init=-1.0,
#           noop_frozen=True,
#           logit_temperature=5.0,
#       ),
#       ...
#   )
# """
# from __future__ import annotations

# from typing import Any, Dict, Tuple, cast, Optional

# import torch as th
# import torch.nn as nn
# from gymnasium import spaces
# from stable_baselines3.common.policies import ActorCriticPolicy
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# from .actor_critic import RTActorCritic


# class IdentityFeaturesExtractor(BaseFeaturesExtractor):
#     """
#     SB3 requires a features extractor, but we do all modeling ourselves in the policy.
#     This extractor just returns a dummy tensor; the policy ignores it.
#     """
#     def __init__(self, observation_space: spaces.Dict, features_dim: int = 1):
#         super().__init__(observation_space, features_dim=features_dim)

#     def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
#         any_tensor = observations["node_features"]
#         B = any_tensor.shape[0]
#         return th.zeros((B, self._features_dim), device=any_tensor.device, dtype=any_tensor.dtype)


# class RTGNNPolicy(ActorCriticPolicy):
#     def __init__(
#         self,
#         observation_space: spaces.Space,
#         action_space: spaces.Space,
#         lr_schedule,
#         in_dim: int,
#         hidden_dim: int = 128,
#         k_max: int = 5,
#         gnn_type: str = "graphsage",
#         num_gnn_layers: int = 2,
#         activation: str = "relu",
#         dropout: float = 0.0,
#         noop_init: float = 0.0,
#         noop_frozen: bool = False, #True
#         logit_temperature: float = 1.0, #5.0,
        
#         *args,
#         **kwargs,
#     ):
#         assert isinstance(action_space, spaces.MultiDiscrete), "RTGNNPolicy requires MultiDiscrete action space"

#         super().__init__(
#             observation_space,
#             action_space,
#             lr_schedule,
#             features_extractor_class=IdentityFeaturesExtractor,
#             features_extractor_kwargs={},
#             *args,
#             **kwargs,
#         )

#         # MultiDiscrete([K+1] * R)
#         self.R = int(len(action_space.nvec))
#         self.Kp1 = int(action_space.nvec[0])
#         self.K = self.Kp1 - 1
#         assert self.K == int(k_max), f"k_max mismatch: action space implies K={self.K}, got k_max={k_max}"
#         self.noop_index = self.K

#         # --- core model ---
#         self.model = RTActorCritic(
#             in_dim=in_dim,
#             hidden_dim=hidden_dim,
#             gnn_type=gnn_type,
#             num_gnn_layers=num_gnn_layers,
#             activation=activation,
#             dropout=dropout,
#         )

#         # --- noop logit (learnable or frozen) ---
#         # shared scalar across batch and robots
#         self.noop_logit = nn.Parameter(th.tensor(float(noop_init), dtype=th.float32), requires_grad=(not noop_frozen))

#         # --- temperature (no grad) ---
#         # if > 1: soften; if == 1: no effect
#         self.logit_temperature: float = float(logit_temperature)

#         # SB3 expects these to exist even if we bypass them
#         self.action_net = nn.Identity()
#         self.value_net = nn.Identity()

#         # Build optimizer etc.
#         self._build(lr_schedule)

#         # CRITICAL: ensure PPO optimizer also updates our GNN + noop_logit
#         # extra_params = list(self.model.parameters()) + [self.noop_logit]
#         # if hasattr(self, "optimizer") and self.optimizer is not None and len(extra_params) > 0:
#         #     self.optimizer.add_param_group({"params": extra_params})
#         extra_params = list(self.model.parameters()) + [self.noop_logit]

#         if hasattr(self, "optimizer") and self.optimizer is not None:
#             # collect existing params already in optimizer
#             existing = set()
#             for g in self.optimizer.param_groups:
#                 for p in g.get("params", []):
#                     existing.add(id(p))

#             new_params = [p for p in extra_params if id(p) not in existing]

#             if len(new_params) > 0:
#                 self.optimizer.add_param_group({"params": new_params})
#                 # Debug buffer: latest per-action logit means (computed on meaningful steps only in callback)
#         # Keys like: action_0_mean, action_1_mean, ..., action_noop_mean
#         self.last_action_logit_means: Dict[str, float] = {}

#     # ---------------- helpers ----------------
#     @th.no_grad()
#     def _stash_action_logit_means(self, logits: th.Tensor, mask_full: th.Tensor) -> None:
#         """
#         logits: [B, R, K+1] already masked with -1e9 for invalid
#         mask_full: [B, R, K+1] bool (True=valid)
#         Stores per-action mean logits into self.last_action_logit_means.
#         """
#         try:
#             x = logits.detach()
#             m = mask_full.detach()

#             if m.dtype != th.bool:
#                 m = m.bool()

#             # Only consider "active" robots: at least one valid non-NOOP action
#             active = m[..., :-1].any(dim=-1)  # [B, R]
#             active3 = active.unsqueeze(-1).expand_as(x)  # [B, R, K+1]

#             valid = m & active3  # [B, R, K+1]
#             if not valid.any():
#                 self.last_action_logit_means = {}
#                 return

#             out: Dict[str, float] = {}
#             Kp1 = x.shape[-1]
#             for a in range(Kp1):
#                 va = valid[..., a]
#                 if va.any():
#                     out[f"action_{a}_mean"] = float(x[..., a][va].mean().cpu().item())

#             # Convenience alias for NOOP
#             out["noop_mean"] = out.get(f"action_{self.noop_index}_mean", float(self.noop_logit.detach().cpu().item()))

#             self.last_action_logit_means = out
#         except Exception:
#             # Never break training because of debug logging
#             self.last_action_logit_means = {}
#             return
#     def _append_noop(self, logits_k: th.Tensor) -> th.Tensor:
#         """
#         logits_k: [B,R,K]
#         returns:  [B,R,K+1] with NOOP logit as a shared scalar parameter.
#         """
#         B, R, _K = logits_k.shape
#         noop_col = self.noop_logit.to(device=logits_k.device, dtype=logits_k.dtype).expand(B, R, 1)
#         return th.cat([logits_k, noop_col], dim=-1)

#     @staticmethod
#     def masked_logprob_entropy(logits: th.Tensor, actions: th.Tensor, active: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
#         """
#         logits:  [B,R,Kp1] already masked with -1e9 for invalid
#         actions: [B,R]
#         active:  [B,R] bool -> include only active robots in sums

#         Returns:
#           log_prob_sum: [B]
#           entropy_sum:  [B]
#         """
#         logp = th.log_softmax(logits, dim=-1)                 # [B,R,Kp1]
#         a = actions.long().unsqueeze(-1)                      # [B,R,1]
#         chosen_logp = logp.gather(-1, a).squeeze(-1)          # [B,R]

#         p = th.softmax(logits, dim=-1)
#         ent = -th.sum(p * logp, dim=-1)                       # [B,R]

#         active_f = active.to(dtype=chosen_logp.dtype)
#         chosen_logp = chosen_logp * active_f
#         ent = ent * active_f

#         return chosen_logp.sum(dim=1), ent.sum(dim=1)

#     def _dist_from_logits(self, logits: th.Tensor):
#         """
#         MultiDiscrete distribution expects flattened logits [B, R*(K+1)].
#         """
#         B = logits.shape[0]
#         logits_flat = logits.reshape(B, -1)
#         return self.action_dist.proba_distribution(action_logits=logits_flat)

#     def _masked_logits_and_value(self, obs_dict: Dict[str, th.Tensor]) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
#         """
#         Returns:
#           logits_masked: [B,R,Kp1]
#           values:        [B,1]
#           active:        [B,R] bool (has any valid candidate excluding NOOP)
#         """
#         x = obs_dict["node_features"]        # [B,N,F]
#         edge_index = obs_dict["edge_index"]  # [B,2,E]
#         num_nodes = obs_dict["num_nodes"]    # [B,1]
#         num_edges = obs_dict["num_edges"]    # [B,1]
#         cand_node_idx = obs_dict["cand_node_idx"]  # [B,R,K]
#         mask_full = obs_dict["action_mask"]         # [B,R,K+1] (0/1 or bool)

#         logits_k, values = self.model(x, edge_index, num_nodes, num_edges, cand_node_idx)  # [B,R,K], [B,1]
#         logits = self._append_noop(logits_k)                                               # [B,R,K+1]

#         # temperature scaling (soften)
#         if self.logit_temperature and self.logit_temperature != 1.0:
#             logits = logits / float(self.logit_temperature)

#         # mask
#         if mask_full.dtype != th.bool:
#             mask_full = mask_full.bool()
#         logits = logits.masked_fill(~mask_full, -1e9)
#         # stash per-action logit means for TensorBoard (callback will read these)
#         self._stash_action_logit_means(logits, mask_full)
#         # active robots = those with at least one valid candidate slot (exclude NOOP)
#         active = mask_full[..., :-1].any(dim=-1)

#         return logits, values, active

#     # ---------------- SB3 API ----------------

#     def forward(self, obs: Any, deterministic: bool = False):
#         obs_dict = cast(Dict[str, th.Tensor], obs)
#         logits, values, active = self._masked_logits_and_value(obs_dict)

#         dist = self._dist_from_logits(logits)
#         actions = dist.get_actions(deterministic=deterministic)  # [B,R]

#         log_prob, _entropy = self.masked_logprob_entropy(logits, actions, active)
#         return actions, values, log_prob

#     def evaluate_actions(self, obs: Any, actions: th.Tensor):
#         obs_dict = cast(Dict[str, th.Tensor], obs)
#         logits, values, active = self._masked_logits_and_value(obs_dict)

#         dist = self._dist_from_logits(logits)
#         log_prob = dist.log_prob(actions)

#         # Replace dist entropy with our "active-only" entropy to avoid inactive robots dominating
#         _lp_sum, entropy_sum = self.masked_logprob_entropy(logits, actions, active)

#         return values, log_prob, entropy_sum

#     def predict_values(self, obs: Any) -> th.Tensor:
#         obs_dict = cast(Dict[str, th.Tensor], obs)
#         # We don't need logits here
#         x = obs_dict["node_features"]
#         edge_index = obs_dict["edge_index"]
#         num_nodes = obs_dict["num_nodes"]
#         num_edges = obs_dict["num_edges"]
#         cand_node_idx = obs_dict["cand_node_idx"]

#         _logits_k, values = self.model(x, edge_index, num_nodes, num_edges, cand_node_idx)
#         return values

from __future__ import annotations

"""
SB3 policy wrapper (colleague-style, copy-ready, syntax-safe).

- Core model produces candidate logits_k: [B,R,K] and value: [B,1]
- Temperature scaling applied to candidate logits only (NOOP not scaled)
- Append shared NOOP scalar logit as last column -> [B,R,K+1]
- Mask invalid candidates using obs['cand_mask'] (or derive from obs['action_mask'])
- Force NOOP always valid
- MultiDiscrete distribution built from flattened logits [B, R*(K+1)]
- Logprob/entropy computed over ACTIVE robots only

Expected obs keys (recommended):
  node_features: [B,N,F]
  edge_index:    [B,2,E]
  num_nodes:     [B,1]
  num_edges:     [B,1]
  cand_node_idx: [B,R,K]
  cand_mask:     [B,R,K]  (0/1 or bool)

Fallback:
  if cand_mask missing but action_mask [B,R,K+1] exists, cand_mask := action_mask[..., :-1]
"""

from typing import Any, Dict, Tuple, cast, Optional

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .actor_critic import RTActorCritic


class DictPassthroughExtractor(BaseFeaturesExtractor):
    """Capture raw dict observation and return dummy features to satisfy SB3."""

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=1)
        self.last_obs: Optional[Dict[str, th.Tensor]] = None

    def forward(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        self.last_obs = obs
        any_tensor = next(iter(obs.values()))
        B = any_tensor.shape[0]
        return th.ones((B, 1), device=any_tensor.device, dtype=any_tensor.dtype)


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
        logit_temperature: float = 5.0,
        noop_init: float = -1.0,
        freeze_noop_logit: bool = False,
        *args,
        **kwargs,
    ):
        assert isinstance(action_space, spaces.MultiDiscrete), "RTGNNPolicy requires MultiDiscrete action space"

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=DictPassthroughExtractor,
            features_extractor_kwargs={},
            *args,
            **kwargs,
        )

        # MultiDiscrete([K+1] * R)
        self.R = int(len(action_space.nvec))
        self.Kp1 = int(action_space.nvec[0])
        self.K = self.Kp1 - 1
        if int(k_max) != self.K:
            raise ValueError(f"k_max mismatch: action space implies K={self.K}, got k_max={k_max}")
        self.noop_index = self.K

        self.model = RTActorCritic(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            gnn_type=gnn_type,
            num_gnn_layers=num_gnn_layers,
            activation=activation,
            dropout=dropout,
        )

        # shared scalar NOOP logit
        self.noop_logit = nn.Parameter(
            th.tensor(float(noop_init), dtype=th.float32),
            requires_grad=(not freeze_noop_logit),
        )

        self.logit_temperature = float(logit_temperature) if logit_temperature is not None else 1.0

        # SB3 expects these modules to exist
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()

        # Build optimizer etc.
        self._build(lr_schedule)

        # Ensure PPO optimizer updates our model params (+ noop if not frozen)
        extra_params = list(self.model.parameters())
        if self.noop_logit.requires_grad:
            extra_params.append(self.noop_logit)

        if hasattr(self, "optimizer") and self.optimizer is not None and len(extra_params) > 0:
            existing = set()
            for g in self.optimizer.param_groups:
                for p in g.get("params", []):
                    existing.add(id(p))
            new_params = [p for p in extra_params if id(p) not in existing]
            if new_params:
                self.optimizer.add_param_group({"params": new_params})

    # ---------------- helpers ----------------

    def _get_cand_mask(self, obs_dict_b: Dict[str, th.Tensor]) -> th.Tensor:
        cand_mask = obs_dict_b.get("cand_mask", None)
        if cand_mask is not None:
            return cand_mask

        action_mask = obs_dict_b.get("action_mask", None)
        if action_mask is None:
            raise KeyError("Observation must contain 'cand_mask' or 'action_mask'.")
        if action_mask.dtype != th.bool:
            action_mask = action_mask.bool()
        return action_mask[..., :-1]

    def _append_noop(self, logits_k: th.Tensor, cand_mask: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        if cand_mask.dtype != th.bool:
            cand_mask = cand_mask.bool()

        B, R, _K = logits_k.shape
        noop_col = self.noop_logit.to(device=logits_k.device, dtype=logits_k.dtype).expand(B, R, 1)
        logits_full = th.cat([logits_k, noop_col], dim=-1)

        ones = th.ones((B, R, 1), dtype=th.bool, device=cand_mask.device)
        mask_full = th.cat([cand_mask, ones], dim=-1)
        return logits_full, mask_full

    @staticmethod
    def masked_logprob_entropy(
        logits: th.Tensor,   # [B,R,K+1] masked with -1e9 for invalid
        actions: th.Tensor,  # [B,R]
        active: th.Tensor,   # [B,R] bool
    ) -> Tuple[th.Tensor, th.Tensor]:
        logp = th.log_softmax(logits, dim=-1)                 # [B,R,K+1]
        a = actions.long().unsqueeze(-1)                      # [B,R,1]
        chosen_logp = logp.gather(-1, a).squeeze(-1)          # [B,R]

        p = th.softmax(logits, dim=-1)
        ent = -th.sum(p * logp, dim=-1)                       # [B,R]

        active_f = active.to(dtype=chosen_logp.dtype)
        chosen_logp = chosen_logp * active_f
        ent = ent * active_f

        return chosen_logp.sum(dim=1), ent.sum(dim=1)

    def _dist_from_logits(self, logits: th.Tensor):
        B = logits.shape[0]
        logits_flat = logits.reshape(B, -1)  # [B, R*(K+1)]
        return self.action_dist.proba_distribution(action_logits=logits_flat)

    def _build_logits_and_value(self, obs_dict_b: Dict[str, th.Tensor]) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        x = obs_dict_b["node_features"]
        edge_index = obs_dict_b["edge_index"]
        num_nodes = obs_dict_b["num_nodes"]
        num_edges = obs_dict_b["num_edges"]
        cand_node_idx = obs_dict_b["cand_node_idx"]

        cand_mask = self._get_cand_mask(obs_dict_b)

        logits_k, values = self.model(x, edge_index, num_nodes, num_edges, cand_node_idx)  # [B,R,K], [B,1]

        # temperature scaling on candidate logits ONLY
        if self.logit_temperature and self.logit_temperature != 1.0:
            logits_k = logits_k / float(self.logit_temperature)

        logits_full, mask_full = self._append_noop(logits_k, cand_mask)
        logits_full = logits_full.masked_fill(~mask_full, -1e9)

        active = cand_mask.bool().any(dim=-1)  # [B,R]
        return logits_full, values, active

    # ---------------- SB3 API ----------------

    def forward(self, obs: Any, deterministic: bool = False):
        _ = self.extract_features(obs, features_extractor=self.features_extractor)
        obs_dict_b = cast(Dict[str, th.Tensor], self.features_extractor.last_obs)
        assert obs_dict_b is not None, "Features extractor did not capture obs dict"

        logits, values, active = self._build_logits_and_value(obs_dict_b)
        dist = self._dist_from_logits(logits)
        actions = dist.get_actions(deterministic=deterministic)  # [B,R]
        log_prob, _entropy = self.masked_logprob_entropy(logits, actions, active)
        return actions, values, log_prob

    def evaluate_actions(self, obs: Any, actions: th.Tensor):
        _ = self.extract_features(obs, features_extractor=self.features_extractor)
        obs_dict_b = cast(Dict[str, th.Tensor], self.features_extractor.last_obs)
        assert obs_dict_b is not None, "Features extractor did not capture obs dict"

        logits, values, active = self._build_logits_and_value(obs_dict_b)
        dist = self._dist_from_logits(logits)

        # SB3 expects log_prob [B]; we return dist.log_prob for compatibility,
        # but entropy is our active-only entropy
        log_prob = dist.log_prob(actions)
        _lp_sum, entropy_sum = self.masked_logprob_entropy(logits, actions, active)
        return values, log_prob, entropy_sum

    def predict_values(self, obs: Any) -> th.Tensor:
        _ = self.extract_features(obs, features_extractor=self.features_extractor)
        obs_dict_b = cast(Dict[str, th.Tensor], self.features_extractor.last_obs)
        assert obs_dict_b is not None

        x = obs_dict_b["node_features"]
        edge_index = obs_dict_b["edge_index"]
        num_nodes = obs_dict_b["num_nodes"]
        num_edges = obs_dict_b["num_edges"]
        cand_node_idx = obs_dict_b["cand_node_idx"]
        _logits_k, values = self.model(x, edge_index, num_nodes, num_edges, cand_node_idx)
        return values

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        actions, _, _ = self.forward(observation, deterministic=deterministic)
        return actions