# """
# Actor-Critic model for robot-task allocation.

# - Runs a GNN over the global graph -> node embeddings
# - Scores candidate tasks per robot using robot embedding + candidate embedding
# - Outputs logits [B, R, K] (NOOP appended in SB3 wrapper)
# - Outputs value [B, 1]
# """
# from __future__ import annotations
# import torch as th
# import torch.nn as nn
# from typing import Tuple

# from .gnn_backbone import build_gnn_backbone


# class RTActorCritic(nn.Module):
#     def __init__(
#         self,
#         in_dim: int,
#         hidden_dim: int,
#         gnn_type: str = "graphsage",
#         num_gnn_layers: int = 2,
#         activation: str = "relu",
#         dropout: float = 0.0,
#         score_hidden: int = 128,
#     ):
#         super().__init__()
#         self.hidden_dim = int(hidden_dim)

#         self.gnn = build_gnn_backbone(
#             gnn_type=gnn_type,
#             input_dim=in_dim,
#             hidden_dim=hidden_dim,
#             num_layers=num_gnn_layers,
#             activation=activation,
#             dropout=dropout,
#         )

#         # score(r,k) = MLP([h_robot[r], h_task[cand_idx]])
#         self.score_mlp = nn.Sequential(
#             nn.Linear(2 * hidden_dim, score_hidden),
#             nn.ReLU(),
#             nn.Linear(score_hidden, 1),
#         )

#         # simple state-value head from mean pooled robot embeddings
#         self.value_head = nn.Sequential(
#             nn.Linear(hidden_dim, score_hidden),
#             nn.ReLU(),
#             nn.Linear(score_hidden, 1),
#         )

#     def forward(
#         self,
#         x: th.Tensor,              # [B, N_max, F]
#         edge_index: th.Tensor,     # [B, 2, E_max]
#         num_nodes: th.Tensor,      # [B, 1]
#         num_edges: th.Tensor,      # [B, 1]
#         cand_node_idx: th.Tensor,  # [B, R, K] values in [-1..N_max-1]
#     ) -> Tuple[th.Tensor, th.Tensor]:
#         """
#         Returns:
#             logits_k: [B, R, K] (NOOP not included)
#             values:   [B, 1]
#         """
#         B, Nmax, _F = x.shape
#         _, R, K = cand_node_idx.shape

#         # Compute node embeddings for each graph in batch separately (because num_nodes/num_edges vary)
#         logits_out = []
#         values_out = []

#         for b in range(B):
#             n = int(num_nodes[b, 0].item())
#             e = int(num_edges[b, 0].item())

#             if n <= 0 or e <= 0:
#                 logits_out.append(th.zeros((R, K), device=x.device))
#                 values_out.append(th.zeros((1,), device=x.device))
#                 continue

#             xb = x[b, :n, :].float()                 # [n, F]
#             eb = edge_index[b, :, :e].long()         # [2, e]

#             hb = self.gnn(xb, eb)                    # [n, H]
#             h_robot = hb[:R, :]                      # [R, H]

#             # value from mean pooled robot embeddings
#             v = self.value_head(h_robot.mean(dim=0, keepdim=True)).squeeze(0)  # [1]
#             values_out.append(v)

#             # gather candidate embeddings per robot/slot
#             cidx = cand_node_idx[b].long()           # [R, K]
#             # clamp invalid to 0 to avoid gather crash; they will be masked later
#             cidx_safe = th.clamp(cidx, min=0)

#             # hb[cidx_safe] -> [R, K, H]
#             h_cand = hb[cidx_safe.view(-1), :].view(R, K, self.hidden_dim)

#             # expand robot embeddings -> [R, K, H]
#             h_r = h_robot.unsqueeze(1).expand(R, K, self.hidden_dim)

#             # score
#             pair = th.cat([h_r, h_cand], dim=-1)     # [R, K, 2H]
#             scores = self.score_mlp(pair).squeeze(-1)  # [R, K]

#             logits_out.append(scores)

#         logits_k = th.stack(logits_out, dim=0)       # [B, R, K]
#         values = th.stack(values_out, dim=0)         # [B, 1]
#         return logits_k, values




#------------------------this verion handel NOOp and replace batch procession
# ---------------------------------------------------------------------
# src/models/actor_critic.py
# """
# Actor-Critic model for robot-task allocation.

# Changes vs your current version (to help logit separation / deterministic behavior):
# - Optional robot-id embedding added to robot node embeddings (breaks symmetry).
# - Candidate scoring uses a bilinear-style score (q_r^T k_c) with temperature,
#   which tends to produce clearer margins than concat-MLP.
# - Optional L2 normalization on q/k vectors.
# - Value head uses mean pooled robot embeddings (as before) but with a slightly stronger head.

# Shapes:
#   logits_k: [B, R, K]
#   values:   [B, 1]
# """

# from __future__ import annotations

# from typing import Tuple

# import torch as th
# import torch.nn as nn
# import torch.nn.functional as F

# from .gnn_backbone import build_gnn_backbone


# class RTActorCritic(nn.Module):
#     def __init__(
#         self,
#         in_dim: int,
#         hidden_dim: int,
#         gnn_type: str = "graphsage",
#         num_gnn_layers: int = 2,
#         activation: str = "relu",
#         dropout: float = 0.0,
#         # scoring
#         score_hidden: int = 128,
#         score_temperature: float = 1.0,  # separate from policy logit_temperature; keep 1.0 initially
#         normalize_qk: bool = True,
#         # symmetry breaking
#         use_robot_id_embed: bool = True,
#         max_robots: int = 64,
#         robot_id_embed_dim: int = 16,
#         # backbone extras
#         norm_type: str = "layer",
#         add_self_loops_flag: bool = True,
#         residual: bool = True,
#     ):
#         super().__init__()
#         self.hidden_dim = int(hidden_dim)
#         self.score_temperature = float(score_temperature)
#         self.normalize_qk = bool(normalize_qk)
#         self.use_robot_id_embed = bool(use_robot_id_embed)
#         self.max_robots = int(max_robots)
#         self.robot_id_embed_dim = int(robot_id_embed_dim)

#         self.gnn = build_gnn_backbone(
#             gnn_type=gnn_type,
#             input_dim=in_dim,
#             hidden_dim=hidden_dim,
#             num_layers=num_gnn_layers,
#             activation=activation,
#             dropout=dropout,
#             norm_type=norm_type,
#             add_self_loops_flag=add_self_loops_flag,
#             residual=residual,
#         )

#         # Optional robot id embedding to break symmetry.
#         # Added only to robot embeddings (first R nodes assumed robots).
#         if self.use_robot_id_embed:
#             self.robot_id_emb = nn.Embedding(self.max_robots, self.robot_id_embed_dim)
#             self.robot_id_proj = nn.Linear(self.robot_id_embed_dim, hidden_dim)
#         else:
#             self.robot_id_emb = None
#             self.robot_id_proj = None

#         # Project robot/task embeddings into query/key spaces
#         self.q_proj = nn.Sequential(
#             nn.Linear(hidden_dim, score_hidden),
#             nn.ReLU(),
#             nn.Linear(score_hidden, score_hidden),
#         )
#         self.k_proj = nn.Sequential(
#             nn.Linear(hidden_dim, score_hidden),
#             nn.ReLU(),
#             nn.Linear(score_hidden, score_hidden),
#         )

#         # Value head (slightly stronger)
#         self.value_head = nn.Sequential(
#             nn.Linear(hidden_dim, score_hidden),
#             nn.ReLU(),
#             nn.Linear(score_hidden, score_hidden),
#             nn.ReLU(),
#             nn.Linear(score_hidden, 1),
#         )

#     def forward(
#         self,
#         x: th.Tensor,              # [B, N_max, F]
#         edge_index: th.Tensor,     # [B, 2, E_max]
#         num_nodes: th.Tensor,      # [B, 1]
#         num_edges: th.Tensor,      # [B, 1]
#         cand_node_idx: th.Tensor,  # [B, R, K] values in [-1..N_max-1]
#     ) -> Tuple[th.Tensor, th.Tensor]:
#         B, Nmax, _F = x.shape
#         _, R, K = cand_node_idx.shape

#         logits_out = []
#         values_out = []

#         device = x.device

#         # robot ids assumed 0..R-1 within each sample (consistent with your slicing hb[:R])
#         if self.use_robot_id_embed:
#             rid = th.arange(R, device=device).clamp_(0, self.max_robots - 1)  # [R]
#             rid_emb = self.robot_id_proj(self.robot_id_emb(rid))  # [R,H]
#         else:
#             rid_emb = None

#         for b in range(B):
#             n = int(num_nodes[b, 0].item())
#             e = int(num_edges[b, 0].item())

#             if n <= 0 or e <= 0:
#                 logits_out.append(th.zeros((R, K), device=device))
#                 values_out.append(th.zeros((1,), device=device))
#                 continue

#             xb = x[b, :n, :].float()         # [n,F]
#             eb = edge_index[b, :, :e].long() # [2,e]

#             hb = self.gnn(xb, eb)            # [n,H]
#             h_robot = hb[:R, :]              # [R,H]

#             # Symmetry breaking: inject robot id embedding
#             if rid_emb is not None:
#                 h_robot = h_robot + rid_emb

#             # Value = head(mean robot embedding)
#             v = self.value_head(h_robot.mean(dim=0, keepdim=True)).squeeze(0)  # [1]
#             values_out.append(v)

#             # Candidates: [R,K]
#             cidx = cand_node_idx[b].long()
#             cidx_safe = th.clamp(cidx, min=0)  # invalid will be masked later
#             h_cand = hb[cidx_safe.view(-1), :].view(R, K, self.hidden_dim)  # [R,K,H]

#             # Project to q/k
#             q = self.q_proj(h_robot)                  # [R,D]
#             k = self.k_proj(h_cand)                   # [R,K,D]

#             if self.normalize_qk:
#                 q = F.normalize(q, p=2, dim=-1)       # [R,D]
#                 k = F.normalize(k, p=2, dim=-1)       # [R,K,D]

#             # Bilinear-ish score: q dot k (per robot)
#             # logits[r,k] = <q[r], k[r,k,:]>
#             scores = th.einsum("rd,rkd->rk", q, k)     # [R,K]

#             # Optional temperature for sharper separation (>1 makes softer; <1 sharper)
#             if self.score_temperature and self.score_temperature != 1.0:
#                 scores = scores / float(self.score_temperature)

#             logits_out.append(scores)

#         logits_k = th.stack(logits_out, dim=0)  # [B,R,K]
#         values = th.stack(values_out, dim=0)    # [B,1]
#         return logits_k, values

# ------------------------this version replace global graph handeling with ego graph----------------
# ---------------------------------------------------------------------------------------------

"""
Actor-Critic for per-robot ego-graphs.

Input shapes (from wrapper, with SB3 batch dim):
  x:          [B, R, N_ego_max, F]
  edge_index: [B, R, 2, E_ego_max]
  num_nodes:  [B, R, 1]
  num_edges:  [B, R, 1]
  cand_idx:   [B, R, K]   (LOCAL indices 0..N_ego_max-1, or -1)

Outputs:
  logits_k: [B, R, K]
  values:   [B, 1]  (mean over robots of per-robot value embeddings)
"""

from __future__ import annotations

from typing import Tuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .gnn_backbone import build_gnn_backbone


class RTActorCritic(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        gnn_type: str = "graphsage",
        num_gnn_layers: int = 2,
        activation: str = "relu",
        dropout: float = 0.0,
        score_hidden: int = 128,
        normalize_qk: bool = True,
        score_temperature: float = 1.0,
        use_robot_id_embed: bool = True,
        max_robots: int = 64,
        robot_id_embed_dim: int = 16,
        # backbone options
        norm_type: str = "layer",
        add_self_loops_flag: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.normalize_qk = bool(normalize_qk)
        self.score_temperature = float(score_temperature)
        self.use_robot_id_embed = bool(use_robot_id_embed)
        self.max_robots = int(max_robots)
        self.robot_id_embed_dim = int(robot_id_embed_dim)

        self.gnn = build_gnn_backbone(
            gnn_type=gnn_type,
            input_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            activation=activation,
            dropout=dropout,
            norm_type=norm_type,
            add_self_loops_flag=add_self_loops_flag,
            residual=residual,
        )

        if self.use_robot_id_embed:
            self.robot_id_emb = nn.Embedding(self.max_robots, self.robot_id_embed_dim)
            self.robot_id_proj = nn.Linear(self.robot_id_embed_dim, hidden_dim)
        else:
            self.robot_id_emb = None
            self.robot_id_proj = None

        # query/key scoring (sharper separation than concat MLP)
        self.q_proj = nn.Sequential(
            nn.Linear(hidden_dim, score_hidden),
            nn.ReLU(),
            nn.Linear(score_hidden, score_hidden),
        )
        self.k_proj = nn.Sequential(
            nn.Linear(hidden_dim, score_hidden),
            nn.ReLU(),
            nn.Linear(score_hidden, score_hidden),
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, score_hidden),
            nn.ReLU(),
            nn.Linear(score_hidden, score_hidden),
            nn.ReLU(),
            nn.Linear(score_hidden, 1),
        )

    # def forward(
    #     self,
    #     x: th.Tensor,              # [B, R, N, F]
    #     edge_index: th.Tensor,     # [B, R, 2, E]
    #     num_nodes: th.Tensor,      # [B, R, 1]
    #     num_edges: th.Tensor,      # [B, R, 1]
    #     cand_node_idx: th.Tensor,  # [B, R, K]
    # ) -> Tuple[th.Tensor, th.Tensor]:
    #     B, R, N, Fdim = x.shape
    #     _, _, _, E = edge_index.shape
    #     _, _, K = cand_node_idx.shape

    #     device = x.device

    #     # robot-id embeddings (shared across batch)
    #     if self.use_robot_id_embed:
    #         rid = th.arange(R, device=device).clamp(0, self.max_robots - 1)  # [R]
    #         rid_emb = self.robot_id_proj(self.robot_id_emb(rid))  # [R,H]
    #     else:
    #         rid_emb = None

    #     logits_out = th.zeros((B, R, K), device=device, dtype=th.float32)
    #     values_out = th.zeros((B, 1), device=device, dtype=th.float32)

    #     # Loop is simplest + matches varying n/e per ego graph.
    #     for b in range(B):
    #         robot_values = []
    #         for r in range(R):
    #             n = int(num_nodes[b, r, 0].item())
    #             e = int(num_edges[b, r, 0].item())

    #             if n <= 0 or e < 0:
    #                 # leave zeros
    #                 robot_values.append(th.zeros((1,), device=device))
    #                 continue

    #             xb = x[b, r, :n, :].float()                  # [n,F]
    #             eb = edge_index[b, r, :, :e].long() if e > 0 else th.zeros((2, 0), device=device, dtype=th.long)

    #             hb = self.gnn(xb, eb)                        # [n,H]

    #             # choose robot embedding = node 0 by convention (because wrapper ensured robot global node included first when possible)
    #             # If your ego graph does not guarantee robot is local 0, we need to pass robot_node_idx.
    #             h_robot = hb[0:1, :].squeeze(0) if hb.size(0) > 0 else th.zeros((self.hidden_dim,), device=device)

    #             if rid_emb is not None:
    #                 h_robot = h_robot + rid_emb[r]

    #             # value per robot
    #             v_r = self.value_head(h_robot.unsqueeze(0)).squeeze(0)  # [1]
    #             robot_values.append(v_r)

    #             # gather candidate embeddings
    #             cidx = cand_node_idx[b, r].long()            # [K]
    #             cidx_safe = th.clamp(cidx, min=0)

    #             # [K,H]
    #             h_cand = hb[cidx_safe, :] if hb.size(0) > 0 else th.zeros((K, self.hidden_dim), device=device)

    #             # project
    #             q = self.q_proj(h_robot)                      # [D]
    #             kvec = self.k_proj(h_cand)                    # [K,D]

    #             if self.normalize_qk:
    #                 q = F.normalize(q, p=2, dim=-1)
    #                 kvec = F.normalize(kvec, p=2, dim=-1)

    #             # dot scores [K]
    #             scores = th.einsum("d,kd->k", q, kvec)

    #             if self.score_temperature and self.score_temperature != 1.0:
    #                 scores = scores / float(self.score_temperature)

    #             logits_out[b, r, :] = scores

    #         # aggregate values across robots (mean)
    #         rv = th.stack(robot_values, dim=0)               # [R,1] effectively
    #         values_out[b, 0] = rv.mean()

    #     return logits_out, values_out
    def forward(self, x, edge_index, num_nodes, num_edges, cand_node_idx):
        B, R, N, Fdim = x.shape
        _, _, _, E = edge_index.shape
        _, _, K = cand_node_idx.shape
        device = x.device

        # robot-id embeddings (shared across batch)
        if self.use_robot_id_embed:
            rid = th.arange(R, device=device).clamp(0, self.max_robots - 1)
            rid_emb = self.robot_id_proj(self.robot_id_emb(rid))  # [R,H]
        else:
            rid_emb = None

        logits_batches = []
        values_batches = []

        for b in range(B):
            robot_logits = []
            robot_values = []

            for r in range(R):
                n = int(num_nodes[b, r, 0].item())
                e = int(num_edges[b, r, 0].item())

                if n <= 0:
                    robot_logits.append(th.zeros((K,), device=device))
                    robot_values.append(th.zeros((1,), device=device))
                    continue

                xb = x[b, r, :n, :].float()  # [n,F]
                eb = edge_index[b, r, :, :e].long() if e > 0 else th.zeros((2, 0), device=device, dtype=th.long)

                hb = self.gnn(xb, eb)  # [n,H]
                h_robot = hb[0] if hb.size(0) > 0 else th.zeros((self.hidden_dim,), device=device)

                if rid_emb is not None:
                    h_robot = h_robot + rid_emb[r]

                v_r = self.value_head(h_robot.unsqueeze(0)).squeeze(0)  # [1]
                robot_values.append(v_r)

                cidx = cand_node_idx[b, r].long()      # [K]
                cidx_safe = th.clamp(cidx, min=0)
                h_cand = hb[cidx_safe, :] if hb.size(0) > 0 else th.zeros((K, self.hidden_dim), device=device)

                q = self.q_proj(h_robot)               # [D]
                kvec = self.k_proj(h_cand)             # [K,D]

                if self.normalize_qk:
                    q = F.normalize(q, p=2, dim=-1)
                    kvec = F.normalize(kvec, p=2, dim=-1)

                scores = th.einsum("d,kd->k", q, kvec)  # [K]
                if self.score_temperature and self.score_temperature != 1.0:
                    scores = scores / float(self.score_temperature)

                robot_logits.append(scores)

            logits_b = th.stack(robot_logits, dim=0)      # [R,K]
            values_b = th.stack(robot_values, dim=0).mean().unsqueeze(0)  # [1]
            logits_batches.append(logits_b)
            values_batches.append(values_b)

        logits_k = th.stack(logits_batches, dim=0)  # [B,R,K]
        values = th.stack(values_batches, dim=0)    # [B,1]
        return logits_k, values
    # ---------------------------------------------------
    # -----------------------------End of this version-----------