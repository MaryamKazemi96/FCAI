"""
Actor-Critic model for robot-task allocation.

- Runs a GNN over the global graph -> node embeddings
- Scores candidate tasks per robot using robot embedding + candidate embedding
- Outputs logits [B, R, K] (NOOP appended in SB3 wrapper)
- Outputs value [B, 1]
"""
from __future__ import annotations
import torch as th
import torch.nn as nn
from typing import Tuple

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
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        self.gnn = build_gnn_backbone(
            gnn_type=gnn_type,
            input_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            activation=activation,
            dropout=dropout,
        )

        # score(r,k) = MLP([h_robot[r], h_task[cand_idx]])
        self.score_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, score_hidden),
            nn.ReLU(),
            nn.Linear(score_hidden, 1),
        )

        # simple state-value head from mean pooled robot embeddings
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, score_hidden),
            nn.ReLU(),
            nn.Linear(score_hidden, 1),
        )

    def forward(
        self,
        x: th.Tensor,              # [B, N_max, F]
        edge_index: th.Tensor,     # [B, 2, E_max]
        num_nodes: th.Tensor,      # [B, 1]
        num_edges: th.Tensor,      # [B, 1]
        cand_node_idx: th.Tensor,  # [B, R, K] values in [-1..N_max-1]
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Returns:
            logits_k: [B, R, K] (NOOP not included)
            values:   [B, 1]
        """
        B, Nmax, _F = x.shape
        _, R, K = cand_node_idx.shape

        # Compute node embeddings for each graph in batch separately (because num_nodes/num_edges vary)
        logits_out = []
        values_out = []

        for b in range(B):
            n = int(num_nodes[b, 0].item())
            e = int(num_edges[b, 0].item())

            if n <= 0 or e <= 0:
                logits_out.append(th.zeros((R, K), device=x.device))
                values_out.append(th.zeros((1,), device=x.device))
                continue

            xb = x[b, :n, :].float()                 # [n, F]
            eb = edge_index[b, :, :e].long()         # [2, e]

            hb = self.gnn(xb, eb)                    # [n, H]
            h_robot = hb[:R, :]                      # [R, H]

            # value from mean pooled robot embeddings
            v = self.value_head(h_robot.mean(dim=0, keepdim=True)).squeeze(0)  # [1]
            values_out.append(v)

            # gather candidate embeddings per robot/slot
            cidx = cand_node_idx[b].long()           # [R, K]
            # clamp invalid to 0 to avoid gather crash; they will be masked later
            cidx_safe = th.clamp(cidx, min=0)

            # hb[cidx_safe] -> [R, K, H]
            h_cand = hb[cidx_safe.view(-1), :].view(R, K, self.hidden_dim)

            # expand robot embeddings -> [R, K, H]
            h_r = h_robot.unsqueeze(1).expand(R, K, self.hidden_dim)

            # score
            pair = th.cat([h_r, h_cand], dim=-1)     # [R, K, 2H]
            scores = self.score_mlp(pair).squeeze(-1)  # [R, K]

            logits_out.append(scores)

        logits_k = th.stack(logits_out, dim=0)       # [B, R, K]
        values = th.stack(values_out, dim=0)         # [B, 1]
        return logits_k, values