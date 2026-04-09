# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, global_mean_pool

# class ActorGNN(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(ActorGNN, self).__init__()
#         self.conv1 = GCNConv(input_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.actor_norm = nn.LayerNorm(hidden_dim)
#         self.actor_head = nn.Linear(hidden_dim, 1)  # Per-node score

#     def forward(self, x, edge_index):
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.relu(self.conv2(x, edge_index))
#         x = self.actor_norm(x)
#         scores = self.actor_head(x).squeeze(-1)  # [num_nodes]
#         return scores


# class CriticGNN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, critic_aggregation="joint_mean"):
#         super(CriticGNN, self).__init__()
#         self.critic_conv1 = GCNConv(input_dim, hidden_dim)
#         self.critic_conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.critic_head = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1)
#         )
#         self.critic_aggregation = critic_aggregation
#         if self.critic_aggregation == "joint_attn":
#             self.attn_score = nn.Linear(hidden_dim, 1)  # Attention weights for robots

#     def forward(self, x, edge_index, batch):
#         x = F.relu(self.critic_conv1(x, edge_index))
#         x = F.relu(self.critic_conv2(x, edge_index))
#         robot_embeds = global_mean_pool(x, batch)  # [num_robots, hidden_dim]

#         if self.critic_aggregation == "per_robot":
#             values = self.critic_head(robot_embeds).squeeze(-1)  # [num_robots]
#         elif self.critic_aggregation == "joint_mean":
#             global_embed = robot_embeds.mean(dim=0, keepdim=True)  # [1, hidden_dim]
#             values = self.critic_head(global_embed).squeeze()  # Scalar
#         elif self.critic_aggregation == "joint_attn":
#             attn_weights = torch.softmax(self.attn_score(robot_embeds).squeeze(-1), dim=0)  # [num_robots]
#             global_embed = (attn_weights.unsqueeze(-1) * robot_embeds).sum(dim=0, keepdim=True)  # [1, hidden_dim]
#             values = self.critic_head(global_embed).squeeze()  # Scalar
#         else:
#             raise ValueError(f"Unknown critic_aggregation '{self.critic_aggregation}'")
#         return values

# (patched CriticGNN.forward)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class ActorGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ActorGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.actor_norm = nn.LayerNorm(hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, 1)  # Per-node score

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.actor_norm(x)
        scores = self.actor_head(x).squeeze(-1)  # [num_nodes]
        return scores


class CriticGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, critic_aggregation="joint_mean"):
        super(CriticGNN, self).__init__()
        self.critic_conv1 = GCNConv(input_dim, hidden_dim)
        self.critic_conv2 = GCNConv(hidden_dim, hidden_dim)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.critic_aggregation = critic_aggregation
        if self.critic_aggregation == "joint_attn":
            self.attn_score = nn.Linear(hidden_dim, 1)  # Attention weights for robots

    def forward(self, x, edge_index, batch=None, num_robots=None):
        """
        Args:
            x: node features [num_nodes, feat]
            edge_index: edge index for the whole graph
            batch: (unused) kept for compatibility
            num_robots: int, number of robot nodes at the start of x.
                        Required for 'per_robot' and 'joint_attn' modes.
        Returns:
            - If critic_aggregation == 'per_robot': Tensor [num_robots] (value per robot)
            - Else: scalar tensor (global value)
        """
        x = F.relu(self.critic_conv1(x, edge_index))
        x = F.relu(self.critic_conv2(x, edge_index))

        # If we want per-robot values, assume the first `num_robots` rows of x correspond to robots
        if self.critic_aggregation == "per_robot":
            if num_robots is None:
                raise ValueError("num_robots must be provided for 'per_robot' critic aggregation")
            robot_embeds = x[:num_robots]  # [num_robots, hidden_dim]
            values = self.critic_head(robot_embeds).squeeze(-1)  # [num_robots]
            return values

        elif self.critic_aggregation == "joint_mean":
            # Global scalar value
            global_embed = x.mean(dim=0, keepdim=True)  # [1, hidden_dim]
            value = self.critic_head(global_embed).squeeze()  # scalar
            return value

        elif self.critic_aggregation == "joint_attn":
            if num_robots is None:
                raise ValueError("num_robots must be provided for 'joint_attn' critic aggregation")
            robot_embeds = x[:num_robots]  # [num_robots, hidden_dim]
            attn_weights = torch.softmax(self.attn_score(robot_embeds).squeeze(-1), dim=0)  # [num_robots]
            global_embed = (attn_weights.unsqueeze(-1) * robot_embeds).sum(dim=0, keepdim=True)  # [1, hidden_dim]
            value = self.critic_head(global_embed).squeeze()
            return value

        else:
            raise ValueError(f"Unknown critic_aggregation '{self.critic_aggregation}'")