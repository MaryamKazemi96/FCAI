# import torch
# import numpy as np
# import time

# def train(env, num_episodes, actors, critic,
#           optimizers_actors, optimizer_critic, gamma=0.99):

#     episode_rewards = []

#     for episode in range(num_episodes):
#         obs, _ = env.reset()
#         ego_graphs, attribute_matrix = obs

#         done = False
#         episode_reward = 0
#         start = time.time()

#         # while not done:
#         for i in range(40):
#             actions = {}
#             log_probs = {}
#             value_preds = {}

#             x = torch.tensor(attribute_matrix, dtype=torch.float)

#             # ---------- ACTOR + CRITIC ----------
#             for rid, ego_list in ego_graphs.items():

#                 # empty ego graph → no available tasks
#                 if len(ego_list) == 0:
#                     actions[rid] = []
#                     continue

#                 edge_index = torch.tensor(
#                     np.concatenate(ego_list, axis=0),
#                     dtype=torch.long
#                 ).t().contiguous()

#                 batch = torch.full((x.size(0),), rid, dtype=torch.long)

#                 logits = actors[rid](x, edge_index)

#                 task_nodes = torch.arange(env.n_robots,
#                                           env.n_robots + env.n_tasks)

#                 task_logits = logits[task_nodes]
#                 probs = torch.softmax(task_logits, dim=-1)

#                 top_idx = torch.argmax(probs)
#                 task_id = task_nodes[top_idx].item()

#                 actions[rid] = [task_id]
#                 log_probs[rid] = torch.log(probs[top_idx] + 1e-8)

#                 value_preds[rid] = critic(x, edge_index, batch)

#             # ---------- ENV STEP ----------
#             next_obs, reward, done, truncated, _ = env.step(actions)
#             ego_graphs, attribute_matrix = next_obs

#             if isinstance(reward, dict):
#                 episode_reward += sum(reward.values())
#             else:
#                 episode_reward += reward

#             # ---------- ADVANTAGES ----------
#             advantages = {}
#             for rid, r in reward.items():
#                 if rid not in value_preds:
#                     continue
#                 advantages[rid] = r + gamma * value_preds[rid].detach() - value_preds[rid]

#             # ---------- CRITIC UPDATE ----------
#             if advantages:
#                 optimizer_critic.zero_grad()
#                 critic_loss = sum(
#                     (reward[rid] - value_preds[rid]) ** 2
#                     for rid in advantages
#                 )
#                 critic_loss.backward()
#                 optimizer_critic.step()

#                 # ---------- ACTOR UPDATE ----------
#                 for rid in advantages:
#                     if rid not in log_probs:
#                         continue
#                     optimizers_actors[rid].zero_grad()
#                     actor_loss = -log_probs[rid] * advantages[rid].detach()
#                     actor_loss.backward()
#                     optimizers_actors[rid].step()

#         print(f"Episode {episode+1}/{num_episodes} "
#               f"Reward={episode_reward:.2f} "
#               f"Time={time.time()-start:.2f}s")

#         episode_rewards.append(episode_reward)

# #     return episode_rewards





# import torch
# import numpy as np
# import time

# def _build_edge_index_from_ego_list(ego_list):
#     """Return torch edge_index (2, E) for given ego_list (list of arrays)."""
#     if len(ego_list) == 0:
#         return torch.empty((2, 0), dtype=torch.long)
#     return torch.tensor(np.concatenate(ego_list, axis=0), dtype=torch.long).t().contiguous()

# def train(env, num_episodes, actors, critic,
#           optimizers_actors, optimizer_critic, gamma=0.99,
#           max_steps_per_episode=200, device=None):

#     episode_rewards = []

#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Put models on device (assumes dict-like actors)
#     for a in actors.values():
#         a.to(device)
#     critic.to(device)

#     for episode in range(num_episodes):
#         obs, _ = env.reset()
#         ego_graphs, attribute_matrix = obs

#         done = False
#         episode_reward = 0.0
#         start = time.time()

#         step = 0
#         while (not done) and step < max_steps_per_episode:
#             actions = {}
#             log_probs = {}
#             value_preds = {}

#             x = torch.tensor(attribute_matrix, dtype=torch.float, device=device)

#             # ---------- ACTOR + CRITIC (current state) ----------
#             for rid, ego_list in ego_graphs.items():
#                 # empty ego graph → no available tasks
#                 if len(ego_list) == 0:
#                     actions[rid] = []
#                     continue

#                 edge_index = _build_edge_index_from_ego_list(ego_list).to(device)
#                 batch = torch.full((x.size(0),), rid, dtype=torch.long, device=device)

#                 logits = actors[rid](x, edge_index)  # assume returns per-node logits on device

#                 # Determine which task nodes are actually present in ego_list.
#                 # ego_list contains edges between node indices; extract task node indices from it.
#                 # Edges are pairs [u, v]; node indices appear in both rows.
#                 nodes_present = torch.unique(edge_index).cpu().numpy().tolist()
#                 # Keep only task nodes (>= n_robots)
#                 task_nodes_present = [n for n in nodes_present if n >= env.n_robots and n < env.n_robots + env.n_tasks]
#                 if len(task_nodes_present) == 0:
#                     actions[rid] = []
#                     continue

#                 task_nodes_tensor = torch.tensor(task_nodes_present, dtype=torch.long, device=device)
#                 task_logits = logits[task_nodes_tensor]
#                 probs = torch.softmax(task_logits, dim=-1)

#                 top_idx = torch.argmax(probs)
#                 task_node = task_nodes_tensor[top_idx].item()

#                 actions[rid] = [task_node]  # we send node index (n_robots + task_index)
#                 log_probs[rid] = torch.log(probs[top_idx] + 1e-8)

#                 # critic value for this robot (current state)
#                 value_preds[rid] = critic(x, edge_index, batch)

#             # ---------- ENV STEP ----------
#             next_obs, reward, done, truncated, _ = env.step(actions)
#             ego_graphs_next, attribute_matrix_next = next_obs

#             if isinstance(reward, dict):
#                 episode_reward += sum(reward.values())
#             else:
#                 episode_reward += reward

#             # ---------- Compute V_next (for bootstrap) ----------
#             # For each robot that we have a value_pred for, compute critic on next state
#             value_next = {}
#             x_next = torch.tensor(attribute_matrix_next, dtype=torch.float, device=device)
#             for rid, ego_list_next in ego_graphs_next.items():
#                 # only compute if we used this robot earlier (i.e., in value_preds)
#                 if rid not in value_preds:
#                     continue
#                 if len(ego_list_next) == 0:
#                     edge_index_next = torch.empty((2, 0), dtype=torch.long, device=device)
#                 else:
#                     edge_index_next = _build_edge_index_from_ego_list(ego_list_next).to(device)
#                 batch_next = torch.full((x_next.size(0),), rid, dtype=torch.long, device=device)
#                 # compute V_next; keep as tensor
#                 value_next[rid] = critic(x_next, edge_index_next, batch_next)

#             # ---------- ADVANTAGES & TARGETS ----------
#             advantages = {}
#             td_targets = {}
#             for rid, r in (reward.items() if isinstance(reward, dict) else enumerate([reward])):
#                 if rid not in value_preds:
#                     continue
#                 v_curr = value_preds[rid]
#                 # if next value exists, bootstrap; else terminal -> no bootstrap
#                 v_next = value_next.get(rid, None)
#                 done_or_trunc = float(done or truncated)  # treat both as terminal for bootstrap
#                 if v_next is not None:
#                     target = r + gamma * v_next.detach() * (1.0 - done_or_trunc)
#                 else:
#                     target = torch.tensor(r, dtype=v_curr.dtype, device=device)
#                 advantages[rid] = target - v_curr
#                 td_targets[rid] = target

#             # ---------- CRITIC UPDATE ----------
#             if td_targets:
#                 optimizer_critic.zero_grad()
#                 critic_loss = sum(((td_targets[rid] - value_preds[rid]) ** 2).mean()
#                                   for rid in td_targets)
#                 critic_loss.backward()
#                 optimizer_critic.step()

#                 # ---------- ACTOR UPDATE ----------
#                 for rid in list(advantages.keys()):
#                     if rid not in log_probs:
#                         continue
#                     optimizers_actors[rid].zero_grad()
#                     # If advantage is a tensor, detach for actor update
#                     adv = advantages[rid].detach()
#                     # log_probs[rid] is scalar tensor
#                     actor_loss = -(log_probs[rid] * adv).mean()
#                     actor_loss.backward()
#                     optimizers_actors[rid].step()

#             # advance
#             ego_graphs = ego_graphs_next
#             attribute_matrix = attribute_matrix_next
#             step += 1

#         print(f"Episode {episode+1}/{num_episodes} "
#               f"Reward={episode_reward:.2f} "
#               f"Time={time.time()-start:.2f}s")

#         episode_rewards.append(episode_reward)

#     return episode_rewards


# import torch
# import numpy as np
# import time

# def _build_edge_index_from_ego_list(ego_list):
#     if len(ego_list) == 0:
#         return torch.empty((2, 0), dtype=torch.long)
#     return torch.tensor(np.concatenate(ego_list, axis=0), dtype=torch.long).t().contiguous()

# def train(env, num_episodes, actors, critic,
#           optimizers_actors, optimizer_critic, gamma=0.99,
#           max_steps_per_episode=200, device=None):

#     episode_rewards = []

#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     for a in actors.values():
#         a.to(device)
#     critic.to(device)

#     for episode in range(num_episodes):
#         obs, _ = env.reset()
#         ego_graphs, attribute_matrix = obs

#         done = False
#         episode_total_sum = 0.0                # sum across robots and steps (as before)
#         per_robot_cumulative = np.zeros(env.n_robots, dtype=np.float32)  # track per-robot totals
#         start = time.time()

#         step = 0
#         while (not done) and step < max_steps_per_episode:
#             actions = {}
#             log_probs = {}
#             value_preds = {}

#             x = torch.tensor(attribute_matrix, dtype=torch.float, device=device)

#             # ---------- ACTOR + CRITIC (current state) ----------
#             for rid, ego_list in ego_graphs.items():
#                 if len(ego_list) == 0:
#                     actions[rid] = []
#                     continue

#                 edge_index = _build_edge_index_from_ego_list(ego_list).to(device)
#                 batch = torch.full((x.size(0),), rid, dtype=torch.long, device=device)

#                 logits = actors[rid](x, edge_index)

#                 nodes_present = torch.unique(edge_index).cpu().numpy().tolist()
#                 task_nodes_present = [n for n in nodes_present if n >= env.n_robots and n < env.n_robots + env.n_tasks]
#                 if len(task_nodes_present) == 0:
#                     actions[rid] = []
#                     continue

#                 task_nodes_tensor = torch.tensor(task_nodes_present, dtype=torch.long, device=device)
#                 # task_logits = logits[task_nodes_tensor]
#                 # probs = torch.softmax(task_logits, dim=-1)

#                 # top_idx = torch.argmax(probs)
#                 # task_node = task_nodes_tensor[top_idx].item()

#                 # actions[rid] = [task_node]
#                 # log_probs[rid] = torch.log(probs[top_idx] + 1e-8)
#                 task_logits = logits[task_nodes_tensor]
#                 probs = torch.softmax(task_logits, dim=-1)

#                 # sample from categorical policy
#                 dist = torch.distributions.Categorical(probs)
#                 sample_idx = dist.sample()                     # index into task_nodes_tensor (tensor)
#                 task_node = task_nodes_tensor[sample_idx].item()

#                 actions[rid] = [task_node]
#                 log_probs[rid] = dist.log_prob(sample_idx)     # scalar tensor
#                 entropies = entropies if 'entropies' in locals() else {}
#                 entropies[rid] = dist.entropy()

#                 value_preds[rid] = critic(x, edge_index, batch)

#             # ---------- ENV STEP ----------
#             # next_obs, reward, done, truncated, _ = env.step(actions)
#             next_obs, reward, done, truncated, _ = env.step(actions)
#             ego_graphs_next, attribute_matrix_next = next_obs

#             # accumulate episode rewards (same as before)
#             if isinstance(reward, dict):
#                 episode_total_sum += sum(reward.values())
#                 # update per-robot cumulative totals for diagnostics
#                 for rid, rv in reward.items():
#                     per_robot_cumulative[rid] += rv
#             else:
#                 episode_total_sum += reward

#             # ---------- Compute V_next ----------
#             value_next = {}
#             x_next = torch.tensor(attribute_matrix_next, dtype=torch.float, device=device)
#             for rid, ego_list_next in ego_graphs_next.items():
#                 if rid not in value_preds:
#                     continue
#                 if len(ego_list_next) == 0:
#                     edge_index_next = torch.empty((2, 0), dtype=torch.long, device=device)
#                 else:
#                     edge_index_next = _build_edge_index_from_ego_list(ego_list_next).to(device)
#                 batch_next = torch.full((x_next.size(0),), rid, dtype=torch.long, device=device)
#                 value_next[rid] = critic(x_next, edge_index_next, batch_next)

#             # ---------- ADVANTAGES & TD TARGETS ----------
#             advantages = {}
#             td_targets = {}
#             # Collect flat list of scalar advantages for normalization later
#             adv_list = []
#             adv_keys = []
#             for rid, r in (reward.items() if isinstance(reward, dict) else enumerate([reward])):
#                 if rid not in value_preds:
#                     continue
#                 v_curr = value_preds[rid]
#                 v_next = value_next.get(rid, None)
#                 done_or_trunc = float(done or truncated)
#                 if v_next is not None:
#                     target = r + gamma * v_next.detach() * (1.0 - done_or_trunc)
#                 else:
#                     target = torch.tensor(r, dtype=v_curr.dtype, device=device)
#                 adv = (target - v_curr)
#                 advantages[rid] = adv
#                 td_targets[rid] = target
#                 # store scalar for normalization (detach to numpy-friendly)
#                 adv_list.append(adv.detach().cpu().item())
#                 adv_keys.append(rid)

#             # Normalize advantages (zero mean, unit std) if we have more than 1
#             if len(adv_list) > 0:
#                 adv_arr = np.array(adv_list, dtype=np.float32)
#                 mean = adv_arr.mean()
#                 std = adv_arr.std() if adv_arr.std() > 1e-8 else 1.0
#                 # map normalized values back into tensors
#                 for i, rid in enumerate(adv_keys):
#                     norm_val = (adv_arr[i] - mean) / std
#                     # replace advantages[rid] with normalized tensor of same shape/dtype/device
#                     advantages[rid] = torch.tensor(norm_val, dtype=advantages[rid].dtype, device=device)

#             # ---------- CRITIC UPDATE ----------
#             if td_targets:
#                 optimizer_critic.zero_grad()
#                 critic_loss = sum(((td_targets[rid] - value_preds[rid]) ** 2).mean()
#                                   for rid in td_targets)
#                 critic_loss.backward()
#                 optimizer_critic.step()

#                 # ---------- ACTOR UPDATE ----------
#                 for rid in list(advantages.keys()):
#                     if rid not in log_probs:
#                         continue
#                     optimizers_actors[rid].zero_grad()
#                     adv = advantages[rid].detach()
#                     # actor_loss = -(log_probs[rid] * adv).mean()
#                     entropy_coef = 0.01  # tune 0.005 - 0.02
#                     entropy_term = entropies.get(rid, torch.tensor(0.0, device=adv.device))
#                     actor_loss = -(log_probs[rid] * adv).mean() - entropy_coef * entropy_term.mean()
#                     actor_loss.backward()
#                     optimizers_actors[rid].step()

#             # advance
#             ego_graphs = ego_graphs_next
#             attribute_matrix = attribute_matrix_next
#             step += 1

#         # Logging: total sum and per-robot average for interpretability
#         avg_per_robot = episode_total_sum / max(1, env.n_robots)
#         print(f"Episode {episode+1}/{num_episodes} Reward(sum)={episode_total_sum:.2f} "
#               f"Reward(avg_per_robot)={avg_per_robot:.2f} Time={time.time()-start:.2f}s")
#         print(" Per-robot cumulative rewards:", per_robot_cumulative.tolist())
#         pickup_rewards_given = sum(1 for t in env.tasks if getattr(t, "pickup_reward_given", False))
#         delivery_rewards_given = sum(1 for t in env.tasks if getattr(t, "delivered_reward_given", False))
#         obsolete_penalties_given = sum(1 for t in env.tasks if getattr(t, "obsolete_penalty_given", False))
#         print(" pickups_paid:", pickup_rewards_given,
#               " deliveries_paid:", delivery_rewards_given,
#               " obsolete_penalties:", obsolete_penalties_given)


#         episode_rewards.append(episode_total_sum)

#     return episode_rewards

# (patched excerpts showing critic usage and robust scalar handling)


# import torch
# import numpy as np
# import time

# def _build_edge_index_from_ego_list(ego_list):
#     if len(ego_list) == 0:
#         return torch.empty((2, 0), dtype=torch.long)
#     return torch.tensor(np.concatenate(ego_list, axis=0), dtype=torch.long).t().contiguous()

# def train(env, num_episodes, actors, critic,
#           optimizers_actors, optimizer_critic, gamma=0.99,
#           max_steps_per_episode=200, device=None):

#     episode_rewards = []

#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     for a in actors.values():
#         a.to(device)
#     critic.to(device)

#     entropy_coef = 0.01  # example; adjust as needed

#     for episode in range(num_episodes):
#         obs, _ = env.reset()
#         ego_graphs, attribute_matrix = obs

#         done = False
#         episode_reward = 0.0
#         start = time.time()

#         step = 0
#         while (not done) and step < max_steps_per_episode:
#             # print(f" Episode {episode+1} Step {step+1} ---------------------")
#             actions = {}
#             log_probs = {}
#             value_preds = {}
#             entropies = {}

#             x = torch.tensor(attribute_matrix, dtype=torch.float, device=device)
#             edge_index_cache = {}  # cache edge_index per robot to avoid repeated builds

#             # ---------- ACTOR + CRITIC (current state) ----------
#             # IMPORTANT: build critic input ONCE per step (shared critic over global graph)
#             # For 'per_robot' critic, we'll request the full per-robot vector and index into it.
#             # Build a single edge_index representing the whole graph — concatenate all ego lists.
#             # But actors still need per-ego edge_index; we reuse _build_edge_index_from_ego_list.
#             for rid, ego_list in ego_graphs.items():
#                 if len(ego_list) == 0:
#                     actions[rid] = []
#                     continue

#                 # Build edge_index for actor (ego graph)
#                 edge_index = _build_edge_index_from_ego_list(ego_list).to(device)
#                 edge_index_cache[rid] = edge_index

#                 # Actor forward (per-robot policy uses full node features x and ego edge_index)
#                 logits = actors[rid](x, edge_index)

#                 # Determine which task nodes are actually present in ego_list.
#                 nodes_present = torch.unique(edge_index).cpu().numpy().tolist()
#                 task_nodes_present = [n for n in nodes_present if n >= env.n_robots and n < env.n_robots + env.n_tasks]
#                 if len(task_nodes_present) == 0:
#                     actions[rid] = []
#                     continue

#                 task_nodes_tensor = torch.tensor(task_nodes_present, dtype=torch.long, device=device)
#                 task_logits = logits[task_nodes_tensor]
#                 probs = torch.softmax(task_logits, dim=-1)

#                 # sample from policy for exploration
#                 dist = torch.distributions.Categorical(probs)
#                 sample_idx = dist.sample()
#                 task_node = task_nodes_tensor[sample_idx].item()

#                 actions[rid] = [task_node]
#                 log_probs[rid] = dist.log_prob(sample_idx)
#                 entropies[rid] = dist.entropy()

#             # ---------- CRITIC EVALUATION ----------
#             # Compute critic outputs once using global graph (edge index built by concatenating all ego lists)
#             # Build a global edge_index by concatenating all ego edge lists if necessary.
#             all_edge_list = []
#             for ego_list in ego_graphs.values():
#                 if len(ego_list) > 0:
#                     all_edge_list.append(np.concatenate(ego_list, axis=0))
#             if all_edge_list:
#                 global_edge_index = torch.tensor(np.concatenate(all_edge_list, axis=0), dtype=torch.long).t().contiguous().to(device)
#             else:
#                 global_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

#             # Call critic ONCE per step. Provide num_robots so critic returns per-robot vector when configured.
#             critic_out = critic(x, global_edge_index, batch=None, num_robots=env.n_robots)

#             # Normalize critic_out handling:
#             # If critic_out is a vector of length num_robots -> use that.
#             # If critic_out is scalar -> treat as global scalar (same for all robots).
#             for rid in range(env.n_robots):
#                 if isinstance(critic_out, torch.Tensor) and critic_out.dim() > 0 and critic_out.numel() > 1:
#                     # per-robot vector case
#                     v = critic_out[rid]
#                 else:
#                     # scalar global value
#                     v = critic_out
#                 value_preds[rid] = v.squeeze()

#             # ---------- ENV STEP ----------
#             next_obs, reward, done, truncated, _ = env.step(actions, assignment_interval=5)
#             ego_graphs_next, attribute_matrix_next = next_obs

#             if isinstance(reward, dict):
#                 episode_reward += sum(reward.values())
#             else:
#                 episode_reward += reward

#             # ---------- Compute critic on next state for bootstrap ----------
#             # Build x_next and global_edge_index_next similarly
#             x_next = torch.tensor(attribute_matrix_next, dtype=torch.float, device=device)
#             all_edge_list_next = []
#             for ego_list in ego_graphs_next.values():
#                 if len(ego_list) > 0:
#                     all_edge_list_next.append(np.concatenate(ego_list, axis=0))
#             if all_edge_list_next:
#                 global_edge_index_next = torch.tensor(np.concatenate(all_edge_list_next, axis=0), dtype=torch.long).t().contiguous().to(device)
#             else:
#                 global_edge_index_next = torch.empty((2, 0), dtype=torch.long, device=device)

#             critic_out_next = critic(x_next, global_edge_index_next, batch=None, num_robots=env.n_robots)

#             value_next = {}
#             for rid in value_preds.keys():
#                 if isinstance(critic_out_next, torch.Tensor) and critic_out_next.dim() > 0 and critic_out_next.numel() > 1:
#                     v_next = critic_out_next[rid]
#                 else:
#                     v_next = critic_out_next
#                 value_next[rid] = v_next.squeeze()

#             # ---------- ADVANTAGES & TD TARGETS ----------
#             advantages = {}
#             td_targets = {}
#             adv_list = []
#             adv_keys = []
#             for rid, r in (reward.items() if isinstance(reward, dict) else enumerate([reward])):
#                 if rid not in value_preds:
#                     continue
#                 v_curr = value_preds[rid]
#                 v_next = value_next.get(rid, None)
#                 done_or_trunc = float(done or truncated)
#                 if v_next is not None:
#                     target = r + gamma * v_next.detach() * (1.0 - done_or_trunc)
#                 else:
#                     target = torch.tensor(r, dtype=v_curr.dtype, device=device)
#                 adv = (target - v_curr)
#                 advantages[rid] = adv
#                 td_targets[rid] = target

#                 # safe scalar extraction for normalization
#                 adv_det = adv.detach()
#                 if adv_det.numel() == 1:
#                     adv_list.append(adv_det.cpu().item())
#                 else:
#                     adv_list.append(float(adv_det.cpu().mean().item()))
#                 adv_keys.append(rid)

#             # Normalize advantages
#             if len(adv_list) > 0:
#                 adv_arr = np.array(adv_list, dtype=np.float32)
#                 mean = adv_arr.mean()
#                 std = adv_arr.std() if adv_arr.std() > 1e-8 else 1.0
#                 for i, rid in enumerate(adv_keys):
#                     norm_val = (adv_arr[i] - mean) / std
#                     advantages[rid] = torch.tensor(norm_val, dtype=advantages[rid].dtype, device=device)

#             # ---------- CRITIC UPDATE ----------
#             if td_targets:
#                 optimizer_critic.zero_grad()
#                 critic_loss = sum(((td_targets[rid] - value_preds[rid]) ** 2).mean()
#                                   for rid in td_targets)
#                 critic_loss.backward()
#                 optimizer_critic.step()

#                 # ---------- ACTOR UPDATE ----------
#                 for rid in list(advantages.keys()):
#                     if rid not in log_probs:
#                         continue
#                     optimizers_actors[rid].zero_grad()
#                     adv = advantages[rid].detach()
#                     entropy_term = entropies.get(rid, torch.tensor(0.0, device=adv.device))
#                     actor_loss = -(log_probs[rid] * adv).mean() - entropy_coef * entropy_term.mean()
#                     actor_loss.backward()
#                     optimizers_actors[rid].step()

#             # advance
#             ego_graphs = ego_graphs_next
#             attribute_matrix = attribute_matrix_next
#             step += 1

#         print(f"Episode {episode+1}/{num_episodes} Reward={episode_reward:.2f} Time={time.time()-start:.2f}s")
#         episode_rewards.append(episode_reward)

#     return episode_rewards

import torch
import numpy as np
import time
import json

def _build_edge_index_from_ego_list(ego_list):
    if len(ego_list) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(np.concatenate(ego_list, axis=0), dtype=torch.long).t().contiguous()

def train(env, num_episodes, actors, critic,
          optimizers_actors, optimizer_critic, gamma=0.99,
          max_steps_per_episode=200, device=None, verbose=True,
          save_dir=None, save_every=10, plot_rewards_fn=None, save_models_fn=None):

    episode_rewards = []

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for a in actors.values():
        a.to(device)
    critic.to(device)

    entropy_coef = 0.01  # example; adjust as needed
    try:
        for episode in range(num_episodes):
            print(f"=== Episode {episode+1}/{num_episodes} ===")
            obs, _ = env.reset()
            ego_graphs, attribute_matrix = obs

            done = False
            episode_reward = 0.0
            start = time.time()

            step = 0
            while (not done) and step < max_steps_per_episode:
                print(f" Episode {episode+1} Step {step+1} ---------------------")
                actions = {}
                # We'll fill log_probs AFTER env.step based on resolved assignments
                log_probs = {}
                entropies = {}

                # For later computing executed action's log_prob
                actor_candidates = {}     # rid -> tensor of node indices
                actor_task_logits = {}    # rid -> tensor of logits aligned with actor_candidates[rid]

                value_preds = {}

                x = torch.tensor(attribute_matrix, dtype=torch.float, device=device)
                edge_index_cache = {}  # cache edge_index per robot to avoid repeated builds

                # ---------- ACTOR (produce top-2 proposals and store logits) ----------
                for rid, ego_list in ego_graphs.items():
                    if rid<1:
                        print(f" Robot {rid} has {len(ego_list), ego_list} ego edges.")
                    if len(ego_list) == 0:
                        actions[rid] = []
                        continue

                    # Build edge_index for actor (ego graph)
                    edge_index = _build_edge_index_from_ego_list(ego_list).to(device)
                    edge_index_cache[rid] = edge_index

                    # Actor forward (per-robot policy uses full node features x and ego edge_index)
                    logits = actors[rid](x, edge_index)  # per-node scores, indexed by global node idx

                    # Determine which task nodes are actually present in ego_list.
                    nodes_present = torch.unique(edge_index).cpu().numpy().tolist()
                    # filter to task nodes in current attribute matrix
                    task_nodes_present = [n for n in nodes_present if n >= env.n_robots and n < env.n_robots + env.n_tasks]
                    if len(task_nodes_present) == 0:
                        actions[rid] = []
                        continue

                    task_nodes_tensor = torch.tensor(task_nodes_present, dtype=torch.long, device=device)
                    task_logits = logits[task_nodes_tensor]  # logits for these candidate nodes

                    # store candidates & logits for later log_prob computation
                    actor_candidates[rid] = task_nodes_tensor  # node indices
                    actor_task_logits[rid] = task_logits       # corresponding logits

                    # choose top-2 by logits to propose (resolve_conflicts can use second choice)
                    k = min(2, task_logits.size(0))
                    topk_indices = torch.topk(task_logits, k=k).indices.cpu().tolist()
                    top_nodes = [int(task_nodes_tensor[i].item()) for i in topk_indices]
                    if len(top_nodes) == 1:
                        top_nodes.append(top_nodes[0])
                    actions[rid] = top_nodes

                # ---------- CRITIC EVALUATION ----------
                # Build global edge_index by concatenating all ego lists if necessary.
                all_edge_list = []
                for ego_list in ego_graphs.values():
                    if len(ego_list) > 0:
                        all_edge_list.append(np.concatenate(ego_list, axis=0))
                if all_edge_list:
                    global_edge_index = torch.tensor(np.concatenate(all_edge_list, axis=0), dtype=torch.long).t().contiguous().to(device)
                else:
                    global_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

                # Call critic ONCE per step. Provide num_robots so critic returns per-robot vector when configured.
                critic_out = critic(x, global_edge_index, batch=None, num_robots=env.n_robots)

                # Normalize critic_out handling:
                for rid in range(env.n_robots):
                    if isinstance(critic_out, torch.Tensor) and critic_out.dim() > 0 and critic_out.numel() > 1:
                        v = critic_out[rid]
                    else:
                        v = critic_out
                    value_preds[rid] = v.squeeze()

                # ---------- ENV STEP ----------
                next_obs, reward, done, truncated, info = env.step(actions, assignment_interval=5)
                ego_graphs_next, attribute_matrix_next = next_obs

                # Get resolved assignments actually applied by the env (rid -> task_identifier)
                resolved = info.get("resolved_assignments", {})

                # For every robot that actually got an assignment, compute the log_prob under actor
                for rid, assigned_task_id in resolved.items():
                    # assigned_task_id might be a node index (n_robots + task_idx) or a unique task id.
                    # Our actor_candidates use node indices. Prefer node-index path.
                    # If assigned_task_id is a float/unique id, try to map to node index via env.taskid_to_task
                    node_assigned = assigned_task_id
                    # Map unique id to node index if necessary
                    if assigned_task_id not in actor_candidates.get(rid, []):
                        # try mapping from task id to node index
                        if assigned_task_id in env.taskid_to_task:
                            # find position of task in self.tasks and compute node index
                            try:
                                t_idx = [t.id for t in env.tasks].index(assigned_task_id)
                                node_assigned = env.n_robots + t_idx
                            except ValueError:
                                node_assigned = assigned_task_id  # leave as is
                    # Now compute log_prob if node_assigned is in actor_candidates
                    if rid in actor_candidates:
                        candidates = actor_candidates[rid]
                        logits = actor_task_logits[rid]
                        # find position of node_assigned in candidates
                        matches = (candidates == node_assigned).nonzero(as_tuple=True)[0]
                        if matches.numel() > 0:
                            pos = matches[0].item()
                            logp_all = torch.log_softmax(logits, dim=-1)
                            log_probs[rid] = logp_all[pos]
                            # compute entropy over this candidate set
                            probs_all = torch.softmax(logits, dim=-1)
                            entropies[rid] = - (probs_all * logp_all).sum()
                        else:
                            # assigned node not in candidate list (rare) -> skip
                            pass

                # accumulate rewards
                if isinstance(reward, dict):
                    episode_reward += sum(reward.values())
                else:
                    episode_reward += reward

                # ---------- Compute critic on next state for bootstrap ----------
                x_next = torch.tensor(attribute_matrix_next, dtype=torch.float, device=device)
                all_edge_list_next = []
                for ego_list in ego_graphs_next.values():
                    if len(ego_list) > 0:
                        all_edge_list_next.append(np.concatenate(ego_list, axis=0))
                if all_edge_list_next:
                    global_edge_index_next = torch.tensor(np.concatenate(all_edge_list_next, axis=0), dtype=torch.long).t().contiguous().to(device)
                else:
                    global_edge_index_next = torch.empty((2, 0), dtype=torch.long, device=device)

                critic_out_next = critic(x_next, global_edge_index_next, batch=None, num_robots=env.n_robots)

                value_next = {}
                for rid in value_preds.keys():
                    if isinstance(critic_out_next, torch.Tensor) and critic_out_next.dim() > 0 and critic_out_next.numel() > 1:
                        v_next = critic_out_next[rid]
                    else:
                        v_next = critic_out_next
                    value_next[rid] = v_next.squeeze()

                # ---------- ADVANTAGES & TD TARGETS ----------
                advantages = {}
                td_targets = {}
                adv_list = []
                adv_keys = []
                for rid, r in (reward.items() if isinstance(reward, dict) else enumerate([reward])):
                    if rid not in value_preds:
                        continue
                    v_curr = value_preds[rid]
                    v_next = value_next.get(rid, None)
                    done_or_trunc = float(done or truncated)
                    if v_next is not None:
                        target = r + gamma * v_next.detach() * (1.0 - done_or_trunc)
                    else:
                        target = torch.tensor(r, dtype=v_curr.dtype, device=device)
                    adv = (target - v_curr)
                    advantages[rid] = adv
                    td_targets[rid] = target

                    # safe scalar extraction for normalization
                    adv_det = adv.detach()
                    if adv_det.numel() == 1:
                        adv_list.append(adv_det.cpu().item())
                    else:
                        adv_list.append(float(adv_det.cpu().mean().item()))
                    adv_keys.append(rid)

                # Normalize advantages
                if len(adv_list) > 0:
                    adv_arr = np.array(adv_list, dtype=np.float32)
                    mean = adv_arr.mean()
                    std = adv_arr.std() if adv_arr.std() > 1e-8 else 1.0
                    for i, rid in enumerate(adv_keys):
                        norm_val = (adv_arr[i] - mean) / std
                        advantages[rid] = torch.tensor(norm_val, dtype=advantages[rid].dtype, device=device)

                # ---------- CRITIC UPDATE ----------
                if td_targets:
                    optimizer_critic.zero_grad()
                    critic_loss = sum(((td_targets[rid] - value_preds[rid]) ** 2).mean()
                                    for rid in td_targets)
                    critic_loss.backward()
                    optimizer_critic.step()

                    # ---------- ACTOR UPDATE ----------
                    for rid in list(advantages.keys()):
                        # only update if we computed a log_prob for executed action
                        if rid not in log_probs:
                            continue
                        optimizers_actors[rid].zero_grad()
                        adv = advantages[rid].detach()
                        entropy_term = entropies.get(rid, torch.tensor(0.0, device=adv.device))
                        actor_loss = -(log_probs[rid] * adv).mean() - entropy_coef * entropy_term.mean()
                        actor_loss.backward()
                        optimizers_actors[rid].step()

                # advance
                ego_graphs = ego_graphs_next
                attribute_matrix = attribute_matrix_next
                step += 1
            episode_rewards.append(episode_reward)

            # Save results every `save_every` episodes
            if save_dir and save_every > 0 and (episode + 1) % save_every == 0:
                print(f"Checkpoint: Saving results at episode {episode+1}...")
                if save_models_fn:
                    save_models_fn(save_dir, actors, critic)
                if plot_rewards_fn:
                    plot_rewards_fn(save_dir, episode_rewards)
                with open(save_dir / "episode_rewards.json", "w") as f:
                    json.dump([float(x) for x in episode_rewards], f)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving progress...")
        if save_dir:
            if save_models_fn:
                save_models_fn(save_dir, actors, critic)
            if plot_rewards_fn:
                plot_rewards_fn(save_dir, episode_rewards)
            with open(save_dir / "episode_rewards.json", "w") as f:
                json.dump([float(x) for x in episode_rewards], f)

    return episode_rewards

            # if verbose:
            #     print(f"Episode {episode+1}/{num_episodes} Reward={episode_reward:.2f} Time={time.time()-start:.2f}s")
            # episode_rewards.append(episode_reward)

        # return episode_rewards