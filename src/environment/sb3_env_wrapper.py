# """
# SB3 wrapper (colleague-style masking):
# - MultiDiscrete per-robot actions
# - action_mask is part of observation (so the policy can mask logits)
# - keeps reward component logging (rew/*) and episode stats
# """
# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# import torch



# class WarehouseEnvSB3Final(gym.Env):
#     def __init__(self, base_env, assignment_interval=50, k_max=5):
#         super().__init__()
#         self.base_env = base_env
#         self.assignment_interval = int(assignment_interval)

#         self.k_max = int(k_max)
#         self.noop_index = self.k_max
#         self.n_robots = base_env.n_robots

#         # Count total tasks
#         total_tasks = 0
#         if hasattr(base_env, 'tasks_batches') and isinstance(base_env.tasks_batches, list):
#             for batch in base_env.tasks_batches:
#                 total_tasks += len(batch)
#         else:
#             total_tasks = base_env.n_tasks

#         max_nodes = base_env.n_robots + total_tasks
#         max_edges = max_nodes * base_env.n_robots * 4
#         self.max_nodes = max_nodes
#         self.max_edges = max_edges
#         self.total_tasks = total_tasks
#         self.n_robots = base_env.n_robots

#         print("Wrapper initialized:")
#         print(f"  Robots: {base_env.n_robots}, Tasks: {total_tasks}")
#         print(f"  Assignment interval: {self.assignment_interval}")
#         print(f"  Action: MultiDiscrete([K+1]*R) with K={self.k_max} and NOOP={self.noop_index}")

#         # Observation space (add action_mask)
#         self.observation_space = spaces.Dict({
#             'node_features': spaces.Box(
#                 low=-np.inf, high=np.inf,
#                 shape=(self.max_nodes, base_env.feature_size),
#                 dtype=np.float32
#             ),
#             'edge_index': spaces.Box(
#                 low=0, high=self.max_nodes - 1,
#                 shape=(2, self.max_edges),
#                 dtype=np.int64
#             ),
#             'num_nodes': spaces.Box(low=0, high=self.max_nodes, shape=(1,), dtype=np.int64),
#             'num_edges': spaces.Box(low=0, high=self.max_edges, shape=(1,), dtype=np.int64),

#             # NEW: per-robot action mask (0/1)
#             'action_mask': spaces.Box(
#                 low=0.0, high=1.0,
#                 shape=(self.n_robots, self.k_max + 1),
#                 dtype=np.float32
#             ),
#             'cand_node_idx': spaces.Box(
#                 low=-1, high=self.max_nodes - 1,
#                 shape=(self.n_robots, self.k_max),
#                 dtype=np.int64
# ),
#         })

#         # CHANGED: MultiDiscrete action
#         self.action_space = spaces.MultiDiscrete([self.k_max + 1] * self.n_robots)

#         self.max_nodes = max_nodes
#         self.max_edges = max_edges
#         self.total_tasks = total_tasks

#         self.step_count = 0
#         self.episode_count = 0
#         self.last_episode_completed = 0
#         self.last_episode_obsolete = 0

#         # Store candidates per robot: list[R][K] task_id or None
#         self._last_cand_task_ids = [[None] * self.k_max for _ in range(self.n_robots)]

#     # -------- candidates / mask / decode --------
#     def _build_id_to_row(self):
#         """
#         Convert trueid_idx_mapping = [row_indices, true_ids] to dict {true_id: row_index}.
#         """
#         mapping = getattr(self.base_env, "trueid_idx_mapping", None)
#         if mapping is None or len(mapping) != 2:
#             return {}
#         rows = np.asarray(mapping[0])
#         ids = np.asarray(mapping[1])
#         return {int(tid): int(r) for r, tid in zip(rows, ids)}


#     def _cand_node_idx_matrix(self):
#         """
#         Return cand_node_idx: shape [R, K] with node indices in attributes_matrix, or -1.
#         Uses self._last_cand_task_ids and base_env.trueid_idx_mapping.
#         """
#         id_to_row = self._build_id_to_row()
#         cand_node_idx = -np.ones((self.n_robots, self.k_max), dtype=np.int64)
#         for r in range(self.n_robots):
#             for k in range(self.k_max):
#                 tid = self._last_cand_task_ids[r][k]
#                 if tid is None:
#                     continue
#                 cand_node_idx[r, k] = int(id_to_row.get(int(tid), -1))
#         return cand_node_idx

#     def _build_candidates(self):
#         """
#         Build up to k_max candidate task IDs per robot, ordered by "best-first".

#         Best-first heuristic (warehouse): smallest Euclidean distance from robot to task pickup.

#         Slot semantics per robot:
#         slot 0 = nearest pickup (greedy-best)
#         slot 1 = 2nd nearest
#         ...
#         slot K-1 = Kth nearest
#         slot K   = NOOP (handled elsewhere)
#         """
#         available_task_ids = list(self.base_env.get_available_task_ids())
#         cand = [[None] * self.k_max for _ in range(self.n_robots)]
#         if not available_task_ids:
#             return cand

#         # fast lookup id -> task object
#         task_map = getattr(self.base_env, "taskid_to_task", {})
#         if not task_map:
#             # fallback: old behavior if map missing
#             top = available_task_ids[: self.k_max]
#             for r in range(self.n_robots):
#                 robot = self.base_env.robots[r]
#                 if robot.capacity >= robot.maxCapacity:
#                     continue
#                 for k, tid in enumerate(top):
#                     cand[r][k] = int(tid)
#             return cand

#         for r in range(self.n_robots):
#             robot = self.base_env.robots[r]
#             if robot.capacity >= robot.maxCapacity:
#                 continue

#             rpos = np.asarray(robot.coordinate[:2], dtype=np.float32)

#             scored: list[tuple[float, int]] = []
#             for tid in available_task_ids:
#                 t = task_map.get(tid, None)
#                 if t is None:
#                     continue
#                 p = np.asarray(t.pick_up_coord[:2], dtype=np.float32)
#                 d = float(np.linalg.norm(rpos - p))
#                 scored.append((d, int(tid)))

#             scored.sort(key=lambda x: x[0])
#             for k, (_, tid) in enumerate(scored[: self.k_max]):
#                 cand[r][k] = int(tid)

#         return cand
#     def _action_mask_matrix(self):
#         """
#         [R, K+1] mask where last column is NOOP (always valid).
#         Candidate slot valid if stored task_id is not None.
#         """
#         mask = np.zeros((self.n_robots, self.k_max + 1), dtype=np.float32)
#         for r in range(self.n_robots):
#             for k in range(self.k_max):
#                 if self._last_cand_task_ids[r][k] is not None:
#                     mask[r, k] = 1.0
#             mask[r, self.noop_index] = 1.0
#         return mask

#     def _decode_action_vec(self, action_vec):
#         """
#         action_vec: shape (R,) each entry in [0..K] where K=NOOP
#         return {robot_id: [task_id]} or None
#         """
#         action_vec = np.asarray(action_vec, dtype=np.int64)
#         assignments = {}
#         for r in range(self.n_robots):
#             a = int(action_vec[r])
#             if a == self.noop_index:
#                 continue
#             if 0 <= a < self.k_max:
#                 task_id = self._last_cand_task_ids[r][a]
#                 if task_id is not None:
#                     assignments[r] = [int(task_id)]
#         return assignments if len(assignments) else None

#     # -------- gym API --------

#     def reset(self, seed=None, options=None):
#         if seed is not None:
#             np.random.seed(seed)
#             torch.manual_seed(seed)

#         if self.episode_count > 0:
#             self.last_episode_completed = sum(1 for t in self.base_env.tasks if t.is_droppedoff)
#             self.last_episode_obsolete = sum(1 for t in self.base_env.tasks if t.is_obsolete(self.base_env.time_count))

#         obs, info = self.base_env.reset()
#         self.step_count = 0
#         self.episode_count += 1

#         self._last_cand_task_ids = self._build_candidates()

#         if not isinstance(info, dict):
#             info = {}
#         info['episode_completed'] = self.last_episode_completed
#         info['episode_obsolete'] = self.last_episode_obsolete
#         info["cand_task_ids"] = self._last_cand_task_ids
#         info["action_mask"] = self._action_mask_matrix()

#         return self._convert_observation(obs), info

#     def step(self, action):
#         self.step_count += 1
#         # is_decision_step = (self.step_count % self.assignment_interval == 0)
#         is_decision_step = ((self.step_count - 1) % self.assignment_interval == 0)

#         if is_decision_step:
#             self._last_cand_task_ids = self._build_candidates()
#             assignments = self._decode_action_vec(action)
#             # print(assignments,'<- decoded assignments from action_vec:', action)
#         else:
#             assignments = None

#         obs, reward, done, truncated, info_reward, info = self.base_env.step(
#             assignments,
#             assignment_interval=self.assignment_interval
#         )
#         # print(f"Step {self.step_count}: reward={reward}, done={done}, truncated={truncated}, info_reward={info_reward}")

#         if isinstance(reward, dict):
#             reward = sum(reward.values())

#         if not isinstance(info, dict):
#             info = {}

#         # reward components
#         if isinstance(info_reward, dict):
#             for k, v in info_reward.items():
#                 info[f"rew/{k}"] = v
#         info["cand_task_ids"] = self._last_cand_task_ids
#         if done or truncated:
#             info['episode_completed'] = sum(1 for t in self.base_env.tasks if t.is_droppedoff)
#             info['episode_obsolete'] = sum(1 for t in self.base_env.tasks if t.is_obsolete(self.base_env.time_count))
#             # info["cand_task_ids"] = self._last_cand_task_ids
#             # info["action_mask"] = self._action_mask_matrix()
#             # info["decoded_assignments"] = assignments  # for debugging
#         info["cand_task_ids"] = self._last_cand_task_ids
#         info["action_mask"] = self._action_mask_matrix()
#         info["decoded_assignments"] = assignments

#         return self._convert_observation(obs), reward, done, truncated, info

#     def _convert_observation(self, obs):
#         ego_graphs, attribute_matrix = obs

#         all_edges = []
#         for robot_id, ego_list in ego_graphs.items():
#             if len(ego_list) > 0:
#                 edges = np.concatenate(ego_list, axis=0)
#                 all_edges.append(edges)

#         if len(all_edges) > 0:
#             edge_index = np.concatenate(all_edges, axis=0).T
#         else:
#             edge_index = np.zeros((2, 0), dtype=np.int64)

#         num_nodes = min(len(attribute_matrix), self.max_nodes)
#         num_edges = min(edge_index.shape[1], self.max_edges)

#         padded_features = np.zeros((self.max_nodes, attribute_matrix.shape[1]), dtype=np.float32)
#         padded_features[:num_nodes] = attribute_matrix[:num_nodes]

#         padded_edges = np.zeros((2, self.max_edges), dtype=np.int64)
#         padded_edges[:, :num_edges] = edge_index[:, :num_edges]

#         # ✅ include mask in obs
#         action_mask = self._action_mask_matrix()

#         cand_node_idx = self._cand_node_idx_matrix()
#         action_mask = self._action_mask_matrix()

#         return {
#             "node_features": padded_features,
#             "edge_index": padded_edges,
#             "num_nodes": np.array([num_nodes], dtype=np.int64),
#             "num_edges": np.array([num_edges], dtype=np.int64),
#             "action_mask": action_mask.astype(np.float32),
#             "cand_node_idx": cand_node_idx,  
#         }



#--------------this version handel batch normalization and NOOP handeling----------------
# ---------------------------------------------------------------------------------------
# """
# SB3 wrapper with per-robot ego-graph observations (colleague-style).

# Key change vs your current wrapper:
# - We DO NOT concatenate ego graphs into one global edge_index.
# - We expose a padded ego graph per robot:
#     node_features: [R, N_ego_max, F]
#     edge_index:    [R, 2, E_ego_max]   (LOCAL node indices)
#     num_nodes:     [R, 1]
#     num_edges:     [R, 1]
#     cand_node_idx: [R, K]              (LOCAL indices)
#     action_mask:   [R, K+1]            (NOOP always valid)
# - We keep your candidate generation & decoding (based on task ids) unchanged.
# """

# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# import torch


# class WarehouseEnvSB3Final(gym.Env):
#     def __init__(self, base_env, assignment_interval=50, k_max=5, ego_max_nodes=None, ego_max_edges=None):
#         super().__init__()
#         self.base_env = base_env
#         self.assignment_interval = int(assignment_interval)

#         self.k_max = int(k_max)
#         self.noop_index = self.k_max
#         self.n_robots = int(base_env.n_robots)

#         # Count total tasks across batches (for legacy global sizing; not used for ego sizing)
#         total_tasks = 0
#         if hasattr(base_env, "tasks_batches") and isinstance(base_env.tasks_batches, list):
#             for batch in base_env.tasks_batches:
#                 total_tasks += len(batch)
#         else:
#             total_tasks = int(getattr(base_env, "n_tasks", 0))

#         # Legacy global bounds (still useful for mapping true_id -> row)
#         max_nodes = int(base_env.n_robots + total_tasks)
#         max_edges = int(max_nodes * base_env.n_robots * 4)
#         self.max_nodes = max_nodes
#         self.max_edges = max_edges
#         self.total_tasks = total_tasks

#         # --- Ego padded sizes (per robot) ---
#         # Tune these. Must be constant for SB3 spaces.
#         if ego_max_nodes is None:
#             # robot + a chunk of local nodes
#             ego_max_nodes = int(self.n_robots + self.k_max + 20)
#         if ego_max_edges is None:
#             ego_max_edges = int(ego_max_nodes * 10)

#         self.ego_max_nodes = int(min(self.max_nodes, max(8, ego_max_nodes)))
#         self.ego_max_edges = int(min(self.max_edges, max(16, ego_max_edges)))

#         print("Wrapper initialized:")
#         print(f"  Robots: {base_env.n_robots}, Tasks: {total_tasks}")
#         print(f"  Assignment interval: {self.assignment_interval}")
#         print(f"  Action: MultiDiscrete([K+1]*R) with K={self.k_max} and NOOP={self.noop_index}")
#         print(f"  Ego obs: node_features [R,{self.ego_max_nodes},F], edge_index [R,2,{self.ego_max_edges}]")

#         F = int(base_env.feature_size)

#         # Observation space (ego-graph per robot)
#         self.observation_space = spaces.Dict({
#             "node_features": spaces.Box(
#                 low=-np.inf, high=np.inf,
#                 shape=(self.n_robots, self.ego_max_nodes, F),
#                 dtype=np.float32,
#             ),
#             "edge_index": spaces.Box(
#                 low=0, high=self.ego_max_nodes - 1,
#                 shape=(self.n_robots, 2, self.ego_max_edges),
#                 dtype=np.int64,
#             ),
#             "num_nodes": spaces.Box(
#                 low=0, high=self.ego_max_nodes,
#                 shape=(self.n_robots, 1),
#                 dtype=np.int64,
#             ),
#             "num_edges": spaces.Box(
#                 low=0, high=self.ego_max_edges,
#                 shape=(self.n_robots, 1),
#                 dtype=np.int64,
#             ),

#             # per-robot action mask (K+1 includes NOOP)
#             "action_mask": spaces.Box(
#                 low=0.0, high=1.0,
#                 shape=(self.n_robots, self.k_max + 1),
#                 dtype=np.float32,
#             ),

#             # LOCAL candidate indices (0..N_ego_max-1) or -1
#             "cand_node_idx": spaces.Box(
#                 low=-1, high=self.ego_max_nodes - 1,
#                 shape=(self.n_robots, self.k_max),
#                 dtype=np.int64,
#             ),
#         })

#         # Action: one discrete per robot
#         self.action_space = spaces.MultiDiscrete([self.k_max + 1] * self.n_robots)

#         self.step_count = 0
#         self.episode_count = 0
#         self.last_episode_completed = 0
#         self.last_episode_obsolete = 0

#         # Store candidates per robot: list[R][K] task_id or None
#         self._last_cand_task_ids = [[None] * self.k_max for _ in range(self.n_robots)]

#     # -------- candidates / mask / decode --------

#     def _build_id_to_row(self):
#         """
#         Convert trueid_idx_mapping = [row_indices, true_ids] to dict {true_id: row_index}.
#         Used to map task true_id -> row index in attribute_matrix.
#         """
#         mapping = getattr(self.base_env, "trueid_idx_mapping", None)
#         if mapping is None or len(mapping) != 2:
#             return {}
#         rows = np.asarray(mapping[0])
#         ids = np.asarray(mapping[1])
#         return {int(tid): int(r) for r, tid in zip(rows, ids)}

#     def _cand_node_idx_matrix_global(self):
#         """
#         Return GLOBAL cand_node_idx: shape [R, K] with node indices in attribute_matrix, or -1.
#         Uses self._last_cand_task_ids and base_env.trueid_idx_mapping.
#         """
#         id_to_row = self._build_id_to_row()
#         cand_node_idx = -np.ones((self.n_robots, self.k_max), dtype=np.int64)
#         for r in range(self.n_robots):
#             for k in range(self.k_max):
#                 tid = self._last_cand_task_ids[r][k]
#                 if tid is None:
#                     continue
#                 cand_node_idx[r, k] = int(id_to_row.get(int(tid), -1))
#         return cand_node_idx

#     def _build_candidates(self):
#         """
#         Build up to k_max candidate task IDs per robot, ordered by smallest distance to pickup.
#         """
#         available_task_ids = list(self.base_env.get_available_task_ids())
#         cand = [[None] * self.k_max for _ in range(self.n_robots)]
#         if not available_task_ids:
#             return cand

#         task_map = getattr(self.base_env, "taskid_to_task", {})
#         if not task_map:
#             top = available_task_ids[: self.k_max]
#             for r in range(self.n_robots):
#                 robot = self.base_env.robots[r]
#                 if robot.capacity >= robot.maxCapacity:
#                     continue
#                 for k, tid in enumerate(top):
#                     cand[r][k] = int(tid)
#             return cand

#         for r in range(self.n_robots):
#             robot = self.base_env.robots[r]
#             if robot.capacity >= robot.maxCapacity:
#                 continue

#             rpos = np.asarray(robot.coordinate[:2], dtype=np.float32)
#             scored = []
#             for tid in available_task_ids:
#                 t = task_map.get(tid, None)
#                 if t is None:
#                     continue
#                 p = np.asarray(t.pick_up_coord[:2], dtype=np.float32)
#                 d = float(np.linalg.norm(rpos - p))
#                 scored.append((d, int(tid)))

#             scored.sort(key=lambda x: x[0])
#             for k, (_, tid) in enumerate(scored[: self.k_max]):
#                 cand[r][k] = int(tid)

#         return cand

#     def _action_mask_matrix(self):
#         """
#         [R, K+1] mask where last column is NOOP (always valid).
#         Candidate slot valid if stored task_id is not None.
#         """
#         mask = np.zeros((self.n_robots, self.k_max + 1), dtype=np.float32)
#         for r in range(self.n_robots):
#             for k in range(self.k_max):
#                 if self._last_cand_task_ids[r][k] is not None:
#                     mask[r, k] = 1.0
#             mask[r, self.noop_index] = 1.0
#         return mask

#     def _decode_action_vec(self, action_vec):
#         """
#         action_vec: shape (R,) each entry in [0..K] where K=NOOP
#         return {robot_id: [task_id]} or None
#         """
#         action_vec = np.asarray(action_vec, dtype=np.int64)
#         assignments = {}
#         for r in range(self.n_robots):
#             a = int(action_vec[r])
#             if a == self.noop_index:
#                 continue
#             if 0 <= a < self.k_max:
#                 task_id = self._last_cand_task_ids[r][a]
#                 if task_id is not None:
#                     assignments[r] = [int(task_id)]
#         return assignments if len(assignments) else None

#     # -------- gym API --------

#     def reset(self, seed=None, options=None):
#         if seed is not None:
#             np.random.seed(seed)
#             torch.manual_seed(seed)

#         if self.episode_count > 0:
#             self.last_episode_completed = sum(1 for t in self.base_env.tasks if t.is_droppedoff)
#             self.last_episode_obsolete = sum(1 for t in self.base_env.tasks if t.is_obsolete(self.base_env.time_count))

#         obs, info = self.base_env.reset()
#         self.step_count = 0
#         self.episode_count += 1

#         self._last_cand_task_ids = self._build_candidates()

#         if not isinstance(info, dict):
#             info = {}
#         info["episode_completed"] = self.last_episode_completed
#         info["episode_obsolete"] = self.last_episode_obsolete
#         info["cand_task_ids"] = self._last_cand_task_ids
#         info["action_mask"] = self._action_mask_matrix()

#         return self._convert_observation(obs), info

#     def step(self, action):
#         self.step_count += 1
#         is_decision_step = ((self.step_count - 1) % self.assignment_interval == 0)

#         if is_decision_step:
#             self._last_cand_task_ids = self._build_candidates()
#             assignments = self._decode_action_vec(action)
#         else:
#             assignments = None

#         obs, reward, done, truncated, info_reward, info = self.base_env.step(
#             assignments,
#             assignment_interval=self.assignment_interval
#         )

#         if isinstance(reward, dict):
#             reward = sum(reward.values())

#         if not isinstance(info, dict):
#             info = {}

#         # reward components flattening (kept for compatibility with your callback)
#         if isinstance(info_reward, dict):
#             for k, v in info_reward.items():
#                 info[f"rew/{k}"] = v

#         info["cand_task_ids"] = self._last_cand_task_ids
#         info["action_mask"] = self._action_mask_matrix()
#         info["decoded_assignments"] = assignments

#         if done or truncated:
#             info["episode_completed"] = sum(1 for t in self.base_env.tasks if t.is_droppedoff)
#             info["episode_obsolete"] = sum(1 for t in self.base_env.tasks if t.is_obsolete(self.base_env.time_count))

#         return self._convert_observation(obs), reward, done, truncated, info

#     def _convert_observation(self, obs):
#         """
#         Convert base_env obs=(ego_graphs, attribute_matrix) into per-robot ego observations.

#         ego_graphs: dict[rid] -> list of edge arrays, each [M,2] (GLOBAL node ids into attribute_matrix)
#         attribute_matrix: [N_global, F]
#         """
#         ego_graphs, attribute_matrix = obs
#         attribute_matrix = np.asarray(attribute_matrix, dtype=np.float32)
#         N_global, F = attribute_matrix.shape

#         R = self.n_robots
#         K = self.k_max

#         # padded outputs
#         node_features = np.zeros((R, self.ego_max_nodes, F), dtype=np.float32)
#         edge_index = np.zeros((R, 2, self.ego_max_edges), dtype=np.int64)
#         num_nodes = np.zeros((R, 1), dtype=np.int64)
#         num_edges = np.zeros((R, 1), dtype=np.int64)

#         # Build GLOBAL candidate indices from task ids mapping
#         cand_global = self._cand_node_idx_matrix_global()  # [R,K] global row indices or -1
#         cand_local = -np.ones((R, K), dtype=np.int64)

#         for rid in range(R):
#             ego_list = ego_graphs.get(rid, [])
#             if ego_list is None or len(ego_list) == 0:
#                 # minimal ego: include robot node itself if possible
#                 g_nodes = [rid] if 0 <= rid < N_global else []
#                 g2l = {int(g): i for i, g in enumerate(g_nodes)}
#                 n_i = min(len(g_nodes), self.ego_max_nodes)
#                 if n_i > 0:
#                     node_features[rid, :n_i, :] = attribute_matrix[np.asarray(g_nodes[:n_i], dtype=np.int64)]
#                 num_nodes[rid, 0] = n_i
#                 num_edges[rid, 0] = 0

#                 for k in range(K):
#                     g = int(cand_global[rid, k])
#                     if g >= 0 and g in g2l and g2l[g] < self.ego_max_nodes:
#                         cand_local[rid, k] = int(g2l[g])
#                 continue

#             edges_g = np.concatenate(ego_list, axis=0).astype(np.int64)  # [E,2] global
#             edges_g = edges_g.reshape(-1, 2)

#             # unique global nodes
#             g_nodes = np.unique(edges_g.reshape(-1)).tolist()

#             # ensure robot node included (assume robot global id == rid)
#             if 0 <= rid < N_global and rid not in g_nodes:
#                 g_nodes = [rid] + g_nodes

#             # truncate nodes
#             if len(g_nodes) > self.ego_max_nodes:
#                 g_nodes = g_nodes[: self.ego_max_nodes]

#             g2l = {int(g): int(i) for i, g in enumerate(g_nodes)}

#             # fill node features (skip invalid globals)
#             valid_g = [g for g in g_nodes if 0 <= int(g) < N_global]
#             n_i = min(len(valid_g), self.ego_max_nodes)
#             if n_i > 0:
#                 node_features[rid, :n_i, :] = attribute_matrix[np.asarray(valid_g[:n_i], dtype=np.int64)]
#             num_nodes[rid, 0] = n_i

#             # remap edges to local indices
#             src_g = edges_g[:, 0]
#             dst_g = edges_g[:, 1]
#             keep = np.array([(int(s) in g2l and int(d) in g2l) for s, d in zip(src_g, dst_g)], dtype=bool)
#             edges_kept = edges_g[keep]

#             if edges_kept.size > 0:
#                 src_l = np.array([g2l[int(s)] for s in edges_kept[:, 0]], dtype=np.int64)
#                 dst_l = np.array([g2l[int(d)] for d in edges_kept[:, 1]], dtype=np.int64)
#                 edges_l = np.stack([src_l, dst_l], axis=0)  # [2,E]
#             else:
#                 edges_l = np.zeros((2, 0), dtype=np.int64)

#             e_i = min(edges_l.shape[1], self.ego_max_edges)
#             if e_i > 0:
#                 edge_index[rid, :, :e_i] = edges_l[:, :e_i]
#             num_edges[rid, 0] = e_i

#             # map candidates global->local
#             for k in range(K):
#                 g = int(cand_global[rid, k])
#                 if g >= 0 and g in g2l:
#                     l = int(g2l[g])
#                     if l < self.ego_max_nodes:
#                         cand_local[rid, k] = l

#         action_mask = self._action_mask_matrix().astype(np.float32)

#         return {
#             "node_features": node_features,
#             "edge_index": edge_index,
#             "num_nodes": num_nodes,
#             "num_edges": num_edges,
#             "action_mask": action_mask,
#             "cand_node_idx": cand_local,
#         }

#-------------------------------------------------------------------------------
# ------------------------------------------------------------------------------



# ------------------------this version replace global graph handeling with ego graph----------------
# ---------------------------------------------------------------------------------------------
"""
SB3 wrapper with per-robot ego-graph observations (colleague-style).

Key change vs your current wrapper:
- We DO NOT concatenate ego graphs into one global edge_index.
- We expose a padded ego graph per robot:
    node_features: [R, N_ego_max, F]
    edge_index:    [R, 2, E_ego_max]   (LOCAL node indices)
    num_nodes:     [R, 1]
    num_edges:     [R, 1]
    cand_node_idx: [R, K]              (LOCAL indices)
    action_mask:   [R, K+1]            (NOOP always valid)
- We keep your candidate generation & decoding (based on task ids) unchanged.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch


class WarehouseEnvSB3Final(gym.Env):
    def __init__(self, base_env, assignment_interval=50, k_max=5, ego_max_nodes=None, ego_max_edges=None):
        super().__init__()
        self.base_env = base_env
        self.assignment_interval = int(assignment_interval)

        self.k_max = int(k_max)
        self.noop_index = self.k_max
        self.n_robots = int(base_env.n_robots)

        # Count total tasks across batches (for legacy global sizing; not used for ego sizing)
        total_tasks = 0
        if hasattr(base_env, "tasks_batches") and isinstance(base_env.tasks_batches, list):
            for batch in base_env.tasks_batches:
                total_tasks += len(batch)
        else:
            total_tasks = int(getattr(base_env, "n_tasks", 0))

        # Legacy global bounds (still useful for mapping true_id -> row)
        max_nodes = int(base_env.n_robots + total_tasks)
        max_edges = int(max_nodes * base_env.n_robots * 4)
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.total_tasks = total_tasks

        # --- Ego padded sizes (per robot) ---
        # Tune these. Must be constant for SB3 spaces.
        if ego_max_nodes is None:
            # robot + a chunk of local nodes
            ego_max_nodes = int(self.n_robots + self.k_max + 20)
        if ego_max_edges is None:
            ego_max_edges = int(ego_max_nodes * 10)

        self.ego_max_nodes = int(min(self.max_nodes, max(8, ego_max_nodes)))
        self.ego_max_edges = int(min(self.max_edges, max(16, ego_max_edges)))

        print("Wrapper initialized:")
        print(f"  Robots: {base_env.n_robots}, Tasks: {total_tasks}")
        print(f"  Assignment interval: {self.assignment_interval}")
        print(f"  Action: MultiDiscrete([K+1]*R) with K={self.k_max} and NOOP={self.noop_index}")
        print(f"  Ego obs: node_features [R,{self.ego_max_nodes},F], edge_index [R,2,{self.ego_max_edges}]")

        F = int(base_env.feature_size)

        # Observation space (ego-graph per robot)
        self.observation_space = spaces.Dict({
            "node_features": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.n_robots, self.ego_max_nodes, F),
                dtype=np.float32,
            ),
            "edge_index": spaces.Box(
                low=0, high=self.ego_max_nodes - 1,
                shape=(self.n_robots, 2, self.ego_max_edges),
                dtype=np.int64,
            ),
            "num_nodes": spaces.Box(
                low=0, high=self.ego_max_nodes,
                shape=(self.n_robots, 1),
                dtype=np.int64,
            ),
            "num_edges": spaces.Box(
                low=0, high=self.ego_max_edges,
                shape=(self.n_robots, 1),
                dtype=np.int64,
            ),

            # per-robot action mask (K+1 includes NOOP)
            "action_mask": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.n_robots, self.k_max + 1),
                dtype=np.float32,
            ),

            # LOCAL candidate indices (0..N_ego_max-1) or -1
            "cand_node_idx": spaces.Box(
                low=-1, high=self.ego_max_nodes - 1,
                shape=(self.n_robots, self.k_max),
                dtype=np.int64,
            ),
        })

        # Action: one discrete per robot
        self.action_space = spaces.MultiDiscrete([self.k_max + 1] * self.n_robots)

        self.step_count = 0
        self.episode_count = 0
        self.last_episode_completed = 0
        self.last_episode_obsolete = 0

        # Store candidates per robot: list[R][K] task_id or None
        self._last_cand_task_ids = [[None] * self.k_max for _ in range(self.n_robots)]

    # -------- candidates / mask / decode --------

    def _build_id_to_row(self):
        """
        Convert trueid_idx_mapping = [row_indices, true_ids] to dict {true_id: row_index}.
        Used to map task true_id -> row index in attribute_matrix.
        """
        mapping = getattr(self.base_env, "trueid_idx_mapping", None)
        if mapping is None or len(mapping) != 2:
            return {}
        rows = np.asarray(mapping[0])
        ids = np.asarray(mapping[1])
        return {int(tid): int(r) for r, tid in zip(rows, ids)}

    def _cand_node_idx_matrix_global(self):
        """
        Return GLOBAL cand_node_idx: shape [R, K] with node indices in attribute_matrix, or -1.
        Uses self._last_cand_task_ids and base_env.trueid_idx_mapping.
        """
        id_to_row = self._build_id_to_row()
        cand_node_idx = -np.ones((self.n_robots, self.k_max), dtype=np.int64)
        for r in range(self.n_robots):
            for k in range(self.k_max):
                tid = self._last_cand_task_ids[r][k]
                if tid is None:
                    continue
                cand_node_idx[r, k] = int(id_to_row.get(int(tid), -1))
        return cand_node_idx

    def _build_candidatesnor_working(self):
        """
        Build up to k_max candidate task IDs per robot, ordered by smallest distance to pickup.
        """
        available_task_ids = list(self.base_env.get_available_task_ids())
        cand = [[None] * self.k_max for _ in range(self.n_robots)]
        if not available_task_ids:
            return cand

        task_map = getattr(self.base_env, "taskid_to_task", {})
        if not task_map:
            top = available_task_ids[: self.k_max]
            for r in range(self.n_robots):
                robot = self.base_env.robots[r]
                if robot.capacity >= robot.maxCapacity:
                    continue
                for k, tid in enumerate(top):
                    cand[r][k] = int(tid)
            return cand

        for r in range(self.n_robots):
            robot = self.base_env.robots[r]
            if robot.capacity >= robot.maxCapacity:
                continue

            rpos = np.asarray(robot.coordinate[:2], dtype=np.float32)
            scored = []
            for tid in available_task_ids:
                t = task_map.get(tid, None)
                if t is None:
                    continue
                p = np.asarray(t.pick_up_coord[:2], dtype=np.float32)
                d = float(np.linalg.norm(rpos - p))
                scored.append((d, int(tid)))

            scored.sort(key=lambda x: x[0])
            for k, (_, tid) in enumerate(scored[: self.k_max]):
                cand[r][k] = int(tid)
        # print("[DEBUG _build_candidates] cand:", cand)

        return cand
    #this version fix the issue with task id to task object mapping and also filter the candidate by release time, assigned/obsolete status, and robot capacity
    def _build_candidates(self):
        """
        Build up to k_max candidate task IDs per robot from the current observed task rows
        in the shared attribute matrix, so candidate IDs always match the current observation.

        This avoids mismatch between:
        - live env tasks (IDs like 10000+)
        - attribute_matrix tasks (IDs currently present in the graph)
        """
        cand = [[None] * self.k_max for _ in range(self.n_robots)]

        # We need the current attribute matrix to define the candidate pool.
        # If not available yet, return empty candidates.
        if not hasattr(self, "_latest_attribute_matrix") or self._latest_attribute_matrix is None:
            return cand

        attr = np.asarray(self._latest_attribute_matrix, dtype=np.float32)
        # print("[DEBUG _build_candidates] attr", attr)
        # print("[DEBUG _build_candidates] self._latest_attribute_matrix", self._latest_attribute_matrix)
        # print("[DEBUG _build_candidates] attr shape:", attr.ndim, attr.shape)
        if attr.ndim != 2 or attr.shape[0] <= self.n_robots:
            return cand

        # Current task rows are rows after the robot rows
        task_rows = np.arange(self.n_robots, attr.shape[0], dtype=np.int64)
        task_ids = attr[task_rows, 0].astype(int)
        # print("[DEBUG TASK DATA]",task_rows)
        # print("[DEBUG _build_candidates] task_ids from attr:", task_ids[:20])
        task_coords = attr[task_rows, 1:3].astype(np.float32)

        # Filter out invalid/empty rows if needed
        valid_mask = np.isfinite(task_coords).all(axis=1)
        task_rows = task_rows[valid_mask]
        task_ids = task_ids[valid_mask]
        task_coords = task_coords[valid_mask]

        if len(task_rows) == 0:
            return cand

        for r in range(self.n_robots):
            robot = self.base_env.robots[r]
            if robot.capacity >= robot.maxCapacity:
                continue

            rpos = np.asarray(robot.coordinate[:2], dtype=np.float32)
            scored = []

            for tid, tpos in zip(task_ids, task_coords):
                d = float(np.linalg.norm(rpos - tpos[:2]))
                scored.append((d, int(tid)))

            scored.sort(key=lambda x: x[0])

            for k, (_, tid) in enumerate(scored[: self.k_max]):
                cand[r][k] = int(tid)
        # print("[DEBUG _build_candidates with wrong task id] cand:", cand)
        return cand
    def _build_candidatesworking_butbuggi(self):
        """
        Build up to k_max candidate task IDs per robot, ordered by smallest distance to pickup.

        IMPORTANT:
        - Uses the current self.base_env.tasks list directly, so candidate task IDs
        match the same task objects/IDs that appear in the current observation.
        - Filters by release time, assigned/obsolete status, and robot capacity.
        """
        cand = [[None] * self.k_max for _ in range(self.n_robots)]

        # Use the current live task objects, not a separate task-id pool
        now = int(getattr(self.base_env, "time_count", 0))
        all_tasks = getattr(self.base_env, "tasks", [])
        if not all_tasks:
            return cand

        available_tasks = []
        for t in all_tasks:
            try:
                if t.release_time <= now and (not t.is_assigned) and (not t.is_obsolete(now)):
                    available_tasks.append(t)
            except Exception:
                # Skip malformed task objects safely
                continue

        if len(available_tasks) == 0:
            return cand

        for r in range(self.n_robots):
            robot = self.base_env.robots[r]
            if robot.capacity >= robot.maxCapacity:
                continue

            rpos = np.asarray(robot.coordinate[:2], dtype=np.float32)

            scored = []
            for t in available_tasks:
                try:
                    p = np.asarray(t.pick_up_coord[:2], dtype=np.float32)
                    d = float(np.linalg.norm(rpos - p))
                    scored.append((d, int(t.id)))
                except Exception:
                    continue

            if len(scored) == 0:
                continue

            scored.sort(key=lambda x: x[0])

            for k, (_, tid) in enumerate(scored[: self.k_max]):
                cand[r][k] = int(tid)
            if not hasattr(self, "_debug_candidate_prints"):
                self._debug_candidate_prints = 0
            if self._debug_candidate_prints < 3:
                self._debug_candidate_prints += 1
                # print("[DEBUG candidates] time_count:", int(getattr(self.base_env, "time_count", 0)))
                # print("[DEBUG candidates] current task ids:", [int(t.id) for t in getattr(self.base_env, "tasks", [])[:20]])
                # print("[DEBUG candidates] available task ids:",
                #     [int(t.id) for t in available_tasks[:20]])
                # print("[DEBUG candidates] cand[0]:", cand[0])
        return cand
    def _action_mask_matrix(self):
        """
        [R, K+1] mask where last column is NOOP (always valid).
        Candidate slot valid if stored task_id is not None.
        """
        mask = np.zeros((self.n_robots, self.k_max + 1), dtype=np.float32)
        for r in range(self.n_robots):
            for k in range(self.k_max):
                if self._last_cand_task_ids[r][k] is not None:
                    mask[r, k] = 1.0
            mask[r, self.noop_index] = 1.0
        return mask

    def _decode_action_vec(self, action_vec):
        """
        action_vec: shape (R,) each entry in [0..K] where K=NOOP
        return {robot_id: [task_id]} or None
        """
        action_vec = np.asarray(action_vec, dtype=np.int64)
        assignments = {}
        for r in range(self.n_robots):
            a = int(action_vec[r])
            if a == self.noop_index:
                continue
            if 0 <= a < self.k_max:
                task_id = self._last_cand_task_ids[r][a]
                if task_id is not None:
                    assignments[r] = [int(task_id)]
        return assignments if len(assignments) else None

    # -------- gym API --------

    def reset(self, seed=None, options=None):
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        if self.episode_count > 0:
            self.last_episode_completed = sum(1 for t in self.base_env.tasks if t.is_droppedoff)
            self.last_episode_obsolete = sum(1 for t in self.base_env.tasks if t.is_obsolete(self.base_env.time_count))

        obs, info = self.base_env.reset()
        # print("[DEBUG reset] obs after calling baseenv.reset:", obs)
        self.step_count = 0
        self.episode_count += 1

        self._last_cand_task_ids = self._build_candidates()

        if not isinstance(info, dict):
            info = {}
        info["episode_completed"] = self.last_episode_completed
        info["episode_obsolete"] = self.last_episode_obsolete
        info["cand_task_ids"] = self._last_cand_task_ids
        info["action_mask"] = self._action_mask_matrix()

        return self._convert_observation(obs), info

    def step(self, action):
        self.step_count += 1
        is_decision_step = ((self.step_count - 1) % self.assignment_interval == 0)

        if is_decision_step:
            self._last_cand_task_ids = self._build_candidates()
            assignments = self._decode_action_vec(action)
            # print(assignments,'<- decoded assignments from action_vec:', action)
        else:
            assignments = None

        obs, reward, done, truncated, info_reward, info = self.base_env.step(
            assignments,
            assignment_interval=self.assignment_interval
        )
        # print("[DEBUG step] obs:", obs)
        if isinstance(reward, dict):
            reward = sum(reward.values())

        if not isinstance(info, dict):
            info = {}

        # reward components flattening (kept for compatibility with your callback)
        if isinstance(info_reward, dict):
            for k, v in info_reward.items():
                info[f"rew/{k}"] = v

        info["cand_task_ids"] = self._last_cand_task_ids
        info["action_mask"] = self._action_mask_matrix()
        info["decoded_assignments"] = assignments

        if done or truncated:
            info["episode_completed"] = sum(1 for t in self.base_env.tasks if t.is_droppedoff)
            info["episode_obsolete"] = sum(1 for t in self.base_env.tasks if t.is_obsolete(self.base_env.time_count))

        return self._convert_observation(obs), reward, done, truncated, info
    #Helper function to get mapping from true_id to row index in attribute_matrix for current observation
    def _build_id_to_row_from_matrix(self, attribute_matrix):
        attribute_matrix = np.asarray(attribute_matrix)
        if attribute_matrix.ndim != 2 or attribute_matrix.shape[1] < 1:
            return {}
        return {int(tid): int(i) for i, tid in enumerate(attribute_matrix[:, 0])}

    def _convert_observation(self, obs):
        """
        Convert base_env obs=(ego_graphs, attribute_matrix) into per-robot ego observations.

        For use_true_id=False:
        - ego_graphs keys are robot row indices (0..R-1)
        - edge endpoints are row indices into attribute_matrix
        - attribute_matrix[:, 0] contains true IDs, used only for candidate mapping
        """
        ego_graphs, attribute_matrix = obs
        # print("[DEBUG _convert_observation] attribute_matrix:", attribute_matrix)
        attribute_matrix = np.asarray(attribute_matrix, dtype=np.float32)
        N_global, F = attribute_matrix.shape
        self._latest_attribute_matrix = attribute_matrix.copy()
        R = self.n_robots
        K = self.k_max

        # padded outputs
        node_features = np.zeros((R, self.ego_max_nodes, F), dtype=np.float32)
        edge_index = np.zeros((R, 2, self.ego_max_edges), dtype=np.int64)
        num_nodes = np.zeros((R, 1), dtype=np.int64)
        num_edges = np.zeros((R, 1), dtype=np.int64)

        # Build task true_id -> row mapping from current matrix
        id_to_row = self._build_id_to_row_from_matrix(attribute_matrix)

        # candidate global rows from task ids
        cand_global = -np.ones((R, K), dtype=np.int64)
        for r in range(R):
            for k in range(K):
                tid = self._last_cand_task_ids[r][k]
                if tid is None:
                    continue
                cand_global[r, k] = int(id_to_row.get(int(tid), -1))

        cand_local = -np.ones((R, K), dtype=np.int64)

        for rid in range(R):
            # use row-index keys directly
            ego_list = ego_graphs.get(rid, [])

            if ego_list is None or len(ego_list) == 0:
                g_nodes = [rid] if 0 <= rid < N_global else []
                g2l = {int(g): i for i, g in enumerate(g_nodes)}

                n_i = min(len(g_nodes), self.ego_max_nodes)
                if n_i > 0:
                    node_features[rid, :n_i, :] = attribute_matrix[np.asarray(g_nodes[:n_i], dtype=np.int64)]
                num_nodes[rid, 0] = n_i
                num_edges[rid, 0] = 0

                for k in range(K):
                    g = int(cand_global[rid, k])
                    if g >= 0 and g in g2l and g2l[g] < self.ego_max_nodes:
                        cand_local[rid, k] = int(g2l[g])
                continue

            # ego_list is list of arrays [M,2] of row indices
            edges_g = np.concatenate(ego_list, axis=0).astype(np.int64).reshape(-1, 2)

            # nodes present in ego edges
            g_nodes = np.unique(edges_g.reshape(-1)).tolist()

            # ensure robot node is first
            if rid in g_nodes:
                g_nodes.remove(rid)
            g_nodes = [rid] + g_nodes

            if len(g_nodes) > self.ego_max_nodes:
                g_nodes = g_nodes[: self.ego_max_nodes]

            g2l = {int(g): int(i) for i, g in enumerate(g_nodes)}

            n_i = len(g_nodes)
            if n_i > 0:
                node_features[rid, :n_i, :] = attribute_matrix[np.asarray(g_nodes, dtype=np.int64)]
            num_nodes[rid, 0] = n_i

            # remap edges to local indices
            src_g = edges_g[:, 0]
            dst_g = edges_g[:, 1]
            keep = np.array([(int(s) in g2l and int(d) in g2l) for s, d in zip(src_g, dst_g)], dtype=bool)
            edges_kept = edges_g[keep]

            if edges_kept.size > 0:
                src_l = np.array([g2l[int(s)] for s in edges_kept[:, 0]], dtype=np.int64)
                dst_l = np.array([g2l[int(d)] for d in edges_kept[:, 1]], dtype=np.int64)
                edges_l = np.stack([src_l, dst_l], axis=0)
            else:
                edges_l = np.zeros((2, 0), dtype=np.int64)

            e_i = min(edges_l.shape[1], self.ego_max_edges)
            if e_i > 0:
                edge_index[rid, :, :e_i] = edges_l[:, :e_i]
            num_edges[rid, 0] = e_i

            # candidates: global row -> local row
            for k in range(K):
                g = int(cand_global[rid, k])
                if g >= 0 and g in g2l:
                    cand_local[rid, k] = int(g2l[g])
        if not hasattr(self, "_debug_wrap2"):
            self._debug_wrap2 = 0
        if self._debug_wrap2 < 3:
            self._debug_wrap2 += 1
            # print("[DEBUG wrapper2] rid0 cand_global:", cand_global[0].tolist())
            # print("[DEBUG wrapper2] rid0 ego_list len:", len(ego_graphs.get(0, [])))
            # print("[DEBUG wrapper2] rid0 g_nodes:", g_nodes[:15] if len(ego_graphs.get(0, [])) > 0 else [])
        action_mask = self._action_mask_matrix().astype(np.float32)

        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "action_mask": action_mask,
            "cand_node_idx": cand_local,
        }
#     def _convert_observation(self, obs):
#         """
#         Convert base_env obs=(ego_graphs, attribute_matrix) into per-robot ego observations.

#         ego_graphs: dict[rid] -> list of edge arrays, each [M,2] (GLOBAL node ids into attribute_matrix)
#         attribute_matrix: [N_global, F]
#         """
#         ego_graphs, attribute_matrix = obs
#         attribute_matrix = np.asarray(attribute_matrix, dtype=np.float32)
#         N_global, F = attribute_matrix.shape

#         R = self.n_robots
#         K = self.k_max

#         # padded outputs
#         node_features = np.zeros((R, self.ego_max_nodes, F), dtype=np.float32)
#         edge_index = np.zeros((R, 2, self.ego_max_edges), dtype=np.int64)
#         num_nodes = np.zeros((R, 1), dtype=np.int64)
#         num_edges = np.zeros((R, 1), dtype=np.int64)

#         # Build GLOBAL candidate indices from task ids mapping
#         cand_global = self._cand_node_idx_matrix_global()  # [R,K] global row indices or -1
#         cand_local = -np.ones((R, K), dtype=np.int64)

#         for rid in range(R):
#             ego_list = ego_graphs.get(rid, [])
#             if ego_list is None or len(ego_list) == 0:
#                 # minimal ego: include robot node itself if possible
#                 g_nodes = [rid] if 0 <= rid < N_global else []
#                 g2l = {int(g): i for i, g in enumerate(g_nodes)}
#                 n_i = min(len(g_nodes), self.ego_max_nodes)
#                 if n_i > 0:
#                     node_features[rid, :n_i, :] = attribute_matrix[np.asarray(g_nodes[:n_i], dtype=np.int64)]
#                 num_nodes[rid, 0] = n_i
#                 num_edges[rid, 0] = 0

#                 for k in range(K):
#                     g = int(cand_global[rid, k])
#                     if g >= 0 and g in g2l and g2l[g] < self.ego_max_nodes:
#                         cand_local[rid, k] = int(g2l[g])
#                 continue

#             edges_g = np.concatenate(ego_list, axis=0).astype(np.int64)  # [E,2] global
#             edges_g = edges_g.reshape(-1, 2)

#             # unique global nodes
#             g_nodes = np.unique(edges_g.reshape(-1)).tolist()

#             # ensure robot node included (assume robot global id == rid)
#             if 0 <= rid < N_global and rid not in g_nodes:
#                 g_nodes = [rid] + g_nodes

#             # truncate nodes
#             if len(g_nodes) > self.ego_max_nodes:
#                 g_nodes = g_nodes[: self.ego_max_nodes]

#             g2l = {int(g): int(i) for i, g in enumerate(g_nodes)}

#             # fill node features (skip invalid globals)
#             valid_g = [g for g in g_nodes if 0 <= int(g) < N_global]
#             n_i = min(len(valid_g), self.ego_max_nodes)
#             if n_i > 0:
#                 node_features[rid, :n_i, :] = attribute_matrix[np.asarray(valid_g[:n_i], dtype=np.int64)]
#             num_nodes[rid, 0] = n_i

#             # remap edges to local indices
#             src_g = edges_g[:, 0]
#             dst_g = edges_g[:, 1]
#             keep = np.array([(int(s) in g2l and int(d) in g2l) for s, d in zip(src_g, dst_g)], dtype=bool)
#             edges_kept = edges_g[keep]

#             if edges_kept.size > 0:
#                 src_l = np.array([g2l[int(s)] for s in edges_kept[:, 0]], dtype=np.int64)
#                 dst_l = np.array([g2l[int(d)] for d in edges_kept[:, 1]], dtype=np.int64)
#                 edges_l = np.stack([src_l, dst_l], axis=0)  # [2,E]
#             else:
#                 edges_l = np.zeros((2, 0), dtype=np.int64)

#             e_i = min(edges_l.shape[1], self.ego_max_edges)
#             if e_i > 0:
#                 edge_index[rid, :, :e_i] = edges_l[:, :e_i]
#             num_edges[rid, 0] = e_i

#             # map candidates global->local
#             for k in range(K):
#                 g = int(cand_global[rid, k])
#                 if g >= 0 and g in g2l:
#                     l = int(g2l[g])
#                     if l < self.ego_max_nodes:
#                         cand_local[rid, k] = l

#         action_mask = self._action_mask_matrix().astype(np.float32)

#         return {
#             "node_features": node_features,
#             "edge_index": edge_index,
#             "num_nodes": num_nodes,
#             "num_edges": num_edges,
#             "action_mask": action_mask,
#             "cand_node_idx": cand_local,
#         }


# -------------------------------------------------------------------------------
# ----------------------------------------End of this version-------------------
