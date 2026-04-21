
"""
SB3 wrapper (colleague-style masking):
- MultiDiscrete per-robot actions
- action_mask is part of observation (so the policy can mask logits)
- keeps reward component logging (rew/*) and episode stats
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch



class WarehouseEnvSB3Final(gym.Env):
    def __init__(self, base_env, assignment_interval=50, k_max=5):
        super().__init__()
        self.base_env = base_env
        self.assignment_interval = int(assignment_interval)

        self.k_max = int(k_max)
        self.noop_index = self.k_max
        self.n_robots = base_env.n_robots

        # Count total tasks
        total_tasks = 0
        if hasattr(base_env, 'tasks_batches') and isinstance(base_env.tasks_batches, list):
            for batch in base_env.tasks_batches:
                total_tasks += len(batch)
        else:
            total_tasks = base_env.n_tasks

        max_nodes = base_env.n_robots + total_tasks
        max_edges = max_nodes * base_env.n_robots * 4
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.total_tasks = total_tasks
        self.n_robots = base_env.n_robots

        print("Wrapper initialized:")
        print(f"  Robots: {base_env.n_robots}, Tasks: {total_tasks}")
        print(f"  Assignment interval: {self.assignment_interval}")
        print(f"  Action: MultiDiscrete([K+1]*R) with K={self.k_max} and NOOP={self.noop_index}")

        # Observation space (add action_mask)
        self.observation_space = spaces.Dict({
            'node_features': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.max_nodes, base_env.feature_size),
                dtype=np.float32
            ),
            'edge_index': spaces.Box(
                low=0, high=self.max_nodes - 1,
                shape=(2, self.max_edges),
                dtype=np.int64
            ),
            'num_nodes': spaces.Box(low=0, high=self.max_nodes, shape=(1,), dtype=np.int64),
            'num_edges': spaces.Box(low=0, high=self.max_edges, shape=(1,), dtype=np.int64),

            # NEW: per-robot action mask (0/1)
            'action_mask': spaces.Box(
                low=0.0, high=1.0,
                shape=(self.n_robots, self.k_max + 1),
                dtype=np.float32
            ),
            'cand_node_idx': spaces.Box(
                low=-1, high=self.max_nodes - 1,
                shape=(self.n_robots, self.k_max),
                dtype=np.int64
),
        })

        # CHANGED: MultiDiscrete action
        self.action_space = spaces.MultiDiscrete([self.k_max + 1] * self.n_robots)

        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.total_tasks = total_tasks

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
        """
        mapping = getattr(self.base_env, "trueid_idx_mapping", None)
        if mapping is None or len(mapping) != 2:
            return {}
        rows = np.asarray(mapping[0])
        ids = np.asarray(mapping[1])
        return {int(tid): int(r) for r, tid in zip(rows, ids)}


    def _cand_node_idx_matrix(self):
        """
        Return cand_node_idx: shape [R, K] with node indices in attributes_matrix, or -1.
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
    # def _build_candidates(self):
    #     """
    #     Build up to k_max candidate task IDs per robot.

    #     Current version is simple (consistent, not optimal):
    #     - If robot full: no candidates
    #     - Otherwise: take the first k_max available tasks (same set for all robots)
    #     Replace with per-robot nearest-task ranking later.
    #     """
    #     available_tasks = list(self.base_env.get_available_task_ids())
    #     cand = [[None] * self.k_max for _ in range(self.n_robots)]
    #     if len(available_tasks) == 0:
    #         return cand

    #     top = available_tasks[: self.k_max]
    #     for r in range(self.n_robots):
    #         robot = self.base_env.robots[r]
    #         if robot.capacity >= robot.maxCapacity:
    #             continue
    #         for k, task_id in enumerate(top):
    #             cand[r][k] = int(task_id)
    #     return cand
    def _build_candidates(self):
        """
        Build up to k_max candidate task IDs per robot, ordered by "best-first".

        Best-first heuristic (warehouse): smallest Euclidean distance from robot to task pickup.

        Slot semantics per robot:
        slot 0 = nearest pickup (greedy-best)
        slot 1 = 2nd nearest
        ...
        slot K-1 = Kth nearest
        slot K   = NOOP (handled elsewhere)
        """
        available_task_ids = list(self.base_env.get_available_task_ids())
        cand = [[None] * self.k_max for _ in range(self.n_robots)]
        if not available_task_ids:
            return cand

        # fast lookup id -> task object
        task_map = getattr(self.base_env, "taskid_to_task", {})
        if not task_map:
            # fallback: old behavior if map missing
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

            scored: list[tuple[float, int]] = []
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
        self.step_count = 0
        self.episode_count += 1

        self._last_cand_task_ids = self._build_candidates()

        if not isinstance(info, dict):
            info = {}
        info['episode_completed'] = self.last_episode_completed
        info['episode_obsolete'] = self.last_episode_obsolete
        info["cand_task_ids"] = self._last_cand_task_ids
        info["action_mask"] = self._action_mask_matrix()

        return self._convert_observation(obs), info

    def step(self, action):
        self.step_count += 1
        # is_decision_step = (self.step_count % self.assignment_interval == 0)
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
        # print(f"Step {self.step_count}: reward={reward}, done={done}, truncated={truncated}, info_reward={info_reward}")

        if isinstance(reward, dict):
            reward = sum(reward.values())

        if not isinstance(info, dict):
            info = {}

        # reward components
        if isinstance(info_reward, dict):
            for k, v in info_reward.items():
                info[f"rew/{k}"] = v
        info["cand_task_ids"] = self._last_cand_task_ids
        if done or truncated:
            info['episode_completed'] = sum(1 for t in self.base_env.tasks if t.is_droppedoff)
            info['episode_obsolete'] = sum(1 for t in self.base_env.tasks if t.is_obsolete(self.base_env.time_count))
            # info["cand_task_ids"] = self._last_cand_task_ids
            # info["action_mask"] = self._action_mask_matrix()
            # info["decoded_assignments"] = assignments  # for debugging
        info["cand_task_ids"] = self._last_cand_task_ids
        info["action_mask"] = self._action_mask_matrix()
        info["decoded_assignments"] = assignments

        return self._convert_observation(obs), reward, done, truncated, info

    def _convert_observation(self, obs):
        ego_graphs, attribute_matrix = obs

        all_edges = []
        for robot_id, ego_list in ego_graphs.items():
            if len(ego_list) > 0:
                edges = np.concatenate(ego_list, axis=0)
                all_edges.append(edges)

        if len(all_edges) > 0:
            edge_index = np.concatenate(all_edges, axis=0).T
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)

        num_nodes = min(len(attribute_matrix), self.max_nodes)
        num_edges = min(edge_index.shape[1], self.max_edges)

        padded_features = np.zeros((self.max_nodes, attribute_matrix.shape[1]), dtype=np.float32)
        padded_features[:num_nodes] = attribute_matrix[:num_nodes]

        padded_edges = np.zeros((2, self.max_edges), dtype=np.int64)
        padded_edges[:, :num_edges] = edge_index[:, :num_edges]

        # ✅ include mask in obs
        action_mask = self._action_mask_matrix()

        cand_node_idx = self._cand_node_idx_matrix()
        action_mask = self._action_mask_matrix()

        return {
            "node_features": padded_features,
            "edge_index": padded_edges,
            "num_nodes": np.array([num_nodes], dtype=np.int64),
            "num_edges": np.array([num_edges], dtype=np.int64),
            "action_mask": action_mask.astype(np.float32),
            "cand_node_idx": cand_node_idx,  
        }