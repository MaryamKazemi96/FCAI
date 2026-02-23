"""
Final working wrapper that stores episode stats.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch


class WarehouseEnvSB3Final(gym.Env):
    """
    Wrapper that properly tracks episode statistics.
    """
    
    def __init__(self, base_env, assignment_interval=5):
        super().__init__()
        
        self.base_env = base_env
        self.assignment_interval = assignment_interval
        
        # Count total tasks
        total_tasks = 0
        if hasattr(base_env, 'tasks_batches') and isinstance(base_env.tasks_batches, list):
            for batch in base_env.tasks_batches:
                total_tasks += len(batch)
        else:
            total_tasks = base_env.n_tasks
        
        max_nodes = base_env.n_robots + total_tasks
        max_edges = max_nodes * base_env.n_robots * 4
        
        print(f"Wrapper initialized:")
        print(f"  Robots: {base_env.n_robots}, Tasks: {total_tasks}")
        print(f"  Assignment interval: {assignment_interval}")
        
        # Observation space
        self.observation_space = spaces.Dict({
            'node_features': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(max_nodes, base_env.feature_size),
                dtype=np.float32
            ),
            'edge_index': spaces.Box(
                low=0, high=max_nodes - 1,
                shape=(2, max_edges),
                dtype=np.int64
            ),
            'num_nodes': spaces.Box(
                low=0, high=max_nodes,
                shape=(1,), dtype=np.int64
            ),
            'num_edges': spaces.Box(
                low=0, high=max_edges,
                shape=(1,), dtype=np.int64
            )
        })
        
        # Action space
        self.action_space = spaces.Discrete(base_env.n_robots * 5 + 1)
        
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.total_tasks = total_tasks
        self.n_robots = base_env.n_robots
        
        # Tracking
        self.step_count = 0
        self.episode_count = 0
        
        # 🔥 CRITICAL: Store episode stats BEFORE reset
        self.last_episode_completed = 0
        self.last_episode_obsolete = 0
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # 🔥 SAVE STATS BEFORE RESET
        if self.episode_count > 0:  # Not first episode
            self.last_episode_completed = sum(1 for t in self.base_env.tasks if t.is_droppedoff)
            self.last_episode_obsolete = sum(1 for t in self.base_env.tasks if t.is_obsolete(self.base_env.time_count))
        
        # Now reset
        obs, info = self.base_env.reset()
        
        self.step_count = 0
        self.episode_count += 1
        
        # 🔥 Add stats to info dict so Monitor can track them
        info['episode_completed'] = self.last_episode_completed
        info['episode_obsolete'] = self.last_episode_obsolete
        
        return self._convert_observation(obs), info
    
    def step(self, action):
        """Step environment."""
        self.step_count += 1
        
        # Decision step logic
        is_decision_step = (self.step_count % self.assignment_interval == 0)
        
        if is_decision_step:
            assignments = self._action_to_assignment(action)
        else:
            assignments = None
        
        # Call base environment
        obs, reward, done, truncated, info_reward, info = self.base_env.step(
            assignments, 
            assignment_interval=self.assignment_interval
        )
        
        # Convert reward
        if isinstance(reward, dict):
            reward = sum(reward.values())
        
        # 🔥 If episode ending, save stats to info
        if done or truncated:
            info['episode_completed'] = sum(1 for t in self.base_env.tasks if t.is_droppedoff)
            info['episode_obsolete'] = sum(1 for t in self.base_env.tasks if t.is_obsolete(self.base_env.time_count))
        
        return self._convert_observation(obs), reward, done, truncated, info
    
    def _convert_observation(self, obs):
        """Convert observation."""
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
        
        return {
            'node_features': padded_features,
            'edge_index': padded_edges,
            'num_nodes': np.array([num_nodes], dtype=np.int64),
            'num_edges': np.array([num_edges], dtype=np.int64)
        }
    
    def _action_to_assignment(self, action):
        """Convert action to assignment."""
        available_tasks = self.base_env.get_available_task_ids()
        
        if len(available_tasks) == 0:
            return None
        
        available_robots = [
            rid for rid, robot in enumerate(self.base_env.robots)
            if robot.capacity < robot.maxCapacity
        ]
        
        if len(available_robots) == 0:
            return None
        
        task_idx = action % len(available_tasks)
        robot_idx = (action // len(available_tasks)) % len(available_robots)
        
        task_id = available_tasks[task_idx]
        robot_id = available_robots[robot_idx]
        
        return {robot_id: [task_id]}