"""
Feature function for the warehouse robot-task allocation domain.

feature_fn(robot, task, node_type) -> np.ndarray[F]

node_type values:
  "robot_ego"   – robot node in its own ego-graph (robot is the center)
  "robot_other" – robot node appearing in another robot's ego-graph (e.g. competitor)
  "task"        – task node; robot-conditioned (includes relative distance)

Feature dimension F = 9  (matches feature_size in training_config.yaml)

Robot feature layout (F=9):
  [0] x  (coordinate[0])
  [1] y  (coordinate[1])
  [2] yaw (coordinate[2])
  [3] capacity_ratio  (robot.capacity / robot.maxCapacity)
  [4..5] padded current task ids (slot 0, slot 1)  – 0 if empty
  [6] is_full_capacity  (1 if capacity >= maxCapacity else 0)
  [7] 0 (reserved)
  [8] 0 (reserved)

Task feature layout (F=9):
  [0] pickup_x
  [1] pickup_y
  [2] pickup_yaw
  [3] ddl_pick      (normalized by dividing by 3000)
  [4] ddl_dropoff   (normalized)
  [5] is_pickedup
  [6] is_assigned
  [7] dist_robot_to_pickup  (Euclidean, not normalized)
  [8] is_obsolete
"""
from __future__ import annotations

import numpy as np
from typing import Any, Optional

F = 9  # global constant – must match feature_size in training_config.yaml
_DDL_NORM = 3000.0  # rough upper bound on deadlines for normalization


def _robot_features(robot) -> np.ndarray:
    """Return 9-dim feature vector for a robot object."""
    coord = np.asarray(robot.coordinate, dtype=np.float32).ravel()
    x = float(coord[0]) if len(coord) > 0 else 0.0
    y = float(coord[1]) if len(coord) > 1 else 0.0
    yaw = float(coord[2]) if len(coord) > 2 else 0.0
    capacity = float(getattr(robot, "capacity", 0))
    max_cap = float(getattr(robot, "maxCapacity", 1))
    capacity_ratio = capacity / max(max_cap, 1.0)

    current_tasks = getattr(robot, "current_tasks_id", [])
    tid0 = float(current_tasks[0]) if len(current_tasks) > 0 else 0.0
    tid1 = float(current_tasks[1]) if len(current_tasks) > 1 else 0.0

    is_full = 1.0 if capacity >= max_cap else 0.0

    vec = np.array([x, y, yaw, capacity_ratio, tid0, tid1, is_full, 0.0, 0.0], dtype=np.float32)
    return vec


def feature_fn(robot: Any, task: Optional[Any], node_type: str) -> np.ndarray:
    """
    Main feature function consumed by build_padded_ego_batch.

    Parameters
    ----------
    robot : Robot object (from environment.py), or None for task-only calls
    task  : Tasks_variable object, or None for robot-only calls
    node_type : one of "robot_ego", "robot_other", "task"

    Returns
    -------
    np.ndarray of shape (F,) = (9,)
    """
    if node_type in ("robot_ego", "robot_other"):
        return _robot_features(robot)

    if node_type == "task":
        # Task node – robot-conditioned
        coord = np.asarray(getattr(task, "coordinate", np.zeros(3)), dtype=np.float32).ravel()
        pickup = np.asarray(getattr(task, "pick_up_coord", np.zeros(3)), dtype=np.float32).ravel()

        px = float(pickup[0]) if len(pickup) > 0 else 0.0
        py = float(pickup[1]) if len(pickup) > 1 else 0.0
        pyaw = float(pickup[2]) if len(pickup) > 2 else 0.0

        ddl_pick = float(getattr(task, "ddl_pick", 0.0)) / _DDL_NORM
        ddl_dropoff = float(getattr(task, "ddl_dropoff", 0.0)) / _DDL_NORM
        is_pickedup = float(getattr(task, "is_pickedup", 0))
        is_assigned = float(getattr(task, "is_assigned", 0))

        # robot-conditioned: distance from robot to task pickup
        if robot is not None:
            r_coord = np.asarray(robot.coordinate, dtype=np.float32).ravel()
            rx = float(r_coord[0]) if len(r_coord) > 0 else 0.0
            ry = float(r_coord[1]) if len(r_coord) > 1 else 0.0
            dist = float(np.sqrt((rx - px) ** 2 + (ry - py) ** 2))
        else:
            dist = 0.0

        is_obsolete_val = 0.0  # we don't have current_time here; set 0 as safe default

        vec = np.array(
            [px, py, pyaw, ddl_pick, ddl_dropoff, is_pickedup, is_assigned, dist, is_obsolete_val],
            dtype=np.float32,
        )
        return vec

    # fallback – return zeros
    return np.zeros(F, dtype=np.float32)


def get_feature_dim() -> int:
    """Return the feature dimension F."""
    return F
