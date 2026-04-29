from __future__ import annotations

import numpy as np
from typing import Any, Optional, Tuple, List, Dict

BASE_ROBOT_FEATURE_NAMES: List[str] = [
    "robot_loc_x",
    "robot_loc_y",
    "robot_free_capacity",
    "pad4",
    "pad5",
    "pad6",
    "pad7",
    "pad8",
    "pad9",
]
BASE_TASK_FEATURE_NAMES: List[str] = [
    "release_time_s",
    "waiting_time_s",
    "est_travel_time_s",
    "pickup_loc_x",
    "pickup_loc_y",
    "drop_loc_x",
    "drop_loc_y",
    "is_obsolete",
    "is_assigned",
]


def compute_feature_dim(
    use_xy_pickup: bool = False,
    use_node_type: bool = False,
    use_edge_rt: bool = False,
    use_ego_robot: bool = False,
    robot_commitment: str = "none",
    route_slots_k: int = 2,
) -> int:
    dim = 9
    if use_xy_pickup and not use_edge_rt:
        dim += 2
    if use_node_type:
        dim += 2
    if use_ego_robot:
        dim += 1
    return dim


def get_feature_names(
    use_xy_pickup: bool = False,
    use_node_type: bool = False,
    use_edge_rt: bool = False,
    use_ego_robot: bool = False,
    robot_commitment: str = "none",
    route_slots_k: int = 2,
) -> Tuple[List[str], List[str]]:
    robot_names = list(BASE_ROBOT_FEATURE_NAMES)
    task_names = list(BASE_TASK_FEATURE_NAMES)

    if use_xy_pickup and not use_edge_rt:
        robot_names += ["pad10", "pad11"]
        task_names = [
            "release_time_s",
            "waiting_time_s",
            "est_travel_time_s",
            "pickup_loc_x",
            "pickup_loc_y",
            "pickup_dx",
            "pickup_dy",
            "drop_loc_x",
            "drop_loc_y",
            "is_obsolete",
            "is_assigned",
        ]

    if use_node_type:
        robot_names += ["is_robot", "is_task"]
        task_names += ["is_robot", "is_task"]

    if use_ego_robot:
        robot_names += ["is_ego_robot"]
        task_names += ["is_ego_robot"]

    return robot_names, task_names


def expand_edge_features(
    edge_features: Optional[List[str]],
    robot_commitment: str = "none",
    route_slots_k: int = 2,
) -> List[str]:
    feats = list(edge_features or [])
    if robot_commitment != "route_slots":
        return feats
    for idx in range(int(route_slots_k)):
        slot_names = [
            f"slot{idx}_pu_dx",
            f"slot{idx}_pu_dy",
            f"slot{idx}_do_dx",
            f"slot{idx}_do_dy",
            f"slot{idx}_valid",
        ]
        for name in slot_names:
            if name not in feats:
                feats.append(name)
    return feats


def make_feature_fn(
    env: Any,
    use_xy_pickup: bool = False,
    normalize_features: bool = False,
    use_node_type: bool = False,
    use_edge_rt: bool = False,
    edge_features: Optional[List[str]] = None,
    use_ego_robot: bool = False,
    robot_commitment: str = "none",
    route_slots_k: int = 2,
):
    feature_dim = compute_feature_dim(
        use_xy_pickup=use_xy_pickup,
        use_node_type=use_node_type,
        use_edge_rt=use_edge_rt,
        use_ego_robot=use_ego_robot,
        robot_commitment=robot_commitment,
        route_slots_k=route_slots_k,
    )
    edge_features = expand_edge_features(edge_features, robot_commitment, route_slots_k)

    pos_scale = max(1.0, float(getattr(env, "radius", 1.0)))
    cap_scale = max(1.0, float(getattr(env, "robot_capacity", 1)))
    wait_scale = max(1.0, float(getattr(env, "batch_time", 1.0)))
    travel_scale = max(1.0, float(getattr(env, "batch_time", 1.0)))
    time_scale = max(1.0, float(getattr(env, "batch_time", wait_scale)))

    def _robot_xy(robot: Any) -> Tuple[float, float]:
        if robot is None:
            return 0.0, 0.0
        coord = np.asarray(getattr(robot, "coordinate", [0.0, 0.0]), dtype=np.float32).ravel()
        x = float(coord[0]) if len(coord) > 0 else 0.0
        y = float(coord[1]) if len(coord) > 1 else 0.0
        return x, y

    def _task_pickup_xy(task: Any) -> Tuple[float, float]:
        if task is None:
            return 0.0, 0.0
        coord = np.asarray(getattr(task, "pick_up_coord", [0.0, 0.0]), dtype=np.float32).ravel()
        x = float(coord[0]) if len(coord) > 0 else 0.0
        y = float(coord[1]) if len(coord) > 1 else 0.0
        return x, y

    def _task_drop_xy(task: Any) -> Tuple[float, float]:
        if task is None:
            return 0.0, 0.0
        coord = np.asarray(getattr(task, "drop_off_coord", [0.0, 0.0]), dtype=np.float32).ravel()
        x = float(coord[0]) if len(coord) > 0 else 0.0
        y = float(coord[1]) if len(coord) > 1 else 0.0
        return x, y

    def _append_node_type(out: np.ndarray, node_type: str) -> None:
        if not use_node_type:
            return
        if use_ego_robot:
            if out.shape[0] >= 3:
                out[-3] = 1.0 if node_type == "robot" else 0.0
                out[-2] = 1.0 if node_type == "task" else 0.0
        else:
            if out.shape[0] >= 2:
                out[-2] = 1.0 if node_type == "robot" else 0.0
                out[-1] = 1.0 if node_type == "task" else 0.0

    def _append_ego_robot(out: np.ndarray, is_ego: bool) -> None:
        if not use_ego_robot:
            return
        if out.shape[0] >= 1:
            out[-1] = 1.0 if is_ego else 0.0

    def _resolve_task(x: Any) -> Optional[Any]:
        if x is None:
            return None
        if hasattr(x, "pick_up_coord"):
            return x
        try:
            tid = int(x)
        except Exception:
            return None
        task_map = getattr(env, "taskid_to_task", {})
        return task_map.get(tid)

    def _edge_rt_features(robot: Any, task: Any) -> np.ndarray:
        out = np.zeros((len(edge_features),), dtype=np.float32)
        if not edge_features:
            return out
        rx, ry = _robot_xy(robot)
        px, py = _task_pickup_xy(task)
        dx = float(px - rx)
        dy = float(py - ry)
        if normalize_features:
            dx /= pos_scale
            dy /= pos_scale
        eta = float(np.sqrt(dx * dx + dy * dy))
        if normalize_features:
            eta = float(np.clip(eta / travel_scale, 0.0, 1.0))

        slot_values: Dict[str, float] = {}
        if robot_commitment == "route_slots":
            for s_idx in range(int(route_slots_k)):
                slot_values[f"slot{s_idx}_pu_dx"] = 0.0
                slot_values[f"slot{s_idx}_pu_dy"] = 0.0
                slot_values[f"slot{s_idx}_do_dx"] = 0.0
                slot_values[f"slot{s_idx}_do_dy"] = 0.0
                slot_values[f"slot{s_idx}_valid"] = 0.0

        for i, name in enumerate(edge_features):
            if name == "dx":
                out[i] = dx
            elif name == "dy":
                out[i] = dy
            elif name == "eta":
                out[i] = eta
            elif name == "is_ego_edge":
                out[i] = 0.0
            elif name in slot_values:
                out[i] = slot_values[name]
        return out

    def feature_fn(obj_a: Any, obj_b: Any, node_type: str) -> np.ndarray:
        out = np.zeros((feature_dim,), dtype=np.float32)
        now = float(getattr(env, "time_count", 0.0))

        if node_type in {"robot", "robot_ego", "robot_other"}:
            is_ego = node_type != "robot_other"
            rx, ry = _robot_xy(obj_a)
            if normalize_features:
                out[0], out[1] = rx / pos_scale, ry / pos_scale
            else:
                out[0], out[1] = rx, ry
            max_cap = float(getattr(obj_a, "maxCapacity", cap_scale))
            capacity = float(getattr(obj_a, "capacity", 0.0))
            free_cap = max(0.0, max_cap - capacity)
            out[2] = free_cap / cap_scale if normalize_features else free_cap
            _append_node_type(out, "robot")
            _append_ego_robot(out, is_ego)
            return out

        if node_type == "task":
            t = _resolve_task(obj_b)
            if t is None:
                return out
            release_time = float(getattr(t, "release_time", getattr(t, "t_release", 0.0)))
            waiting = max(0.0, now - release_time)
            est_travel = float(getattr(t, "estimatedTravelTime", 0.0))
            if normalize_features:
                out[0] = release_time / time_scale
                out[1] = waiting / wait_scale
                out[2] = est_travel / travel_scale
            else:
                out[0] = release_time
                out[1] = waiting
                out[2] = est_travel

            px, py = _task_pickup_xy(t)
            dx, dy = _task_drop_xy(t)
            if normalize_features:
                out[3], out[4] = px / pos_scale, py / pos_scale
            else:
                out[3], out[4] = px, py

            if use_xy_pickup and not use_edge_rt:
                rx, ry = _robot_xy(obj_a)
                if normalize_features:
                    out[5] = (px - rx) / pos_scale
                    out[6] = (py - ry) / pos_scale
                    out[7], out[8] = dx / pos_scale, dy / pos_scale
                else:
                    out[5] = px - rx
                    out[6] = py - ry
                    out[7], out[8] = dx, dy
                out[9] = 1.0 if bool(t.is_obsolete(now)) else 0.0
                out[10] = 1.0 if bool(getattr(t, "is_assigned", False)) else 0.0
            else:
                if normalize_features:
                    out[5], out[6] = dx / pos_scale, dy / pos_scale
                else:
                    out[5], out[6] = dx, dy
                out[7] = 1.0 if bool(t.is_obsolete(now)) else 0.0
                out[8] = 1.0 if bool(getattr(t, "is_assigned", False)) else 0.0

            _append_node_type(out, "task")
            _append_ego_robot(out, False)
            return out

        if node_type == "edge_rt":
            t = _resolve_task(obj_b)
            if t is None:
                return np.zeros((len(edge_features),), dtype=np.float32)
            return _edge_rt_features(obj_a, t)

        return out

    return feature_fn
