"""
Graph construction utilities for batched ego-graphs.

build_padded_ego_batch(...) converts per-robot candidate lists into a fixed-shape
batch of ego-graphs (robot + candidate tasks) suitable for GNN-based policies.
"""
from __future__ import annotations
from typing import Callable, Sequence, Any, List, Tuple, Optional, Dict
import numpy as np


def build_padded_ego_batch(
    *,
    robots: Sequence[Optional[Any]],
    tasks: Sequence[Any],
    candidate_lists: Sequence[Sequence[int]],  # per-robot, indices into `tasks`
    N_max: int,
    E_max: int,
    K_max: int,
    F: int,
    G: int,
    feature_fn: Callable[[Any, Any, str], np.ndarray],
    two_hop: bool = False,
    two_hop_directed: bool = False,
    normalize_features: bool = False,
    vicinity_m: float = 0.0,
    use_edge_rt: bool = False,
    edge_feat_dim: int = 0,
    edge_features: Optional[List[str]] = None,
) -> Tuple[Dict[str, np.ndarray], List[List[Optional[int]]]]:
    """
    Build per-robot ego-graphs as star graphs:
      node 0 = robot, nodes 1..M = that robot's candidate tasks (truncated to fit N_max).
      Edges: undirected star (robot<->task).

    Returns:
      obs dict and a parallel list cand_task_ids[R][K] mapping each slot to a task_id (or None if padded).
    """
    R = len(robots)
    x = np.zeros((R, N_max, F), np.float32)
    node_mask = np.zeros((R, N_max), np.uint8)
    edge_index = np.zeros((R, 2, E_max), np.int64)
    edge_mask = np.zeros((R, E_max), np.uint8)
    edge_attr = np.zeros((R, E_max, edge_feat_dim), np.float32) if edge_feat_dim > 0 else None
    edge_features = list(edge_features or [])
    ego_edge_idx = edge_features.index("is_ego_edge") if "is_ego_edge" in edge_features else None
    cand_idx = np.zeros((R, K_max), np.int64)
    cand_mask = np.zeros((R, K_max), np.uint8)

    cand_task_ids: List[List[Optional[int]]] = [[None] * K_max for _ in range(R)]

    vicinity_threshold = float(vicinity_m)
    pos_scale = max(1.0, float(vicinity_m))

    def _scale_position(xy: Tuple[float, float]) -> Tuple[float, float]:
        if normalize_features:
            return xy[0] * pos_scale, xy[1] * pos_scale
        return xy

    robot_xy_cache: List[Tuple[float, float]] = []
    for rid in robots:
        try:
            rf = feature_fn(rid, None, "robot_other")
            robot_xy_cache.append(_scale_position((float(rf[0]), float(rf[1]))))
        except Exception:
            robot_xy_cache.append((0.0, 0.0))

    for i in range(R):
        rid = robots[i]
        try:
            x[i, 0, :] = feature_fn(rid, None, "robot_ego")
        except Exception:
            pass
        node_mask[i, 0] = 1

        cands = list(candidate_lists[i]) if i < len(candidate_lists) else []
        max_tasks_here = max(0, N_max - 1)
        cands = cands[: min(K_max, max_tasks_here)]

        e_ptr = 0
        next_node_id = 1 + len(cands)
        competitor_nodes: Dict[str, int] = {}
        for local_slot, task_idx in enumerate(cands):
            node_id = 1 + local_slot
            if node_id >= N_max or task_idx >= len(tasks):
                break

            t = tasks[task_idx]
            try:
                x[i, node_id, :] = feature_fn(rid, t, "task")
            except Exception:
                pass
            node_mask[i, node_id] = 1

            task_xy = None
            if two_hop:
                try:
                    tf = feature_fn(rid, t, "task")
                    task_xy = _scale_position((float(tf[3]), float(tf[4])))
                except Exception:
                    task_xy = None

            cand_idx[i, local_slot] = node_id
            cand_mask[i, local_slot] = 1

            try:
                task_id = getattr(t, "id", None)
                cand_task_ids[i][local_slot] = int(task_id) if task_id is not None else None
            except Exception:
                cand_task_ids[i][local_slot] = None

            if e_ptr + 2 <= E_max:
                edge_index[i, 0, e_ptr] = 0
                edge_index[i, 1, e_ptr] = node_id
                edge_mask[i, e_ptr] = 1
                if edge_attr is not None and use_edge_rt:
                    try:
                        edge_attr[i, e_ptr, :] = feature_fn(rid, t, "edge_rt")
                        if ego_edge_idx is not None:
                            edge_attr[i, e_ptr, ego_edge_idx] = 1.0
                    except Exception:
                        pass
                e_ptr += 1

                edge_index[i, 0, e_ptr] = node_id
                edge_index[i, 1, e_ptr] = 0
                edge_mask[i, e_ptr] = 1
                if edge_attr is not None and use_edge_rt:
                    try:
                        edge_attr[i, e_ptr, :] = feature_fn(rid, t, "edge_rt")
                        if ego_edge_idx is not None:
                            edge_attr[i, e_ptr, ego_edge_idx] = 1.0
                    except Exception:
                        pass
                e_ptr += 1

            if two_hop and task_xy is not None:
                for j, other_rid in enumerate(robots):
                    if other_rid is None or j == i:
                        continue
                    rx, ry = robot_xy_cache[j]
                    dx = rx - task_xy[0]
                    dy = ry - task_xy[1]
                    if (dx * dx + dy * dy) > (vicinity_threshold * vicinity_threshold):
                        continue

                    other_key = str(getattr(other_rid, "robot_id", j))
                    if other_key in competitor_nodes:
                        other_node_id = competitor_nodes[other_key]
                    else:
                        if next_node_id >= N_max:
                            continue
                        other_node_id = next_node_id
                        next_node_id += 1
                        competitor_nodes[other_key] = other_node_id
                        try:
                            x[i, other_node_id, :] = feature_fn(other_rid, None, "robot_other")
                        except Exception:
                            pass
                        node_mask[i, other_node_id] = 1

                    if e_ptr + 1 <= E_max:
                        edge_index[i, 0, e_ptr] = node_id
                        edge_index[i, 1, e_ptr] = other_node_id
                        edge_mask[i, e_ptr] = 1
                        if edge_attr is not None and use_edge_rt:
                            try:
                                edge_attr[i, e_ptr, :] = feature_fn(other_rid, t, "edge_rt")
                                if ego_edge_idx is not None:
                                    edge_attr[i, e_ptr, ego_edge_idx] = 0.0
                            except Exception:
                                pass
                        e_ptr += 1

                    if (not two_hop_directed) and (e_ptr + 1 <= E_max):
                        edge_index[i, 0, e_ptr] = other_node_id
                        edge_index[i, 1, e_ptr] = node_id
                        edge_mask[i, e_ptr] = 1
                        if edge_attr is not None and use_edge_rt:
                            try:
                                edge_attr[i, e_ptr, :] = feature_fn(other_rid, t, "edge_rt")
                                if ego_edge_idx is not None:
                                    edge_attr[i, e_ptr, ego_edge_idx] = 0.0
                            except Exception:
                                pass
                        e_ptr += 1

    obs = dict(
        x=x,
        node_mask=node_mask,
        edge_index=edge_index,
        edge_mask=edge_mask,
        **({"edge_attr": edge_attr} if edge_attr is not None else {}),
        cand_idx=cand_idx,
        cand_mask=cand_mask,
    )
    return obs, cand_task_ids
