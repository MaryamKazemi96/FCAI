import numpy as np
import gym
from gym import spaces
from pathlib import Path
import sys
import yaml
import random
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.graph_utils import get_edge_idx_graph, update_shared_attribute_matrix
# from utils.utils import enlarge_obstacles
from utils import utils as ut
from PIL import Image


class Planner:
    def __init__(self):
        # Load the map and create an obstacle grid
        root_path = Path(__file__).resolve().parent.parent.parent / "env"
        config_path = root_path / "ATC_wed.yaml"
        # print(config_path)
        with open(config_path, 'r') as file:
            params = yaml.safe_load(file)
        map_path = root_path / params['map_filename']
        self.map_img = Image.open(map_path).convert('L')
        self.map_resolution = params['map_resolution']
        self.Planning_resolution = params['Planning_resolution']
        self.threshold = params['obstacle_threshold']
        self.origin_x = params['origin_x'] 
        self.origin_y = params['origin_y']
        self.average_velocity = params['average_velocity']

    def get_obstacle_grid(self):
        img_w, img_h = self.map_img.size
        scale = self.map_resolution / self.Planning_resolution 
        grid_height, grid_width =  int(img_h * scale), int(img_w * scale)
        # print(scale, grid_height, grid_width, img_w, img_h) 
        grid = np.zeros((grid_height, grid_width), dtype=np.uint8)  # (rows, cols)
        for row in range(grid_height):
            for col in range(grid_width):
                px = int((col + 0.5) * img_w / grid_width)
                py = int((row + 0.5) * img_h / grid_height)
                grid[row, col] = 1 if self.map_img.getpixel((px, py)) < (self.threshold *255) else 0
        return grid
    
    def is_point_valid(self, point):
        # print(f"Checking validity of point: {point}")
        grid = self.get_obstacle_grid()
        w = point[1]
        h = point[0] 
        # print( row, col, grid.shape)
        if 0 <= h < grid.shape[0] and 0 <= w < grid.shape[1]:
            # print(f"Checking validity of point: {point} , valid")
            return grid[h,w] == 0
        # print(f"Checking validity of point: {point} , out of bounds")
        return False
    
    def get_plan(self, start, end):
        # print(f"Planning from {start} to {end}, obstacle grid shape: {self.get_obstacle_grid().shape}")
        """Generate a trajectory from origin to goal using A*."""
        grid = self.get_obstacle_grid()
        # print(grid, 'obstacle grid in get plan')
        # Use A* to find a path
        found, path = ut.astar(grid, start, end)
        # print(f"A* found: {found} after astar before smoothing, path: {path}")
        # if found:
        #     trajectory = ut.smooth_astar_path(path, self.Planning_resolution, self.origin_x, self.origin_y, self.average_velocity)
        # else:
        #         print(f"No valid path found")
        #         return 0, []
        return found, path # this path is all cell with [h,w] order
    
class Robot:
    def __init__(self, id, init_coordinates, maxCapacity=3, feature_size=9):
        self.robot_id = id
        self.coordinate = np.array([init_coordinates[1], init_coordinates[0], init_coordinates[2]], dtype=np.float32).reshape(1, -1)
        self.coordinate = self.coordinate[0]
        self.capacity = 0
        self.current_task_stage= []  # "pickup" or "dropoff"
        self.maxCapacity = maxCapacity
        self.feature_size = feature_size
        self.needs_replan = False
        self.reset()

    def reset(self):
        self.current_tasks_coords = [] # list(np.zeros((self.capacity, 2)))
        self.current_dropoff_coords = [] #list(np.zeros((self.capacity, 2)))
        self.current_tasks_id = [] #list(np.zeros(self.maxCapacity))
        self.trajectory = []
        self.goal_list = []
        self.capacity = 0 
        self.current_task_stage = []
        self.needs_replan = False

    def assign_trajectory(self, trajectory):
        self.trajectory = list(trajectory)
    
    def _task_reached(self, reach_tol=0.5):
        """
        Check whether the robot has reached any pickup or dropoff location.

        Returns:
            reached_pickup_ids: list[int]
            reached_dropoff_ids: list[int]
        """
        reached_pickup_ids = []
        reached_dropoff_ids = []

        if not self.current_tasks_coords:
            return reached_pickup_ids, reached_dropoff_ids

        # Convert to array safely
        coords = np.array(self.current_tasks_coords, dtype=np.float32)

        # Euclidean distance
        dists = np.linalg.norm(self.coordinate[:2] - coords[:, :2], axis=1)

        # Iterate BACKWARDS so pop() is safe
        for i in range(len(dists) - 1, -1, -1):

            # use tolerance (<= reach_tol means it's considered reached)
            if dists[i] > reach_tol:
                continue

            task_id = self.current_tasks_id[i]

            # -------- PICKUP --------
            if self.current_task_stage[i] == "pickup":
                reached_pickup_ids.append(task_id)

                # Switch target to dropoff
                self.current_tasks_coords[i] = self.current_dropoff_coords[i]
                self.goal_list[i] = self.current_dropoff_coords[i]
                self.current_task_stage[i] = "dropoff"
                # print(f"Robot {self.robot_id} picked up task {task_id}")

                # Re-order goals and mark for replan so the robot will head to the nearest goal next
                self.reorder_goals_by_distance()
                self.needs_replan = True

            # -------- DROPOFF --------
            else:
                reached_dropoff_ids.append(task_id)

                # Remove task completely
                self.current_tasks_coords.pop(i)
                self.current_dropoff_coords.pop(i)
                self.current_tasks_id.pop(i)
                self.current_task_stage.pop(i)
                self.goal_list.pop(i)
                self.capacity -= 1
                self.needs_replan = True
                # print(f"Robot {self.robot_id} dropped off task {task_id}")

        return reached_pickup_ids, reached_dropoff_ids

    def _task_reachedold(self):
        reached_tasks_dropoff_id = []
        reached_tasks_pickup_id = []

        # Calculate distances to all current task coordinates
        # print(self.coordinate, 'robot coord in task reached')
        # print(self.current_tasks_coords, 'current task coords in task reached')
        reached_tasks = (self.coordinate[:2] - np.array(self.current_tasks_coords)[:, :2]).sum(-1)
        # print(reached_tasks, 'reached tasks in task reached')

        # Iterate in reverse order to avoid index shifting issues
        for i in range(len(reached_tasks) - 1, -1, -1):
            is_reached = reached_tasks[i]
            if is_reached == 0:  # Robot has reached the task location
                # print(f"Robot reached task location for task {self.current_tasks_id[i]}")
                # print(i, self.current_tasks_coords, self.current_dropoff_coords, 'current task coord and drop off in task reached')
                if np.allclose(self.current_tasks_coords[i], self.current_dropoff_coords[i]):
                    # If the robot is at the drop-off location
                    reached_tasks_dropoff_id.append(self.current_tasks_id[i])
                    self.current_tasks_coords.pop(i)
                    self.current_dropoff_coords.pop(i)
                    self.current_tasks_id.pop(i)
                    # print(f"Robot dropped off task {reached_tasks_dropoff_id[-1]}")
                    self.goal_list.pop(0)
                    self.capacity -= 1
                else:
                    # If the robot is at the pickup location
                    reached_tasks_pickup_id.append(self.current_tasks_id[i])
                    # print(f"Robot reached pickup location for task {self.current_tasks_id[i]}")
                    self.current_tasks_coords[i] = self.current_dropoff_coords[i]  # Update target to drop-off location
                    self.goal_list[i] = self.current_dropoff_coords[i]
        # print(reached_tasks_pickup_id,reached_tasks_dropoff_id, 'reached pickup and dropoff ids in task reached')
        return reached_tasks_pickup_id, reached_tasks_dropoff_id



    def move(self):
        if self.trajectory:
            self.coordinate[:2] = np.array(self.trajectory.pop(0))
        return self.coordinate, self._task_reached()

    def add_task(self, task_id, task_pickup_coord, task_dropoff_coord):
        if self.capacity >= self.maxCapacity:
            return False
        self.current_tasks_id.append(task_id)
        self.current_tasks_coords.append(task_pickup_coord)
        self.current_dropoff_coords.append(task_dropoff_coord)
        self.goal_list.append(task_pickup_coord)
        self.current_task_stage.append("pickup")
        self.capacity += 1
        self.needs_replan = True
        return True
    
    def reorder_goals_by_distance(self):
        if not self.goal_list:
            return
        # Simple nearest-goal ordering
        current_pos = self.coordinate[:2]
        # sort goal_list and keep other related lists aligned
        order = sorted(range(len(self.goal_list)), key=lambda i: np.linalg.norm(current_pos - self.goal_list[i][:2]))
        # reorder goal_list and all parallel arrays to maintain consistency
        self.goal_list = [self.goal_list[i] for i in order]
        self.current_tasks_coords = [self.current_tasks_coords[i] for i in order]
        self.current_dropoff_coords = [self.current_dropoff_coords[i] for i in order]
        self.current_tasks_id = [self.current_tasks_id[i] for i in order]
        self.current_task_stage = [self.current_task_stage[i] for i in order]

    def get_attribute_array(self):
        coord_size = 3
        other_f_sizes = 1 + self.maxCapacity
        current_size = coord_size + other_f_sizes

        att = np.zeros(self.feature_size)
        # print(self.coordinate, 'robot coord in get attribute array')
        att[:3] = self.coordinate
        att[3] = self.capacity

        # Pad task IDs to maxCapacity
        padded_task_ids = np.zeros(self.maxCapacity)
        if len(self.current_tasks_id) > 0:
            padded_task_ids[:len(self.current_tasks_id)] = self.current_tasks_id

        att[4:4 + self.maxCapacity] = padded_task_ids
        return att

    # def get_attribute_array(self):
    #     coord_size = 3
    #     other_f_sizes = 1 + self.maxCapacity
    #     current_size = coord_size + other_f_sizes
    #     if current_size > self.feature_size:
    #         return False
    #     att = np.zeros(self.feature_size)
    #     att[:3] = self.coordinate
    #     att[3] = self.capacity
    #     att[4:current_size] = self.current_tasks_id
    #     return att
    
class Tasks_variable:
    def __init__(self, task_info_array, idle_allowance_time=60, init_time=0, feature_size=9):
        [task_id, w_origin, h_origin, yaw_origin,
         w_destination, h_destination, yaw_destination,
         t_release, pickupddl, estimatedTravelTime, dropoff_deadline] = task_info_array

        self.pick_up_coord = np.array([h_origin, w_origin, yaw_origin])
        self.drop_off_coord = np.array([h_destination, w_destination, yaw_destination])
        self.t_release = t_release
        self.id = task_id
        self.estimatedTravelTime = estimatedTravelTime
        self.idle_allowance_time = idle_allowance_time
        self.init_time = init_time
        self.feature_size = feature_size
        self.is_assigned = False
        self.reset(init_time=0)

    @property
    def is_active(self):
        if self.is_pickedup:
            return 0
        else:
            return 1
    
    def is_available(self, current_time=0):
        """Check if task is available based on release_time and current state."""
        if self.is_pickedup:
            return False
        # Task is available if current time has reached its release time
        return current_time >= self.release_time


    def is_obsolete(self, current_time=0):
        if (not self.is_pickedup) and (current_time > self.ddl_pick*10):
            return 1
        # if picked up but dropoff deadline passed and still not dropped
        if self.is_pickedup and (not self.is_droppedoff) and (current_time > self.ddl_dropoff * 10):
            return 1
        return 0
        # if self.ddl_pick > current_time and not self.is_pickedup:
        #     return 0
        # elif self.ddl_dropoff > current_time and not self.is_droppedoff:
        #     return 0

    def picked_up(self):
        self.is_pickedup = 1
        self.node_type = "pick_up"
        self.coordinate = self.drop_off_coord

    def drop_off(self):
        self.is_droppedoff = 1
        # self.is_active = 0

    def reset(self, init_time=0):
        if init_time != 0:
            self.init_time = init_time
        self.is_pickedup = 0
        self.is_droppedoff = 0
        self.release_time = int(self.t_release)
        self.ddl_pick = self.release_time + int(self.idle_allowance_time)
        self.ddl_dropoff = self.ddl_pick + int(self.idle_allowance_time)
        self.coordinate = np.array(self.pick_up_coord, dtype=np.float32).reshape(1, -1)
        # bookkeeping for one-time rewards / penalties
        self.picked_by = None                 # which robot picked it up
        self.pickup_reward_given = False      # one-time pickup reward flag
        self.delivered_by = None              # set at dropoff
        self.delivered_reward_given = False  # one-time delivery reward flag
        self.assigned_to = None               # robot id assigned (set when assigned)
        self.obsolete_penalty_given = False   # one-time obsolete penalty flag

    def picked_up(self):
        self.is_pickedup = 1
        self.node_type = "pick_up"
        self.coordinate = self.drop_off_coord
        # picked_by will be set by the environment (step) to the robot id that picked it

    def drop_off(self):
        self.is_droppedoff = 1
        # delivered_by will be set by the environment (step) to the robot id that dropped it

    def get_attribute_array(self):
        coord_size = 3
        other_f_sizes = 6
        current_size = coord_size + other_f_sizes
        if current_size > self.feature_size:
            return False
        att = np.zeros(self.feature_size)
        att[:3] = self.coordinate
        att[3:current_size] = [
            self.ddl_pick,
            self.ddl_dropoff,
            float(self.is_obsolete(self.init_time)),
            float(self.is_pickedup),
            float(self.is_droppedoff),
            float(self.is_assigned) 
        ]
        return att
class MultiTaskAllocationEnv(gym.Env):
    def __init__(self, agents_cont_coord_array, task_cont_coord_array, radius=2000, feature_size=9, use_true_id=False):
        super(MultiTaskAllocationEnv, self).__init__()
        self.planner = Planner()
        self.tasks_batches = task_cont_coord_array
        self.robot_capacity = 5
        self.radius = radius
        self.feature_size = feature_size
        self.agents_cont_coord_array = agents_cont_coord_array
        self.n_robots = len(self.agents_cont_coord_array)
        self.n_tasks = len(task_cont_coord_array)
        self.task_cont_coord_array = task_cont_coord_array
        self.time_count = 0
        self.batch_time = 180
        self.use_true_id = use_true_id
        self.current_traj = {i: [] for i in range(self.n_robots)}
        self.observation_space = spaces.Box(0, self.n_robots, shape=(feature_size,), dtype=int)
        self.action_space = spaces.Discrete(self.n_robots)
        self.reset()

    # Insert this inside the MultiTaskAllocationEnv class (e.g., after reset_tasks)

    def set_batch(self, task_batch):
        """
        Replace environment's current tasks with `task_batch` and reset task/robot
        state so an episode starts cleanly on this batch.

        task_batch should be an iterable (list/ndarray) of task_info rows
        in the same format your Tasks_variable expects (the same as
        elements of task_cont_coord_array).
        """
        # Replace the container used by reset_tasks
        self.task_cont_coord_array = task_batch

        # Rebuild the tasks list using your existing reset_tasks() helper
        # which reads from self.task_cont_coord_array
        self.reset_tasks()

        # Reset robots (positions/capacities). We reset robots so episodes
        # start from the same robot initialization each episode.
        self.reset_robot()

        # Rebuild the shared attribute matrix (robots then tasks)
        self._init_attribute_matrix()

        # reset time counter so deadlines / time-dependent behavior start fresh
        self.time_count = 0

    def set_multi_batch(self, task_batches_with_release_times):
        """
        Set multiple task batches with their release times for concurrent training.
        
        Args:
            task_batches_with_release_times: List of tuples [(batch_tasks, release_time), ...]
                where batch_tasks is an array of task_info rows and release_time is when 
                tasks in that batch become available.
        """
        # Combine all batches into a single task array
        # For each batch, modify the t_release field to be the batch's release_time
        all_tasks = []
        for batch_tasks, batch_release_time in task_batches_with_release_times:
            for task_info in batch_tasks:
                # task_info structure: [task_id, w_origin, h_origin, yaw_origin,
                #                       w_destination, h_destination, yaw_destination,
                #                       t_release, pickupddl, estimatedTravelTime, dropoff_deadline]
                # Replace t_release (index 7) with batch_release_time
                modified_task_info = task_info.copy()
                modified_task_info[7] = batch_release_time
                all_tasks.append(modified_task_info)
        
        # Set the combined tasks as the environment's task array
        if len(all_tasks) > 0:
            self.task_cont_coord_array = np.array(all_tasks)
        else:
            self.task_cont_coord_array = np.zeros((0, 11))
        
        # Rebuild tasks and reset state
        self.reset_tasks()
        self.reset_robot()
        self._init_attribute_matrix()
        self.time_count = 0
    
    def reset(self):
        self.time_count = 0
        self.assign_traj = []
        self.reset_tasks()
        self.reset_robot()
        self._init_attribute_matrix()
        return self._get_observations(update_node_att=False), {}

    def reset_robot(self):
        self.robots = []
        self.robots_id = []
        self.robots_info = []

        for oneagent in self.agents_cont_coord_array:
            rid = oneagent[0]
            coord = oneagent[1:]
            agent = Robot(rid, coord, self.robot_capacity)
            self.robots.append(agent)
            self.robots_id.append(rid)
            self.robots_info.append(agent.get_attribute_array())
        self.robots_info = np.array(self.robots_info)

    # def reset_tasksmain(self):
    #     self.tasks = []
    #     self.tasks_id = []
    #     self.tasks_info = []
    #     for tasks_info in self.task_cont_coord_array:
    #         one_task = Tasks_variable(tasks_info)
    #         self.tasks.append(one_task)
    #         self.tasks_id.append(one_task.id)
    #         self.tasks_info.append(one_task.get_attribute_array())
    #     self.tasks_info = np.array(self.tasks_info)
    #     self.taskid_to_task = {t.id: t for t in self.tasks}

    def reset_tasks(self):
        self.tasks = []
        self.tasks_id = []
        self.tasks_info = []
        for tasks_info in self.task_cont_coord_array:
            one_task = Tasks_variable(tasks_info)
            self.tasks.append(one_task)
            self.tasks_id.append(one_task.id)
            self.tasks_info.append(one_task.get_attribute_array())

        # Ensure tasks_info is 2D even when there are zero tasks
        if len(self.tasks_info) == 0:
            self.tasks_info = np.zeros((0, self.feature_size), dtype=np.float32)
        else:
            self.tasks_info = np.array(self.tasks_info, dtype=np.float32)

        # Update number of tasks and mapping
        self.n_tasks = len(self.tasks)
        self.taskid_to_task = {t.id: t for t in self.tasks}

    def _init_attribute_matrix(self):
        self.attributes_matrix = np.row_stack((self.robots_info, self.tasks_info))

    def _get_observations(self, update_node_att=True):
        if update_node_att:
            self.update_nodes_attr()
        self.update_graph()
        return self.list_ego_graphs, self.attributes_matrix

    def update_nodes_attr(self):
        self.attributes_matrix, self.trueid_idx_mapping = update_shared_attribute_matrix(
            self.attributes_matrix, self.robots_info, self.tasks_info
        )
    def update_nodes_attr2(self):
        """
        Update the shared attributes_matrix and trueid->index mapping.

        Try the incremental merge for efficiency. If anything looks incompatible
        (different id sets, shape mismatch, or an exception), fall back to a
        full rebuild that stacks robots_info and tasks_info from scratch.
        """
        # Recompute canonical robots_info and tasks_info arrays for the current objects
        robots_info = np.array([r.get_attribute_array() for r in self.robots], dtype=np.float32)

        if hasattr(self, "tasks") and len(self.tasks) > 0:
            tasks_info = np.array([t.get_attribute_array() for t in self.tasks], dtype=np.float32)
        else:
            tasks_info = np.zeros((0, self.feature_size), dtype=np.float32)

        # Fast path: try incremental update only if we already have an attributes_matrix
        try:
            if hasattr(self, "attributes_matrix") and self.attributes_matrix is not None and self.attributes_matrix.size != 0:
                updated_attr, updated_mapping = update_shared_attribute_matrix(
                    self.attributes_matrix, robots_info, tasks_info
                )
                # Sanity check: mapping size must match current number of tasks
                if isinstance(updated_mapping, dict) and len(updated_mapping) == tasks_info.shape[0]:
                    # Accept incremental result
                    self.attributes_matrix = updated_attr
                    self.trueid_idx_mapping = updated_mapping
                    self.robots_info = robots_info
                    self.tasks_info = tasks_info
                    return
                # else fall through to full rebuild
        except Exception:
            # any failure -> fall back to full rebuild
            pass

        # Full rebuild (safe)
        self.robots_info = robots_info
        self.tasks_info = tasks_info
        # Stack robots then tasks into attributes_matrix (handles zero tasks safely)
        self.attributes_matrix = np.row_stack((self.robots_info, self.tasks_info))
        # Rebuild mapping from true task id to row index (task rows start at self.n_robots)
        self.trueid_idx_mapping = {t.id: (i + self.n_robots) for i, t in enumerate(self.tasks)}
    def update_graph(self):
        self.list_ego_graphs, _, self.trueid_idx_mapping = get_edge_idx_graph(
            self.attributes_matrix, self.n_tasks, self.n_robots, self.radius, self.use_true_id
        )
        assigned_task_ids = [
        t.id for t in self.tasks if t.is_assigned
        ]
        id_to_index = {
        task_id: idx 
        for idx, task_id in enumerate(self.taskid_to_task.keys())
    }   
        # print(id_to_index, 'id to index in update graph')
        # print(assigned_task_ids, 'assigned task ids in update graph')
        mapped_indices = [id_to_index[t_id] for t_id in assigned_task_ids]
        # print(mapped_indices, 'mapped indices in update graph')
        # print(len(self.robots_id), 'len robots id in update graph')
        id_to_remove = [i + len(self.robots_id) for i in mapped_indices]
        # print(id_to_remove, 'mapped indices in update graph')
        # print(self.taskid_to_task)
        
        
        # print(self.list_ego_graphs, 'ego graphs before deleting assigned tasks')

        from utils.graph_utils import delete_taskid_in_graph
        delete_taskid_in_graph(self.list_ego_graphs, id_to_remove)
        # print(self.list_ego_graphs, 'ego graphs after deleting assigned tasks')
        # print(self.list_ego_graphs, 'ego graphs after deleting assigned tasks')

        return self.list_ego_graphs
    
    def resolve_conflicts(self, assignments):
        """
        Resolve conflicts using top-2 task choices per robot.
        Returns a dict {robot_id: assigned_task_id}.
        """
        final_assignments = {}
        taken_tasks = set()

        # Process robots in order of their IDs
        for rid in sorted(assignments.keys()):
            top2_tasks = assignments[rid]
            if not top2_tasks or top2_tasks[0] is None:
                continue  # no task available

            first_choice = top2_tasks[0]
            second_choice = top2_tasks[1] if len(top2_tasks) > 1 else None

            if first_choice not in taken_tasks:
                final_assignments[rid] = first_choice
                taken_tasks.add(first_choice)
            elif second_choice is not None and second_choice not in taken_tasks:
                final_assignments[rid] = second_choice
                taken_tasks.add(second_choice)
            else:
                # no available task
                final_assignments[rid] = None

        # Remove robots that couldn't be assigned any task
        final_assignments = {rid: tid for rid, tid in final_assignments.items() if tid is not None}
        return final_assignments

    def _get_task_from_assignment_id(self, task_identifier):
        """
        Accept either:
         - unique task id (task.id), or
         - node index used in graph (n_robots + task_index)
        and return the corresponding Tasks_variable instance or None.
        """
        # If it's exactly a unique task id in our mapping, return it directly
        if task_identifier in self.taskid_to_task:
            return self.taskid_to_task[task_identifier]

        # If it looks like a node index (>= n_robots), map to task_index
        if isinstance(task_identifier, int) and task_identifier >= len(self.robots_id):
            task_idx = task_identifier - len(self.robots_id)
            if 0 <= task_idx < len(self.tasks):
                return self.tasks[task_idx]
        # no mapping found
        return None
    
    def get_available_task_ids(self):
        # print([self.taskid_to_task[t_id].is_assigned for t_id in self.taskid_to_task], 'task assigned status in get available task ids')
        return [
            tid for tid, t in self.taskid_to_task.items()
            if t.is_available(self.time_count) and not t.is_assigned
        ]

    def _plan_robot_trajectory(self, robot):
        """
        Build a full trajectory for `robot` by planning from the robot's current position
        through each goal in robot.goal_list in order. Returns a list (possibly empty).
        """
        full_trajectory = []
        start = (int(robot.coordinate[0]), int(robot.coordinate[1]))
        for goal in robot.goal_list:
            end = (int(goal[0]), int(goal[1]))
            try:
                found, traj = self.planner.get_plan(start, end)
            except Exception as e:
                print(f"Planner error when planning {start} -> {end}: {e}")
                continue
            if found and traj:
                full_trajectory.extend(traj)
                start = end
            else:
                # if planning to this goal segment failed, skip it and continue
                continue
        return full_trajectory

    def step(self, list_t2r_assignments=None, assignment_interval=5):
        # print(f"\n=== Step {self.time_count} ===")
        # ensure we always have a dict to report which assignments were applied this step
        final_assignments_for_step = {}
        for ridx, robot in enumerate(self.robots):
            if len(robot.trajectory) > 0:
                coord, reached = robot.move()
                reached_pick, reached_drop = reached
            else:
                reached_pick, reached_drop = robot._task_reached()

            # Handle pickups
            for task_id in reached_pick:
                # map task identifier to task object (see mapping strategy below)
                task = self._get_task_from_assignment_id(task_id)
                if task is not None:
                    # print(f"Robot {robot.robot_id} or {ridx} picked up task {task.id} at step {self.time_count}")
                    # record which robot picked it up
                    task.picked_by = ridx
                    # try:
                    #     task.picked_by = robot.robot_id
                    # except Exception:
                    #     task.picked_by = ridx

                    # call object's picked_up() to flip flags and update coord
                    task.picked_up()
                    # allow reward() to grant pickup reward once
                    task.pickup_reward_given = False
                    # task.is_assigned = True

            # Handle drop-offsdef st
            for task_id in reached_drop:
                task = self._get_task_from_assignment_id(task_id)
                if task is not None:
                    # print(f"Robot {robot.robot_id} or {ridx} dropped off task {task.id} at step {self.time_count}")
                 # record which robot delivered this task so rewards can attribute properly
                    # robot here is the variable from the per-robot loop (same scope)
                    task.delivered_by = ridx
                    # try:
                    #     task.delivered_by = robot.robot_id
                    # except Exception:
                    #     # fall back to index if robot.robot_id not set
                    #     task.delivered_by = ridx
                    task.drop_off()
                    # allow reward() to grant delivery reward once
                    task.delivered_reward_given = False
                    # mark task no longer assigned (optional)
                    # task.is_assigned = False
            
            # After pickups or dropoffs, replan if needed (centralized place)
            if robot.needs_replan or (len(robot.trajectory) == 0 and robot.goal_list):
                full_trajectory = self._plan_robot_trajectory(robot)
                if full_trajectory:
                    robot.assign_trajectory(full_trajectory)
                # clear the replan flag even if planning failed; we'll retry next step if needed
                robot.needs_replan = False

        if self.time_count % assignment_interval == 0:
            # print(f"\n--- Assignment at Step {self.time_count} ---")
            
            # self.update_nodes_attr()
            # self.update_graph()

            # print(f"Step {self.time_count}: Running assignment")
            if list_t2r_assignments is None:
                list_t2r_assignments = {}
                available_tasks = self.get_available_task_ids()
                # print([t.id for t in available_tasks], "unassigned tasks before assignment")
                # print(available_tasks, "available tasks before assignment")
                for rid in range(self.n_robots):
                    if not available_tasks:
                        list_t2r_assignments[rid] = [None, None]
                        continue

                    top2 = np.random.choice(
                        available_tasks,
                        size=min(2, len(available_tasks)),
                        replace=False
                    ).tolist()

                    if len(top2) == 1:
                        top2.append(top2[0])

                    list_t2r_assignments[rid] = top2

            print("Assignments before conflict resolution:", list_t2r_assignments)
            # print(f"\n=== Step {self.time_count} Assignment proposals (per robot) ===")
            # for rid in sorted(list_t2r_assignments.keys()):
            #     print(f"  R{rid}: {list_t2r_assignments[rid]}")

            # show available task count
            available_tasks = self.get_available_task_ids()
            # print(f"  Available tasks count: {len(available_tasks)} -> ids sample: {available_tasks[:20]}")

            # Resolve conflicts using top-2 list ---
            resolved_assignments = self.resolve_conflicts(list_t2r_assignments)
            print("Assignments after conflict resolution:", resolved_assignments)
            self._get_final_assigment(resolved_assignments)
            # after assignment application (one-shot or iterative)
            final_assignments_for_step = resolved_assignments.copy() if 'resolved_assignments' in locals() else {}
            # if you implemented iterative assignment, you can accumulate all applied assignments into this dict
            # print(list_t2r_assignments, 'list_t2r_assignments in step')
        # 4Compute reward
        reward = self.reward()
        # terminated = all([not t.is_active for t in self.tasks])
        # Terminate only when all tasks are either dropped off or obsolete
        terminated = all([t.is_droppedoff or t.is_obsolete(self.time_count) for t in self.tasks])
        # if self.time_count %20==0:
        #     print(f"Terminated: id dropedoff, is obsolete, is_assigned, is picked up {[[t.is_droppedoff , t.is_obsolete(self.time_count), t.is_assigned, t.is_pickedup] for t in self.tasks]}")

        truncated = self.time_count >= self.batch_time

        obs = self._get_observations(update_node_att=True)  # graph already updated
        self.time_count += 1
        info = {"resolved_assignments": final_assignments_for_step}
        return obs, reward, terminated, truncated, info

    def _get_task_from_assignment_id(self, task_identifier):
        """
        Accept either:
         - unique task id (task.id), or
         - node index used in graph (n_robots + task_index)
        and return the corresponding Tasks_variable instance or None.
        """
        # If it's exactly a unique task id in our mapping, return it directly
        if task_identifier in self.taskid_to_task:
            return self.taskid_to_task[task_identifier]

        # If it looks like a node index (>= n_robots), map to task_index
        if isinstance(task_identifier, int) and task_identifier >= len(self.robots_id):
            task_idx = task_identifier - len(self.robots_id)
            if 0 <= task_idx < len(self.tasks):
                return self.tasks[task_idx]
        # no mapping found
        return None
    
    def _get_final_assigment(self, assignments):
        """
        assignments: dict {robot_id: task_identifier}
          - task_identifier may be either the unique task.id or a graph node index (n_robots + task_index)
        Behavior:
          - For each robot, attempt to add the task to the robot (respecting capacity).
          - If add_task succeeds, mark task.is_assigned = True.
          - Reorder robot.goal_list by distance and set robot.needs_replan = True
            (actual planning happens in step(), centrally).
        """
        # Process in deterministic order
        for rid in sorted(assignments.keys()):
            # validate robot id
            if rid < 0 or rid >= len(self.robots):
                # print(f"_get_final_assigment: invalid robot id {rid}, skipping")
                continue

            robot = self.robots[rid]
            task_identifier = assignments[rid]

            # skip if robot is full
            if robot.capacity >= robot.maxCapacity:
                # print(f"Robot {rid} at capacity {robot.capacity}/{robot.maxCapacity}, skipping assignment {task_identifier}")
                continue

            # Map identifier to task object (handles both unique task ids and graph node indices)
            task = self._get_task_from_assignment_id(task_identifier)
            if task is None:
                # print(f"_get_final_assigment: could not map task identifier {task_identifier} to a task object")
                continue

            # Skip if not available (release_time) or already assigned
            if not task.is_available(self.time_count) or task.is_assigned:
                # print(f"Task {task.id} not available or already assigned; skipping")
                continue

            # Attempt to add the task to the robot (this respects robot.maxCapacity)
            success = robot.add_task(task.id, task.pick_up_coord, task.drop_off_coord)
            if not success:
                # couldn't add task (capacity race), skip marking is_assigned
                # print(f"Robot {rid} failed to add task {task.id} (maybe capacity).")
                continue

            task.is_assigned = True
            task.assigned_to = rid 
            # mark the task assigned only after successful add_task
            task.is_assigned = True

            # Reorder goals so the robot will next go to the closest goal (could be another pickup)
            robot.reorder_goals_by_distance()

            # Mark that a replan is required; actual planning will be done in step()
            robot.needs_replan = True
    
    def _get_final_assigment2(self, list_t2r_assignments):

        for rid, task_id in list_t2r_assignments.items():
            # print(list_t2r_assignments, 'assignments in final assignment')
            robot = self.robots[rid]

            # Skip if robot is at max capacity
            if robot.capacity >= robot.maxCapacity:
                continue

            task_idx = task_id - len(self.robots_id)
            # print(task_idx, task_id, len(self.robots_id),'task idx and task_id in final assignment')
            # print(len(self.robots_id), 'len robots id in final assignment')
            if task_idx < 0 or task_idx >= len(self.tasks):
                continue
            task = self.tasks[task_idx]
            if not task.is_available(self.time_count):
                continue

            # Add task to robot's goal list
            success = robot.add_task(task_id, task.pick_up_coord, task.drop_off_coord)
            if not success:
                continue
            # Mark as assigned
            task.is_assigned = True

            # Reorder goals based on distance
            robot.reorder_goals_by_distance()

            # Plan trajectory to first goal in goal_list
            full_trajectory = []
            start = (int(robot.coordinate[0]), int(robot.coordinate[1]))
            for goal in robot.goal_list:
                end = (int(goal[0]), int(goal[1]))
                found, traj = self.planner.get_plan(start, end)
                if found and traj is not None:
                    full_trajectory.extend(traj)
                    start = end
            robot.assign_trajectoryrewards(full_trajectory)
            # print(f"Robot {rid} assigned tasks {robot.current_tasks_id}, trajectory length {len(robot.trajectory)}")
    
    
    def _get_final_assigmentOld(self, list_t2r_assignments):
        # print(list_t2r_assignments, 'list_t2r_assignments in final assignment')
        # for robot_ids, task_id in list_t2r_assignments.items():
            # print(f"Task {task_id} assigned to robots {robot_ids}")
        # print()
        assignments = [
        (task_id, rid)
        for task_id, robot_ids in list_t2r_assignments.items()
        for rid in (robot_ids if isinstance(robot_ids, (list, set)) else [robot_ids])  # Ensure robot_ids is iterable
    ]
        for rid , task_id in assignments:
            # robot_idx = list(self.robots_id).index(rid)
            # robot_idx = int(self.robots_id[rid])
            # print(robot_idx, 'robot_idx')
            # robot = self.robots[robot_idx]
            robot = self.robots[rid]

            # task_idx = list(self.tasks_id).index(task_id)
            # task = self.tasks[task_idx]
            task = self.tasks[task_id-len(self.robots_id)]
            robot.add_task(task_id, task.pick_up_coord, task.drop_off_coord)
            # print(robot.coordinate[0][:2], 'robot.coordinate')
            # print(task.pick_up_coord[:2], 'task.pick_up_coord')
            # self.current_traj[robot_idx] 
            # print(robot.coordinate[0][:2], task.pick_up_coord[:2], 'robot and task coord in final assignment')
            # print(robot.coordinate, task.coordinate, 'robot and task coord int in final assignment')
            found,trajectory= self.planner.get_plan(start=(int(robot.coordinate[0]),int(robot.coordinate[1]))
                                                                 , end=(int(task.pick_up_coord[0]), int(task.pick_up_coord[1])))
            # print(f"Planning trajectory for robot {rid} to task {task_id}: Found={found}")
            if trajectory is None:
                robot.trajectory = []
            else:                
                robot.assign_trajectory(trajectory)
        # print(f"Assigned trajectory to robot {rid}: {trajectory}")
    def _reward_tasks_completion_per_robot(self, robot):
        """
        Count tasks delivered by this robot (uses task.delivered_by set at dropoff).
        Returns integer count.
        """
        delivered_by_id = getattr(robot, "robot_id", None)
        return sum(1 for t in self.tasks if t.is_droppedoff and getattr(t, "delivered_by", None) == delivered_by_id)

    # (4) Replace your reward() with the following function

    def reward(self, debug=False):
        """
        One-time-event reward function.

        - pickup_reward: one-time (+1) awarded to robot that actually picked the task.
        - completion_reward_per_task: one-time (+20) awarded to delivering robot when drop_off() occurs.
        - obsolete_penalty: one-time (-2) awarded to the robot the task was assigned to if it becomes obsolete while assigned.
        - step_penalty_per_robot: small per-step penalty (optional, set to 0 to disable).

        This function mutates task flags (pickup_reward_given / delivered_reward_given /
        obsolete_penalty_given) so each event is only paid once.
        """
        n_robots = max(1, len(self.robots))

        # Tunable constants
        pickup_reward = 1.0
        completion_reward_per_task = 10.0
        obsolete_penalty_amount = -10.0
        step_penalty_per_robot = -0.1  # set to 0.0 if you don't want any per-step penalty

        # Initialize per-robot rewards
        rewards = {rid: 0.0 for rid in range(n_robots)}

        # 1) One-time pickup rewards
        for task in self.tasks:
            if task.is_pickedup and (not getattr(task, "pickup_reward_given", False)):
                pid = getattr(task, "picked_by", None) or getattr(task, "assigned_to", None)
                if pid is not None and 0 <= pid < n_robots:
                    rewards[pid] += pickup_reward
                task.pickup_reward_given = True

        # 2) One-time delivery rewards
        for task in self.tasks:
            if task.is_droppedoff and (not getattr(task, "delivered_reward_given", False)):
                did = getattr(task, "delivered_by", None)
                if did is not None and 0 <= did < n_robots:
                    rewards[did] += completion_reward_per_task
                task.delivered_reward_given = True

        # 3) One-time obsolete penalty (apply only if task was assigned and not picked up)
        for task in self.tasks:
            if (not task.is_pickedup) and task.is_assigned and task.is_obsolete(self.time_count):
                if not getattr(task, "obsolete_penalty_given", False):
                    assigned = getattr(task, "assigned_to", None)
                    if assigned is None:
                        # fallback: try to find robot that had this task in its list
                        for rid, robot in enumerate(self.robots):
                            if task.id in robot.current_tasks_id:
                                assigned = robot.robot_id if hasattr(robot, "robot_id") else rid
                                break
                    if assigned is not None and 0 <= assigned < n_robots:
                        rewards[assigned] += obsolete_penalty_amount
                    task.obsolete_penalty_given = True
                    # Optionally clear is_assigned so it doesn't appear in future assignment lists
                    task.is_assigned = False

        # 4) Small per-step penalty if you want to encourage speed
        if step_penalty_per_robot != 0.0:
            for rid in range(n_robots):
                rewards[rid] += step_penalty_per_robot

        if debug:
            info = {
                "sum_rewards": sum(rewards.values()),
                "pickup_reward": pickup_reward,
                "completion_reward": completion_reward_per_task,
                "obsolete_penalty_amount": obsolete_penalty_amount,
                "step_penalty_per_robot": step_penalty_per_robot,
                "n_robots": n_robots,
                "abandoned_count": sum(1 for t in self.tasks if t.is_obsolete(self.time_count) and not t.is_pickedup)
            }
            return rewards, info

        return rewards
    
    def rewardOld(self, debug=False):
        """
        Return per-robot reward dictionary with reasonable scaling:
         - completion_reward_per_task: positive reward for tasks this robot delivered.
         - step_penalty_per_robot: small negative per step (not multiplied by time_count).
         - capacity_reward: small positive per carried task (encourages using capacity).
         - abandoned_task_penalty: global penalty distributed evenly across robots.
        """
        n_robots = max(1, len(self.robots))
        # Tunable constants
        completion_reward_per_task = 20.0
        step_penalty_per_robot = -0.1   # per step, per robot
        capacity_reward_weight = 1   # per carried item
        abandoned_task_penalty_total_weight = -2.0  # total penalty per abandoned task

        # global abandoned count
        abandoned_count = sum(1 for t in self.tasks if t.is_obsolete(self.time_count) and not t.is_pickedup)
        total_abandon_penalty = abandoned_count * abandoned_task_penalty_total_weight
        # distribute the global penalty evenly across robots (avoids n_robots multiplicative effect)
        abandon_penalty_per_robot = total_abandon_penalty / n_robots

        rewards = {}
        total_completed = 0
        for rid, robot in enumerate(self.robots):
            r = 0.0
            # reward for tasks completed by this robot
            completed = self._reward_tasks_completion_per_robot(robot)
            r += completed * completion_reward_per_task

            # small per-step penalty (not cumulative time_count) to encourage speed
            r += step_penalty_per_robot

            # reward for using capacity (optional)
            r += robot.capacity * capacity_reward_weight

            # distribute abandoned penalty
            r += abandon_penalty_per_robot

            rewards[rid] = r
        if debug:
            info = {
                "n_robots": n_robots,
                "total_completed": total_completed,
                "abandoned_count": abandoned_count,
                "total_abandon_penalty": total_abandon_penalty,
                "per_robot_step_penalty": step_penalty_per_robot,
                "completion_reward_per_task": completion_reward_per_task,
                "sum_rewards": sum(rewards.values())
            }
            return rewards, info

        return rewards
  

    # def _reward_tasks_completion(self):s
    #     return sum([1 for t in self.tasks if t.is_droppedoff])
    def _reward_tasks_completion_per_robot(self, robot):
        # Count tasks completed by this robot
        completed_task_ids = set(robot.current_tasks_id)
        reward = 0
        for task in self.tasks:
            if task.is_droppedoff and task.id in completed_task_ids:
                reward += 1
        return reward

    def _reward_steps(self):
        return -self.time_count * 1

    def _reward_agent_capacity(self):
        # print(f"{[r.capacity for r in self.robots]} robot capacities in reward agent capacity")
        return [max(0, r.capacity) for r in self.robots]
        # return max(0, self.robots[rid].capacity)

    def _reward_tasks_abandone_penalty(self):
        return -sum(1 for t in self.tasks if t.is_obsolete(self.time_count) and not t.is_pickedup)

    def render(self, mode="human"):
        pass