
# import random
# import math
# import sys
# from pathlib import Path
# import yaml
# from PIL import Image
# import numpy as np
# sys.path.append(str(Path(__file__).resolve().parent.parent))

# from utils.utils import astar, discritized_path  # Ensure astar is implemented in utils.py
# from environment.environment import Planner
# class DataGenerator:
#     def __init__(self, x_min, x_max, y_min, y_max, max_waiting_time,
#                   max_travel_delay_percentage,planning_resolution,
#                     planner, origin_x, origin_y ):
#         self.planning_resolution = planning_resolution
#         self.h_min = 0
#         self.h_max = abs(y_min - y_max) / planning_resolution
#         self.w_min = 0
#         self.w_max = abs(x_min - x_max) / planning_resolution
#         self.origin_x =origin_x
#         self.origin_y =origin_y
#         self.max_waiting_time = max_waiting_time
#         self.max_travel_delay_percentage = max_travel_delay_percentage
#         self.planner = planner  
        

#     def generate_agents(self, num_agents):
#         agents = []
#         for i in range(num_agents):
#             while True:
#                 h = random.randint(self.h_min, self.h_max)
#                 w = random.randint(self.w_min, self.w_max)
#                 if self.planner.is_point_valid((h, w)) == True:
#                     unique_id =  int(f"2{i:02d}") # Hex ID (000 to FFF)
#                     yaw = random.random() * 2 * math.pi - math.pi
#                     agents.append([unique_id, w, h, yaw])
#                     break
#                     # print(f"Generated agent {unique_id} at ({w}, {h}) with yaw {yaw}")
#         return np.array(agents)

#     def generate_tasks(self, n_batches, n_points):
#         all_batches = []  # Store all batches of tasks
#         for i in range(n_batches):
#             tasks = []
#             task_num = 0  # Initialize task number for the batch
#             while len(tasks) < n_points:
#                 while True:
                    
#                     h_origin = random.randint(self.h_min, self.h_max)
#                     w_origin = random.randint(self.w_min, self.w_max)
#                     if self.planner.is_point_valid((h_origin,w_origin)):
#                         break 
#                 # Generate random coordinates for the task destination
#                 while True:
#                     h_destination = random.randint(self.h_min, self.h_max)
#                     w_destination = random.randint(self.w_min, self.w_max)
#                     if self.planner.is_point_valid((h_destination, w_destination)):
#                         break  # Exit the loop if the destination is valid
#                 yaw_origin = random.random() * 2 * math.pi - math.pi
#                 yaw_destination = random.random() * 2 * math.pi - math.pi
#                 t_release = random.randint(0, 0)
#                 # Calculate estimated travel time using A*
#                 start = (h_origin,w_origin)
#                 goal = (h_destination, w_destination)
#                 found, path = self.planner.get_plan(start, goal)
#                 if not found:
#                     continue  # Skip if no path found
#                 # solution_lines = [f"{x} {y} {theta}" for x, y, theta in path]
#                 # tra, tra_index = discritized_path(solution_lines, average_velocity = 1 ,
#                 #                                    resolution=self.planning_resolution,
#                 #                                    origin_x =self.origin_x, origin_y =self.origin_y)
#                 # print(tra, f'tra from h , w {start} to {goal}')
#                 estimated_travel_time = len(path) if found else float('inf')  # Use path length as travel time

#                 # Calculate deadlines
#                 pickup_deadline = t_release + self.max_waiting_time
#                 drop_off_deadline = pickup_deadline + (estimated_travel_time * (1 + self.max_travel_delay_percentage))

#                 # Generate a unique ID
#                 unique_id = int(f"1{i:02d}{task_num:02d}")
#                 # print(f"Generated task {unique_id} from ({w_origin}, {h_origin}) to ({w_destination}, {h_destination})")
#                 tasks.append([
#                     unique_id, 
#                     w_origin, h_origin, yaw_origin,  # Origin (x, y)
#                     w_destination, h_destination, yaw_destination,  # Destination (x, y)
#                     t_release,  # Release time
#                     pickup_deadline,  # Pickup deadline
#                     drop_off_deadline,  # Drop-off deadline
#                     estimated_travel_time  # Estimated travel time
#                 ])
#                 task_num += 1  # Increment task number for the batch

#             # Save the tasks for the current batch
#             all_batches.append(tasks)

#         return all_batches
    
# if __name__ == "__main__":
#     # Load configuration from ATC_wed.yaml
#     config_path = Path(__file__).resolve().parent.parent.parent / "env" / "ATC_wed.yaml"
#     with open(config_path, 'r') as file:
#         params = yaml.safe_load(file)

#     # Extract grid bounds and other parameters from the YAML file
#     x_min, x_max = params['x_min'], params['x_max']
#     y_min, y_max = params['y_min'], params['y_max']
#     map_resolution = params['map_resolution']
#     Planning_resolution = params['Planning_resolution']
#     max_waiting_time = 10
#     max_travel_delay_percentage = 10 / 100

#     # Load the map (osaka2d.png) and initialize the Planner
#     map_path = Path(__file__).resolve().parent.parent.parent / "env" / "osaka2dfree.png"
#     # planner = Planner(map_path, map_resolution, Planning_resolution, threshold=0.5, origin_x=-60, origin_y=20)
#     planner = Planner()

#     # Initialize the data generator
#     generator = DataGenerator(x_min, x_max, y_min, y_max, max_waiting_time, 
#                               max_travel_delay_percentage, Planning_resolution, 
#                               planner,origin_x=-60, origin_y=20)

#     # Generate tasks
#     n_batches = 10
#     n_tasks = 10
#     n_robots = 5
#     agents = generator.generate_agents(n_robots)
#     # print("Generated Agents:", agents)
#     tasks = generator.generate_tasks(n_batches, n_tasks)
#     # print("Generated Tasks:", tasks)

#     output_dir = Path(__file__).resolve().parent.parent.parent / "data"
#     output_dir.mkdir(exist_ok=True)  # Create the directory if it doesn't exist

#     # Save agents
#     agents_file = output_dir / "agents.npy"
#     np.save(agents_file, agents)
#     print(f"Agents saved to {agents_file}")

#     # Save tasks
#     for i, batch in enumerate(tasks):
#         tasks_file = output_dir / f"tasks_batch_{i}.npy"
#         np.save(tasks_file, batch)
#         print(f"Tasks for batch {i} saved to {tasks_file}")


import random
import math
import sys
from pathlib import Path
import yaml
from PIL import Image
import numpy as np
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.utils import astar, discritized_path  # Ensure astar is implemented in utils.py
from environment.environment import Planner

class DataGenerator:
    def __init__(self, x_min, x_max, y_min, y_max, max_waiting_time,
                  max_travel_delay_percentage, planning_resolution,
                    planner, origin_x, origin_y ):
        self.planning_resolution = planning_resolution
        self.h_min = 0
        # ensure integer max indices
        self.h_max = int(abs(y_min - y_max) / planning_resolution)
        self.w_min = 0
        self.w_max = int(abs(x_min - x_max) / planning_resolution)
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.max_waiting_time = max_waiting_time
        self.max_travel_delay_percentage = max_travel_delay_percentage
        self.planner = planner  
        

    def generate_agents(self, num_agents):
        agents = []
        for i in range(num_agents):
            while True:
                h = random.randint(self.h_min, self.h_max)
                w = random.randint(self.w_min, self.w_max)
                if self.planner.is_point_valid((h, w)) == True:
                    unique_id =  int(f"2{i:02d}") # Hex-like ID (e.g. 200, 201...)
                    yaw = random.random() * 2 * math.pi - math.pi
                    agents.append([unique_id, w, h, yaw])
                    break
        return np.array(agents)

    def generate_tasks(self, n_batches, n_points):
        all_batches = []  # Store all batches of tasks
        for i in range(n_batches):
            tasks = []
            task_num = 0  # Initialize task number for the batch
            while len(tasks) < n_points:
                # origin
                while True:
                    h_origin = random.randint(self.h_min, self.h_max)
                    w_origin = random.randint(self.w_min, self.w_max)
                    if self.planner.is_point_valid((h_origin, w_origin)):
                        break 
                # destination
                while True:
                    h_destination = random.randint(self.h_min, self.h_max)
                    w_destination = random.randint(self.w_min, self.w_max)
                    if self.planner.is_point_valid((h_destination, w_destination)):
                        break
                yaw_origin = random.random() * 2 * math.pi - math.pi
                yaw_destination = random.random() * 2 * math.pi - math.pi
                t_release = 0  # currently always 0

                # Calculate estimated travel time using Planner (A*)
                start = (h_origin, w_origin)
                goal = (h_destination, w_destination)
                found, path = self.planner.get_plan(start, goal)
                if not found:
                    continue  # Skip if no path found

                # Use path length as estimated travel time
                estimated_travel_time = len(path) if found else float('inf')

                # Calculate deadlines
                pickup_deadline = t_release + self.max_waiting_time
                drop_off_deadline = pickup_deadline + (estimated_travel_time * (1 + self.max_travel_delay_percentage))

                # Generate a unique ID (task)
                unique_id = int(f"1{i:02d}{task_num:02d}")

                # IMPORTANT: the Tasks_variable expects the order:
                # [task_id, w_origin, h_origin, yaw_origin,
                #  w_destination, h_destination, yaw_destination,
                #  t_release, pickupddl, estimatedTravelTime, dropoff_deadline]
                tasks.append([
                    unique_id,
                    w_origin, h_origin, yaw_origin,
                    w_destination, h_destination, yaw_destination,
                    t_release,
                    pickup_deadline,
                    estimated_travel_time,   # estimated travel time BEFORE dropoff_deadline
                    drop_off_deadline
                ])
                task_num += 1

            all_batches.append(tasks)

        return all_batches
    
if __name__ == "__main__":
    # Load configuration from ATC_wed.yaml
    config_path = Path(__file__).resolve().parent.parent.parent / "env" / "ATC_wed.yaml"
    with open(config_path, 'r') as file:
        params = yaml.safe_load(file)

    # Extract grid bounds and other parameters from the YAML file
    x_min, x_max = params['x_min'], params['x_max']
    y_min, y_max = params['y_min'], params['y_max']
    map_resolution = params['map_resolution']
    Planning_resolution = params['Planning_resolution']
    max_waiting_time = 10
    max_travel_delay_percentage = 10 / 100

    # Initialize planner (expects Planner to load map from env/ATC_wed.yaml)
    planner = Planner()

    # Initialize the data generator
    generator = DataGenerator(x_min, x_max, y_min, y_max, max_waiting_time, 
                              max_travel_delay_percentage, Planning_resolution, 
                              planner, origin_x=-60, origin_y=20)

    # Generate tasks
    n_batches = 10
    n_tasks = 10
    n_robots = 5
    agents = generator.generate_agents(n_robots)
    tasks = generator.generate_tasks(n_batches, n_tasks)

    output_dir = Path(__file__).resolve().parent.parent.parent / "data"
    output_dir.mkdir(exist_ok=True)

    # Save agents
    agents_file = output_dir / "agents.npy"
    np.save(agents_file, agents)
    print(f"Agents saved to {agents_file}")

    # Save tasks batches
    for i, batch in enumerate(tasks):
        tasks_file = output_dir / f"tasks_batch_{i}.npy"
        np.save(tasks_file, batch)
        print(f"Tasks for batch {i} saved to {tasks_file}")