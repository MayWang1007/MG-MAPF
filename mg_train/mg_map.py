import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from pathlib import Path
import yaml
import random
from functools import lru_cache


@lru_cache(maxsize=16)
def _goal_placement_data(positions):
    walkable = set(positions)
    neighbors = {
        pos: [neighbor for neighbor in (
            (pos[0] - 1, pos[1]), (pos[0] + 1, pos[1]),
            (pos[0], pos[1] - 1), (pos[0], pos[1] + 1),
        ) if neighbor in walkable]
        for pos in positions
    }
    component = {}
    discovery = {}
    low = {}
    parent = {}
    articulation = set()
    time = 0
    component_id = 0

    for root in positions:
        if root in discovery:
            continue
        parent[root] = None
        component[root] = component_id
        discovery[root] = low[root] = time
        time += 1
        root_children = 0
        stack = [(root, iter(neighbors[root]))]

        while stack:
            pos, adjacent = stack[-1]
            neighbor = next(adjacent, None)
            if neighbor is None:
                stack.pop()
                previous = parent[pos]
                if previous is not None:
                    low[previous] = min(low[previous], low[pos])
                    if parent[previous] is not None and low[pos] >= discovery[previous]:
                        articulation.add(previous)
                continue
            if neighbor not in discovery:
                parent[neighbor] = pos
                component[neighbor] = component_id
                discovery[neighbor] = low[neighbor] = time
                time += 1
                if pos == root:
                    root_children += 1
                stack.append((neighbor, iter(neighbors[neighbor])))
            elif neighbor != parent[pos]:
                low[pos] = min(low[pos], discovery[neighbor])

        if root_children > 1:
            articulation.add(root)
        component_id += 1

    safe_goals = {
        pos for pos in positions
        if len(neighbors[pos]) >= 3 and pos not in articulation
    }
    return component, safe_goals

def load_map_from_yaml(yaml_file, map_name):
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    map_str = data.get(map_name)
    if map_str is None:
        raise KeyError(f"Map '{map_name}' not found in YAML file.")
    return map_str

def generate_task_instance_list(
    num_instances: int,
    fixed_goal_num: int,
    valid_starts: list[tuple],
    obstacles: list[tuple],
    boundaries: list[tuple],
    seed: int = None
) -> tuple[list[list[tuple]], list[tuple]]:
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random

    if not valid_starts:
        print("Error: valid_starts is empty")
        return [], []
    invalid_set = set(obstacles) | set(boundaries)
    pool = [pos for pos in valid_starts if pos not in invalid_set]

    component, safe_goals = _goal_placement_data(tuple(valid_starts))
    total_goals = num_instances * fixed_goal_num
    by_component = {}
    for pos in pool:
        by_component.setdefault(component[pos], []).append(pos)

    eligible = []
    for cells in by_component.values():
        goal_cells = [pos for pos in cells if pos in safe_goals]
        if len(cells) >= num_instances + total_goals and len(goal_cells) >= total_goals:
            eligible.append((cells, goal_cells))
    if not eligible:
        raise ValueError('Not enough connected, non-bottleneck cells for unique starts and goals')

    pool, goal_pool = max(eligible, key=lambda item: (len(item[0]), len(item[1])))
    non_goal_starts = [pos for pos in pool if pos not in safe_goals]
    required_non_goal_starts = max(0, num_instances - (len(goal_pool) - total_goals))
    start_positions = rng.sample(non_goal_starts, required_non_goal_starts)
    selected_starts = set(start_positions)
    start_positions.extend(rng.sample(
        [pos for pos in pool if pos not in selected_starts],
        num_instances - required_non_goal_starts,
    ))
    rng.shuffle(start_positions)

    selected_starts = set(start_positions)
    available_goal_pool = [pos for pos in goal_pool if pos not in selected_starts]
    rng.shuffle(available_goal_pool)
    task_instances_rc = [
        [convert_xy_to_rowcol(pos) for pos in available_goal_pool[i * fixed_goal_num:(i + 1) * fixed_goal_num]]
        for i in range(num_instances)
    ]
    start_positions_rc = [convert_xy_to_rowcol(pos) for pos in start_positions]
    return task_instances_rc, start_positions_rc

def get_map_info(yaml_file_path: str, map_name: str):
    obstacle_symbols = {'#'}
    valid_passable_symbols = {'!', '$', '@', '.'}

    with open(yaml_file_path, 'r', encoding='utf-8') as f:
        map_data = yaml.safe_load(f)

    map_str = map_data[map_name]
    map_lines = [line.strip('\n') for line in map_str.split('\n') if line.strip('\n')]

    valid_positions = []
    valid_start_positions = []
    grid_obstacles = []

    for y, line in enumerate(map_lines):
        for x, char in enumerate(line):
            current_coords = (x, y)
            if char in obstacle_symbols:
                grid_obstacles.append(current_coords)
            elif char in valid_passable_symbols:
                valid_positions.append(current_coords)
                valid_start_positions.append(current_coords)
            else:
                continue

    boundaries = []
    if map_lines:
        max_y = len(map_lines) - 1
        max_x = len(map_lines[0]) - 1
        for x in range(max_x + 1):
            boundaries.append((x, 0))
        for x in range(max_x + 1):
            boundaries.append((x, max_y))
        for y in range(1, max_y):
            boundaries.append((0, y))
        for y in range(1, max_y):
            boundaries.append((max_x, y))

    return valid_positions, valid_start_positions, grid_obstacles, boundaries

def convert_xy_to_rowcol(pos_xy: tuple) -> tuple:
    x, y = pos_xy
    return (y, x)

def convert_rowcol_to_xy(pos_rc: tuple) -> tuple:
    row, col = pos_rc
    return (col, row)
