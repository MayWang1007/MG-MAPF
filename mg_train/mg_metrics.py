import time
from typing import List, Tuple, Dict
import numpy as np
import csv
from pathlib import Path
from datetime import datetime

class PerformanceMetrics:
    def __init__(self, num_agents: int, task_instances: List[List[Tuple]]):
        self.num_agents = num_agents
        self.task_instances = task_instances
        self.total_goals = sum(len(tasks) for tasks in task_instances)

        self.start_time = None
        self.end_time = None

        self.goals_completed = [0] * num_agents

        self.total_steps = 0
        self.agent_steps = {}
        self.algorithm_runtime = 0.0
        self._collision_agents_sum = 0

    def start(self):
        self.start_time = time.time()
        for i in range(self.num_agents):
            self.agent_steps[i] = 0

    def record_goal_completion(self, agent_id: int, step: int):
        self.goals_completed[agent_id] += 1

    def record_step(self, agent_id: int, moved: bool = True):
        self.agent_steps[agent_id] += 1

    def record_collisions(self, old_positions: List[Tuple], new_positions: List[Tuple], active_mask=None):
        colliding = set()
        n = len(new_positions)
        active = [i for i in range(n) if active_mask is None or active_mask[i]]

        pos_to_agents = {}
        for aid in active:
            pos = new_positions[aid]
            if pos not in pos_to_agents:
                pos_to_agents[pos] = []
            pos_to_agents[pos].append(aid)
        for pos, agents in pos_to_agents.items():
            if len(agents) > 1:
                colliding.update(agents)

        for offset, i in enumerate(active):
            for j in active[offset + 1:]:
                if (new_positions[i] == old_positions[j] and
                        new_positions[j] == old_positions[i]):
                    colliding.add(i)
                    colliding.add(j)

        self._collision_agents_sum += len(colliding)

    def record_step_runtime(self, step_runtime: float):
        self.algorithm_runtime += step_runtime

    def finish(self, total_steps: int):
        self.end_time = time.time()
        self.total_steps = total_steps

    def calculate_metrics(self) -> Dict:
        completed_goals = sum(self.goals_completed)

        makespan = self.total_steps

        soc = sum(self.agent_steps.values())

        avg_throughput_time = completed_goals / self.algorithm_runtime if self.algorithm_runtime > 0 else 0
        avg_throughput_step = completed_goals / self.total_steps if self.total_steps > 0 else 0


        if self.total_steps > 0 and self.num_agents > 0:
            collision_rate = self._collision_agents_sum / (self.total_steps * self.num_agents)
        else:
            collision_rate = 0.0

        return {
            'num_agents': self.num_agents,
            'total_goals': self.total_goals,
            'runtime': self.algorithm_runtime,
            'makespan': makespan,
            'sum_of_costs': soc,
            'avg_throughput_time': avg_throughput_time,
            'avg_throughput_step': avg_throughput_step,
            'collision_rate': collision_rate
        }

    def print_summary(self):
        metrics = self.calculate_metrics()
        print(f"{'='*70}")
        print(f"num_agents: {metrics['num_agents']}")
        print(f"total_goals: {metrics['total_goals']}")
        print(f"runtime(s): {metrics['runtime']:.3f}")
        print(f"makespan: {metrics['makespan']}")
        print(f"sum_of_costs: {metrics['sum_of_costs']}")
        print(f"avg_throughput(goals/s): {metrics['avg_throughput_time']:.3f}")
        print(f"avg_throughput(goals/step): {metrics['avg_throughput_step']:.3f}")
        print(f"collision_rate: {metrics['collision_rate']:.4f}")
        print(f"{'='*70}")

        return metrics


def save_metrics_to_csv(metrics: Dict, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    csv_file = Path(output_path).with_suffix('.csv')
    file_exists = csv_file.exists()

    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                'timestamp', 'num_agents', 'total_goals', 'runtime(s)',
                'makespan', 'sum_of_costs', 'avg_throughput(goals/s)',
                'avg_throughput(goals/step)', 'collision_rate'
            ])

        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            metrics['num_agents'],
            metrics['total_goals'],
            round(metrics['runtime'], 3),
            metrics['makespan'],
            metrics['sum_of_costs'],
            round(metrics['avg_throughput_time'], 3),
            round(metrics['avg_throughput_step'], 3),
            round(metrics.get('collision_rate', 0.0), 4)
        ])

    print(f"Results saved to: {csv_file}")
