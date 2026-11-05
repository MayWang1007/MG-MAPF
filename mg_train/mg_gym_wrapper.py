import gymnasium as gym
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import yaml
import json
from pathlib import Path
from mg_train.mg_env import MGMapfEnv, CompletedAgentManager, MultiGoalManager
from mg_train.mg_map import get_map_info, generate_task_instance_list
from scipy.spatial.distance import cdist
import itertools
from mg_train.mg_preprocessing import MGFollowerWrapper, MGPreprocessorConfig
import threading

class MGMapfMultiTrainEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    _logging_active = False
    _logging_lock = None

    def __init__(
        self,
        map_name: str = 'maze-32-32-4',
        num_agents: int = 32,
        num_goals: int = 8,
        obs_radius: int = 5,
        max_episode_steps: int = 2048,
        goal_reward: float = 1.0,
        multi_goal_bonus: float = 0.5,
        collision_penalty: float = -0.05,
        movement_penalty: float = -0.001,
        backtrack_penalty: float = -0.02,
        include_id_map: bool = False,
        collision_mode: str = 'hard',
        maps_yaml: str = './env/training-maps.yaml',
    ):
        super().__init__()
        if not MGMapfMultiTrainEnv._logging_active:
            if MGMapfMultiTrainEnv._logging_lock is None:
                MGMapfMultiTrainEnv._logging_lock = threading.Lock()

            with MGMapfMultiTrainEnv._logging_lock:
                if not MGMapfMultiTrainEnv._logging_active:
                    MGMapfMultiTrainEnv._logging_active = True
                    self.should_log = True
                else:
                    self.should_log = False
        else:
            self.should_log = False

        self.map_name = map_name
        self.base_num_agents = num_agents
        self.base_num_goals = num_goals
        self.obs_radius = obs_radius
        self.max_episode_steps = max_episode_steps

        self.goal_reward = goal_reward
        self.multi_goal_bonus = multi_goal_bonus
        self.collision_penalty = collision_penalty
        self.movement_penalty = movement_penalty
        self.backtrack_penalty = backtrack_penalty
        self.collision_mode = collision_mode

        self.num_agents = num_agents
        self.num_goals = num_goals

        _maps_yaml = maps_yaml
        if not Path(_maps_yaml).is_file() and Path('./env/maps.yaml').is_file():
            _maps_yaml = './env/maps.yaml'
        valid_positions, valid_starts, obstacles, boundaries = get_map_info(
            _maps_yaml, map_name
        )
        self.valid_starts = valid_starts
        self.obstacles = obstacles
        self.boundaries = boundaries

        with open(_maps_yaml, 'r') as f:
            maps = yaml.safe_load(f)
        self.map_str = maps[map_name]
        map_lines = [line.strip() for line in self.map_str.strip().splitlines() if line.strip()]
        self.map_shape = (len(map_lines), len(map_lines[0]))

        self.include_id_map = include_id_map
        obs_size = 2 * obs_radius + 1
        obs_channels = 3 if include_id_map else 2
        self.observation_space = gym.spaces.Dict({
            'obs': gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_channels, obs_size, obs_size), dtype=np.float32),
            'xy': gym.spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32),
            'target_xy': gym.spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32),
            'neighbor_targets': gym.spaces.Box(low=-1e6, high=1e6, shape=(20, 2), dtype=np.float32),
        })

        self.action_space = gym.spaces.Discrete(5)

        self.env: Optional[MGMapfEnv] = None
        self.goal_manager: Optional[MultiGoalManager] = None
        self.step_count = 0
        self.episode_count = 0
        self.total_env_steps = 0
        self.current_paths: List[List[Tuple]] = []

        self.consecutive_goals = 0
        self.agent_consecutive_goals: List[int] = []

        self.prev_positions: List[Tuple] = []
        self.just_completed: List[bool] = []

        self.episode_trajectories: Dict[int, List[Tuple[int, int]]] = {}
        self.recording_trajectory = False

        self.episode_rewards = []
        self.recent_episode_rewards = []
        self.max_episodes_to_track = 100

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)

        task_instances, start_positions = generate_task_instance_list(
            self.num_agents,
            self.num_goals,
            self.valid_starts,
            self.obstacles,
            self.boundaries
        )
        self.goal_manager = MultiGoalManager(
            task_instances, start_positions,
            avoid_blocking=True,
            obstacles=set(self.obstacles),
            map_shape=self.map_shape,
        )
        self.env = MGMapfEnv(
            self.map_str,
            start_positions,
            self.obs_radius,
            use_improved_planner=True,
            static_cost_weight=1.0,
            dynamic_cost_weight=2.0,
            completion_manager=self.goal_manager.completion_manager,
            collision_mode=self.collision_mode,
        )

        self.step_count = 0
        self.episode_count += 1
        self.consecutive_goals = 0
        self.agent_consecutive_goals = [0] * self.num_agents

        self.max_team_milestone = 0

        self.prev_positions = [tuple(pos) for pos in start_positions]
        self.just_completed = [False] * self.num_agents

        self.episode_trajectories = {i: [tuple(start_positions[i])] for i in range(self.num_agents)}
        self.recording_trajectory = True

        self.episode_rewards = [0.0] * self.num_agents

        obs_list = self._get_observations_with_paths()
        obs_list = self._pad_observations(obs_list)

        return obs_list, {}

    def _check_backtrack(self, agent_id: int, old_pos: Tuple, new_pos: Tuple) -> bool:
        if not self.prev_positions or agent_id >= len(self.prev_positions):
            return False

        prev_pos = self.prev_positions[agent_id]

        if new_pos != prev_pos:
            return False
        if self.just_completed[agent_id]:
            return False
        row, col = old_pos
        adjacent_positions = [
            (row-1, col),
            (row+1, col),
            (row, col-1),
            (row, col+1),
        ]
        passable_count = 0
        for pos in adjacent_positions:
            if (0 <= pos[0] < self.env.height and
                0 <= pos[1] < self.env.width and
                pos not in self.env.obstacles):
                passable_count += 1
        has_alternatives = passable_count >= 3
        return has_alternatives

    def step(self, actions):
        self.step_count += 1
        self.total_env_steps += 1

        active_actions = np.array(actions[:self.num_agents], copy=True)
        for agent_id in range(self.num_agents):
            if self.goal_manager.completion_manager.should_force_action_zero(agent_id):
                active_actions[agent_id] = 0

        old_positions = [tuple(pos) for pos in self.env.current_positions]

        new_positions, conflicts = self.env.step(active_actions)
        completed = self.goal_manager.check_and_update(new_positions)

        self.just_completed = [False] * self.num_agents
        for agent_id in completed:
            self.just_completed[agent_id] = True

        rewards = []
        backtrack_count = 0

        for i in range(self.num_agents):
            r = 0.0
            if i in completed:
                r += self.goal_reward
                self.agent_consecutive_goals[i] += 1
                self.consecutive_goals += 1
                if self.agent_consecutive_goals[i] >= 2:
                    consecutive_bonus = self.goal_reward * (self.agent_consecutive_goals[i] - 1)
                    r += consecutive_bonus

            if i in conflicts['vertex'] or i in conflicts['edge']:
                r += self.collision_penalty

            rewards.append(r)
            self.episode_rewards[i] += r
        self.prev_positions = old_positions

        all_done = all(
            self.goal_manager.current_goal_idx[i] >= len(self.goal_manager.task_instances[i])
            for i in range(self.num_agents)
        )
        timeout = self.step_count >= self.max_episode_steps

        dones = [all_done or timeout] * self.base_num_agents
        truncated = [timeout and not all_done] * self.base_num_agents


        min_goals_completed = min(self.goal_manager.goals_completed)
        if min_goals_completed > self.max_team_milestone:
            self.max_team_milestone = min_goals_completed

            team_bonus = self.multi_goal_bonus
            for i in range(self.num_agents):
                rewards[i] += team_bonus

        while len(rewards) < self.base_num_agents:
            rewards.append(0.0)

        obs_list = self._get_observations_with_paths()
        obs_list = self._pad_observations(obs_list)

        infos = self._build_infos(completed, backtrack_count)

        if self.should_log and self.step_count % 100 == 0:
            agent_completions = self.goal_manager.goals_completed
            avg_completion = sum(agent_completions) / len(agent_completions) if agent_completions else 0
            max_completion = max(agent_completions) if agent_completions else 0
            min_completion = min(agent_completions) if agent_completions else 0

            total_completed = sum(agent_completions)
            total_goals = self.num_agents * self.num_goals
            completion_rate = 100 * total_completed / total_goals if total_goals > 0 else 0

            if self.recent_episode_rewards:
                recent_avg = sum(self.recent_episode_rewards) / len(self.recent_episode_rewards)
                reward_str = f"Recent Avg: {recent_avg:.4f}"
            else:
                current_avg = sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0.0
                reward_str = f"Current Avg: {current_avg:.4f}"


        if all_done or timeout:
            if self.should_log:
                episode_avg_reward = sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0.0
                self.recent_episode_rewards.append(episode_avg_reward)

                if len(self.recent_episode_rewards) > self.max_episodes_to_track:
                    self.recent_episode_rewards.pop(0)

                total_completed = sum(self.goal_manager.goals_completed)
                total_goals = self.num_agents * self.num_goals


            terminal_obs = obs_list
            self.reset()
            obs_list = terminal_obs

        return obs_list, rewards, dones, truncated, infos

    def _pad_observations(self, obs_list: List[Dict]) -> List[Dict]:
        if len(obs_list) >= self.base_num_agents:
            return obs_list[:self.base_num_agents]

        dummy_obs = {
            'obstacles': np.zeros((2 * self.obs_radius + 1, 2 * self.obs_radius + 1), dtype=np.float32),
            'agents': np.zeros((2 * self.obs_radius + 1, 2 * self.obs_radius + 1), dtype=np.float32),
            'xy': np.array([0, 0], dtype=np.int32),
            'target_xy': np.array([0, 0], dtype=np.int32),
        }

        while len(obs_list) < self.base_num_agents:
            obs_list.append(dummy_obs.copy())

        return obs_list

    def _get_observations_with_paths(self) -> List[Dict]:
        current_positions = self.env.current_positions
        current_targets = self.goal_manager.get_current_targets(current_positions)

        for agent_id in range(self.num_agents):
            if self.goal_manager.completion_manager.should_skip_planning(agent_id):
                continue
            if self.env.plan_path(agent_id, current_targets[agent_id]) is None:
                if self.env.repark_blocking_agent(agent_id, current_targets[agent_id]):
                    current_targets = self.goal_manager.get_current_targets(current_positions)

        observations = []
        paths = []

        for agent_id in range(self.num_agents):
            if self.goal_manager.completion_manager.should_skip_planning(agent_id):
                path = []
            else:
                path = self.env.plan_path(agent_id, current_targets[agent_id])
                path = path if path else []

            paths.append(path)
            obs = self._build_observation(agent_id, current_targets[agent_id], path,
                                          current_targets=current_targets)
            observations.append(obs)

        self.current_paths = paths
        return observations

    def _build_observation(self, agent_id: int, target: Tuple, path: List[Tuple],
                           current_targets: List[Tuple] = None) -> Dict:
        obs_dict = self.env.get_observation(agent_id, target, path,
                                            include_id_map=self.include_id_map,
                                            current_targets=current_targets)
        obs = {
            'obs': obs_dict['obs'],
            'xy': np.array(self.env.current_positions[agent_id], dtype=np.float32),
            'target_xy': np.array(target, dtype=np.float32),
            'neighbor_targets': obs_dict['neighbor_targets'],
        }
        return obs

    def _build_infos(self, completed: List[int], backtrack_count: int = 0) -> List[Dict]:
        infos = []
        total_completed = sum(self.goal_manager.goals_completed)
        total_goals = self.num_agents * self.num_goals

        actual_success_rate = total_completed / total_goals if total_goals > 0 else 0.0
        success_with_recency = actual_success_rate + (self.total_env_steps / 1e7)

        for i in range(self.base_num_agents):
            if i < self.num_agents:
                info = {
                    'goals_completed': self.goal_manager.goals_completed[i],
                    'episode_extra_stats': {
                        'agent_goals_completed': self.goal_manager.goals_completed[i],
                        'agent_completion_rate': self.goal_manager.goals_completed[i] / self.num_goals if self.num_goals > 0 else 0,
                        'total_goals_completed': total_completed,
                        'global_completion_rate': total_completed / total_goals if total_goals > 0 else 0,
                        'episode_length': self.step_count,
                        'success': actual_success_rate,
                        'success_with_recency': success_with_recency,
                        'consecutive_goals': self.consecutive_goals,
                        'agent_consecutive_goals': self.agent_consecutive_goals[i],
                        'backtrack_count': backtrack_count,
                        'team_milestones': self.max_team_milestone,
                    }
                }
            else:
                info = {
                    'goals_completed': 0,
                    'episode_extra_stats': {
                        'agent_goals_completed': 0,
                        'agent_completion_rate': 0.0,
                        'total_goals_completed': total_completed,
                        'global_completion_rate': total_completed / total_goals if total_goals > 0 else 0,
                        'episode_length': self.step_count,
                        'success': actual_success_rate,
                        'success_with_recency': success_with_recency,
                        'consecutive_goals': 0,
                        'agent_consecutive_goals': 0,
                        'backtrack_count': 0,
                        'team_milestones': self.max_team_milestone,
                    }
                }
            infos.append(info)
        return infos

    def get_paths(self) -> List[List[Tuple]]:
        return self.current_paths

    def get_num_agents(self):
        return self.base_num_agents

    def render(self):
        pass

    def close(self):
        pass


def make_mg_mapf_env(full_env_name, cfg=None, env_config=None, render_mode=None):
    import random as _random


    map_pool   = getattr(cfg, 'map_pool',   None)
    agent_pool = getattr(cfg, 'agent_pool', None)
    goal_pool  = getattr(cfg, 'goal_pool',  None)

    if map_pool and isinstance(map_pool, (list, tuple)) and len(map_pool) > 0:
        map_name = _random.choice(map_pool)
    else:
        map_name = getattr(cfg, 'map_name', 'maze-32-32-4')

    if agent_pool and isinstance(agent_pool, (list, tuple)) and len(agent_pool) > 0:
        num_agents = _random.choice(agent_pool)
    else:
        num_agents = getattr(cfg, 'num_agents', 32)

    if goal_pool and isinstance(goal_pool, (list, tuple)) and len(goal_pool) > 0:
        num_goals = _random.choice(goal_pool)
    else:
        num_goals = getattr(cfg, 'num_goals', 8)

    obs_radius        = getattr(cfg, 'obs_radius', 5)
    max_episode_steps = getattr(cfg, 'max_episode_steps', 2048)
    goal_reward            = getattr(cfg, 'goal_reward', 1.0)
    multi_goal_bonus       = getattr(cfg, 'multi_goal_bonus', 0.5)
    collision_penalty      = getattr(cfg, 'collision_penalty', -0.05)
    movement_penalty       = getattr(cfg, 'movement_penalty', -0.001)
    backtrack_penalty      = getattr(cfg, 'backtrack_penalty', -0.02)
    collision_mode         = getattr(cfg, 'collision_mode', 'hard')
    intrinsic_target_reward = getattr(cfg, 'intrinsic_target_reward', 0.1)

    include_id_map = getattr(cfg, 'prediction_head', 'map') == 'action_id'
    maps_yaml = getattr(cfg, 'maps_yaml', None) or './env/training-maps.yaml'
    if not Path(maps_yaml).is_file() and Path('./env/maps.yaml').is_file():
        maps_yaml = './env/maps.yaml'


    try:
        import yaml as _yaml
        with open(maps_yaml, 'r') as _f:
            _maps = _yaml.safe_load(_f)

        _pool = map_pool if (map_pool and isinstance(map_pool, (list, tuple))) else [map_name]
        _max_side = 0
        for _mn in _pool:
            if _mn in _maps:
                _lines = [l for l in _maps[_mn].strip().split('\n') if l.strip()]
                _h, _w = len(_lines), len(_lines[0].strip())
                _max_side = max(_max_side, _h, _w)
        if _max_side > 0:
            cfg.map_size = _max_side
    except Exception as _e:
        pass

    env = MGMapfMultiTrainEnv(
        map_name=map_name,
        num_agents=num_agents,
        num_goals=num_goals,
        obs_radius=obs_radius,
        max_episode_steps=max_episode_steps,
        goal_reward=goal_reward,
        multi_goal_bonus=multi_goal_bonus,
        collision_penalty=collision_penalty,
        movement_penalty=movement_penalty,
        backtrack_penalty=backtrack_penalty,
        include_id_map=include_id_map,
        collision_mode=collision_mode,
        maps_yaml=maps_yaml,
    )
    env._maps_yaml = maps_yaml

    preproc_config = MGPreprocessorConfig()
    preproc_config.network_input_radius = obs_radius
    preproc_config.intrinsic_target_reward = intrinsic_target_reward

    env = MGFollowerWrapper(env, preproc_config)
    return env
