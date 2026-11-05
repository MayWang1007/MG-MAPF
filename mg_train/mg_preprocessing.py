import numpy as np
import gymnasium
from gymnasium import ObservationWrapper, Wrapper
from gymnasium.spaces import Box, Dict


class MGPreprocessorConfig:
    network_input_radius: int = 5
    intrinsic_target_reward: float = 0.01


def mg_preprocessor(env, config: MGPreprocessorConfig):
    env = MGFollowerWrapper(env=env, config=config)
    env = CutObservationWrapper(env, target_observation_radius=config.network_input_radius)
    env = ConcatPositionalFeatures(env)
    return env


class MGFollowerWrapper(Wrapper):
    def __init__(self, env, config: MGPreprocessorConfig):
        super().__init__(env)
        self._cfg: MGPreprocessorConfig = config

        self.current_subgoals = None

    def _extract_subgoals_from_paths(self, paths):
        subgoals = []
        for path in paths:
            if path and len(path) > 1:
                subgoals.append(tuple(path[1]))
            elif path and len(path) == 1:
                subgoals.append(tuple(path[0]))
            else:
                subgoals.append(None)
        return subgoals

    def _compute_intrinsic_rewards(self, current_positions, num_agents):
        rewards = []

        for i in range(num_agents):
            reward = 0.0

            if (self.current_subgoals is not None and
                i < len(self.current_subgoals) and
                self.current_subgoals[i] is not None):
                current_pos = tuple(current_positions[i])
                subgoal = self.current_subgoals[i]

                if current_pos == subgoal:
                    reward = self._cfg.intrinsic_target_reward

            rewards.append(reward)

        return rewards

    def step(self, action):
        observations, extrinsic_rewards, dones, truncated, infos = self.env.step(action)

        current_positions = []
        for obs in observations:
            if 'xy' in obs:
                current_positions.append(tuple(obs['xy']))
            else:
                current_positions.append(None)

        intrinsic_rewards = self._compute_intrinsic_rewards(
            current_positions, len(observations)
        )

        paths = self.env.get_paths() if hasattr(self.env, 'get_paths') else []
        self.current_subgoals = self._extract_subgoals_from_paths(paths)

        combined_rewards = [
            extrinsic_rewards[i] + intrinsic_rewards[i]
            for i in range(len(extrinsic_rewards))
        ]

        return observations, combined_rewards, dones, truncated, infos

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)

        paths = self.env.get_paths() if hasattr(self.env, 'get_paths') else []
        self.current_subgoals = self._extract_subgoals_from_paths(paths)

        return observations, infos


class CutObservationWrapper(ObservationWrapper):
    def __init__(self, env, target_observation_radius):
        super().__init__(env)
        self._target_obs_radius = target_observation_radius
        self._initial_obs_radius = self.env.observation_space['obstacles'].shape[0] // 2

        for key, value in self.observation_space.items():
            d = self._initial_obs_radius * 2 + 1
            if value.shape == (d, d):
                r = self._target_obs_radius
                self.observation_space[key] = Box(0.0, 1.0, shape=(r * 2 + 1, r * 2 + 1))

    def observation(self, observations):
        tr = self._target_obs_radius
        ir = self._initial_obs_radius
        d = ir * 2 + 1

        for obs in observations:
            for key, value in obs.items():
                if key == 'neighbor_targets':
                    continue
                if hasattr(value, 'shape') and value.shape == (d, d):
                    obs[key] = value[ir - tr:ir + tr + 1, ir - tr:ir + tr + 1]

        return observations


class ConcatPositionalFeatures(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.to_concat = []

        observation_space = Dict()
        full_size = self.env.observation_space['obstacles'].shape[0]

        for key, value in self.observation_space.items():
            if key == 'neighbor_targets':
                observation_space[key] = value
                continue
            if value.shape == (full_size, full_size):
                self.to_concat.append(key)
            else:
                observation_space[key] = value

        obs_shape = (len(self.to_concat), full_size, full_size)
        observation_space['obs'] = Box(0.0, 1.0, shape=obs_shape)
        self.to_concat.sort(key=self.key_comparator)
        self.observation_space = observation_space

    def observation(self, observations):
        for agent_idx, obs in enumerate(observations):
            main_obs = np.concatenate([obs[key][None] for key in self.to_concat])

            for key in self.to_concat:
                del obs[key]

            for key in obs:
                if key == 'neighbor_targets':
                    obs[key] = np.array(obs[key], dtype=np.float32)
                    continue
                obs[key] = np.array(obs[key], dtype=np.float32)

            observations[agent_idx]['obs'] = main_obs.astype(np.float32)

        return observations

    @staticmethod
    def key_comparator(x):
        if x == 'obstacles':
            return '0_' + x
        elif x == 'agents' or (x != 'agent_id' and 'agents' in x):
            return '1_' + x
        elif x == 'agent_id':
            return '2_' + x
        return '3_' + x
