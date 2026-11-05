import torch
import numpy as np
import json
from collections import OrderedDict
from pathlib import Path
import yaml
from typing import List, Tuple, Dict, Optional, Set

from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.utils.attr_dict import AttrDict
from mg_train.mg_gym_wrapper import make_mg_mapf_env
from .mg_metrics import PerformanceMetrics, save_metrics_to_csv
from .mg_visualizer import create_svg_animation
from .mg_env import MGMapfEnv, MultiGoalManager

from sample_factory.cfg.arguments import parse_sf_args, parse_full_cfg
from argparse import Namespace
from sample_factory.envs.env_utils import register_env
from sample_factory.algo.utils.make_env import make_env_func_batched

class SFInference:
    def __init__(self, model_path, device='cuda'):
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else 'cpu'

        try:
            from mg_train.mg_model import make_adaptive_conv_encoder
            global_model_factory().register_encoder_factory(make_adaptive_conv_encoder)
        except:
            pass


        checkpoint = torch.load(model_path, map_location=self.device)
        checkpoint_dir = Path(model_path).parent.parent
        config_json_path = checkpoint_dir / 'config.json'

        cfg = None
        if config_json_path.exists():
            with open(config_json_path, 'r') as f:
                cfg_dict = json.load(f)
                cfg = AttrDict(cfg_dict)
        else:
            if 'cfg' in checkpoint:
                cfg = checkpoint['cfg']
            elif 'config' in checkpoint:
                cfg = checkpoint['config']
            elif 'full_config' in checkpoint:
                cfg = checkpoint['full_config']

        if cfg is None:
            parser, partial_cfg = parse_sf_args(argv=['--algo=APPO', '--env=mg_mapf'], evaluation=False)
            cfg = parse_full_cfg(parser, argv=['--algo=APPO', '--env=mg_mapf'])
            cfg = AttrDict(vars(cfg)) if not isinstance(cfg, AttrDict) else cfg

        if not isinstance(cfg, AttrDict):
            if isinstance(cfg, dict):
                cfg = AttrDict(cfg)
            else:
                cfg = AttrDict(vars(cfg))

        if not hasattr(cfg, 'encoder') or cfg.encoder is None:
            cfg.encoder = {
                'extra_fc_layers': 1,
                'num_filters': 64,
                'num_res_blocks': 8,
                'activation_func': 'ReLU',
                'hidden_size': 512
            }

        if not hasattr(cfg, 'rnn_size'):
            cfg.rnn_size = 256
        if not hasattr(cfg, 'policy_rnn_size'):
            cfg.policy_rnn_size = 256
        if not hasattr(cfg, 'pred_rnn_size'):
            cfg.pred_rnn_size = 128
        cfg.use_rnn = True
        cfg.rnn_type = getattr(cfg, 'rnn_type', 'gru')
        cfg.normalize_input = False
        cfg.normalize_returns = False


        if not hasattr(cfg, 'env'):
            cfg.env = 'mg_mapf'
        if not hasattr(cfg, 'num_envs'):
            cfg.num_envs = 1

        self.cfg = cfg


        _ckpt_conv_in_ch = None
        for k, v in checkpoint['model'].items():
            if 'conv_head' in k and 'weight' in k and v.dim() == 4:
                _ckpt_conv_in_ch = v.shape[1]
                break


        register_env('mg_mapf', make_mg_mapf_env)
        cfg.num_envs = 1
        temp_env = make_env_func_batched(cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0))
        obs_space = temp_env.observation_space
        action_space = temp_env.action_space
        temp_env.close()


        if _ckpt_conv_in_ch is not None:
            from gymnasium import spaces as gym_spaces


            if isinstance(obs_space, gym_spaces.Dict):
                _env_obs_ch = obs_space['obs'].shape[0]
                _obs_shape   = obs_space['obs'].shape
            else:
                _env_obs_ch = obs_space.shape[0]
                _obs_shape   = obs_space.shape

            _n_pred = _ckpt_conv_in_ch - _env_obs_ch

            if _n_pred > 0:
                cfg.use_neighbor_predictor = True
                cfg.prediction_steps       = _n_pred
            elif _n_pred == 0:
                cfg.use_neighbor_predictor = False
            else:
                _correct_ch = _ckpt_conv_in_ch
                cfg.use_neighbor_predictor = False
                import numpy as np
                _H, _W = _obs_shape[1], _obs_shape[2]
                _new_obs_box = gym_spaces.Box(
                    low=-1.0, high=1.0,
                    shape=(_correct_ch, _H, _W),
                    dtype=np.float32,
                )
                obs_space = gym_spaces.Dict({**obs_space.spaces, 'obs': _new_obs_box})
        self.actor_critic = create_actor_critic(cfg, obs_space, action_space)
        self.actor_critic.to(self.device)
        self.actor_critic.load_state_dict(checkpoint['model'])
        self.actor_critic.eval()

        self.rnn_states = None
        self.rnn_hidden_size = cfg.rnn_size

        self._uses_predictor: bool = bool(
            getattr(cfg, 'use_neighbor_predictor', False)
        )
        self._use_id_map: bool = (
            self._uses_predictor and
            getattr(cfg, 'prediction_head', 'map') == 'action_id'
        )

    def reset_states(self):
        self.rnn_states = None

    def act(self, observations):
        if self.rnn_states is None:
            batch_size = len(observations)
            rnn_out_size = self.actor_critic.core.get_core_out_size() if hasattr(self.actor_critic.core, 'get_core_out_size') else self.rnn_hidden_size
            self.rnn_states = torch.zeros(
                batch_size,
                rnn_out_size,
                dtype=torch.float32,
                device=self.device
            )
        obs_dict = self.transform_dict_observations(observations)
        obs_tensor = {
            'obs': torch.FloatTensor(obs_dict['obs']).to(self.device)
        }


        if self._uses_predictor:
            if 'xy' in obs_dict:
                obs_tensor['xy'] = torch.FloatTensor(obs_dict['xy']).to(self.device)
            if 'neighbor_targets' in obs_dict:
                obs_tensor['neighbor_targets'] = torch.FloatTensor(
                    obs_dict['neighbor_targets']).to(self.device)

        encoder = self.actor_critic.encoder
        if self._uses_predictor:
            encoder._prev_rnn_states = self.rnn_states

        with torch.no_grad():
            policy_outputs = self.actor_critic(obs_tensor, self.rnn_states)
            action_logits = policy_outputs['action_logits']
            actions = torch.argmax(action_logits, dim=-1)
            self.rnn_states = policy_outputs['new_rnn_states']

        return actions.cpu().numpy()

    @staticmethod
    def transform_dict_observations(observations):
        obs_dict = {}

        if isinstance(observations[0], (dict, OrderedDict)):
            for key in observations[0].keys():
                if not isinstance(observations[0][key], str):
                    obs_dict[key] = [o[key] for o in observations]
        else:
            obs_dict['obs'] = observations

        for key, x in obs_dict.items():
            obs_dict[key] = np.stack(x)

        return obs_dict

    def get_model_parameters(self):
        return sum(p.numel() for p in self.actor_critic.parameters() if p.requires_grad)

def run_mgmapf_python(
    map_name: str,
    task_instances: List[List[Tuple]],
    start_positions: List[Tuple],
    obs_radius: int = 5,
    max_steps: int = 1024,
    use_improved_planner: bool = True,
    static_cost_weight: float = 1.0,
    dynamic_cost_weight: float = 2.0,
    avoid_blocking: bool = True,
    save_animation: bool = True,
    animation_path: str = 'renders/animation.svg',
    save_metrics: bool = True,
    metrics_path: str = 'results/metrics.csv',
    ckp_path: str = '',
    collision_mode: str = 'hard',
):
    with open('./env/maps.yaml', 'r') as f:
        maps = yaml.safe_load(f)
    map_str = maps[map_name]

    _map_lines = [line.strip() for line in map_str.strip().split('\n') if line.strip()]
    _obstacles_set = set()
    for _r, _line in enumerate(_map_lines):
        for _c, _ch in enumerate(_line):
            if _ch == '#':
                _obstacles_set.add((_r, _c))

    goal_manager = MultiGoalManager(
        task_instances,
        start_positions,
        avoid_blocking=avoid_blocking,
        obstacles=_obstacles_set,
        map_shape=(len(_map_lines), len(_map_lines[0])),
    )
    env = MGMapfEnv(
        map_str,
        start_positions,
        obs_radius,
        use_improved_planner=use_improved_planner,
        static_cost_weight=static_cost_weight,
        dynamic_cost_weight=dynamic_cost_weight,
        completion_manager=goal_manager.completion_manager,
        collision_mode=collision_mode,
    )

    from .mg_infer import SFInference
    algo = SFInference(
        model_path=ckp_path,
        device='cuda'
    )
    algo.reset_states()

    metrics = PerformanceMetrics(env.num_agents, goal_manager.task_instances)
    metrics.start()

    agent_trajectories = {i: [] for i in range(env.num_agents)}

    step_count = 0

    while step_count < max_steps:
        step_count += 1
        for agent_id in range(env.num_agents):
            agent_trajectories[agent_id].append(env.current_positions[agent_id])

        current_targets = goal_manager.get_current_targets(env.current_positions)
        task_active = [
            not goal_manager.completion_manager.agent_completed[agent_id]
            for agent_id in range(env.num_agents)
        ]

        for agent_id in range(env.num_agents):
            if goal_manager.completion_manager.should_skip_planning(agent_id):
                continue
            if env.plan_path(agent_id, current_targets[agent_id]) is None:
                if env.repark_blocking_agent(agent_id, current_targets[agent_id]):
                    current_targets = goal_manager.get_current_targets(env.current_positions)


        paths = []
        for agent_id in range(env.num_agents):
            if goal_manager.completion_manager.should_skip_planning(agent_id):
                paths.append([])
            else:
                path = env.plan_path(agent_id, current_targets[agent_id])
                paths.append(path if path else [])


        observations = []
        for agent_id in range(env.num_agents):
            obs = env.get_observation(
                agent_id,
                current_targets[agent_id],
                paths[agent_id],
                include_id_map=algo._use_id_map,
                current_targets=current_targets
            )
            obs['xy'] = obs['xy'].astype(np.int32)
            obs['target_xy'] = obs['target_xy'].astype(np.int32)
            observations.append(obs)


        actions = algo.act(observations)

        for agent_id in range(env.num_agents):
            if goal_manager.completion_manager.should_force_action_zero(agent_id):
                actions[agent_id] = 0


        prev_positions = list(env.current_positions)
        new_positions, conflicts = env.step(actions)
        metrics.record_collisions(prev_positions, new_positions, task_active)
        for agent_id in range(env.num_agents):
            if task_active[agent_id]:
                metrics.record_step(agent_id)
        completed = goal_manager.check_and_update(new_positions)
        for agent_id in completed:
            metrics.record_goal_completion(agent_id, step_count)


        if goal_manager.all_completed():
            break

    for agent_id in range(env.num_agents):
        agent_trajectories[agent_id].append(env.current_positions[agent_id])

    metrics.finish(step_count)


    result_metrics = metrics.print_summary()
    if save_metrics:
        save_metrics_to_csv(result_metrics, metrics_path)

    if save_animation:
        print(f"\nGenerating SVG...")
        create_svg_animation(
            map_size=(env.height, env.width),
            obstacles=env.obstacles,
            agent_trajectories=agent_trajectories,
            task_instances=goal_manager.task_instances,
            output_path=animation_path,
            cell_size=100
        )

    return {
        'total_steps': step_count,
        'goals_completed': goal_manager.goals_completed,
        'success': goal_manager.all_completed(),
        'trajectories': agent_trajectories,
        'metrics': result_metrics
    }
