import sys
import yaml
from pathlib import Path
import numpy as np
import torch

sys.path.append('.')
sys.path.append('./mg_train')

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import make_runner
from sample_factory.algo.utils.misc import ExperimentStatus

from mg_train.mg_gym_wrapper import make_mg_mapf_env
from mg_train.mg_model import make_adaptive_conv_encoder, prepare_pretrained_init


def register_custom_components():
    register_env('mg_mapf', make_mg_mapf_env)
    global_model_factory().register_encoder_factory(make_adaptive_conv_encoder)


USE_MG_LEARNER = False

try:
    from sample_factory.algo.learning.learner import Learner

    class MGLearner(Learner):
        def _calculate_loss(self, cfg, mb, valids, num_invalids):
            if not hasattr(self, '_pred_log_step'):
                self._pred_log_step = 0
                self._pred_log_path = 'pred_loss_log.txt'
                with open(self._pred_log_path, 'a') as f:
                    f.write(f"[called] use_aux_loss={getattr(cfg, 'use_aux_loss', 'MISSING')} "
                            f"use_neighbor_predictor={getattr(cfg, 'use_neighbor_predictor', 'MISSING')} "
                            f"has_predictor={hasattr(self.actor_critic.encoder, 'predictor')}\n")


            encoder = self.actor_critic.encoder
            if getattr(cfg, 'use_neighbor_predictor', False):
                encoder._prev_rnn_states = mb.rnn_states


            loss, stats = super()._calculate_loss(cfg, mb, valids, num_invalids)


            if not (getattr(cfg, 'use_neighbor_predictor', False)
                    and getattr(cfg, 'use_aux_loss', False)
                    and hasattr(encoder, 'predictor')):
                return loss, stats

            with torch.enable_grad():
                obs_x    = mb.obs['obs']
                prev_rnn = mb.rnn_states
                T = obs_x.shape[0]
                n_steps  = int(getattr(cfg, 'prediction_steps', 1))
                recurrence = int(getattr(cfg, 'recurrence', 32))
                device = obs_x.device
                H, W = obs_x.shape[2], obs_x.shape[3]


                _, future_pos, key_mask = encoder.predictor(obs_x, prev_rnn)


                sigma = 0.5
                coords = torch.linspace(-1.0, 1.0, H, device=device)
                gr, gc = torch.meshgrid(coords, coords, indexing='ij')

                valid_nb = (~key_mask).float()


                soft_maps = []
                for k in range(n_steps):
                    py = future_pos[:, :, k, 0]
                    px = future_pos[:, :, k, 1]

                    dy = gr[None, None] - py[:, :, None, None]
                    dx = gc[None, None] - px[:, :, None, None]
                    gauss = torch.exp(-(dy ** 2 + dx ** 2) / (2.0 * sigma ** 2))
                    gauss = gauss * valid_nb[:, :, None, None]
                    soft_maps.append(gauss.sum(dim=1))


                any_done = mb.dones.bool()
                chunk_end = torch.zeros(T, dtype=torch.bool, device=device)
                last_in_chunk = torch.arange(recurrence - 1, T, recurrence, device=device)
                if last_in_chunk.numel() > 0:
                    chunk_end[last_in_chunk] = True


                aux_loss_total = torch.tensor(0.0, device=device)
                valid_steps = 0

                for k in range(n_steps):
                    offset = k + 1

                    if T <= offset:
                        continue


                    invalid = torch.zeros(T - offset, dtype=torch.bool, device=device)
                    for j in range(offset):
                        invalid = invalid | any_done[j: T - offset + j]
                        invalid = invalid | chunk_end[j: T - offset + j]
                    valid_mask_k = ~invalid

                    if valid_mask_k.sum() == 0:
                        continue

                    pred_k = soft_maps[k][:T - offset]
                    gt_k   = obs_x[offset:, 1, :, :]

                    step_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        pred_k[valid_mask_k],
                        gt_k[valid_mask_k],
                    )
                    aux_loss_total = aux_loss_total + step_loss
                    valid_steps += 1
                    stats[f'pred_loss_step{offset}'] = step_loss.item()
                    stats[f'pred_valid_frac_step{offset}'] = valid_mask_k.float().mean().item()

                if valid_steps == 0:
                    return loss, stats

                aux_loss = aux_loss_total / valid_steps
                coeff = float(getattr(cfg, 'pred_loss_coeff', 0.05))
                loss = loss + coeff * aux_loss
                stats['pred_aux_loss'] = aux_loss.item()


                self._pred_log_step += 1
                if self._pred_log_step % 1000 == 0:
                    step_strs = ' | '.join(
                        f"step{k+1}={stats.get(f'pred_loss_step{k+1}', float('nan')):.4f}"
                        for k in range(n_steps)
                    )
                    log_line = (
                        f"[step {self._pred_log_step:08d}] "
                        f"pred_aux_loss={aux_loss.item():.4f} | {step_strs}\n"
                    )
                    with open(self._pred_log_path, 'a') as f:
                        f.write(log_line)

            return loss, stats

except Exception as e:
    print(f"[sf_train] MGLearner load failed: {e}")
    import traceback; traceback.print_exc()
    MGLearner = None


def load_config_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def apply_config_to_cfg(cfg, config):
    cfg.env = config.get('env', 'mg_mapf')
    cfg.map_name = config.get('map_name', 'maze-32-32-4')
    cfg.maps_yaml = config.get('maps_yaml', './env/maps.yaml')
    cfg.num_agents = config.get('num_agents', 4)
    cfg.num_goals = config.get('num_goals', 4)
    cfg.obs_radius = config.get('obs_radius', 5)
    cfg.max_episode_steps = config.get('max_episode_steps', 2048)
    cfg.collision_mode = config.get('collision_mode', 'hard')

    cfg.enable_curriculum = config.get('enable_curriculum', True)
    cfg.curriculum_agents = config.get('curriculum_agents', None)
    cfg.curriculum_goals = config.get('curriculum_goals', None)
    cfg.curriculum_steps = config.get('curriculum_steps', None)


    cfg.goal_reward = config.get('goal_reward', 1.0)
    cfg.multi_goal_bonus = config.get('multi_goal_bonus', 0.5)
    cfg.collision_penalty = config.get('collision_penalty', -0.05)
    cfg.movement_penalty = config.get('movement_penalty', -0.001)
    cfg.backtrack_penalty = config.get('backtrack_penalty', -0.02)
    cfg.stay_penalty = config.get('stay_penalty', -0.01)
    cfg.stay_penalty_threshold = config.get('stay_penalty_threshold', 3)

    encoder_cfg = config.get('encoder', {})
    cfg.encoder = {
        'extra_fc_layers': encoder_cfg.get('extra_fc_layers', 1),
        'num_filters': encoder_cfg.get('num_filters', 64),
        'num_res_blocks': encoder_cfg.get('num_res_blocks', 8),
        'activation_func': encoder_cfg.get('activation_func', 'ReLU'),
        'hidden_size': encoder_cfg.get('hidden_size', 512)
    }


    rnn_cfg = config.get('rnn', {})
    cfg.use_rnn = rnn_cfg.get('use_rnn', True)
    cfg.rnn_size = rnn_cfg.get('rnn_size', 256)
    cfg.policy_rnn_size = rnn_cfg.get('policy_rnn_size', 256)
    cfg.pred_rnn_size   = rnn_cfg.get('pred_rnn_size', 128)
    cfg.recurrence = rnn_cfg.get('recurrence', 32)
    cfg.rnn_type = rnn_cfg.get('rnn_type', 'gru')


    training_cfg = config.get('training', {})
    cfg.train_for_env_steps = training_cfg.get('train_for_env_steps', 10000000)
    cfg.rollout = training_cfg.get('rollout', 32)
    cfg.num_workers = training_cfg.get('num_workers', 8)
    cfg.num_envs_per_worker = training_cfg.get('num_envs_per_worker', 4)
    cfg.worker_num_splits = training_cfg.get('worker_num_splits', 1)
    cfg.max_policy_lag = training_cfg.get('max_policy_lag', 1)
    cfg.force_envs_single_thread = training_cfg.get('force_envs_single_thread', True)


    ppo_cfg = config.get('ppo', {})
    cfg.ppo_clip_ratio = ppo_cfg.get('ppo_clip_ratio', 0.1)
    cfg.batch_size = ppo_cfg.get('batch_size', 16384)
    cfg.num_batches_per_epoch = ppo_cfg.get('num_batches_per_epoch', 16)
    cfg.num_batches_to_accumulate = ppo_cfg.get('num_batches_to_accumulate', 1)


    optimizer_cfg = config.get('optimizer', {})
    cfg.optimizer = optimizer_cfg.get('type', 'adam')
    cfg.learning_rate = optimizer_cfg.get('learning_rate', 0.000146)
    cfg.gamma = optimizer_cfg.get('gamma', 0.9756)
    cfg.lr_schedule = optimizer_cfg.get('lr_schedule', 'kl_adaptive_minibatch')


    norm_cfg = config.get('normalization', {})
    cfg.normalize_input = norm_cfg.get('normalize_input', False)
    cfg.normalize_returns = norm_cfg.get('normalize_returns', False)


    other_cfg = config.get('other', {})
    cfg.exploration_loss_coeff = other_cfg.get('exploration_loss_coeff', 0.03)
    cfg.async_rl = other_cfg.get('async_rl', False)
    cfg.shuffle_minibatches = other_cfg.get('shuffle_minibatches', False)
    cfg.decoder_mlp_layers = other_cfg.get('decoder_mlp_layers', [])
    cfg.value_bootstrap = other_cfg.get('value_bootstrap', True)
    cfg.save_milestones_sec = other_cfg.get('save_milestones_sec', -1)
    cfg.keep_checkpoints = other_cfg.get('keep_checkpoints', 3)
    cfg.stats_avg = other_cfg.get('stats_avg', 10)
    cfg.save_best_metric = other_cfg.get('save_best_metric', 'success')


    pred_cfg = config.get('predictor', {})

    if 'map_pool' in config:
        cfg.map_pool   = config['map_pool']
    if 'agent_pool' in config:
        cfg.agent_pool = config['agent_pool']
    if 'goal_pool' in config:
        cfg.goal_pool  = config['goal_pool']

    cfg.use_neighbor_predictor = pred_cfg.get('use_neighbor_predictor', False)
    cfg.prediction_steps = pred_cfg.get('prediction_steps', 1)
    cfg.pred_loss_coeff = pred_cfg.get('pred_loss_coeff', 0.05)
    cfg.use_aux_loss = pred_cfg.get('use_aux_loss', False)
    cfg.prediction_head = pred_cfg.get('prediction_head', 'action_ste')
    cfg.history_steps   = pred_cfg.get('history_steps', 3)
    cfg.max_neighbors   = pred_cfg.get('max_neighbors', 20)

    _yaml_map_size = pred_cfg.get('map_size', 32)
    try:
        import yaml as _yaml
        _maps_yaml = cfg.maps_yaml
        with open(_maps_yaml, 'r') as _f:
            _maps = _yaml.safe_load(_f)
        _pool = config.get('map_pool', None)
        if not _pool:
            _pool = [config.get('map_name', 'maze-32-32-4')]
        _max_side = 0
        for _mn in _pool:
            if _mn in _maps:
                _lines = [l for l in _maps[_mn].strip().split('\n') if l.strip()]
                _h, _w = len(_lines), len(_lines[0].strip())
                _max_side = max(_max_side, _h, _w)
        cfg.map_size = _max_side if _max_side > 0 else _yaml_map_size
    except Exception:
        cfg.map_size = _yaml_map_size
    cfg.confidence_weight = pred_cfg.get('confidence_weight', 0.3)
    cfg.collab_penalty    = pred_cfg.get('collab_penalty', 0.1)


    cfg.train_dir = config.get('train_dir', './train_dir')


    if 'experiment' not in getattr(cfg, 'cli_args', {}):
        cfg.experiment = config.get('experiment', cfg.experiment)


    cfg.with_wandb = config.get('with_wandb', False)

    return cfg


def add_custom_args(parser):
    parser.add_argument('--config_yaml', type=str, default='train_new.yaml',
                       help='Path to YAML configuration file')
    parser.add_argument('--map_name', type=str, default=None)
    parser.add_argument('--num_agents', type=int, default=None)
    parser.add_argument('--num_goals', type=int, default=None)
    parser.add_argument('--obs_radius', type=int, default=None)
    parser.add_argument('--max_episode_steps', type=int, default=None)
    parser.add_argument('--enable_curriculum', type=bool, default=None)
    parser.add_argument('--init_checkpoint', type=str, default=None)

    return parser


def main():
    register_custom_components()

    import sys
    if '--env' not in ' '.join(sys.argv):
        sys.argv.extend(['--env', 'mg_mapf'])

    parser, cfg = parse_sf_args(evaluation=False)
    parser = add_custom_args(parser)
    cfg = parse_full_cfg(parser)

    yaml_path = cfg.config_yaml
    if Path(yaml_path).exists():
        config = load_config_from_yaml(yaml_path)
        cfg = apply_config_to_cfg(cfg, config)

    if cfg.init_checkpoint:
        if cfg.restart_behavior != 'resume':
            raise ValueError('Pretrained initialization requires restart_behavior=resume')
        checkpoint_path = (Path(cfg.train_dir) / cfg.experiment / 'checkpoint_p0'
                           / 'checkpoint_000000000_0.pth')
        prepare_pretrained_init(cfg.init_checkpoint, checkpoint_path, cfg.learning_rate)


    if getattr(cfg, 'use_aux_loss', False) and MGLearner is not None:
        import sample_factory.algo.learning.learner_worker as _lw_mod
        _original_learner_cls = _lw_mod.Learner
        _lw_mod.Learner = MGLearner
        print("[sf_train] MGLearner monkey-patched into LearnerWorker")
    else:
        _original_learner_cls = None
        if getattr(cfg, 'use_aux_loss', False) and MGLearner is None:
            print("[sf_train] WARNING: MGLearner could not be imported. "
                  "Falling back to PPO-only training.")

    cfg, runner = make_runner(cfg)

    status = runner.init()
    if status == ExperimentStatus.SUCCESS:
        status = runner.run()


    if _original_learner_cls is not None:
        import sample_factory.algo.learning.learner_worker as _lw_mod
        _lw_mod.Learner = _original_learner_cls

    return status


if __name__ == '__main__':
    status = main()
    exit_code = status.value if hasattr(status, 'value') else status
    sys.exit(exit_code)
