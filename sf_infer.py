import argparse
from pathlib import Path
import yaml
from mg_train.mg_infer import run_mgmapf_python


def convert_xy_to_rowcol(pos_xy: tuple) -> tuple:
    x, y = pos_xy
    return (y, x)


def load_task_from_yaml(task_yaml_path: str):
    with open(task_yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f'Task YAML must contain a mapping: {task_yaml_path}')
    agents = data.get('agents')
    if not isinstance(agents, list) or not agents:
        raise ValueError(f'Task YAML must contain a non-empty agents list: {task_yaml_path}')
    task_instances = []
    start_positions = []
    for agent in agents:
        if not isinstance(agent, dict) or 'start' not in agent or 'goals' not in agent:
            raise ValueError('Each YAML agent must define start and goals')
        start_xy = tuple(agent['start'])
        if len(start_xy) != 2:
            raise ValueError('Each YAML start must contain two coordinates')
        start_rc = convert_xy_to_rowcol(start_xy)
        start_positions.append(start_rc)
        goals_xy = [tuple(goal) for goal in agent['goals']]
        if not goals_xy or any(len(goal) != 2 for goal in goals_xy):
            raise ValueError('Each YAML agent must define goals as coordinate pairs')
        goals_rc = [convert_xy_to_rowcol(goal) for goal in goals_xy]
        task_instances.append(goals_rc)
    return task_instances, start_positions


def resolve_checkpoint(checkpoint_path: str) -> str:
    path = Path(checkpoint_path)
    if path.is_file():
        return str(path)
    if not path.is_dir():
        return checkpoint_path

    checkpoint_dir = path / 'checkpoint_p0'
    candidates = list(checkpoint_dir.glob('best_*.pth'))
    if not candidates:
        candidates = list(checkpoint_dir.glob('checkpoint_*.pth'))
    if not candidates:
        raise FileNotFoundError(f'No checkpoint found under {checkpoint_dir}')
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def load_eval_config(args):
    if not args.eval_yaml:
        return args
    with open(args.eval_yaml, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def run_single(args):
    if args.one_shot_mapf:
        args.num_goals = 1

    if not args.task_yaml:
        raise ValueError('Testing requires --task_yaml; random task generation is disabled')
    print(f"\nLoading tasks from {args.task_yaml}")
    task_instances, start_positions = load_task_from_yaml(args.task_yaml)
    if args.one_shot_mapf:
        task_instances = [goals[:1] for goals in task_instances]
    num_agents = len(start_positions)
    goal_counts = {len(goals) for goals in task_instances}
    if len(goal_counts) != 1:
        raise ValueError('All agents in the task YAML must have the same number of goals')
    num_goals = next(iter(goal_counts))
    if args.agent_counts and (
        len(args.agent_counts) != 1 or int(args.agent_counts[0]) != num_agents
    ):
        raise ValueError('--agent_counts must match the agent count in --task_yaml')
    print(f"Loaded {num_agents} agents with {num_goals} goals each")

    print(f"Start testing...")
    results = run_mgmapf_python(
        map_name=args.map_name,
        task_instances=task_instances,
        start_positions=start_positions,
        max_steps=args.max_episode_steps,
        obs_radius=5,
        save_animation=args.save_animation,
        avoid_blocking=True,
        animation_path=f'{args.animation_dir}/{args.map_name}_{num_agents}_{num_goals}_seed{args.seed}.svg',
        metrics_path=args.metrics_path,
        ckp_path=args.checkpoint,
        collision_mode=args.collision_mode,
    )

    if results.get('success'):
        print(f"\nAll succeeded")
    print(f"{'='*70}")


def main(args):
    args = load_eval_config(args)
    if not args.task_yaml:
        raise ValueError(
            'Testing only reads task data from YAML. Supply --task_yaml '
            'or set task_yaml in --eval_yaml.'
        )
    if not Path(args.task_yaml).is_file():
        raise FileNotFoundError(f'Task YAML not found: {args.task_yaml}')
    args.checkpoint = resolve_checkpoint(args.checkpoint)
    run_single(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Multi-Agent Path Planning Test (tasks loaded from YAML only)'
    )
    parser.add_argument('--num_agents', type=int, default=10,
                        help='Legacy option; actual count is read from --task_yaml')
    parser.add_argument('--agent_counts', type=int, nargs='+', default=None,
                        help='Agent counts for batch evaluation')
    parser.add_argument('--num_goals', type=int, default=10,
                        help='Legacy option; actual goals are read from --task_yaml')
    parser.add_argument('--seed', type=int, default=256, help='Random seed')
    parser.add_argument('--map_name', type=str, default='maze-32-32-4', help='Map name')
    parser.add_argument('--max_episode_steps', type=int, default=2000, help='Max steps')
    parser.add_argument('--save_animation', action='store_true', default=True,
                        help='Save SVG animation (default: True)')
    parser.add_argument('--no_animation', dest='save_animation', action='store_false',
                        help='Disable SVG animation')
    parser.add_argument('--show_map_names', action='store_true', help='Show available maps')
    parser.add_argument('--checkpoint', type=str,
                        default='train_dir/mg_mapf_final/checkpoint_p0/checkpoint_000000051_104448.pth',
                        help='Checkpoint路径')
    parser.add_argument('--task_yaml', type=str, default=None,
                        help='Required task YAML file; test tasks are never generated randomly')
    parser.add_argument('--eval_yaml', type=str, default=None,
                        help='Path to evaluation YAML file')
    parser.add_argument('--metrics_path', type=str, default='results/metrics_follower.csv',
                        help='Path to metrics CSV file')
    parser.add_argument('--animation_dir', type=str, default='renders',
                        help='Directory to save SVG animations')
    parser.add_argument('--one_shot_mapf', action='store_true',
                        help='Run standard single-goal MAPF evaluation')
    parser.add_argument('--collision_mode', choices=['hard', 'soft', 'pogema'], default='hard')

    args = parser.parse_args()

    if args.show_map_names:
        with open('./env/maps.yaml', 'r', encoding='utf-8') as f:
            maps = yaml.safe_load(f)
        for idx, map_name in enumerate(sorted(maps.keys()), 1):
            print(f"  {idx}. {map_name}")
        exit()

    main(args)
