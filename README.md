# IMMP: Intent-aware Multi-Goal Multi-Agent Pathfinding

## Environment setup

The reference project uses Python 3.10:

```bash
conda create -n immp python=3.10
conda activate immp

pip install torch torchvision
pip install sample-factory==2.0
pip install gymnasium numpy pyyaml matplotlib scipy
```

## Training

Training continues to use the environment's task generation utilities.

```bash
python sf_train.py --config_yaml ./path/to/task.yaml
```

## Evaluation

Evaluation requires `--task_yaml`, or a `task_yaml` entry in the file passed to `--eval_yaml`.
```bash
python sf_infer.py \
  --checkpoint train_dir/<experiment>/checkpoint_p0/best.pth \
  --map_name maze-32-32-4 \
  --task_yaml ./path/to/maze-32-32-4_16_4_0.yaml \
  --max_episode_steps 2000 \
  --no_animation
```

Each task YAML must contain a non-empty `agents` list. For example:

```yaml
agents:
  - name: agent0
    start: [4, 7]
    goals:
      - [12, 9]
      - [16, 3]
```
