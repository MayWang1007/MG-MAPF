import torch
import inspect
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from sample_factory.model.encoder import Encoder
from sample_factory.utils.typing import Config, ObsSpace
from sample_factory.algo.utils.torch_utils import calc_num_elements
from gymnasium import spaces
from typing import Literal, Tuple as TTuple, List, Tuple, Optional
from pydantic import BaseModel
import heapq
import numpy as np

class AStarPlanner:
    def __init__(
        self,
        obstacles: List[Tuple],
        map_size: Tuple[int, int],
        use_static_cost: bool = True,
        use_dynamic_cost: bool = True,
        static_cost_weight: float = 1.0,
        dynamic_cost_weight: float = 2.0,
        reset_dynamic_cost: bool = True,
    ):
        self.obstacles = set(obstacles)
        self.height, self.width = map_size
        self.use_static_cost = use_static_cost
        self.use_dynamic_cost = use_dynamic_cost
        self.static_cost_weight = static_cost_weight
        self.reset_dynamic_cost = reset_dynamic_cost
        self.static_penalty = self._precompute_static_penalty() if use_static_cost else None
        self.num_occupations = np.zeros((self.height, self.width), dtype=np.float32)
        self.current_goal = None

    def _precompute_static_penalty(self, max_penalty: float = 5.0) -> np.ndarray:
        penalty_map = np.zeros((self.height, self.width), dtype=np.float32)
        for row in range(self.height):
            for col in range(self.width):
                if (row, col) in self.obstacles:
                    penalty_map[row, col] = np.inf
                else:
                    min_dist = float('inf')
                    for obs_row, obs_col in self.obstacles:
                        dist = abs(row - obs_row) + abs(col - obs_col)
                        min_dist = min(min_dist, dist)
                    if min_dist <= 3:
                        penalty_map[row, col] = max_penalty * np.exp(-min_dist / 2.0)
        return penalty_map

    def update_dynamic_occupations(
        self,
        agent_positions: List[Tuple],
        current_goal: Optional[Tuple] = None,
        exclude_agent: int = None,
    ):
        if self.reset_dynamic_cost and current_goal is not None:
            if self.current_goal != current_goal:
                self.current_goal = current_goal
        self.num_occupations = np.zeros((self.height, self.width), dtype=np.float32)
        for i, pos in enumerate(agent_positions):
            if exclude_agent is None or i != exclude_agent:
                if 0 <= pos[0] < self.height and 0 <= pos[1] < self.width:
                    self.num_occupations[pos] += 1.0

    def heuristic(self, pos: Tuple, goal: Tuple) -> float:
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def get_cost(self, pos: Tuple) -> float:
        cost = 1.0
        if self.use_static_cost and self.static_penalty is not None:
            row, col = pos
            if 0 <= row < self.height and 0 <= col < self.width:
                static_cost = self.static_cost_weight * self.static_penalty[row, col]
                if np.isinf(static_cost):
                    return np.inf
                cost += static_cost
        if self.use_dynamic_cost:
            cost += self.num_occupations[pos]
        return cost

    def get_neighbors(self, pos: Tuple) -> List[Tuple]:
        row, col = pos
        neighbors = []
        for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < self.height
                    and 0 <= new_col < self.width
                    and (new_row, new_col) not in self.obstacles):
                neighbors.append((new_row, new_col))
        return neighbors

    def find_path(
        self,
        start: Tuple,
        goal: Tuple,
        agent_positions: Optional[List[Tuple]] = None,
        current_agent_id: Optional[int] = None,
    ) -> Optional[List[Tuple]]:
        if start == goal:
            return [start]
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + self.get_cost(neighbor)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None


class NeighborAttentionPredictor(nn.Module):
    def __init__(
        self,
        pred_rnn_size:  int = 128,
        d_model:        int = 64,
        n_heads:        int = 4,
        n_steps:        int = 1,
        grid:           int = 11,
        max_neighbors:  int = 20,
        map_size:       int = 32,
    ):
        super().__init__()
        self.d_model       = d_model
        self.n_steps       = n_steps
        self.grid          = grid
        self.max_neighbors = max_neighbors
        self.center        = (grid - 1) / 2.0
        self.pred_rnn_size = pred_rnn_size
        self.map_size      = float(map_size)


        self.neighbor_embed = nn.Linear(2, d_model)
        self.goal_embed     = nn.Linear(2, d_model)


        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=0.0
        )


        self.pred_gru = nn.GRUCell(d_model, pred_rnn_size)


        self.action_head = nn.Linear(pred_rnn_size, 5)


        self.register_buffer('action_offsets', torch.tensor([
            [0,  0],
            [-1, 0],
            [1,  0],
            [0, -1],
            [0,  1],
        ], dtype=torch.long))


    def _extract_neighbors(
        self,
        ch1:             torch.Tensor,
        my_world_pos:    torch.Tensor,
        neighbor_targets: torch.Tensor,
    ):
        B, H, W = ch1.shape
        device  = ch1.device


        rows = torch.arange(H, device=device).float()
        cols = torch.arange(W, device=device).float()
        gr, gc = torch.meshgrid(rows, cols, indexing='ij')
        dist_flat = ((gr - self.center)**2 + (gc - self.center)**2).sqrt()                     .reshape(1, H * W).expand(B, -1)


        flat_ch1 = ch1.reshape(B, H * W).clone()
        _cidx = int(self.center) * W + int(self.center)
        flat_ch1[:, _cidx] = 0.0

        scores = torch.where(
            flat_ch1 > 0.5,
            -dist_flat,
            torch.full_like(dist_flat, -1e9),
        )

        k = min(self.max_neighbors, H * W)
        top_vals, top_idx = torch.topk(scores, k, dim=1)
        key_mask = top_vals <= -1e8


        rows_k = (top_idx // W).float()
        cols_k = (top_idx  % W).float()


        local_offset_row = rows_k - self.center
        local_offset_col = cols_k - self.center
        nb_world_row = my_world_pos[:, 0:1] + local_offset_row
        nb_world_col = my_world_pos[:, 1:2] + local_offset_col


        pos_rel = torch.stack([
            (nb_world_row - my_world_pos[:, 0:1]) / self.map_size,
            (nb_world_col - my_world_pos[:, 1:2]) / self.map_size,
        ], dim=-1)
        pos_rel = pos_rel * (~key_mask).unsqueeze(-1).float()


        goal_world = neighbor_targets[:, :k, :]
        goal_rel = torch.stack([
            (goal_world[:, :, 0] - my_world_pos[:, 0:1]) / self.map_size,
            (goal_world[:, :, 1] - my_world_pos[:, 1:2]) / self.map_size,
        ], dim=-1)
        goal_rel = goal_rel * (~key_mask).unsqueeze(-1).float()


        pos_tokens  = self.neighbor_embed(pos_rel)
        goal_tokens = self.goal_embed(goal_rel)

        return pos_tokens, goal_tokens, key_mask, top_idx, rows_k.long(), cols_k.long()


    def forward(
        self,
        obs:              torch.Tensor,
        g_states:         torch.Tensor,
        my_world_pos:     torch.Tensor,
        neighbor_targets: torch.Tensor,
        training:         bool = True,
    ):
        ch1 = obs[:, 1, :, :]
        B   = obs.shape[0]
        K   = self.max_neighbors

        pos_tokens, goal_tokens, key_mask, top_idx, rows_k, cols_k =             self._extract_neighbors(ch1, my_world_pos, neighbor_targets)


        safe_mask = key_mask.clone()
        safe_mask[key_mask.all(dim=1), 0] = False
        attn_out, attn_weights = self.self_attn(
            pos_tokens, pos_tokens, goal_tokens,
            key_padding_mask=safe_mask,
            need_weights=True,
            average_attn_weights=True,
        )


        self._last_attn_weights = attn_weights.detach()
        self._last_norm_pos     = (pos_tokens * (~key_mask).unsqueeze(-1).float()).detach()
        self._last_key_mask     = key_mask.detach()
        self._last_rows_k       = rows_k.detach()
        self._last_cols_k       = cols_k.detach()

        g_flat = g_states.view(B, K, self.pred_rnn_size)


        new_g_flat = torch.zeros_like(g_flat)
        for i in range(K):
            new_g_flat[:, i, :] = self.pred_gru(
                attn_out[:, i, :], g_flat[:, i, :]
            )
        new_g_states_flat = new_g_flat


        act_logits = self.action_head(new_g_states_flat)
        self._last_action_logits = act_logits.detach()


        pred_occ_maps = []
        cur_rows_k = rows_k.clone()
        cur_cols_k = cols_k.clone()

        for step in range(self.n_steps):
            if step == 0:
                step_logits = act_logits
            else:
                step_logits = self.action_head(new_g_states_flat)

            if training:
                soft = torch.softmax(step_logits, dim=-1)
                hard = torch.zeros_like(soft).scatter_(
                    -1, step_logits.argmax(-1, keepdim=True), 1.0
                )
                ste_out = hard - soft.detach() + soft

                offsets_f = self.action_offsets.float()
                delta = (ste_out.unsqueeze(-1) * offsets_f).sum(-2)
                pred_rows_f = cur_rows_k.float() + delta[:, :, 0]
                pred_cols_f = cur_cols_k.float() + delta[:, :, 1]

                _coords = torch.arange(self.grid, device=obs.device).float()
                _gr, _gc = torch.meshgrid(_coords, _coords, indexing='ij')
                _dy = _gr[None, None] - pred_rows_f[:, :, None, None]
                _dx = _gc[None, None] - pred_cols_f[:, :, None, None]
                _sigma = 0.8
                _gauss = torch.exp(-(_dy**2 + _dx**2) / (2 * _sigma**2))
                _valid = (~key_mask).float()
                occ_map = (_gauss * _valid[:, :, None, None]).sum(dim=1)
            else:
                pred_actions = step_logits.argmax(dim=-1)
                dr = self.action_offsets[pred_actions, 0]
                dc = self.action_offsets[pred_actions, 1]
                pred_rows = (cur_rows_k + dr).clamp(0, self.grid-1)
                pred_cols = (cur_cols_k + dc).clamp(0, self.grid-1)
                occ_map = torch.zeros(B, self.grid, self.grid, device=obs.device)
                _valid_k = ~key_mask
                _b_idx = torch.arange(B, device=obs.device).unsqueeze(1).expand_as(pred_rows)
                occ_map[_b_idx[_valid_k], pred_rows[_valid_k], pred_cols[_valid_k]] = -1.0
                cur_rows_k = pred_rows
                cur_cols_k = pred_cols

            pred_occ_maps.append(occ_map)

        occupancy_maps = torch.stack(pred_occ_maps, dim=1)


        new_g_states = new_g_states_flat.view(B, K * self.pred_rnn_size)

        return occupancy_maps, new_g_states


class EncoderConfig(BaseModel):
    extra_fc_layers: int = 1
    num_filters:     int = 64
    num_res_blocks:  int = 8
    activation_func: Literal['ReLU', 'ELU', 'Mish'] = 'ReLU'
    hidden_size:     int = 512


def activation_func(cfg: EncoderConfig) -> nn.Module:
    if cfg.activation_func == "ELU":
        return nn.ELU(inplace=True)
    elif cfg.activation_func == "ReLU":
        return nn.ReLU(inplace=True)
    elif cfg.activation_func == "Mish":
        return nn.Mish(inplace=True)
    raise ValueError(f"Unknown activation_func: {cfg.activation_func}")


class ResBlock(nn.Module):
    def __init__(self, cfg: EncoderConfig, input_ch: int, output_ch: int):
        super().__init__()
        self.res_block_core = nn.Sequential(
            activation_func(cfg),
            nn.Conv2d(input_ch, output_ch, kernel_size=3, stride=1, padding=1),
            activation_func(cfg),
            nn.Conv2d(output_ch, output_ch, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return x + self.res_block_core(x)


class ResnetEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)

        if hasattr(cfg, 'encoder') and isinstance(cfg.encoder, dict):
            self.encoder_cfg = EncoderConfig(**cfg.encoder)
        else:
            self.encoder_cfg = EncoderConfig()

        if isinstance(obs_space, spaces.Dict):
            actual_obs_space = obs_space['obs']
            obs_input_ch     = actual_obs_space.shape[0]
            obs_grid_shape   = actual_obs_space.shape[1:]
        elif hasattr(obs_space, 'shape'):
            obs_input_ch   = obs_space.shape[0]
            obs_grid_shape = obs_space.shape[1:]
        else:
            obs_input_ch   = 2

            _r = int(getattr(cfg, 'obs_radius', 5))
            obs_grid_shape = (2 * _r + 1, 2 * _r + 1)

        self.obs_input_ch = obs_input_ch
        grid_h, grid_w    = obs_grid_shape

        self.use_predictor:   bool = bool(getattr(cfg, 'use_neighbor_predictor', False))
        self.n_pred_steps:    int  = int(getattr(cfg, 'prediction_steps', 1))
        self.policy_rnn_size: int  = int(getattr(cfg, 'policy_rnn_size', 256))
        self.pred_rnn_size:   int  = int(getattr(cfg, 'pred_rnn_size', 128))
        self.max_neighbors:   int  = int(getattr(cfg, 'max_neighbors', 20))
        self.map_size:        int  = int(getattr(cfg, 'map_size', 32))


        self.g_states_size: int = self.max_neighbors * self.pred_rnn_size


        self._prev_rnn_states: Optional[torch.Tensor] = None

        if self.use_predictor:
            self.predictor = NeighborAttentionPredictor(
                pred_rnn_size=self.pred_rnn_size,
                d_model=64,
                n_heads=4,
                n_steps=self.n_pred_steps,
                grid=grid_h,
                max_neighbors=self.max_neighbors,
                map_size=self.map_size,
            )
            conv_input_ch = obs_input_ch + self.n_pred_steps
        else:
            self.predictor = None
            conv_input_ch  = obs_input_ch

        conv_test_shape = (conv_input_ch, grid_h, grid_w)
        resnet_conf = [[self.encoder_cfg.num_filters, self.encoder_cfg.num_res_blocks]]
        curr_ch = conv_input_ch
        layers  = []
        for out_channels, res_blocks in resnet_conf:
            layers.append(
                nn.Conv2d(curr_ch, out_channels, kernel_size=3, stride=1, padding=1)
            )
            layers.extend(
                [ResBlock(self.encoder_cfg, out_channels, out_channels)
                 for _ in range(res_blocks)]
            )
            curr_ch = out_channels
        layers.append(activation_func(self.encoder_cfg))
        self.conv_head = nn.Sequential(*layers)

        self.conv_head_out_size = calc_num_elements(self.conv_head, conv_test_shape)
        self.encoder_out_size   = self.conv_head_out_size

        if self.encoder_cfg.extra_fc_layers:
            self.extra_linear = nn.Sequential(
                nn.Linear(self.encoder_out_size, self.encoder_cfg.hidden_size),
                activation_func(self.encoder_cfg),
            )
            self.encoder_out_size = self.encoder_cfg.hidden_size

    def get_out_size(self) -> int:
        return self.encoder_out_size

    def forward(self, obs_dict):
        if isinstance(obs_dict, dict):
            x = obs_dict['obs']
        else:
            x = obs_dict

        if self.use_predictor:
            B, C, H, W = x.shape

            rnn_ok = (
                self._prev_rnn_states is not None
                and self._prev_rnn_states.shape[0] == B
            )

            if rnn_ok:
                g_states_flat = self._prev_rnn_states[
                    :, self.policy_rnn_size:self.policy_rnn_size + self.g_states_size
                ]


                if isinstance(obs_dict, dict):
                    my_world_pos     = obs_dict.get('xy', torch.zeros(B, 2, device=x.device))
                    neighbor_targets = obs_dict.get('neighbor_targets',
                                                    torch.zeros(B, self.max_neighbors, 2, device=x.device))
                    if not isinstance(my_world_pos, torch.Tensor):
                        my_world_pos = torch.tensor(my_world_pos, dtype=torch.float32, device=x.device)
                    if not isinstance(neighbor_targets, torch.Tensor):
                        neighbor_targets = torch.tensor(neighbor_targets, dtype=torch.float32, device=x.device)
                    my_world_pos     = my_world_pos.float().to(x.device)
                    neighbor_targets = neighbor_targets.float().to(x.device)
                else:
                    my_world_pos     = torch.zeros(B, 2, device=x.device)
                    neighbor_targets = torch.zeros(B, self.max_neighbors, 2, device=x.device)

                is_training = self.training
                occupancy_maps, new_g_states_flat = self.predictor(
                    x, g_states_flat, my_world_pos, neighbor_targets, training=is_training
                )
                self._last_pred_maps = occupancy_maps.detach()
            else:
                occupancy_maps = torch.zeros(
                    B, self.n_pred_steps, H, W, device=x.device, dtype=x.dtype
                )

            x = torch.cat([x, occupancy_maps], dim=1)
            self._last_obs_concat = x.detach()

        x = self.conv_head(x)
        x = x.contiguous().view(-1, self.conv_head_out_size)

        if self.encoder_cfg.extra_fc_layers:
            x = self.extra_linear(x)

        return x

def make_adaptive_conv_encoder(cfg: Config, obs_space: ObsSpace):
    return ResnetEncoder(cfg, obs_space)


def prepare_pretrained_init(source, destination, learning_rate):
    source = Path(source)
    destination = Path(destination)
    if destination.exists() or (destination.parent.parent / 'config.json').exists():
        raise FileExistsError(destination)
    if destination.parent.exists() and any(destination.parent.glob('*.pth')):
        raise FileExistsError(destination.parent)

    load_args = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        load_args['weights_only'] = False
    checkpoint = torch.load(source, **load_args)
    if not {'model', 'optimizer'} <= checkpoint.keys():
        raise ValueError('Source is not a Sample Factory training checkpoint')

    optimizer = checkpoint['optimizer']
    optimizer['state'] = {}
    for group in optimizer['param_groups']:
        group['lr'] = learning_rate
        if 'initial_lr' in group:
            group['initial_lr'] = learning_rate

    initialized = {
        'train_step': 0,
        'env_steps': 0,
        'best_performance': -1e9,
        'model': checkpoint['model'],
        'optimizer': optimizer,
        'curr_lr': learning_rate,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(initialized, destination)
