from pathlib import Path
from typing import List, Tuple, Dict

class SVGAnimationGenerator:
    def __init__(self, map_size: Tuple[int, int], cell_size: int = 100):
        self.height, self.width = map_size
        self.cell_size = cell_size
        self.canvas_width  = self.width  * cell_size
        self.canvas_height = self.height * cell_size
        self.agent_colors = [
            '#c1433c', '#2e6f9e', '#6e81af', '#00b9c8',
            '#f39c12', '#27ae60', '#8e44ad', '#e67e22',
            '#16a085', '#d35400', '#2980b9', '#c0392b'
        ]

    def _coord_to_pixel(self, row: int, col: int) -> Tuple[float, float]:
        return (col * self.cell_size + self.cell_size / 2,
                row * self.cell_size + self.cell_size / 2)

    def _generate_svg_header(self) -> str:
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     width="512" height="512" viewBox="0 0 {self.canvas_width} {self.canvas_height}">
<defs>
    <rect id="obstacle" width="{self.cell_size*0.7}" height="{self.cell_size*0.7}"
          fill="#84A1AE" rx="{self.cell_size*0.15}"/>
    <style>
        .grid-line {{stroke: #84A1AE; stroke-width: {self.cell_size*0.1}; opacity: 0.3;}}
        .agent {{r: {self.cell_size*0.35};}}
        .target {{fill: none; stroke-width: {self.cell_size*0.1}; r: {self.cell_size*0.35};}}
        .path-line {{fill: none; stroke-width: {self.cell_size*0.15}; opacity: 0.4;}}
    </style>
</defs>
'''

    def _generate_grid(self) -> str:
        lines = []
        for col in range(self.width + 1):
            x = col * self.cell_size
            lines.append(f'<line class="grid-line" x1="{x}" x2="{x}" '
                         f'y1="0" y2="{self.canvas_height}" />')
        for row in range(self.height + 1):
            y = row * self.cell_size
            lines.append(f'<line class="grid-line" x1="0" x2="{self.canvas_width}" '
                         f'y1="{y}" y2="{y}" />')
        return '\n'.join(lines)

    def _generate_obstacles(self, obstacles: List[Tuple[int, int]]) -> str:
        elements = []
        for row, col in obstacles:
            x, y = self._coord_to_pixel(row, col)
            elements.append(f'<use href="#obstacle" x="{x - self.cell_size*0.35}" '
                            f'y="{y - self.cell_size*0.35}" />')
        return '\n'.join(elements)

    def _generate_path_trace(self, trajectory: List[Tuple[int, int]], color: str) -> str:
        if len(trajectory) < 2:
            return ""
        pts = [f"{self._coord_to_pixel(r,c)[0]},{self._coord_to_pixel(r,c)[1]}"
               for r, c in trajectory]
        return f'<path class="path-line" stroke="{color}" d="M {" L ".join(pts)}" />'

    def _generate_animations(self, agent_id: int, trajectory: List[Tuple[int, int]],
                             goals: List[Tuple[int, int]], color: str,
                             total_steps: int) -> str:
        if not trajectory:
            return ""
        duration = max(10.0, total_steps * 0.05)
        n = len(trajectory)
        key_times = [f"{i/(n-1):.10f}" if n > 1 else "0.0000000000" for i in range(n)]
        x_vals = [str(self._coord_to_pixel(r, c)[0]) for r, c in trajectory]
        y_vals = [str(self._coord_to_pixel(r, c)[1]) for r, c in trajectory]
        key_times += ["1.0"]; x_vals += [x_vals[-1]]; y_vals += [y_vals[-1]]
        kt = ";".join(key_times); xv = ";".join(x_vals); yv = ";".join(y_vals)
        sx, sy = self._coord_to_pixel(*trajectory[0])
        out = (f'<circle class="agent" cx="{sx}" cy="{sy}" fill="{color}" '
               f'r="{self.cell_size*0.35}">\n'
               f'    <animate attributeName="cx" dur="{duration}s" keyTimes="{kt}" '
               f'repeatCount="indefinite" values="{xv}"/>\n'
               f'    <animate attributeName="cy" dur="{duration}s" keyTimes="{kt}" '
               f'repeatCount="indefinite" values="{yv}"/>\n'
               f'    <animate attributeName="visibility" dur="{duration}s" '
               f'keyTimes="0.0;{key_times[-2]};1.0" repeatCount="indefinite" '
               f'values="visible;visible;hidden"/>\n</circle>')
        for goal_row, goal_col in goals:
            gx, gy = self._coord_to_pixel(goal_row, goal_col)
            dt = next((i/(n-1) for i, (tr, tc) in enumerate(trajectory)
                       if (tr, tc) == (goal_row, goal_col) and n > 1), 1.0)
            nt = min(1.0, dt + 0.01)
            out += (f'\n<circle class="target" cx="{gx}" cy="{gy}" '
                    f'r="{self.cell_size*0.35}" stroke="{color}">\n'
                    f'    <animate attributeName="visibility" dur="{duration}s" '
                    f'keyTimes="0.0;{dt:.10f};{nt:.10f};1.0" repeatCount="indefinite" '
                    f'values="visible;visible;hidden;hidden"/>\n</circle>')
        return out

    def generate(self, obstacles: List[Tuple[int, int]],
                 agent_trajectories: Dict[int, List[Tuple[int, int]]],
                 task_instances: List[List[Tuple[int, int]]],
                 output_path: str = 'animation.svg') -> str:
        total_steps = max((len(t) for t in agent_trajectories.values()), default=1)
        svg_parts = [self._generate_svg_header(), self._generate_grid(),
                     self._generate_obstacles(obstacles)]
        for aid in sorted(agent_trajectories.keys()):
            p = self._generate_path_trace(agent_trajectories[aid],
                                          self.agent_colors[aid % len(self.agent_colors)])
            if p:
                svg_parts.append(p)
        for aid in sorted(agent_trajectories.keys()):
            svg_parts.append(self._generate_animations(
                aid, agent_trajectories[aid],
                task_instances[aid] if aid < len(task_instances) else [],
                self.agent_colors[aid % len(self.agent_colors)], total_steps))
        svg_parts.append('</svg>')
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text('\n'.join(svg_parts), encoding='utf-8')
        print(f"SVG output: {output_path}")
        return str(out)


def create_svg_animation(map_size: Tuple[int, int],
                         obstacles: List[Tuple[int, int]],
                         agent_trajectories: Dict[int, List[Tuple[int, int]]],
                         task_instances: List[List[Tuple[int, int]]],
                         output_path: str = 'renders/animation.svg',
                         cell_size: int = 100) -> str:
    return SVGAnimationGenerator(map_size, cell_size).generate(
        obstacles, agent_trajectories, task_instances, output_path)
