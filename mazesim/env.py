# env.py
from typing import List, Tuple
from pathlib import Path
import re

import numpy as np
from types import SimpleNamespace

_ENV = None

def _require_env():
    if _ENV is None:
        raise RuntimeError("Maze env has not been initialized, please call set_maze_env(...) first")
    return _ENV

def get_env():
    return _require_env()

def _layout_26() -> List[str]:
    return [
        '##########################',
        '#                        #',
        '#                        #',
        '#                        #',
        '#                        #',
        '#                        #',
        '#     ##############     #',
        '#     #            #     #',
        '#     #            #     #',
        '#     #            #     #',
        '#     #            #     #',
        '#     #            #     #',
        '#     #            #     #',
        '#     #            #     #',
        '#     #            #     #',
        '#     #            #     #',
        '#     #            #     #',
        '#     #            #     #',
        '#     #            #     #',
        '#     ##############     #',
        '#                        #',
        '#                        #',
        '#                        #',
        '#                        #',
        '#                        #',
        '##########################',
    ]

def _layout_38() -> List[str]:
    return [
        '######################################',
        '#                                    #',
        '#                                    #',
        '#                                    #',
        '#                                    #',
        '#                                    #',
        '#     ##########################     #',
        '#     #                        #     #',
        '#     #                        #     #',
        '#     #                        #     #',
        '#     #                        #     #',
        '#     #                        #     #',
        '#     #                        #     #',
        '#     #                        #     #',
        '#     #                        #     #',
        '#     #                        #     #',
        '#     #                        #     #',
        '#     #                        #     #',
        '#     #                        #     #',
        '#     #                        #     #',
        '#     #                        #     #',
        '#     #                        #     #',
        '#     #                        #     #',
        '#     #                        #     #',
        '#     #                        #     #',
        '#     #                        #     #',
        '#     #                        #     #',
        '#     #                        #     #',
        '#     #                        #     #',
        '#     #                        #     #',
        '#     #                        #     #',
        '#     ##########################     #',
        '#                                    #',
        '#                                    #',
        '#                                    #',
        '#                                    #',
        '#                                    #',
        '######################################',
    ]

def set_maze_env(inner_size: int = 24, world_min: int = 7):
    """
    Create the minimal environment (only the attributes/methods you actually use).
    """
    global _ENV
    inner_size = int(inner_size)
    world_min  = int(world_min)
    world_max  = world_min + inner_size + 1

    env = SimpleNamespace()

    # --- 基本設定 ---
    env.inner_size = inner_size
    env.world_min  = world_min
    env.world_max  = world_max

    # 可動區（排除牆）：8..19 或 8..31
    env.move_min = world_min + 1
    env.move_max = world_max - 1

    # ★ 讓 disk 1..inner_size 對應可動區 move_min..move_max
    env.coord_offset = env.move_min - 1   # e.g. 8 -> 1, 19/31 -> 12/24

    # 幾何中心（世界座標）
    env.center = (world_min + inner_size // 2,
                  world_min + inner_size // 2)

    # 版面（ASCII 內框）
    env.base_layout = _layout_26() if inner_size == 12 else _layout_38()

    # 世界座標夾界（含牆）
    def clamp_world(pos):
        a = np.asarray(pos, dtype=float).copy()
        a[..., 0] = np.clip(a[..., 0], env.world_min, env.world_max)
        a[..., 1] = np.clip(a[..., 1], env.world_min, env.world_max)
        return a
    env.clamp_world = clamp_world

    # 轉換：世界 -> 磁碟 / 磁碟 -> 世界
    def world_to_disk(pos):
        a = np.asarray(pos, dtype=float) - float(env.coord_offset)
        a[..., 0] = np.clip(a[..., 0], 1, env.inner_size)
        a[..., 1] = np.clip(a[..., 1], 1, env.inner_size)
        return a
    env.world_to_disk = world_to_disk

    def disk_to_world(pos):
        return np.asarray(pos, dtype=float) + float(env.coord_offset)
    env.disk_to_world = disk_to_world

    # 範圍檢查（世界座標）
    def assert_world(pos, msg=''):
        a = np.asarray(pos, dtype=float)
        if not ((env.world_min <= a[..., 0]).all() and (a[..., 0] <= env.world_max).all()):
            raise AssertionError(f'X out of range {env.world_min}..{env.world_max} {msg}')
        if not ((env.world_min <= a[..., 1]).all() and (a[..., 1] <= env.world_max).all()):
            raise AssertionError(f'Y out of range {env.world_min}..{env.world_max} {msg}')
    env.assert_world = assert_world

    _ENV = env


def _require_env() -> SimpleNamespace:
    if _ENV is None:
        raise RuntimeError("Maze env has not been initialized, please call set_maze_env(...) first")
    return _ENV

def assert_world(pos, msg=''):  
    env = _require_env()
    return env.assert_world(pos, msg)
def clamp_world(pos):       
    env = _require_env()
    return env.clamp_world(pos)
def world_to_disk(pos):         
    env = _require_env()
    return env.world_to_disk(pos)
def disk_to_world(pos):         
    env = _require_env()
    return env.disk_to_world(pos)

def _get_env_bounds():
    # (x_min, x_max, y_min, y_max, coord_offset, inner_size)
    return 8, 31, 8, 31, 7, 24

def find_inner_bounds(base_maze_layout: List[str], inner_size: int = None) -> Tuple[int,int,int,int,int]:
    """
    From ASCII maze, infer inner bounds.
    Returns (x_left, x_right, y_top, y_bottom, inner_size)
        - x_left/x_right/y_top/y_bottom are the indices of wall '#' (inclusive)
        - The first walkable cell is at (x_left+1, y_top+1)
        - If inner_size is not given, it will be detected from the pattern
    """
    sizes = [inner_size] if inner_size in (12, 24) else [24, 12]
    for inn in sizes:
        pat = re.compile(r"#" + r"\s" * inn + r"#")
        y_hits = []
        x_left = x_right = None
        for y, row in enumerate(base_maze_layout):
            m = pat.search(row)
            if m:
                y_hits.append(y)
                if x_left is None:
                    x_left = m.start()    
                    x_right = m.end() - 1  
        if y_hits:
            y_top, y_bottom = min(y_hits), max(y_hits)
            return x_left, x_right, y_top, y_bottom, inn
    raise ValueError("Cannot find inner bounds (# + space*{12/24} + #), please check the ASCII.")


def detect_coord_system_and_inner(coords_np: np.ndarray) -> Tuple[str, int]:
    """
    Roughly determine if TXT coordinates are 'disk'(1..12/24) or 'world'(8..19 / 8..31).
    Returns (coord_system, inner_size), where coord_system ∈ {'disk','world'}.
    """
    a = np.asarray(coords_np, dtype=float)
    mn = float(np.nanmin(a)); mx = float(np.nanmax(a))
    # Like disk: completely within 1..24
    if 1.0 <= mn and mx <= 24.0:
        return ('disk', 12 if mx <= 12.0 else 24)
    # Like world: movable area 8..19 or 8..31
    if 8.0 <= mn and mx <= 19.0:
        return ('world', 12)
    if 8.0 <= mn and mx <= 31.0:
        return ('world', 24)
    return ('disk', 24)


def world_to_disk_if_needed(arr: np.ndarray) -> np.ndarray:
    """If coordinates are in world (7..30), convert to disk (1..24); if already 1..24, return as is."""
    mn, mx = float(arr.min()), float(arr.max())
    if mn >= 7 - 1e-6 and mx <= 30 + 1e-6:
        return arr - 6.0
    return arr

def read_floats_after_maze(txt_path: Path) -> np.ndarray:
    """Read past 'Maze:' and ASCII maze, return a (K,2) array of coordinates (disk or world coordinates)."""
    with open(txt_path, encoding='utf-8') as f:
        lines = f.read().splitlines()

    def _find_data_start(lines_after_header: List[str]) -> int:
        for i, ln in enumerate(lines_after_header):
            if '[' in ln:
                return i
        return len(lines_after_header)

    if lines and lines[0].strip() == 'Maze:':
        maze_end   = 1 + _find_data_start(lines[1:])
        data_lines = lines[maze_end:]
    else:
        data_lines = lines

    nums = re.findall(r'-?\d+(?:\.\d+)?', ''.join(data_lines))
    if len(nums) % 2 != 0 or len(nums) == 0:
        raise ValueError(f"{txt_path}: cannot parse an even number of floats (length={len(nums)})")
    return np.asarray(list(map(float, nums)), dtype=float).reshape(-1, 2)

def to_world_from_txt(arr: np.ndarray,
                      coord_system: str,
                      x_left_wall: int,
                      y_top_wall: int) -> np.ndarray:
    """
    TXT -> world coordinates:
        - disk -> world: world_x = disk_x + (x_left_wall + 1)
                        world_y = disk_y + (y_top_wall  + 1)
        - world -> world: unchanged
    """
    a = np.asarray(arr, dtype=float)
    if coord_system == 'disk':
        out = a.copy()
        out[..., 0] += float(x_left_wall + 1)
        out[..., 1] += float(y_top_wall  + 1)
        return out
    return a
