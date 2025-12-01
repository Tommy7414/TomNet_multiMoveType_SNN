# gridizer_multitrack.py
import re
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List

def _parse_xy_list(txt_path: Path) -> np.ndarray:
    s = Path(txt_path).read_text(encoding="utf-8")
    nums = re.findall(r"[-+]?\d*\.?\d+", s)
    arr = np.array([float(x) for x in nums], dtype=float).reshape(-1, 2)
    if arr.size == 0:
        raise ValueError(f"Cannot parse coordinates from {txt_path}")
    return arr  # (N,2) 1-based [x,y]

def _xy1_to_rc0(xy_1based: Tuple[float,float], H: int, W: int) -> Tuple[int,int]:
    x1, y1 = xy_1based
    r = int(round(y1)) - 1
    c = int(round(x1)) - 1
    r = max(0, min(H-1, r))
    c = max(0, min(W-1, c))
    return r, c

def build_channel_map(n_agents: int) -> Dict[str, Dict]:
    assert n_agents in (3,5), f"n_agents should be 3 or 5, got {n_agents}"
    init_idx = {u: u for u in range(n_agents)}
    act_base = {u: n_agents + u*5 for u in range(n_agents)}
    action_offset = {"right":0, "left":1, "up":2, "down":3, "goal":4}
    C = 6 * n_agents
    agent_names = {0:"S", 1:"A", 2:"B", 3:"C", 4:"D"}
    return {"C": C, "init_idx": init_idx, "act_base": act_base,
            "action_offset": action_offset, "agent_names": agent_names}

def coords_to_grid_seq_multitrack(
    trail_txt_path: Path,
    n_agents: int,
    H: int = 24, W: int = 24,
    init_values: List[float] = None,      # 初始平面用的數值，預設 [1,2,3,4,5][:n_agents]
    keep_init_every_frame: bool = True    # True=每幀都有；False=只在 t=0
) -> np.ndarray:
    """
    多軌座標（每幀 n_agents 個點）-> (T,H,W,C)，C=6*n_agents，無障礙通道。
    - 初始平面在 (r0,c0) 放 'init_values[u]'（預設 u+1）
    - 動作通道 0/1（right/left/up/down），最後一幀只點各自的 goal
    """
    coords = _parse_xy_list(Path(trail_txt_path))  # (N_all, 2), 1-based
    N = coords.shape[0]
    assert N % n_agents == 0, f"rows ({N}) must be a multiple of n_agents ({n_agents})"
    T = N // n_agents

    chm = build_channel_map(n_agents)
    C = chm["C"]; init_idx = chm["init_idx"]; act_base = chm["act_base"]; off = chm["action_offset"]

    # 預設初始值為 [1,2,3,4,5][:n_agents]
    if init_values is None:
        init_values = [float(u+1) for u in range(n_agents)]
    else:
        assert len(init_values) >= n_agents, "init_values length must cover all agents"

    # 拆軌：track[u] 形狀 (T,2)
    tracks = [coords[u::n_agents] for u in range(n_agents)]

    seq = np.zeros((T, H, W, C), dtype=np.float32)

    # 初始位置平面
    for u in range(n_agents):
        r0, c0 = _xy1_to_rc0(tracks[u][0], H, W)
        val = float(init_values[u])
        if keep_init_every_frame:
            seq[:, r0, c0, init_idx[u]] = val
        else:
            seq[0, r0, c0, init_idx[u]] = val

    # 動作（各軌 t→t+1）
    for t in range(T-1):
        for u in range(n_agents):
            r, c   = _xy1_to_rc0(tracks[u][t],   H, W)
            r2, c2 = _xy1_to_rc0(tracks[u][t+1], H, W)
            dr, dc = r2 - r, c2 - c
            base = act_base[u]
            # 水平位移
            if dc > 0: seq[t, r, c, base + off["right"]] = 1.0
            if dc < 0: seq[t, r, c, base + off["left"]]  = 1.0
            # 垂直位移（row 增加視為 down）
            if dr < 0: seq[t, r, c, base + off["up"]]    = 1.0
            if dr > 0: seq[t, r, c, base + off["down"]]  = 1.0

    # 最後一幀：各自 goal
    t_last = T - 1
    for u in range(n_agents):
        rf, cf = _xy1_to_rc0(tracks[u][t_last], H, W)
        seq[t_last, rf, cf, act_base[u] + off["goal"]] = 1.0

    return seq  # (T,H,W,C)
