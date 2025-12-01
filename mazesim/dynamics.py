# dynamics.py
import numpy as np
import random
import pandas as pd
from pathlib import Path
from typing import Tuple

from .env import clamp_world, world_to_disk, disk_to_world, set_maze_env  # 視實際需要

def generate_social_score(out_dir: Path, fname='sim1', n_agents=3,
                          MIN=-30, MAX=0, MEAN=-10, SD=10):
    csv_dir  = out_dir
    csv_dir.mkdir(parents=True, exist_ok=True)
    out_path = csv_dir / f'{fname}.csv'

    # make the matrix
    raw   = np.random.randint(MIN, MAX + 1, (n_agents, n_agents))
    np.fill_diagonal(raw, 0)

    z     = (raw - raw.mean()) / raw.std()
    score = z * SD + MEAN

    # header row
    labels = [str(i+1) for i in range(n_agents)]
    df = pd.DataFrame(score, columns=labels, index=labels)

    df.to_csv(out_path) 
    print(f'✓ social score saved -> {out_path}')
    return df

def make_role_labels(move_type: str, n: int) -> list[str]:
    move_type = move_type.lower()
    if move_type.startswith('intermediate'):
        return [chr(65 + i) for i in range(n)]          # A,B,C,...
    else:
        if n <= 0: return []
        return ['S'] + [chr(65 + i) for i in range(n-1)]  # S,A,B,...

def Center_Agent(fix_central_index, maze_total, n_agents):
    return (fix_central_index[maze_total % len(fix_central_index)]
            if fix_central_index else (maze_total % n_agents))

def inputs(agent_place, x_random, y_random, social_reward, n_chosen_agents):
    """Initialize positions + calculate pairwise distances/forces (world coordinates)."""
    agent_place = np.asarray(agent_place, dtype=float)
    if np.all(agent_place == 0):
        for i in range(n_chosen_agents):
            agent_place[i] = np.array([x_random[i], y_random[i]], dtype=float)

    distance = np.zeros((n_chosen_agents, n_chosen_agents), dtype=float)
    for i in range(n_chosen_agents):
        for j in range(n_chosen_agents):
            distance[i, j] = np.hypot(agent_place[i,0]-agent_place[j,0],
                                      agent_place[i,1]-agent_place[j,1])

    dist_sq = np.square(distance) + 1e-6
    force   = social_reward / dist_sq
    return agent_place, distance, force

def _nth_perm(multiset, k):
    from itertools import islice
    from more_itertools import distinct_permutations

    it = islice(distinct_permutations(multiset), k, k+1)
    try: return list(next(it))
    except StopIteration: raise ValueError("direc_case 超出可用排列索引範圍")
def _round_step_from_avg(avg_vec: np.ndarray) -> np.ndarray:
    avg = np.asarray(avg_vec, dtype=float)
    return (np.sign(avg) * np.floor(np.abs(avg) + 0.5)).astype(int)
def _safe_vec(v) -> np.ndarray:
    if v is None: return np.array([0,0], dtype=int)
    a = np.asarray(v)
    return a.astype(int) if a.size==2 else np.array([0,0], dtype=int)
def _round_half_up_vec(v: np.ndarray) -> np.ndarray:
    """
    Element-wise rounding half up, where 0.5 always rounds up towards 1
    """
    v = np.asarray(v, dtype=float)
    return np.floor(v + 0.5)

_DIR_DXY = {
    'L': (-1, 0),
    'R': ( 1, 0),
    'U': ( 0,-1),
    'D': ( 0, 1),
}

DIR8 = [
    (0,-1), (1,0), (0,1), (-1,0),
    (1,-1), (1,1), (-1,1), (-1,-1),
]

def move_rulemap(env, agent_place, force, n_chosen_agents):
    agent_place  = np.asarray(agent_place, dtype=float)
    force_vector = np.zeros((n_chosen_agents, 2 * n_chosen_agents), dtype=float)

    for i in range(n_chosen_agents):
        for j in range(n_chosen_agents):
            if i == j: 
                continue
            dx = agent_place[j,0] - agent_place[i,0]
            dy = agent_place[j,1] - agent_place[i,1]
            dist = np.hypot(dx, dy)
            if dist > 0:
                f = force[i,j] / (dist**2 + 1e-6)
                force_vector[i, 2*j  ] = (dx/dist) * f
                force_vector[i, 2*j+1] = (dy/dist) * f

    force_sum   = force_vector.reshape(n_chosen_agents, n_chosen_agents, 2).sum(axis=1)
    move_vector = np.zeros_like(force_sum, dtype=int)
    norms = np.linalg.norm(force_sum, axis=1)

    for i in range(n_chosen_agents):
        ux, uy = force_sum[i] / (norms[i] if norms[i] > 0 else 1)
        move_vector[i,0] =  1 if ux >=  0.5 else (-1 if ux <= -0.5 else 0)
        move_vector[i,1] =  1 if uy >=  0.5 else (-1 if uy <= -0.5 else 0)

    proposed = []
    for i in range(n_chosen_agents):
        x0, y0 = agent_place[i]
        x1 = min(env.move_max, max(env.move_min, x0 + move_vector[i, 0]))
        y1 = min(env.move_max, max(env.move_min, y0 + move_vector[i, 1]))
        proposed.append((x1, y1))

    from collections import defaultdict
    buckets = defaultdict(list)
    for i, pos in enumerate(proposed):
        buckets[pos].append(i)

    new_place = agent_place.copy()
    for pos, idxs in buckets.items():
        if len(idxs) == 1:
            new_place[idxs[0]] = pos
        else:
            strengths = [norms[i] for i in idxs]
            winner = idxs[int(np.argmax(strengths))]
            new_place[winner] = pos

    return env.clamp_world(new_place)


def move_random(agent_place, n_chosen_agents: int, mu: int, *, env=None):
    if env is None:
        raise ValueError("move_random requires env to be passed")

    cur = np.array(agent_place, dtype=int)
    N   = int(n_chosen_agents)
    if N <= 0:
        return cur

    K = max(1, min(N, int(mu)))
    moving_idx = np.random.choice(N, K, replace=False)


    xmin, xmax = int(env.move_min), int(env.move_max)

    occupied = { (int(x), int(y)) for x,y in cur.tolist() }
    prop = cur.copy()

    for i in moving_idx:
        order = np.random.permutation(8)
        moved = False
        for k in order:
            dx, dy = DIR8[k]
            x = int(np.clip(cur[i,0] + dx, xmin, xmax))
            y = int(np.clip(cur[i,1] + dy, xmin, xmax))
            if (x, y) != (cur[i,0], cur[i,1]) and (x, y) not in occupied:
                occupied.discard((int(cur[i,0]), int(cur[i,1])))
                occupied.add((x, y))
                prop[i] = [x, y]
                moved = True
                break

    return env.clamp_world(prop)


def move_logic(env, positions: np.ndarray, direc_case: int) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must be (n_agents, 2)")

    n_agents = positions.shape[0]
    if n_agents not in (3, 5):
        raise ValueError("move_logic only supports 3 or 5")
    LRU_SPECIAL = {  # for n=3
        7:['L','R','U'], 8:['U','L','R'], 9:['U','R','L'],
        10:['L','U','R'], 11:['R','U','L'], 0:['R','L','U'],
    }

    if n_agents == 3:
        if 1 <= direc_case <= 3:
            rule = _nth_perm(['L','L','D'], direc_case - 1)
        elif 4 <= direc_case <= 6:
            rule = _nth_perm(['R','R','D'], direc_case - 4)
        elif direc_case in LRU_SPECIAL:
            rule = LRU_SPECIAL[direc_case]
        else:
            raise ValueError("3 agents' direc_case only supports 1..11 and 0")
    else:
        if   1 <= direc_case <= 10:
            rule = _nth_perm(['L','L','L','D','D'], direc_case - 1)
        elif 11 <= direc_case <= 20:
            rule = _nth_perm(['R','R','R','D','D'], direc_case - 11)
        elif 21 <= direc_case <= 50:
            rule = _nth_perm(['L','L','R','U','U'], direc_case - 21)
        elif 51 <= direc_case <= 80:
            rule = _nth_perm(['R','R','L','U','U'], direc_case - 51)
        else:
            raise ValueError("5 agents' direc_case only supports 1..80")

    deltas = np.array([_DIR_DXY[d] for d in rule], dtype=float)
    return env.clamp_world(positions + deltas)

def move_intermediate(env, pos: np.ndarray, case_type: int, n_agents: int):
    """
    Intermediate move with fixed role assignment
    """
    pos = np.asarray(pos, dtype=float)
    if pos.shape != (n_agents, 2):
        raise ValueError(f'Pos shape mismatch: {pos.shape} != ({n_agents}, 2)')

    new_pos = pos.copy()

    if n_agents == 3:
        A, B, C = 0, 1, 2
    elif n_agents == 5:
        A, B, C, D, E = 0, 1, 2, 3, 4
    else:
        raise ValueError("move_intermediate only supports N=3 or N=5")

    # A, B random 8-direction move
    dA = np.array(random.choice(DIR8), dtype=float)
    dB = np.array(random.choice(DIR8), dtype=float)
    new_pos[A] += dA
    new_pos[B] += dB

    dC = np.zeros(2, dtype=float)
    
    if case_type == 1:
        dC = _round_step_from_avg((dA + dB) / 2.0)
        new_pos[C] += dC
    elif case_type == 2:
        mid_ab = (new_pos[A] + new_pos[B]) / 2.0
        target_c = _round_half_up_vec(mid_ab)
        dC_full = target_c - pos[C]
        dC_step = np.clip(dC_full, -1, 1)
        new_pos[C] = pos[C] + dC_step
        dC = dC_step

    # 5-agent logic
    if n_agents == 5:
        dD = np.array(random.choice(DIR8), dtype=float)
        new_pos[D] += dD
        if case_type == 1:
            dE = _round_step_from_avg((dC + dD) / 2.0)
            new_pos[E] += dE
        elif case_type == 2:
            mid_cd = (new_pos[C] + new_pos[D]) / 2.0
            target_e = _round_half_up_vec(mid_cd)
            dE_full = target_e - pos[E]
            dE_step = np.clip(dE_full, -1, 1)
            new_pos[E] = pos[E] + dE_step

    # Clamp to movable area
    new_pos[..., 0] = np.clip(new_pos[..., 0], env.move_min, env.move_max)
    new_pos[..., 1] = np.clip(new_pos[..., 1], env.move_min, env.move_max)

    return new_pos

def overlay_agents_on_maze(env, init_positions_world, labels):
    """
    env: maze environment
    labels: list of str, length N
    init_positions_world: (N,2) world coordinates
    """
    maze = [list(r) for r in env.base_layout]
    H = len(maze); W = len(maze[0]) if H else 0
    pts = np.asarray(init_positions_world, dtype=float)

    for lab, (xw, yw) in zip(labels, pts):
        # world (8..19/31) to ASCII index (7..18/30): -1
        xi = int(round(xw)) - 1
        yi = int(round(yw)) - 1
        if not (0 <= yi < H and 0 <= xi < W):
            raise ValueError(f"overlay OOB for {lab}: world=({xw},{yw}) -> idx=({xi},{yi}), grid={W}x{H}")
        maze[yi][xi] = lab

    return [''.join(r) for r in maze]

