# scripts/plot_trails.py
import re
import sys
import json
import hashlib
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
"""
Pipeline 3
Plot trails and options from data/{sim_name}/{move_type}/*.txt
Make sure the hyperparameters in SimConfig are consistent with those in gen_trails.py
You can set plot_counts to limit number of plots per move_type( Not needed for all)
"""
sys.path.append(str(Path(__file__).parent.parent))
from mazesim.config import SimConfig

ODD_LABELS  = ['1', '3', '5', '7', '9']
EVEN_LABELS = ['0', '2', '4', '6', '8']

def _find_data_start(lines_after_header):
    for i, ln in enumerate(lines_after_header):
        if "[" in ln:
            return i
    return len(lines_after_header)

def _parse_all_floats(lines_after_maze):
    return list(map(float, re.findall(r"-?\d+(?:\.\d+)?", "".join(lines_after_maze))))

def _find_inner_bounds(base_maze_layout, inner_size=None):
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
    return 0, 0, 0, 0, inner_size or 24

def _detect_coord_system_and_inner(coords_np):
    a = np.asarray(coords_np, dtype=float)
    mn = float(np.nanmin(a))
    mx = float(np.nanmax(a))
    if 1.0 <= mn and mx <= 24.0:
        return ("disk", 12 if mx <= 12.0 else 24)
    if 8.0 <= mn and mx <= 31.0:
        return ("world", 24 if mx > 19 else 12)
    return ("disk", 24)

def _to_world_from_txt(arr, coord_system, x_left_wall, y_top_wall):
    a = np.asarray(arr, dtype=float)
    if coord_system == "disk":
        out = a.copy()
        out[..., 0] += float(x_left_wall + 1)
        out[..., 1] += float(y_top_wall + 1)
        return out
    return a

def _read_xy_block(text: str) -> np.ndarray:
    nums = list(map(float, re.findall(r"-?\d+(?:\.\d+)?", text)))
    if len(nums) == 0 or len(nums) % 2 != 0:
        raise ValueError("block cannot be parsed as (x,y) pairs")
    return np.asarray(nums, dtype=float).reshape(-1, 2)

def parse_choices_file(choice_txt: Path) -> dict[str, np.ndarray]:
    txt = choice_txt.read_text(encoding="utf-8")
    pieces = re.split(r"Choice\s+([ABCD])\s*:\s*", txt)
    out: dict[str, np.ndarray] = {}
    for i in range(1, len(pieces), 2):
        letter = pieces[i]
        block = pieces[i + 1]
        out[letter] = _read_xy_block(block)
    return out

# ================== Logic / Parity Functions ==================

def parse_case_type_from_name(path: Path) -> Optional[int]:
    m = re.search(r'case(\d+)', path.stem, re.I)
    return int(m.group(1)) if m else None

def parity_by_role(case_type: int, N: int, role_letters: list[str]) -> dict:
    if N == 3:
        norm_case = ((case_type - 1) % 36) + 1
        grp = (norm_case - 1) // 12 + 1
        m = {}
        if grp == 1:   m = {'S':'even','A':'odd','B':'odd'}
        elif grp == 2: m = {'S':'odd','A':'even','B':'odd'}
        elif grp == 3: m = {'S':'odd','A':'odd','B':'even'}
        return {r: m.get(r, 'odd') for r in role_letters[:N]}
    elif N == 5:
        grp = ((case_type - 1) // 80) + 1
        idxs = list(range(N))
        parity_vec = ['even'] * N
        if 1 <= grp <= 10:
            pair = list(itertools.combinations(idxs, 2))[grp - 1]
            for i in pair: parity_vec[i] = 'odd'
        elif 11 <= grp <= 15:
            ev_i = list(itertools.combinations(idxs, 1))[grp - 11][0]
            parity_vec = ['odd'] * N
            parity_vec[ev_i] = 'even'
        elif grp == 16:
            parity_vec = ['even'] * N
        else:
            parity_vec = ['odd'] * N
        return {role_letters[i]: parity_vec[i] for i in range(N)}
    else:
        return {r: 'odd' for r in role_letters}

def find_role_positions(ascii_lines, role_letters):
    roles = {}
    valid = set(role_letters)
    for y, row in enumerate(ascii_lines):
        for x, ch in enumerate(row):
            if ch in valid:
                roles[ch] = (x, y)
    return roles

def map_roles_to_indices(pos0_world: np.ndarray, role_pos: dict):
    idx_to_role = {}
    used_roles  = set()
    p0 = np.rint(pos0_world).astype(int)
    for i, (x, y) in enumerate(p0):
        if i in idx_to_role: continue
        candidates = []
        for role, (rx, ry) in role_pos.items():
            if role in used_roles: continue
            ex_wx, ex_wy = rx + 1, ry + 1
            dist = abs(ex_wx - x) + abs(ex_wy - y)
            candidates.append((dist, role))
        if candidates:
            _, best_role = min(candidates, key=lambda t: t[0])
            idx_to_role[i] = best_role
            used_roles.add(best_role)
    return idx_to_role

def stable_labels_for_step(stem: str, step_identifier, idx_to_role, parity_role, N: int):
    seed_str = f"{stem}:{step_identifier}"
    seed = int(hashlib.sha1(seed_str.encode()).hexdigest()[:8], 16)
    rng  = random.Random(seed)
    labels = []
    for i in range(N):
        role = idx_to_role.get(i, None)
        parity = parity_role.get(role, 'odd') if role else 'odd'
        pool = ODD_LABELS if parity == 'odd' else EVEN_LABELS
        labels.append(rng.choice(pool))
    return labels

# ================== Plotting ==================

COLOR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
LABEL_FONTSIZE = 12
LABEL_COLOR = "white"
LABEL_WEIGHT = "bold"
CELL_EDGE_LW = 1.2
ALPHA_FILL = 0.9

def _draw_one_step(ax, pos_world, labels, colors, bounds):
    x_min, x_max, y_min, y_max = bounds
    for (x, y), c, lbl in zip(pos_world, colors, labels):
        if x_min <= x <= x_max and y_min <= y <= y_max:
            ax.add_patch(
                plt.Rectangle(
                    (x - 0.5, y - 0.5), 1, 1,
                    facecolor=c, edgecolor="black", alpha=ALPHA_FILL, lw=CELL_EDGE_LW,
                )
            )
            ax.text(
                x, y, lbl, ha="center", va="center",
                color=LABEL_COLOR, weight=LABEL_WEIGHT, fontsize=LABEL_FONTSIZE,
            )

def _export_png(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def _new_axes(bounds):
    x_min, x_max, y_min, y_max = bounds
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor("white")
    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_ylim(y_max + 0.5, y_min - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(x_min - 0.5, x_max + 0.5, 1))
    ax.set_yticks(np.arange(y_min - 0.5, y_max + 0.5, 1))
    ax.grid(True, lw=0.8, color="lightgrey")
    ax.tick_params(length=0, labelleft=False, labelbottom=False)
    return fig, ax

def _plot_single_file(txt_path: Path, move_type: str, config: SimConfig):
    lines = txt_path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "Maze:":
        print(f"[{move_type}] Skip (No Maze header) -> {txt_path.name}")
        return

    maze_end = 1 + _find_data_start(lines[1:])
    ascii_lines = lines[1:maze_end]
    data_lines = lines[maze_end:]

    floats = _parse_all_floats(data_lines)
    N = config.n_agents
    if not floats or len(floats) % (2 * N) != 0:
        print(f"[{move_type}] Skip (Data mismatch) -> {txt_path.name}")
        return

    traj = np.asarray(floats, dtype=float).reshape(-1, N, 2)
    T = traj.shape[0]

    # Bounds
    try:
        xL, xR, yT, yB, inner = _find_inner_bounds(ascii_lines, inner_size=config.maze_size)
    except ValueError:
        xL, xR, yT, yB, inner = 0, 0, 0, 0, config.maze_size

    x0 = (xL + 1) + 1
    y0 = (yT + 1) + 1
    bounds = (x0, x0 + inner - 1, y0, y0 + inner - 1)
    
    coord_system, _ = _detect_coord_system_and_inner(traj)
    colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(N)]
    
    # Logic Setup: Roles & Parity
    is_logic = (move_type.lower() == 'logic')
    idx_to_role = {}
    parity_role = {}
    
    if is_logic:
        role_letters = ['S'] + [chr(ord('A') + i) for i in range(N-1)]
        case_type = parse_case_type_from_name(txt_path) or 1
        
        # 1. Find roles in ASCII
        role_pos = find_role_positions(ascii_lines, role_letters)
        # 2. Convert Step 0 to World to align
        pos0_world = _to_world_from_txt(traj[0], coord_system, xL, yT)
        # 3. Map
        idx_to_role = map_roles_to_indices(pos0_world, role_pos)
        # 4. Parity
        parity_role = parity_by_role(case_type, N, role_letters)

    # Output Dir
    plot_dir = txt_path.parent / "plot" / f"trail_{txt_path.stem}"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 1. Plot Trajectory (Modified: Plot T-1 steps to hide the final answer step)
    #    T-1 because the last step is often the goal position which may reveal the answer
    plot_T = max(1, T - 1) if T > 1 else T
    for t in range(plot_T):
        pos_world = _to_world_from_txt(traj[t], coord_system, xL, yT)
        
        # Determine labels
        if is_logic:
            labels = stable_labels_for_step(txt_path.stem, t, idx_to_role, parity_role, N)
        else:
            labels = [str(i + 1) for i in range(N)]
            
        fig, ax = _new_axes(bounds)
        _draw_one_step(ax, pos_world, labels, colors, bounds)
        _export_png(fig, plot_dir / f"{t+1}.png")

    # 2. Plot Options (Modified: Plot for ALL types if choices file exists)
    answers_dir = txt_path.parent / "answers"
    choice_txt = answers_dir / f"{txt_path.stem}_choices.txt"
    
    if choice_txt.exists():
        try:
            choices_xy = parse_choices_file(choice_txt)
            for ch in ["A", "B", "C", "D"]:
                if ch not in choices_xy: continue
                
                arr = choices_xy[ch]
                # Assume options use same coord system as trail
                pos_world = _to_world_from_txt(arr, coord_system, xL, yT)
                
                # Determine labels for option
                if is_logic:
                    # Logic: use randomized stable labels
                    labels = stable_labels_for_step(txt_path.stem, f"choice_{ch}", idx_to_role, parity_role, N)
                else:
                    # Others: use standard 1..N labels (consistent with trajectory)
                    labels = [str(i + 1) for i in range(N)]
                
                fig, ax = _new_axes(bounds)
                _draw_one_step(ax, pos_world, labels, colors, bounds)
                _export_png(fig, plot_dir / f"{ch}.png")
        except Exception as e:
            print(f"[{move_type}] Choice parse error {choice_txt.name}: {e}")

    print(f"✓ [{move_type}] {txt_path.name} -> {plot_dir} (T={plot_T} frames + Options)")

def plot_all_trails(config: SimConfig, plot_counts: dict):
    data_root = config.data_root

    for move_type in config.move_types:
        count = plot_counts.get(move_type, 0)
        if count == 0: continue
        
        type_dir = data_root / move_type
        if not type_dir.exists(): continue

        all_txts = sorted([p for p in type_dir.glob("*.txt") if "choices" not in p.name and "FINAL" not in p.name])
        
        if not all_txts:
            print(f"[{move_type}] No txt files found.")
            continue

        target_txts = all_txts[:count] if count > 0 else all_txts
        print(f"--- Plotting {move_type} ({len(target_txts)} files) ---")
        
        for txt_path in target_txts:
            _plot_single_file(txt_path, move_type, config)

def main():
    config = SimConfig(
        sim_name="5_24",
        n_agents=5,
        maze_size=24,
        n_steps=10,
        move_types=["logic", "rulemap", "random", "intermediate_case1", "intermediate_case2"],
        n_trials_per_type={"logic": 10, "rulemap": 10, "random":10, "intermediate_case1":10, "intermediate_case2":10},
        project_root=Path("/Users/Jer_ry/Desktop/script_tom"),
    )

    plot_counts = {
        "logic": 5,
        "rulemap": 5,
        "random": 5,
        "intermediate_case1": 5,
        "intermediate_case2": 5
    }

    plot_all_trails(config, plot_counts)

if __name__ == "__main__":
    main()