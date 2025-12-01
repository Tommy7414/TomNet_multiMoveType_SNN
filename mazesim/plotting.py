# plotting.py
import re, glob, itertools, hashlib, random, os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

from .env import find_inner_bounds, detect_coord_system_and_inner, _get_env_bounds

from .config import SimConfig

COLOR_PALETTE = [
    '#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
    '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'
]
LABEL_FONTSIZE = 12
LABEL_COLOR    = 'white'
LABEL_WEIGHT   = 'bold'
CELL_EDGE_LW   = 1.2
ALPHA_FILL     = 0.9

ODD  = ['1','3','5','7','9']
EVEN = ['0','2','4','6','8']

# ---------- A. Inner bounds detection & data reading ----------
def _find_data_start(lines_after_header: List[str]) -> int:
    """After 'Maze:', find the start of the numeric data block (the first line containing '[')."""
    for i, ln in enumerate(lines_after_header):
        if '[' in ln:
            return i
    return len(lines_after_header)

def parse_all_floats(lines_after_maze: List[str]) -> List[float]:
    return list(map(float, re.findall(r'-?\d+(?:\.\d+)?',
                                      ''.join(lines_after_maze))))

def find_inner_bounds(base_maze_layout: List[str], inner_size: int = None) -> Tuple[int,int,int,int,int]:
    """
    From ASCII maze, infer inner bounds.
    Returns (x_left, x_right, y_top, y_bottom, inner_size)
    - x_left/x_right/y_top/y_bottom are the indices of wall '#' (inclusive)
    - The first walkable cell is at (x_left+1, y_top+1)
    - If inner_size is not given, it will be detected from the pattern
    """
    # Try matching 12 / 24 automatically
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
                    x_left = m.start()       # 左牆 '#'
                    x_right = m.end() - 1    # 右牆 '#'
        if y_hits:
            y_top, y_bottom = min(y_hits), max(y_hits)
            return x_left, x_right, y_top, y_bottom, inn
    raise ValueError("找不到內圈（# + 空白*{12/24} + #），請檢查 ASCII。")

def detect_coord_system_and_inner(coords_np: np.ndarray) -> Tuple[str, int]:
    """
    Roughly determine if TXT coordinates are 'disk' (1..12/24) or 'world' (8..19 / 8..31).
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
# ---------- B. Plotting ----------
def draw_one_step(ax, pos_world, labels, colors, bounds):
    x_min, x_max, y_min, y_max = bounds
    for (x, y), c, lbl in zip(pos_world, colors, labels):
        if x_min <= x <= x_max and y_min <= y <= y_max:
            ax.add_patch(plt.Rectangle((x-.5, y-.5), 1, 1,
                                       facecolor=c,
                                       edgecolor='black',
                                       alpha=ALPHA_FILL,
                                       lw=CELL_EDGE_LW))
            ax.text(x, y, lbl, ha='center', va='center',
                    color=LABEL_COLOR, weight=LABEL_WEIGHT, fontsize=LABEL_FONTSIZE)

def export_png(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# ---------- C. Logic labels ----------
def parse_case_type_from_name(path: Path) -> Optional[int]:
    m = re.search(r'case(\d+)', path.stem, re.I)
    return int(m.group(1)) if m else None

def parity_by_role(case_type: int, N: int, role_letters: List[str]) -> Dict[str, str]:
    """
    Returns {role -> 'odd'/'even'}
    N=3：36 種 case，每 12 個一組：
      grp1: S=even, A=odd,  B=odd
      grp2: S=odd,  A=even, B=odd
      grp3: S=odd,  A=odd,  B=even
    """
    norm = ((case_type - 1) % 36) + 1
    grp  = (norm - 1) // 12 + 1
    if grp == 1: base = {'S':'even','A':'odd','B':'odd'}
    elif grp == 2: base = {'S':'odd','A':'even','B':'odd'}
    else:          base = {'S':'odd','A':'odd','B':'even'}
    return {r: base.get(r, 'odd') for r in role_letters[:N]}

def find_role_positions_any(ascii_lines: List[str], role_letters: List[str]) -> Dict[str, Tuple[int,int]]:
    roles, valid = {}, set(role_letters)
    for y, row in enumerate(ascii_lines):
        for x, ch in enumerate(row):
            if ch in valid:
                roles[ch] = (x, y) 
    return roles

def map_roles_to_indices(pos0_world: np.ndarray,
                         role_pos: Dict[str,Tuple[int,int]]) -> Dict[int,str]:
    """
    Convert first step in "world coordinates" -> first convert back to ASCII index (world-1), then match with role characters.
    Fix: Previously directly using world coordinates to match ASCII was off by 1.
    """
    idx_to_role: Dict[int,str] = {}
    used: set[str] = set()
    p0 = (np.rint(pos0_world).astype(int) - 1)  # world → ASCII index

    for role, (rx, ry) in role_pos.items():
        for i, (x, y) in enumerate(p0):
            if i in idx_to_role:
                continue
            if x == rx and y == ry:
                idx_to_role[i] = role
                used.add(role)
                break

    for i, (x, y) in enumerate(p0):
        if i in idx_to_role:
            continue
        rest = [(abs(rx-x)+abs(ry-y), r) for r,(rx,ry) in role_pos.items() if r not in used]
        if rest:
            _, r = min(rest, key=lambda t: t[0])
            idx_to_role[i] = r
            used.add(r)
    return idx_to_role

def stable_labels_for_step(stem: str, step_idx: int,
                           idx_to_role: Dict[int,str],
                           parity_role: Dict[str,str],
                           N: int) -> List[str]:
    seed = int(hashlib.sha1(f"{stem}:{step_idx}".encode()).hexdigest()[:8], 16)
    rng  = random.Random(seed)
    out  = []
    for i in range(N):
        role = idx_to_role.get(i, None)
        parity = parity_role.get(role, 'odd') if role else 'odd'
        out.append(rng.choice(ODD if parity=='odd' else EVEN))
    return out

# ---------- D. Infer N ----------
def _infer_N(ascii_lines: List[str], floats: List[float], guess_max: int = 16) -> int:
    letters = {ch for row in ascii_lines for ch in row if ch == 'S' or 'A' <= ch <= 'Z'}
    letter_cnt = len(letters)

    L = len(floats)
    candidates = [n for n in range(1, guess_max+1) if (L % (2*n) == 0)]

    if letter_cnt in candidates and letter_cnt != 0:
        return letter_cnt
    if candidates:
        if 3 in candidates: return 3
        if letter_cnt > 0:  return min(candidates, key=lambda n: abs(n - letter_cnt))
        return min(candidates)
    raise ValueError(f"無法推斷 N：len(floats)={L}, letters={letters}")

# ---------- E. Batch plotting main function ----------
def batch_plot_general(
    move_type: str,
    src_dir: Path,
    dst_dir: Path,
    preserve_subdirs: bool = False,
    keep_all_steps: bool = False,
    n_override: Optional[int] = None,
    coord_system_for_txt: str = 'disk',
):
    txt_list = sorted(
        t for t in glob.glob(str(src_dir/'**'/'*.txt'), recursive=True)
        if 'FINAL' not in os.path.basename(t)
    )
    if not txt_list:
        print(f'CANT FIND：{src_dir}')
        return

    for txt in txt_list:
        txt_path = Path(txt)
        with open(txt_path, encoding='utf-8') as f:
            lines = f.read().splitlines()

        if not lines or lines[0].strip() != 'Maze:':
            print(f'WARN: File does not start with "Maze:", skipping → {txt_path}')
            continue

        maze_end    = 1 + _find_data_start(lines[1:])
        ascii_lines = lines[1:maze_end]
        data_lines  = lines[maze_end:]

        floats = parse_all_floats(data_lines)
        if not floats:
            print(f'WARN: No numeric data found, skipping → {txt_path}')
            continue

        # Inner wall positioning (ASCII indices)
        xL, xR, yT, yB, inner = find_inner_bounds(ascii_lines)
        # world boundaries (movable area)
        x0 = (xL + 1) + 1
        y0 = (yT + 1) + 1
        x_min, y_min = x0, y0
        x_max, y_max = x0 + inner - 1, y0 + inner - 1
        bounds = (x_min, x_max, y_min, y_max)

        # N
        if n_override is not None:
            N = int(n_override)
            if len(floats) % (2*N) != 0:
                raise ValueError(
                    f"{txt_path.name}: len(floats)={len(floats)} 不能被 2*N={2*N} 整除，請確認 N 或檔案內容。"
                )
        else:
            N = _infer_N(ascii_lines, floats)

        colors = list(itertools.islice(itertools.cycle(COLOR_PALETTE), N))

        try:
            traj = np.asarray(floats, dtype=float).reshape(-1, N, 2)  # (T, N, 2)
        except Exception as e:
            raise ValueError(
                f"reshape failed: file={txt_path.name}, len(floats)={len(floats)}, N={N}"
            ) from e

        # Decide coordinate system, and convert "step 0" to world coordinates (note y uses yT, not inner)
        if coord_system_for_txt not in ('disk','world','auto'):
            raise ValueError("coord_system_for_txt must be 'disk'/'world'/'auto'")
        if coord_system_for_txt == 'auto':
            cs, _ = detect_coord_system_and_inner(traj)
            coord_system_for_txt = cs
        pos0_world = to_world_from_txt(traj[0], coord_system_for_txt, xL, yT)  # ★ Defined before is_logic

        # logic: do one-time role alignment
        is_logic    = (move_type.lower() == 'logic')
        case_type   = parse_case_type_from_name(txt_path) if is_logic else None
        idx_to_role = {}
        parity_role = {}
        step_labels_map: Dict[str, List[str]] = {}

        if is_logic:
            role_letters = ['S'] + [chr(ord('A') + i) for i in range(max(0, N-1))]
            role_pos     = find_role_positions_any(ascii_lines, role_letters)
            idx_to_role  = map_roles_to_indices(pos0_world, role_pos)  # ★ 用上面已定義的 pos0_world
            parity_role  = parity_by_role(case_type or 1, N, role_letters)

        # Output directory
        out_base  = (dst_dir / txt_path.parent.relative_to(src_dir)) if preserve_subdirs else dst_dir
        out_trial = out_base / f'step_plot_{txt_path.stem}'
        out_trial.mkdir(parents=True, exist_ok=True)

        # Plot each step
        STEP_LIMIT = None
        max_s = len(traj) if STEP_LIMIT is None else min(len(traj), STEP_LIMIT+1)
        for s in range(max_s):
            # ★ Use (xL, yT) here, not inner
            pos_world = to_world_from_txt(traj[s], coord_system_for_txt, xL, yT)

            if (not keep_all_steps) and s > 0:
                prev_world = to_world_from_txt(traj[s-1], coord_system_for_txt, xL, yT)
                if np.array_equal(pos_world, prev_world):
                    continue

            fig, ax = plt.subplots(figsize=(7,7))
            ax.set_facecolor('white')
            ax.set_xlim(x_min-.5, x_max+.5)
            ax.set_ylim(y_max+.5, y_min-.5)
            ax.set_aspect('equal')
            ax.set_xticks(np.arange(x_min-.5, x_max+.5, 1))
            ax.set_yticks(np.arange(y_min-.5, y_max+.5, 1))
            ax.grid(True, lw=.8, color='lightgrey')
            ax.tick_params(length=0, labelleft=False, labelbottom=False)

            if is_logic and idx_to_role and parity_role:
                labels_this_step = stable_labels_for_step(txt_path.stem, s, idx_to_role, parity_role, N)
                step_labels_map[str(s)] = labels_this_step
                draw_one_step(ax, pos_world, labels_this_step, colors, bounds)
            else:
                default_labels = [str(i+1) for i in range(N)]
                draw_one_step(ax, pos_world, default_labels, colors, bounds)

            export_png(fig, out_trial / f'step_{s:03d}.png')

        if is_logic:
            import json
            meta = {
                "case_type": case_type,
                "parity": parity_role,
                "idx_to_role": {str(k): v for k, v in (idx_to_role or {}).items()},
                "step_labels": step_labels_map
            }
            with open(out_trial / "labels.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        prefix = '' if not preserve_subdirs else ('' if str(txt_path.parent.relative_to(src_dir))=='.' else f'{txt_path.parent.relative_to(src_dir)}/')
        print(f'✓ {move_type:<12} {prefix}{txt_path.stem} → {out_trial}')

def gen_plots(config: SimConfig):
    """
    According to config.move_types, generate corresponding step_plot_XXX folders (including png / labels.json) from .txt files in data_root.
    """
    DATA_ROOT = config.data_root   # /.../data/5_24
    PLOT_ROOT = config.plot_root   # /.../training_data/5_24
    N = config.n_agents            # override N（#agents）

    if "rulemap" in config.move_types:
        batch_plot_general(
            move_type="rulemap",
            src_dir=DATA_ROOT / "rulemap",
            dst_dir=PLOT_ROOT / "rulemap_plot",
            preserve_subdirs=False,
            keep_all_steps=True,
            n_override=N, 
            coord_system_for_txt="disk",
        )

    if "random" in config.move_types:
        batch_plot_general(
            move_type="random",
            src_dir=DATA_ROOT / "random",
            dst_dir=PLOT_ROOT / "random_plot",
            preserve_subdirs=False,
            keep_all_steps=True,
            n_override=N,
            coord_system_for_txt="disk",
        )

    if "intermediate_case1" in config.move_types:
        # raw txt path: data_root / intermediate_case1 / case1 / *.txt
        batch_plot_general(
            move_type="intermediate_case1",
            src_dir=DATA_ROOT / "intermediate_case1",
            dst_dir=PLOT_ROOT / "intermediate_case1_plot",
            preserve_subdirs=False,  
            keep_all_steps=True,
            n_override=N,
            coord_system_for_txt="disk",
        )

    if "intermediate_case2" in config.move_types:
        batch_plot_general(
            move_type="intermediate_case2",
            src_dir=DATA_ROOT / "intermediate_case2",
            dst_dir=PLOT_ROOT / "intermediate_case2_plot",
            preserve_subdirs=False,  
            keep_all_steps=True,
            n_override=N,
            coord_system_for_txt="disk",
        )

    if "logic" in config.move_types:
        batch_plot_general(
            move_type="logic",
            src_dir=DATA_ROOT / "logic",
            dst_dir=PLOT_ROOT / "logic_plot",
            preserve_subdirs=False,
            keep_all_steps=True,
            n_override=N,
            coord_system_for_txt="disk",
        )
