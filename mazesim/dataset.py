# dataset.py
import os, re, csv, json, glob, hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from numpy.random import default_rng

from .env import read_floats_after_maze, world_to_disk_if_needed
from .config import SimConfig


PAD_ID = 576

def disk_to_cell_ids(arr_3n2: np.ndarray) -> List[int]:
    """change (K,2) 的 1..24 coordinate to 0..575 cell id (1D index)."""
    out: List[int] = []
    for x, y in arr_3n2:
        xi, yi = int(round(x)), int(round(y))
        # allow out-of-bound but clamp back to 1..24 to avoid crashing
        if not (1 <= xi <= 24 and 1 <= yi <= 24):
            xi = min(24, max(1, xi))
            yi = min(24, max(1, yi))
        cid = (yi - 1) * 24 + (xi - 1)
        out.append(int(cid))
    return out

def cid_to_xy(cid):
    y = cid // 24 + 1
    x = cid % 24 + 1
    return x, y

def xy_to_cid(x, y):
    return (y-1)*24 + (x-1)

def check_train_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    assert {"trial_id","mode","seq_len_T","xy_json","cell_json"}.issubset(df.columns)

    for i,row in df.iterrows():
        cells = json.loads(row["cell_json"])
        xys   = np.array(json.loads(row["xy_json"]), dtype=int)
        T     = int(row["seq_len_T"])
        assert len(cells) == 3*T, f"{csv_path}:{i} cell_json len != 3*T"
        assert xys.shape == (3*T, 2), f"{csv_path}:{i} xy_json shape error"
        assert all(0 <= c <= 575 for c in cells), f"{csv_path}:{i} cid out of range"
        assert (1 <= xys[:,0]).all() and (xys[:,0] <= 24).all(), f"{csv_path}:{i} x out of range"
        assert (1 <= xys[:,1]).all() and (xys[:,1] <= 24).all(), f"{csv_path}:{i} y out of range"
        # 一致性檢查
        cells2 = [xy_to_cid(int(x),int(y)) for x,y in xys]
        assert cells2 == cells, f"{csv_path}:{i} xy_json <-> cell_json mismatch"

def check_answers_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    assert {"trial_id","mode","correct"}.issubset(df.columns)

    for i,row in df.iterrows():
        assert row["correct"] in ["A","B","C","D"], f"{csv_path}:{i} correct not in ABCD"
        for ch in "ABCD":
            cc  = json.loads(row[f"choice{ch}_cell"])
            cxy = np.array(json.loads(row[f"choice{ch}_xy"]), dtype=int)
            assert len(cc) == 3, f"{csv_path}:{i} choice{ch}_cell len != 3"
            assert cxy.shape == (3,2), f"{csv_path}:{i} choice{ch}_xy shape error"
            assert all(0 <= c <= 575 for c in cc), f"{csv_path}:{i} choice{ch} cid out of range"
            assert (1 <= cxy[:,0]).all() and (cxy[:,0] <= 24).all(), f"{csv_path}:{i} choice{ch} x out of range"
            assert (1 <= cxy[:,1]).all() and (cxy[:,1] <= 24).all(), f"{csv_path}:{i} choice{ch} y out of range"
            cc2 = [xy_to_cid(int(x),int(y)) for x,y in cxy]
            assert cc2 == cc, f"{csv_path}:{i} choice{ch} xy <-> cell mismatch"

def disk_to_cell_ids(arr_3n2: np.ndarray) -> List[int]:
    out = []
    for x, y in arr_3n2:
        xi, yi = int(round(x)), int(round(y))
        # 允許 0..24（題目裡的「改其中一個到 [0,0]~[24,24]」），但 token 只支援 1..24
        if not (1 <= xi <= 24 and 1 <= yi <= 24):
            # 將 0 夾回 1；>24 夾回 24，避免崩
            xi = min(24, max(1, xi))
            yi = min(24, max(1, yi))
        cid = (yi - 1) * 24 + (xi - 1)
        out.append(int(cid))
    return out

def load_answer_json(step_dir: Path) -> Optional[str]:
    """若存在 training_data/.../step_plot_*/answer.json，讀取 'correct' 的 'A'..'D'。"""
    ans = step_dir / "answer.json"
    if ans.exists():
        try:
            with open(ans, "r", encoding="utf-8") as f:
                j = json.load(f)
            cor = j.get("correct", None)
            if isinstance(cor, str) and cor in ("A", "B", "C", "D"):
                return cor
        except Exception:
            pass
    return None

def parse_choices(choice_txt_path: Path) -> Dict[str, np.ndarray]:
    """
    read *_choice.txt，return {'A':(3,2), 'B':(3,2), 'C':(3,2), 'D':(3,2)}, coordinates remain as original (usually 1..24).
    Compatible format:
        Choice A:
        [[x y]
         [x y]
         [x y]]
    """
    with open(choice_txt_path, encoding='utf-8') as f:
        content = f.read()
    chunks = {}
    for label in ['A', 'B', 'C', 'D']:
        m = re.search(rf'Choice\s+{label}\s*:\s*(\[\[.*?\]\])', content, flags=re.S)
        if not m:
            continue
        num_pat = re.compile(r'-?\d+(?:\.\d+)?')
        arr = np.asarray(list(map(float, num_pat.findall(m.group(1)))), dtype=float).reshape(-1, 2)
        chunks[label] = arr
    if len(chunks) != 4:
        raise ValueError(f"{choice_txt_path} failed to parse choices, found {len(chunks)}")
    return chunks

def read_float_array(txt_path: Path) -> np.ndarray:
    """Read the trailing [..] from trail.txt or data file, output (K,2)"""
    with open(txt_path, encoding='utf-8') as f:
        text = f.read()
    num_pat = re.compile(r'-?\d+(?:\.\d+)?')
    nums = list(map(float, num_pat.findall(text)))
    if len(nums) % 2 != 0 or len(nums) == 0:
        raise ValueError(f"{txt_path} failed to parse an even number of numbers")
    arr = np.asarray(nums, dtype=float).reshape(-1, 2)
    return arr

def infer_original_txt_path(mode_name: str, orig_dir: Path, step_dir: Path) -> Optional[Path]:
    """
    Given a step_plot_* directory and the original data/<mode>/ directory,
    infer the corresponding original .txt path.
    Returns the Path if found, else None.
    """
    stem = step_dir.name.replace("step_plot_", "")
    cand = orig_dir / f"{stem}.txt"
    return cand if cand.exists() else None

def get_last3_from_original(orig_txt: Path) -> np.ndarray:
    """從 data/sim1/... 的原檔取得「最後三個點」(世界或磁碟都吃，輸出磁碟 1..24)"""
    arr = read_float_array(orig_txt)  # 這會把整份 [..] 都讀進來
    last3_world_or_disk = arr[-3:].copy()
    last3_disk = world_to_disk_if_needed(last3_world_or_disk)
    return last3_disk

def np_equal_rounded(a: np.ndarray, b: np.ndarray) -> bool:
    """四捨五入成 int 後逐元素相等"""
    return np.array_equal(np.rint(a).astype(int), np.rint(b).astype(int))

def ensure_list_jsonable(x):
    """把 list 轉成 JSON 字串（確保都是可序列化）"""
    return json.dumps(x, ensure_ascii=False)

def parse_choices_from_step_dir(step_dir: Path) -> Dict[str, np.ndarray]:
    """
    從 step_plot_* 目錄裡的 label.json 讀出四個選項：
      return {'A': (3,2), 'B': (3,2), 'C': (3,2), 'D': (3,2)}

    假設 label.json 長這樣：
      {
        "A": [[x,y],[x,y],[x,y]],
        "B": [[x,y],[x,y],[x,y]],
        "C": ...,
        "D": ...
      }

    👉 如果你實際的 label.json 結構不一樣，只要在這裡改 mapping 即可。
    """
    path = step_dir / "label.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在")

    data = json.loads(path.read_text(encoding="utf-8"))
    choices: Dict[str, np.ndarray] = {}

    for ch in ["A", "B", "C", "D"]:
        if ch not in data:
            raise ValueError(
                f"label.json 裡找不到 key '{ch}'，請依實際格式修改 parse_choices_from_step_dir()"
            )
        arr = np.asarray(data[ch], dtype=float).reshape(-1, 2)
        choices[ch] = arr

    return choices

def keys(root_dir: Path) -> set[str]:
    ks = set()
    NAMES = ["rulemap","random","logic","intermediate_case1","intermediate_case2"]
    for n in NAMES:
        for suffix in ["_train.csv", "_answers.csv"]:
            p = root_dir / f"{n}{suffix}"
            if not p.exists(): 
                continue
            df = pd.read_csv(p, usecols=["trial_id"])
            for t in df["trial_id"].astype(str):
                ks.add(f"{n}|{t}")  # ← 以 (mode|trial_id) 當唯一鍵
    return ks

def no_leakage(train_ids, val_ids, test_ids):
    assert train_ids.isdisjoint(val_ids) and train_ids.isdisjoint(test_ids) and val_ids.isdisjoint(test_ids)

def read_traj_from_data_txt(txt_path: Path) -> np.ndarray:
    """
    Read all coordinates after "Maze" from data/.../*.txt and convert to disk (1..24).
    """
    raw = read_floats_after_maze(txt_path)      # (K,2)，world coordinates or disk coordinates
    disk = world_to_disk_if_needed(raw)         # ensure converted to disk coordinates 1..24

    if disk.ndim != 2 or disk.shape[1] != 2:
        raise ValueError(f"{txt_path}: abnormal trajectory shape {disk.shape}")

    return disk

def generate_choices_for_traj(
    traj_disk: np.ndarray,
    n_agents: int,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], str]:
    """
    Given a full trajectory (K,2) and number of agents n_agents:
      - reshape to (T, n_agents, 2)
      - use the last time step as the correct answer
      - automatically generate 3 incorrect options that are not equal to the correct answer
      - return choices: {'A':(n_agents,2), ...} and correct_letter e.g. 'C'
    """
    a = np.asarray(traj_disk, dtype=float)
    if a.size == 0 or a.shape[1] != 2:
        raise ValueError(f"traj_disk shape must be (K,2), got {a.shape}")
    if a.shape[0] % n_agents != 0:
        raise ValueError(f"Total points {a.shape[0]} is not divisible by n_agents={n_agents}")

    T = a.shape[0] // n_agents
    steps = a.reshape(T, n_agents, 2)   # (T, N, 2)

    true_pos = steps[-1].copy()   

    rng = default_rng(seed)

    candidates: List[np.ndarray] = []
    if T >= 2:
        candidates.append(steps[0])
    if T >= 3:
        candidates.append(steps[T // 2])
    if T >= 4:
        candidates.append(steps[-2])

    wrong: List[np.ndarray] = []

    for cand in candidates:
        if not np_equal_rounded(cand, true_pos) and all(
            not np_equal_rounded(cand, w) for w in wrong
        ):
            wrong.append(cand.copy())
        if len(wrong) == 3:
            break

    while len(wrong) < 3:
        jitter = rng.integers(-1, 2, size=true_pos.shape)  # -1,0,1
        cand = true_pos + jitter
        # 1..24
        cand[..., 0] = np.clip(cand[..., 0], 1, 24)
        cand[..., 1] = np.clip(cand[..., 1], 1, 24)

        if not np_equal_rounded(cand, true_pos) and all(
            not np_equal_rounded(cand, w) for w in wrong
        ):
            wrong.append(cand)

    # Randomly insert the correct answer into one of A/B/C/D
    letters = ["A", "B", "C", "D"]
    correct_idx = int(rng.integers(0, 4))
    correct_letter = letters[correct_idx]

    choices: Dict[str, np.ndarray] = {}
    wrong_iter = iter(wrong)
    for i, ch in enumerate(letters):
        if i == correct_idx:
            choices[ch] = true_pos.copy()
        else:
            choices[ch] = next(wrong_iter).copy()

    return choices, correct_letter

def write_choice_txt(path: Path, choices: Dict[str, np.ndarray]):
    """
    Write choices {'A':(N,2),...} to a simple txt file,
    which can be read back using your original parse_choices_file().
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for ch in ["A", "B", "C", "D"]:
            arr = np.rint(choices[ch]).astype(int)
            f.write(f"Choice {ch}:\n")
            for x, y in arr:
                f.write(f"{int(x)} {int(y)}\n")
            f.write("\n")

def process_mode_data(config: SimConfig, mode_name: str) -> Tuple[int, int]:
    """
    Read raw .txt files from data/<sim_name>/<mode>/...,
    generate:
      - data/<sim_name>/<mode>/answers/<trial>_choices.txt
      - data/<sim_name>/<mode>/answers/<trial>_answer.json
      - data/<sim_name>/csv/<mode>_train.csv
      - data/<sim_name>/csv/<mode>_answers.csv
    """
    DATA_ROOT = config.data_root

    if mode_name == "rulemap":
        mode_dir = DATA_ROOT / "rulemap"
    elif mode_name == "random":
        mode_dir = DATA_ROOT / "random"
    elif mode_name == "logic":
        mode_dir = DATA_ROOT / "logic"
    elif mode_name == "intermediate_case1":
        mode_dir = DATA_ROOT / "intermediate_case1"
    elif mode_name == "intermediate_case2":
        mode_dir = DATA_ROOT / "intermediate_case2"
    else:
        raise ValueError(f"Unknown mode_name: {mode_name}")

    if not mode_dir.exists():
        print(f"[{mode_name}] Directory does not exist, skipping: {mode_dir}")
        return 0, 0

    txt_files = sorted(
        p for p in mode_dir.glob("*.txt")
        if p.is_file() and "FINAL" not in p.name
    )
    if not txt_files:
        print(f"[{mode_name}] No .txt files found in {mode_dir}")
        return 0, 0

    csv_root = DATA_ROOT / "csv"
    csv_root.mkdir(parents=True, exist_ok=True)

    seq_rows: List[Dict] = []
    ans_rows: List[Dict] = []

    for txt_path in txt_files:
        try:
            traj_disk = read_traj_from_data_txt(txt_path)  # (K,2)
        except Exception as e:
            print(f"[{mode_name}] Failed to read {txt_path.name}: {e}")
            continue

        K = traj_disk.shape[0]
        n_agents = config.n_agents
        if K % n_agents != 0 or K == 0:
            print(f"[{mode_name}] {txt_path.name}: Total points {K} is not divisible by n_agents={n_agents}, skipping")
            continue

        T = K // n_agents
        cells = disk_to_cell_ids(traj_disk)

        trial_id    = txt_path.stem  # 例如 5_24_1, 5_24_case13_1
        relpath_txt = txt_path.relative_to(DATA_ROOT).as_posix()

        # ---- one row for train CSV ----
        seq_rows.append({
            "trial_id":    trial_id,
            "mode":        mode_name,
            "relpath_txt": relpath_txt,
            "n_agents":    n_agents,
            "seq_len_T":   T,
            "xy_json":     ensure_list_jsonable(
                np.rint(traj_disk).astype(int).tolist()
            ),
            "cell_json":   ensure_list_jsonable(cells),
        })

        # ---- generate four choices + correct answer ----
        seed = int(hashlib.sha1(trial_id.encode("utf-8")).hexdigest()[:8], 16)
        try:
            choices, correct_letter = generate_choices_for_traj(traj_disk, n_agents, seed)
        except Exception as e:
            print(f"[{mode_name}] Failed to generate choices for {txt_path.name}: {e}")
            continue

        # Write answers files (for later visualization or other pipeline use)
        ans_dir = txt_path.parent / "answers"
        ans_dir.mkdir(parents=True, exist_ok=True)

        choice_txt_path = ans_dir / f"{trial_id}_choices.txt"
        write_choice_txt(choice_txt_path, choices)

        ans_json_path = ans_dir / f"{trial_id}_answer.json"
        with open(ans_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "trial_id":    trial_id,
                    "mode":        mode_name,
                    "n_agents":    n_agents,
                    "relpath_txt": relpath_txt,
                    "correct":     correct_letter,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        # ---- answers CSV 的一列 ----
        row_ans: Dict[str, object] = {
            "trial_id":    trial_id,
            "mode":        mode_name,
            "relpath_txt": relpath_txt,
            "n_agents":    n_agents,
            "correct":     correct_letter,
        }

        for ch in ["A", "B", "C", "D"]:
            arr = np.rint(choices[ch]).astype(int)
            row_ans[f"choice{ch}_xy"]   = ensure_list_jsonable(arr.tolist())
            row_ans[f"choice{ch}_cell"] = ensure_list_jsonable(disk_to_cell_ids(arr))

        ans_rows.append(row_ans)

    # ---- write out CSV ----
    if seq_rows:
        seq_csv = csv_root / f"{mode_name}_train.csv"
        with open(seq_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(seq_rows[0].keys()))
            w.writeheader()
            w.writerows(seq_rows)
        print(f"[{mode_name}] ✓ wrote {seq_csv} ({len(seq_rows)} rows)")
    else:
        print(f"[{mode_name}] ✗ no sequence rows")

    if ans_rows:
        ans_csv = csv_root / f"{mode_name}_answers.csv"
        with open(ans_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(ans_rows[0].keys()))
            w.writeheader()
            w.writerows(ans_rows)
        print(f"[{mode_name}] ✓ wrote {ans_csv} ({len(ans_rows)} rows)")
    else:
        print(f"[{mode_name}] ✗ no answer rows")

    return len(seq_rows), len(ans_rows)

def build_all_csv(config: SimConfig):
    """
    This version only does:
      data/<sim_name>/<mode>/...  ->  answers files + data/<sim_name>/csv/*.csv
    No visualization, no touching training_data.
    """
    total_seq = 0
    total_ans = 0
    for mode_name in config.move_types:
        n_seq, n_ans = process_mode_data(config, mode_name)
        total_seq += n_seq
        total_ans += n_ans

    print(f"Total sequence rows = {total_seq}, answer rows = {total_ans}")
