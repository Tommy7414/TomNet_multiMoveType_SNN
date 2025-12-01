# scripts/gen_grid.py
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Dict, Tuple
import json
import re
import os
import numpy as np

from gridizer_multitrack import coords_to_grid_seq_multitrack, _parse_xy_list
"""
Pipeline 5
From data/{sim_name}/{split}/{mode}/*.txt, generate
    - grid_seq.npy
    - balanced_choice.txt
    - label.json
For model training and evaluation.
"""

# =========element setting =========
PROJECT_ROOT = Path("/Users/Jer_ry/Desktop/script_tom")
SIM_NAME     = "3_12"        # corresponds to data/3_12
SPLITS       = ["training", "validation", "testing"]
H = W        = 12    

# all move_types
MODES = ["rulemap", "random", "intermediate_case1", "intermediate_case2", "logic"]

_num_pat = re.compile(r"-?\d+(?:\.\d+)?")

# =========element tools =========

def _ensure_disk_xy(arr: np.ndarray) -> np.ndarray:
    """
    if the coordinates look like world (approx. 8..31), subtract 6 to convert to disk (1..24).
    """
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return a
    mn, mx = float(a.min()), float(a.max())
    if mn >= 6.5 and mx <= 30.5:
        a = a - 6.0
    return a


def _read_xy_block(text: str) -> np.ndarray:
    nums = list(map(float, _num_pat.findall(text)))
    if len(nums) == 0 or len(nums) % 2 != 0:
        raise ValueError("block cant be parsed into (x,y) pairs")
    return np.asarray(nums, dtype=float).reshape(-1, 2)


def parse_choices_file(choice_txt: Path) -> Tuple[Dict[str, np.ndarray], int]:
    """
    analyzed answer choices file: {choice_txt}
        Choice A:
        [[x y]
        [x y]
        [x y]]
        .....
    Returns:
        choices_xy: {'A':(N,2), 'B':..., 'C':..., 'D':...} (all in disk coordinates)
        n_agents:   N
    """
    txt = choice_txt.read_text(encoding="utf-8")
    pieces = re.split(r"Choice\s+([ABCD])\s*:\s*", txt)
    out_xy: Dict[str, np.ndarray] = {}

    for i in range(1, len(pieces), 2):
        letter = pieces[i]
        block  = pieces[i+1]
        arr_xy = _ensure_disk_xy(_read_xy_block(block))
        out_xy[letter] = arr_xy

    need = set("ABCD")
    if set(out_xy) != need:
        raise ValueError(f"{choice_txt}: analyze in {list(out_xy.keys())}, need A,B,C,D")

    ns = {k: v.shape[0] for k, v in out_xy.items()}
    if len(set(ns.values())) != 1:
        raise ValueError(f"{choice_txt}: inconsistent number of pairs across choices {ns}")
    n_agents = next(iter(ns.values()))
    return out_xy, n_agents

def safe_save_npy(out_path: Path, arr: np.ndarray):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.is_dir():
        raise IsADirectoryError(f"Target is a directory: {out_path}")
    arr = np.ascontiguousarray(arr.astype(np.float32, copy=False))
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_path)

def load_correct_letter(answer_json: Path) -> str:
    """
    From answers/XXX_answer.json, load the correct letter 'A'..'D'.
    Try to be compatible with several formats:
        - "A"
        - {"correct": "A"}
        - {"answer": "A"} / {"label": "A"}
    """
    data = json.loads(answer_json.read_text(encoding="utf-8"))
    if isinstance(data, str):
        letter = data.strip()
    elif isinstance(data, dict):
        letter = (
            data.get("correct")
            or data.get("answer")
            or data.get("label")
        )
    else:
        letter = None
    if letter not in ("A", "B", "C", "D"):
        raise ValueError(f"{answer_json}: unable to parse correct letter, got {letter!r}")
    return letter


# ========= main pipeline =========

def process_one_txt(mode_dir: Path, answers_dir: Path, mode: str, txt_path: Path):
    """
    For one data/5_24/<split>/<mode>/XXX.txt:
      - Read <mode>/answers/XXX_choices.txt & XXX_answer.json
      - Create <mode>/step_plot_XXX/
      - Write grid_seq.npy, balanced_choice.txt, label.json
    """
    stem = txt_path.stem  # e.g. "5_24_case2_2" 或 "5_24_1" 等
    choice_txt = answers_dir / f"{stem}_choices.txt"
    answer_js  = answers_dir / f"{stem}_answer.json"

    if not choice_txt.exists():
        print(f"  [skip] {mode}:{stem}: Cannot find {choice_txt.name}")
        return
    if not answer_js.exists():
        print(f"  [skip] {mode}:{stem}: Cannot find {answer_js.name}")
        return

    # 1) Parse choices to get n_agents
    try:
        choices_xy, n_agents = parse_choices_file(choice_txt)
    except Exception as e:
        print(f"  [skip] {mode}:{stem}: Failed to parse choices → {e}")
        return

    # 2) New trial directory: step_plot_XXX
    trial_dir = mode_dir / f"step_plot_{stem}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # 3) Generate grid_seq.npy
    grid_path = trial_dir / "grid_seq.npy"
    try:
        # coords_to_grid_seq_multitrack will use _parse_xy_list(txt_path) to get [x,y] coordinates
        init_vals = [1.0, 2.0, 3.0, 4.0, 5.0][:n_agents]
        seq = coords_to_grid_seq_multitrack(
            txt_path,
            n_agents=n_agents,
            H=H,
            W=W,
            init_values=init_vals,
            keep_init_every_frame=True,
        )
        safe_save_npy(grid_path, seq)
    except Exception as e:
        print(f"  [skip] {mode}:{stem}: Failed to generate grid_seq → {e}")
        return

    # 4) balanced_choice.txt (directly copy *_choices.txt)
    balanced_path = trial_dir / "balanced_choice.txt"
    if not balanced_path.exists():
        balanced_path.write_text(choice_txt.read_text(encoding="utf-8"), encoding="utf-8")

    # 5) label.json (answer_idx + answer_letter)
    try:
        letter = load_correct_letter(answer_js)
    except Exception as e:
        print(f"  [warn] {mode}:{stem}: Failed to load correct letter → {e}")
        letter = None

    if letter is not None:
        idx = "ABCD".index(letter)  # A=0,B=1,C=2,D=3
        label_payload = {
            "answer_idx": idx,
            "answer_letter": letter,
            "stem": stem,
            "mode": mode,
        }
        (trial_dir / "label.json").write_text(
            json.dumps(label_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"  [+] {mode}:{stem} → {trial_dir}")


def process_split(split: str):
    """
    For one split (training / validation / testing),
    process all .txt files under MODES in order.
    """
    base = PROJECT_ROOT / "data" / SIM_NAME / split
    if not base.exists():
        print(f"[{split}] base does not exist: {base} (skip entire split)")
        return

    print(f"\n=== split = {split} @ {base} ===")
    for mode in MODES:
        mode_dir = base / mode
        if not mode_dir.exists():
            print(f"  [{split}] No {mode} directory, skip: {mode_dir}")
            continue

        answers_dir = mode_dir / "answers"
        if not answers_dir.exists():
            print(f"  [{split}:{mode}] No answers directory, skip: {answers_dir}")
            continue

        txt_list = sorted(
            p for p in mode_dir.glob("*.txt")
            if p.is_file()
        )
        if not txt_list:
            print(f"  [{split}:{mode}] (no .txt files to process)")
            continue

        print(f"  >>> mode = {mode} ({len(txt_list)} txt files)")
        for txt in txt_list:
            process_one_txt(mode_dir, answers_dir, mode, txt)


def main():
    for sp in SPLITS:
        process_split(sp)


if __name__ == "__main__":
    main()
