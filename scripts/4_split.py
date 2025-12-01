# scripts/split.py
# -*- coding: utf-8 -*-
import shutil, re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
"""
Pipeline 4:
Split data/{sim_name}/csv/*.csv into
  data/{sim_name}/training/csv/*.csv
  data/{sim_name}/validation/csv/*.csv
  data/{sim_name}/testing/csv/*.csv
And copy corresponding .txt and answers files into each split folder.

Make sure the hyperparameters in SimConfig are consistent with those in gen_trails.py
"""
PROJECT_ROOT = Path("/Users/Jer_ry/Desktop/script_tom")
SIM_NAME     = "5_12"      
N_AGENTS   = 5      
H = W = 12      

DATA_ROOT = PROJECT_ROOT / "data" / SIM_NAME
CSV_ROOT  = DATA_ROOT / "csv"

# YOU CAN ADJUST THE SPLIT RATIOS HERE
SPLIT_RATIOS: Dict[str, float] = {
    "training":   0.8,
    "validation": 0.1,
    "testing":    0.1,
}

RANDOM_SEED = 1234 

_num_pat = re.compile(r"-?\d+(?:\.\d+)?")

def _assign_splits(trial_ids: List[str]) -> Dict[str, List[str]]:
    """base on SPLIT_RATIOS slice trial_ids into train/val/test randomly."""
    n = len(trial_ids)
    if n == 0:
        return {k: [] for k in SPLIT_RATIOS}

    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(n)

    n_train = int(round(n * SPLIT_RATIOS["training"]))
    n_val   = int(round(n * SPLIT_RATIOS["validation"]))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    n_test  = n - n_train - n_val

    idx_train = perm[:n_train]
    idx_val   = perm[n_train:n_train + n_val]
    idx_test  = perm[n_train + n_val:]

    out = {
        "training":   [trial_ids[i] for i in idx_train],
        "validation": [trial_ids[i] for i in idx_val],
        "testing":    [trial_ids[i] for i in idx_test],
    }
    return out


def _infer_n_agents_from_traj(arr: np.ndarray) -> int:
    """For the current SIM_NAME, always use N_AGENTS."""
    K = arr.shape[0]
    if K % N_AGENTS != 0:
        raise ValueError(
            f"Trajectory length {K} is not divisible by N_AGENTS={N_AGENTS},"
            f" please check if the original txt is correct (SIM_NAME={SIM_NAME})"
        )
    return N_AGENTS


def _build_step_plots_for_txt(dest_txt: Path) -> None:
    """
    Given a data/3_12/<split>/<mode>/<trial>.txt,
    create two directories at the same level:

        step_plot_<trial>_1/trail.txt  (history: use up to the second last time step)
        step_plot_<trial>_2/trail.txt  (query: use the last time step, the entire trajectory)
    D_build_tomnet2_pairs.py will:
      - use the last 1 / 2 in the directory name as the odd/even step_id
      - read the coordinates in trail.txt and perform the world→disk conversion itself
    """
    if not dest_txt.exists():
        return

    text = dest_txt.read_text(encoding="utf-8")
    nums = list(map(float, _num_pat.findall(text)))
    if len(nums) == 0 or len(nums) % 2 != 0:
        print(f"[step_plot] Skip {dest_txt}, no valid (x,y) coordinates found")
        return

    arr = np.asarray(nums, dtype=float).reshape(-1, 2)  # (K,2)

    try:
        n_agents = _infer_n_agents_from_traj(arr)
    except ValueError as e:
        print(f"[step_plot] Skip {dest_txt.name}: {e}")
        return

    T = arr.shape[0] // n_agents
    if T < 2:
        print(f"[step_plot] Skip {dest_txt.name}: T={T} < 2 cannot split history/query")
        return

    # history: use up to the second last time step
    hist_T    = max(1, T - 1)
    arr_hist  = arr[: hist_T * n_agents, :].copy()
    arr_query = arr.copy()  # full length

    base_dir = dest_txt.parent          # e.g. data/3_12/training/logic
    stem     = dest_txt.stem            # e.g. 3_12_case1_793

    step1_dir = base_dir / f"step_plot_{stem}_1"   # odd → history
    step2_dir = base_dir / f"step_plot_{stem}_2"   # even → query

    for d, sub_arr in [(step1_dir, arr_hist), (step2_dir, arr_query)]:
        d.mkdir(parents=True, exist_ok=True)
        out_txt = d / "trail.txt"
        with out_txt.open("w", encoding="utf-8") as f:
            for x, y in sub_arr:
                f.write(f"{float(x)} {float(y)}\n")


def _split_one_mode(mode_name: str):
    train_csv = CSV_ROOT / f"{mode_name}_train.csv"
    ans_csv   = CSV_ROOT / f"{mode_name}_answers.csv"

    if not train_csv.exists():
        print(f"[{mode_name}] Cannot find {train_csv}, skipping...")
        return

    df_train = pd.read_csv(train_csv)
    df_ans   = pd.read_csv(ans_csv) if ans_csv.exists() else None

    if "trial_id" not in df_train.columns or "relpath_txt" not in df_train.columns:
        raise ValueError(f"{train_csv} missing trial_id or relpath_txt columns")

    trial_ids = df_train["trial_id"].astype(str).tolist()
    split_ids = _assign_splits(trial_ids)

    for split_name, ids in split_ids.items():
        if not ids:
            print(f"[{mode_name}] split={split_name} no data, skipping")
            continue

        # ---- 該 split 的 CSV ----
        mask_train = df_train["trial_id"].astype(str).isin(ids)
        df_train_split = df_train.loc[mask_train].copy()

        split_root    = DATA_ROOT / split_name
        split_csv_root = split_root / "csv"
        split_csv_root.mkdir(parents=True, exist_ok=True)

        out_train_csv = split_csv_root / f"{mode_name}_train.csv"
        df_train_split.to_csv(out_train_csv, index=False)

        if df_ans is not None:
            mask_ans      = df_ans["trial_id"].astype(str).isin(ids)
            df_ans_split  = df_ans.loc[mask_ans].copy()
            out_ans_csv   = split_csv_root / f"{mode_name}_answers.csv"
            df_ans_split.to_csv(out_ans_csv, index=False)
        for _, row in df_train_split.iterrows():
            rel    = row["relpath_txt"]            # e.g. 'logic/3_12_case1_793.txt'
            src_txt = DATA_ROOT / rel             # original txt
            if not src_txt.exists():
                print(f"[{mode_name}] WARNING: Cannot find original file {src_txt}")
                continue

            dest_txt = split_root / rel           # e.g. data/3_12/training/logic/3_12_case1_793.txt
            dest_txt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_txt, dest_txt)

            # Copy answers metadata
            stem        = src_txt.stem       
            src_ans_dir = src_txt.parent / "answers"

            for suffix in ("_choices.txt", "_answer.json"):
                src_meta = src_ans_dir / f"{stem}{suffix}"
                if not src_meta.exists():
                    continue

                dest_meta = (
                    split_root
                    / src_ans_dir.relative_to(DATA_ROOT)
                    / f"{stem}{suffix}"
                )
                dest_meta.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_meta, dest_meta)

            # Create two step_plot_*/trail.txt for each trial in the split
            _build_step_plots_for_txt(dest_txt)

        print(f"[{mode_name}] split={split_name}: {len(df_train_split)} trials")


def main():
    MODES = ["rulemap", "random", "intermediate_case1", "intermediate_case2", "logic"]
    for m in MODES:
        _split_one_mode(m)


if __name__ == "__main__":
    main()
