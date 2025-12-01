# grid_dataset.py
### function: To 
from pathlib import Path
import json
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional

def _coord_to_cid(x: float, y: float, W: int = 24) -> int:
    """Convert 1..24 (x,y) to 0-based cell-id [0,575]."""
    r = int(round(y)) - 1
    c = int(round(x)) - 1
    if not (0 <= r < 24 and 0 <= c < 24):
        raise ValueError(f"cell out of 24x24: {(x,y)} -> (r={r},c={c})")
    return r * W + c

def _parse_choices_txt(choice_txt: Path) -> dict:
    """
    Read 'Choice A/B/C/D:' file and return N cell-ids (0-based) for each choice.
    The file format is output by np.array2string, e.g. [[x y], [x y], ..., [x y]], N can be 3, 5, etc.
    """
    txt = choice_txt.read_text(encoding="utf-8")
    pieces = re.split(r"Choice\s+([ABCD])\s*:\s*", txt)
    out = {}
    n_per_choice = set()

    for i in range(1, len(pieces), 2):
        letter = pieces[i]
        block  = pieces[i+1]
        nums = re.findall(r"[-+]?\d*\.?\d+", block)
        if len(nums) % 2 != 0:
            raise ValueError(f"{choice_txt}: {letter} has odd number of coords (need even), got {len(nums)} numbers")

        coords = []
        for j in range(0, len(nums), 2):
            x, y = float(nums[j]), float(nums[j+1]) 
            coords.append(_coord_to_cid(x, y, W=24))
        out[letter] = coords
        n_per_choice.add(len(coords))

    need = set("ABCD")
    if set(out) != need:
        raise ValueError(f"{choice_txt}: choices parsed={list(out.keys())}, expect A/B/C/D")

    if len(n_per_choice) != 1:
        raise ValueError(f"{choice_txt}: inconsistent number of points per choice: {sorted(n_per_choice)}")

    return out  # dict{'A':[cid]*N, 'B':[cid]*N, 'C':[cid]*N, 'D':[cid]*N}


class GridSeqDataset(Dataset):
    def __init__(self, root: Path, sim_name: str, split: str, mode: str):
        """
        root: e.g. /Users/xxx/Desktop/scripts
        split: 'train' | 'val' | 'test'
        mode : 'rulemap' | 'random' | 'intermediate_case1' | 'intermediate_case2' | 'logic'
        """
        self.root = Path(root)
        self.sim_name = sim_name
        self.split = split
        self.mode = mode
        self.items = self._scan()
        # --- in __init__: 若全部都是 balanced_choice，關掉內部置換 ---
        self._train_perms = None
        if self.split == "train":
            # 只要有樣本不是 balanced，就用內部 round-robin 當保險
            need_internal = any(not it.get("is_balanced", False) for it in self.items)
            if need_internal:
                import itertools
                perms = list(itertools.permutations(range(4)))  # 24
                offset = (hash((self.sim_name, self.mode)) % 24)
                self._train_perms = [list(perms[(i + offset) % 24]) for i in range(len(self.items))]

    def _scan(self):
        split_map = {
            "train": "training",    
            "val": "validation",      
            "test": "testing"       
        }
        base_root = self.root / "data" / self.sim_name / split_map[self.split]
        
        if self.mode == "rulemap":
            search_dirs = [base_root / "rulemap"]    
        elif self.mode == "random":
            search_dirs = [base_root / "random"] 
        elif self.mode == "intermediate_case1":
            search_dirs = [base_root / "intermediate_case1"]
        elif self.mode == "intermediate_case2":
            search_dirs = [base_root / "intermediate_case2"]
        elif self.mode == "logic":
            search_dirs = [base_root / "logic"]      
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        items = []
        for dpath in search_dirs:
            if not dpath.exists():
                continue
            for trial_dir in sorted(dpath.glob("step_plot_*")):
                grid   = trial_dir / "grid_seq.npy"
                labelf = trial_dir / "label.json"
                ct_all = sorted(trial_dir.glob("*choice*.txt"))
                choicef = None
                for c in ct_all:
                    if c.name == "balanced_choice.txt":
                        choicef = c
                        break
                if choicef is None and ct_all:
                    choicef = ct_all[0]

                if grid.exists() and labelf.exists() and choicef is not None:
                    items.append({
                        "dir": trial_dir, "grid": grid, "label": labelf, "choice": choicef,
                        "is_balanced": (choicef.name == "balanced_choice.txt")
                    })

        if not items:
            searched = ", ".join(str(p) for p in search_dirs)
            raise FileNotFoundError(f"No trials found for split={self.split}, mode={self.mode} under {searched}")
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        arr = np.load(it["grid"]).astype(np.float32, copy=False)  # [T,H,W,C]
        arr = arr[:-1, :, :, :]  # delete last frame -> [T-1,H,W,C]
        T, H, W, C = arr.shape
        grid_seq = torch.from_numpy(arr)

        if C % 6 != 0:
            raise ValueError(f"{it['dir']}: channels C={C} not divisible by 6 → cannot infer n_agents")
        n_from_grid = C // 6

        tmask = torch.ones(T, dtype=torch.bool)

        meta = json.loads(Path(it["label"]).read_text(encoding="utf-8"))
        label_idx = int(meta["answer_idx"])
        labels = torch.tensor(label_idx, dtype=torch.long)

        ch = _parse_choices_txt(it["choice"])  # dict{'A':[cid]*N, ...}
        n_from_choice = len(ch["A"])
        if any(len(ch[k]) != n_from_choice for k in "BCD"):
            raise ValueError(f"{it['choice']}: A/B/C/D lengths mismatch")

        if n_from_choice != n_from_grid:
            raise ValueError(
                f"{it['dir']}: n_agents mismatch between grid (C={C}→{n_from_grid}) "
                f"and choices (N={n_from_choice})."
            )

        choices_ids = torch.tensor([ch["A"], ch["B"], ch["C"], ch["D"]], dtype=torch.long)  # [4,N]


        return {
            "grid_seq": grid_seq,        # [T,H,W,C] float32
            "tmask": tmask,              # [T] bool
            "choices_ids": choices_ids,  # [4,3] long (cell-id)
            "labels": labels,            # () long
            "trial_dir": str(it["dir"]),
        }

 
@dataclass
class StepPlotEntry:
    step_dir: Path
    grid_path: Path
    choices_path: Path
    label_path: Path
    choices_mask_path: Optional[Path] = None


def scan_step_plot_dir(step_root: Path) -> list[StepPlotEntry]:
    """
    掃描 data/.../step_plot_5_24_1 底下所有 trial 目錄，
    找出 grid_seq / choices / label 等檔案位置。
    """
    entries: list[StepPlotEntry] = []

    for d in sorted(step_root.iterdir()):
        if not d.is_dir():
            continue

        grid = d / "grid_seq.npy"
        choices = d / "choices_ids.npy"
        label_txt = d / "label_idx.txt"
        choices_mask = d / "choices_mask.npy"

        if not grid.exists() or not choices.exists() or not label_txt.exists():
            # 不完整就跳過
            continue

        entries.append(
            StepPlotEntry(
                step_dir=d,
                grid_path=grid,
                choices_path=choices,
                label_path=label_txt,
                choices_mask_path=choices_mask if choices_mask.exists() else None,
            )
        )
    return entries


# -------------------------------
# 1) manifest for SNN / Transformer
# -------------------------------

def export_mcq_manifest(step_root: Path,
                        out_path: Path,
                        drop_last_frame: bool = True) -> None:
    """
    Generate mcq_train_manifest_v1.jsonl for SNN / Transformer.

    Each line of JSON looks like this:
    {
        "step_dir": "trial_0001",
        "grid": "grid_seq.npy",
        "choices_ids": "choices_ids.npy",
        "choices_mask": "choices_mask.npy" | null,
        "label_idx": 2,
        "drop_last_frame": true
    }
    """
    entries = scan_step_plot_dir(step_root)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for e in entries:
            with e.label_path.open("r", encoding="utf-8") as lf:
                label_idx = int(lf.read().strip())

            row = {
                "step_dir": e.step_dir.name,
                # Use relative paths, DataLoader will join them later
                "grid": e.grid_path.name,
                "choices_ids": e.choices_path.name,
                "choices_mask": e.choices_mask_path.name if e.choices_mask_path else None,
                "label_idx": label_idx,
                "drop_last_frame": bool(drop_last_frame),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# -------------------------------
# 2) manifest for ToMNet2 pairs
# -------------------------------

def _infer_agent_id_from_dirname(d: Path) -> str:
    """
    According to your step_dir naming convention, group trials from the same agent together.

    Here we assume the directory name looks like:   
        A0_case1_trial003
    or:
        A0_XXXX_XXXX

    Then take the first part as the agent id:
    """
    name = d.name
    parts = name.split("_")
    return parts[0]  # <-- If your actual naming is different, please modify this line accordingly

def export_tomnet2_pairs_manifest(
    step_root: Path,
    out_path: Path,
    infer_agent_id: Callable[[Path], str] = _infer_agent_id_from_dirname,
    max_pairs_per_agent: Optional[int] = None,
    drop_last_frame: bool = True,
) -> None:
    """
    Generate tomnet2_train_pairs_v1.jsonl for ToMNet2JsonlDataset.

    Each line of JSON looks like this:
    {
      "step_dir_j": "A0_trial0001",
      "step_dir_k": "A0_trial0002",
      "grid_j": "grid_seq.npy",
      "grid_k": "grid_seq.npy",
      "choices_ids": "choices_ids.npy",
      "choices_mask": "choices_mask.npy" | null,
      "label_idx": 1,
      "drop_last_frame": true
    }

    Pair generation strategy (default):
      - Group trials by agent_id.
      - For each agent, generate all ordered pairs (j, k) within the group, where j != k.
      - If max_pairs_per_agent is not None, only take the first N pairs.
    """
    entries = scan_step_plot_dir(step_root)

    # 1) Group by agent_id
    groups: dict[str, list[StepPlotEntry]] = defaultdict(list)
    for e in entries:
        aid = infer_agent_id(e.step_dir)
        groups[aid].append(e)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for aid, trials in groups.items():
            trials = sorted(trials, key=lambda e: e.step_dir.name)
            pairs_written = 0

            # 所有有序 pair (j, k)
            for j_idx, e_j in enumerate(trials):
                for k_idx, e_k in enumerate(trials):
                    if j_idx == k_idx:
                        continue

                    with e_k.label_path.open("r", encoding="utf-8") as lf:
                        label_idx = int(lf.read().strip())

                    row = {
                        "agent_id": aid,
                        "step_dir_j": e_j.step_dir.name,
                        "step_dir_k": e_k.step_dir.name,
                        "grid_j": e_j.grid_path.name,
                        "grid_k": e_k.grid_path.name,
                        "choices_ids": e_k.choices_path.name,
                        "choices_mask": (
                            e_k.choices_mask_path.name if e_k.choices_mask_path else None
                        ),
                        "label_idx": label_idx,
                        "drop_last_frame": bool(drop_last_frame),
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    pairs_written += 1

                    if max_pairs_per_agent is not None and pairs_written >= max_pairs_per_agent:
                        break
                if max_pairs_per_agent is not None and pairs_written >= max_pairs_per_agent:
                    break
