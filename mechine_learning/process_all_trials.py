# process_all_trials.py
import re, json, hashlib
from itertools import permutations
# Path
from pathlib import Path    
import numpy as np

def _parse_choices_xy(choice_txt: Path) -> dict:
    """讀取 *_choice.txt，回傳 {'A': np.ndarray(N,2), 'B':..., 'C':..., 'D':...}（1-based座標）"""
    s = choice_txt.read_text(encoding="utf-8")
    out = {}
    for L in "ABCD":
        m = re.search(rf"Choice\s+{L}\s*:\s*(\[\[.*?\]\])", s, flags=re.S)
        if not m:
            raise ValueError(f"{choice_txt}: missing Choice {L}")
        nums = list(map(float, re.findall(r"-?\d+(?:\.\d+)?", m.group(1))))
        arr = np.asarray(nums, dtype=float).reshape(-1, 2)  # [[x,y],...]
        out[L] = arr
    return out

def _write_balanced_choice_txt(step_dir: Path, abcd_xy: dict):
    """以固定格式覆寫/寫入 balanced_choice.txt（座標保留原 1-based、整數/小數皆可）"""
    def fmt(arr):
        return np.array2string(arr, separator=' ', formatter={'float_kind':lambda x: ('%.0f'%x if x.is_integer() else '%.6f'%x)})
    lines = []
    for L in "ABCD":
        lines.append(f"Choice {L}: {fmt(abcd_xy[L])}\n")
    (step_dir / "balanced_choice.txt").write_text("".join(lines), encoding="utf-8")

def _load_correct_letter(step_dir: Path) -> str:
    a = step_dir / "answer.json"
    if a.exists():
        j = json.loads(a.read_text(encoding="utf-8"))
        c = j.get("correct", None)
        if c in ("A","B","C","D"): return c
    b = step_dir / "label.json"
    if b.exists():
        j = json.loads(b.read_text(encoding="utf-8"))
        c = j.get("answer_letter", None)
        if c in ("A","B","C","D"): return c
    raise FileNotFoundError(f"{step_dir}: no answer.json/label.json with correct letter")

def _save_correct(step_dir: Path, new_idx: int):
    """更新 label.json 的 answer_idx 與 answer_letter；若有 answer.json 也同步"""
    idx2L = "ABCD"
    # label.json
    lab = step_dir / "label.json"
    if lab.exists():
        meta = json.loads(lab.read_text(encoding="utf-8"))
        meta["answer_idx"] = int(new_idx)
        meta["answer_letter"] = idx2L[new_idx]
        lab.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    # answer.json（若存在）
    ans = step_dir / "answer.json"
    if ans.exists():
        meta = json.loads(ans.read_text(encoding="utf-8"))
        meta["correct"] = idx2L[new_idx]
        ans.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def _stable_offset(sim_name: str, mode: str) -> int:
    s = f"{sim_name}::{mode}"
    h = hashlib.sha1(s.encode()).hexdigest()
    return int(h[:4], 16) % 24

def apply_balanced_choice_permutations(root: Path, sim_name: str,
                                       modes=("rulemap","random","intermediate_case1","intermediate_case2")):
    """
    只處理 training_data 下的各模式；對每個 step_plot_*：
      - 讀 *_choice.txt
      - 指定 24 種全排列 round-robin 置換 ABCD（平均）
      - 寫 balanced_choice.txt
      - 同步更新 label.json / answer.json 的正解
    """
    base = root / "training_data" / sim_name
    mode_to_dir = {
        "rulemap":            base / "rulemap_plot",
        "random":             base / "random_plot",
        "intermediate_case1": base / "intermediate_plot" / "case1",
        "intermediate_case2": base / "intermediate_plot" / "case2",
        # "logic": base / "logic_plot",   # 如需也可加入
    }
    perms = list(permutations(range(4)))  # 24 種
    for mode in modes:
        droot = mode_to_dir.get(mode, None)
        if not droot or not droot.exists(): 
            continue
        steps = sorted(droot.glob("step_plot_*"))
        if not steps: 
            continue
        off = _stable_offset(sim_name, mode)
        for i, step in enumerate(steps):
            # 找原始 choice 檔
            cand = sorted(step.glob("*choice*.txt"))
            if not cand:
                continue
            src = cand[0]
            abcd = _parse_choices_xy(src)          # dict of np.ndarray
            # 原始正解 idx
            L2idx = {"A":0,"B":1,"C":2,"D":3}
            old_letter = _load_correct_letter(step)
            old_idx = L2idx[old_letter]
            # 指派平均置換
            perm = list(perms[(i + off) % 24])     # 例如 [2,0,3,1]
            # 套用置換 → 新 ABCD
            Ls = ["A","B","C","D"]
            new_abcd = { Ls[j]: abcd[Ls[perm[j]]] for j in range(4) }
            _write_balanced_choice_txt(step, new_abcd)
            # 新正解 idx（在新順序中的位置）
            new_idx = perm.index(old_idx)
            _save_correct(step, new_idx)
            print(f"[balanced] {step.name}: perm={perm} old={old_letter} -> new_idx={new_idx}")
    print("[OK] training_data/* 已寫入 balanced_choice.txt 並更新 label/answer")
