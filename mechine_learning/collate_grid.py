# collate_grid.py
import torch

def collate_grid(batch):
    """
    Collate variable-length time sequences by padding zeros on the right;
    support choices of shape [B,4,N] (N=number of agents).
    Required keys returned by Dataset:
      - "grid_seq": [T,H,W,C] float32
      - "tmask":    [T] bool
      - "choices_ids": [4,N] long
      - "labels": () long
      - (optional) "trial_dir": str
    """
    B = len(batch)
    Tm = max(b["grid_seq"].shape[0] for b in batch)
    H, W, C = batch[0]["grid_seq"].shape[1:]

    # Time sequence alignment
    grid  = torch.zeros((B, Tm, H, W, C), dtype=batch[0]["grid_seq"].dtype)
    tmask = torch.zeros((B, Tm), dtype=torch.bool)
    for i, b in enumerate(batch):
        T = b["grid_seq"].shape[0]
        grid[i, :T]  = b["grid_seq"]
        tmask[i, :T] = b["tmask"]

    # choices: assume the same N within a batch, otherwise pad zeros on the right (cell-id=0)
    Ns = [b["choices_ids"].shape[1] for b in batch]
    maxN = max(Ns)
    choices = torch.zeros((B, 4, maxN), dtype=batch[0]["choices_ids"].dtype)
    for i, b in enumerate(batch):
        N = b["choices_ids"].shape[1]
        choices[i, :, :N] = b["choices_ids"]

    # labels (support old key name 'label')
    labels = torch.stack([ (b["labels"] if "labels" in b else b["label"]) for b in batch ], dim=0)

    out = {
        "grid_seq": grid,       # [B,Tm,H,W,C]
        "tmask": tmask,         # [B,Tm]
        "choices_ids": choices, # [B,4,N]
        "labels": labels,       # [B]
    }

    # Keep trial_dir for debugging (if present)
    if "trial_dir" in batch[0]:
        out["trial_dir"] = [b["trial_dir"] for b in batch]

    return out
