# model_evaluate.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Callable, List
import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    r2_score, 
    confusion_matrix, 
    precision_recall_fscore_support,
    accuracy_score
)
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Physical Parameters (45nm CMOS) ----
ENERGY_PER_MAC = 4.6e-12  # 4.6 pJ -> Joule
ENERGY_PER_SOP = 0.9e-12  # 0.9 pJ -> Joule

@dataclass
class EnergyReport:
    macs_per_sample: Optional[float] = None
    ann_energy_per_sample: Optional[float] = None
    sops_per_sample: Optional[float] = None
    snn_energy_per_sample: Optional[float] = None

@dataclass
class PerformanceMetrics:
    """Comprehensive performance evaluation metrics for the model"""
    # Basic metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Advanced metrics
    roc_auc: float
    r2_score: float
    
    # Confusion matrix
    confusion_matrix: np.ndarray
    
    # Entropy-related metrics
    prediction_entropy: float  # Prediction uncertainty
    confidence_scores: np.ndarray  # Confidence for each sample
    
    # Energy-related metrics
    energy_report: EnergyReport
    
    # Computational efficiency
    macs_per_accuracy: float  # MACs per unit accuracy
    energy_per_accuracy: float  # Energy per unit accuracy

def calculate_prediction_entropy(probabilities: np.ndarray) -> float:
    """Calculate prediction entropy (uncertainty)"""
    epsilon = 1e-12  # 避免log(0)
    entropy = -np.sum(probabilities * np.log(probabilities + epsilon), axis=1)
    return float(np.mean(entropy))

def calculate_confidence(probabilities: np.ndarray) -> np.ndarray:
    """Calculate confidence for each sample (maximum probability)"""
    return np.max(probabilities, axis=1)

def _reset_macs(model: nn.Module):
    for m in model.modules():
        if hasattr(m, "_macs"):
            delattr(m, "_macs")

def _ann_macs_hook(m: nn.Module, inp, out):
    """
    inp: tuple of tensors
    out: tensor or tuple
    """
    out_t = out[0] if isinstance(out, tuple) else out

    macs = 0

    # ----- Conv2d -----
    if isinstance(m, nn.Conv2d):
        # out: [B, Cout, H, W]
        B, Cout, H, W = out_t.shape
        Cin = m.in_channels
        Kh, Kw = m.kernel_size
        groups = m.groups
        macs = B * Cout * H * W * (Cin // groups) * Kh * Kw

    # ----- Linear: 支援任意維度 (*, Dout) -----
    elif isinstance(m, nn.Linear):
        # [FIX] 加入 try-except 來處理 NestedTensor 不支援 .shape 的問題
        try:
            if out_t.ndim == 0:
                macs = 0
            else:
                Dout = out_t.shape[-1]
                # e.g. out_t.shape = [B, 4, D] → batch_elems = B*4
                batch_elems = out_t.numel() // Dout
                macs = batch_elems * m.in_features * Dout
        except RuntimeError:
            # 如果是 NestedTensor，.shape 會報錯，但 .numel() 通常可用
            # Linear 層運算量 = 輸出元素總數 * 輸入特徵數
            # (因為每個輸出元素都是一次長度為 in_features 的 dot product)
            try:
                macs = out_t.numel() * m.in_features
            except:
                macs = 0

    if macs > 0:
        m._macs = getattr(m, "_macs", 0) + int(macs)

ForwardFn = Callable[[nn.Module, Dict[str, torch.Tensor]], None]

def estimate_ann_macs(model: nn.Module,
                      batch: Dict[str, torch.Tensor],
                      device: str = "cpu",
                      forward_fn: Optional[ForwardFn] = None) -> int:
    """
    Run a forward pass on a batch to estimate the total MACs for the ANN (including batch)
    """
    model.eval()
    model.to(device)

    hooks = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(_ann_macs_hook))

    _reset_macs(model)

    with torch.no_grad():
        x = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
             for k, v in batch.items()}
        if forward_fn is None:
            # Default behavior: adapt to old Transformer model interface
            # Note: assumes input contains these keys
            if "grid_seq" in x:
                model(x["grid_seq"], x["tmask"], x["choices_ids"])
            elif "grid_seq_j" in x:
                # ToMNet2 interface
                model(x["grid_seq_j"], x["tmask_j"], x["grid_seq_k"], x["tmask_k"], x["choices_ids"], x["choices_mask"])
        else:
            forward_fn(model, x)

    total_macs = 0
    for m in model.modules():
        total_macs += getattr(m, "_macs", 0)

    for h in hooks:
        h.remove()

    return total_macs

def ann_energy_from_macs(macs: int) -> float:
    """
    Return energy (Joule)
    """
    return macs * ENERGY_PER_MAC

def evaluate_ann_energy(model: nn.Module,
                        batch: Dict[str, torch.Tensor],
                        device: str = "cpu",
                        forward_fn: Optional[ForwardFn] = None) -> EnergyReport:
    """
    Calculate MACs and Energy (per sample) specifically for ANN (conv / transformer / conv+pred)
    """
    macs = estimate_ann_macs(model, batch, device=device, forward_fn=forward_fn)
    
    # Try to get Batch Size
    if "grid_seq" in batch:
        B = batch["grid_seq"].size(0)
    elif "grid_seq_j" in batch:
        B = batch["grid_seq_j"].size(0)
    else:
        # Fallback
        B = next(iter(batch.values())).size(0)
        
    macs_per_sample = macs / max(B, 1)
    e_per_sample = ann_energy_from_macs(macs_per_sample)
    return EnergyReport(macs_per_sample=macs_per_sample,
                        ann_energy_per_sample=e_per_sample)

def evaluate_model_performance(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
    forward_fn: Optional[Callable] = None,
    energy_forward_fn: Optional[Callable] = None
) -> PerformanceMetrics:
    """
    Comprehensive evaluation of model performance
    """
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch_data = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            if forward_fn:
                logits = forward_fn(model, batch_data)
            else:
                # Default forward logic
                if "grid_seq_j" in batch_data:
                    logits = model(
                        batch_data["grid_seq_j"],
                        batch_data["tmask_j"], 
                        batch_data["grid_seq_k"],
                        batch_data["tmask_k"],
                        batch_data["choices_ids"],
                        batch_data["choices_mask"]
                    )
                else:
                    logits = model(
                        batch_data["grid_seq"],
                        batch_data["tmask"],
                        batch_data["choices_ids"]
                    )
            
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = batch_data["labels"].cpu().numpy()
            
            all_probabilities.append(probabilities)
            all_predictions.append(predictions)
            all_labels.append(labels)
    
    all_probabilities = np.vstack(all_probabilities)
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    try:
        labels_one_hot = np.eye(all_probabilities.shape[1])[all_labels]
        roc_auc = roc_auc_score(labels_one_hot, all_probabilities, average='weighted', multi_class='ovr')
    except:
        roc_auc = 0.5  # If calculation fails (e.g., only one class)
    
    r2 = r2_score(all_labels, all_predictions)
    
    cm = confusion_matrix(all_labels, all_predictions)
    
    prediction_entropy = calculate_prediction_entropy(all_probabilities)
    confidence_scores = calculate_confidence(all_probabilities)
    
    # 計算能耗
    if energy_forward_fn:
        # 使用第一個batch來估算能耗
        dataloader_iter = iter(dataloader)
        first_batch = next(dataloader_iter)
        # 確保 batch 在正確的 device 上 (evaluate_ann_energy 會處理)
        energy_report = evaluate_ann_energy(
            model, first_batch, device, forward_fn=energy_forward_fn
        )
    else:
        energy_report = EnergyReport()
    
    # 計算效率指標
    macs_per_accuracy = (
        energy_report.macs_per_sample / accuracy 
        if energy_report.macs_per_sample and accuracy > 0 
        else float('inf')
    )
    energy_per_accuracy = (
        energy_report.ann_energy_per_sample / accuracy 
        if energy_report.ann_energy_per_sample and accuracy > 0 
        else float('inf')
    )
    
    return PerformanceMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        roc_auc=roc_auc,
        r2_score=r2,
        confusion_matrix=cm,
        prediction_entropy=prediction_entropy,
        confidence_scores=confidence_scores,
        energy_report=energy_report,
        macs_per_accuracy=macs_per_accuracy,
        energy_per_accuracy=energy_per_accuracy
    )

def plot_performance_comparison(
    metrics_dict: Dict[str, PerformanceMetrics],
    save_path: Optional[str] = None
):
    """Plot performance comparison of multiple models"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    model_names = list(metrics_dict.keys())
    
    # 1. Accuracy comparison
    accuracies = [m.accuracy for m in metrics_dict.values()]
    axes[0, 0].bar(model_names, accuracies)
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    
    # 2. ROC AUC comparison
    roc_aucs = [m.roc_auc for m in metrics_dict.values()]
    axes[0, 1].bar(model_names, roc_aucs)
    axes[0, 1].set_title('ROC AUC Comparison')
    axes[0, 1].set_ylabel('ROC AUC')
    
    # 3. Energy efficiency comparison
    energy_per_acc = [m.energy_per_accuracy for m in metrics_dict.values()]
    axes[0, 2].bar(model_names, energy_per_acc)
    axes[0, 2].set_title('Energy per Accuracy')
    axes[0, 2].set_ylabel('Joules per Accuracy Unit')
    
    # 4. Prediction entropy comparison
    entropies = [m.prediction_entropy for m in metrics_dict.values()]
    axes[1, 0].bar(model_names, entropies)
    axes[1, 0].set_title('Prediction Entropy')
    axes[1, 0].set_ylabel('Entropy')
    
    # 5. MACs comparison
    macs = [m.energy_report.macs_per_sample or 0 for m in metrics_dict.values()]
    axes[1, 1].bar(model_names, macs)
    axes[1, 1].set_title('MACs per Sample')
    axes[1, 1].set_ylabel('MACs')
    
    # 6. F1 score comparison
    f1_scores = [m.f1_score for m in metrics_dict.values()]
    axes[1, 2].bar(model_names, f1_scores)
    axes[1, 2].set_title('F1 Score Comparison')
    axes[1, 2].set_ylabel('F1 Score')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

# SNN energy meter (unchanged)
class SNNEnergyMeter:
    def __init__(self, fan_out: Dict[str, int]):
        self.fan_out = fan_out
        self.spike_counts: Dict[str, int] = {k: 0 for k in fan_out.keys()}

    def add_spikes(self, name: str, spikes: torch.Tensor):
        if name not in self.spike_counts:
            raise KeyError(f"Unknown layer name for energy meter: {name}")
        self.spike_counts[name] += int(spikes.detach().sum().item())

    def total_sops(self) -> int:
        total = 0
        for name, n_spike in self.spike_counts.items():
            fan = self.fan_out[name]
            total += n_spike * fan
        return total

    def energy(self, e_leak: float = 0.0) -> float:
        return self.total_sops() * ENERGY_PER_SOP + e_leak