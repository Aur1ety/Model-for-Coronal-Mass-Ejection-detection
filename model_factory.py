"""
model_factory.py
================
Temporal Convolutional Network (TCN) Architecture for CME Detection
Aditya-L1 ASPEX-SWIS Pipeline v2.0

Architecture highlights:
  - Dilated causal convolutions for long-range temporal receptive field
  - Residual blocks with skip connections (no vanishing gradients)
  - Weighted Binary Cross-Entropy loss (handles 95% quiet / 5% CME imbalance)
  - Full W&B experiment tracking with hyperparameter sweeps
  - Model serialisation to ONNX for cross-platform deployment

Author : CME Detection Pipeline v2.0
Python : 3.10+ | PyTorch 2.x
"""

from __future__ import annotations

import math
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score,
)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("model_factory")

# ---------------------------------------------------------------------------
# SECTION 1: TCN BUILDING BLOCKS
# ---------------------------------------------------------------------------

class CausalConv1d(nn.Module):
    """
    1-D causal dilated convolution.

    'Causal' means the output at time t depends ONLY on inputs ≤ t —
    no future data leaks into the prediction. Dilation expands the
    effective receptive field exponentially without increasing parameters.

    Receptive field = 1 + (kernel_size - 1) × dilation
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int = 3,
        dilation:     int = 1,
        dropout:      float = 0.1,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation  # causal padding

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation,
            padding=self.padding,
        )
        self.norm    = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.act     = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        out = self.conv(x)
        # Remove future leakage (causal: slice off right padding)
        out = out[:, :, : x.size(2)]
        # LayerNorm expects (batch, time, channels)
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        out = self.act(out)
        out = self.dropout(out)
        return out


class ResidualTCNBlock(nn.Module):
    """
    TCN Residual Block: two stacked causal-dilated convolutions + skip connection.

    Skip connection design:
      - If in_channels == out_channels: identity skip
      - Otherwise: 1×1 conv to match dimensions

    This mirrors the ResNet principle: the block learns the RESIDUAL
    (what to ADD to the input), not a full transformation. This makes
    gradient flow trivial even with 8+ stacked blocks.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int   = 3,
        dilation:     int   = 1,
        dropout:      float = 0.1,
    ) -> None:
        super().__init__()

        self.conv1 = CausalConv1d(in_channels,  out_channels, kernel_size, dilation, dropout)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation, dropout)

        # Skip connection
        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return self.act(out + residual)


# ---------------------------------------------------------------------------
# SECTION 2: FULL TCN MODEL
# ---------------------------------------------------------------------------

class CMEDetectorTCN(nn.Module):
    """
    Temporal Convolutional Network for CME binary detection.

    Architecture
    ------------
    Input  → n_blocks × ResidualTCNBlock (exponentially increasing dilation)
           → Global Average Pooling
           → Classifier head (FC → Dropout → FC → Sigmoid)
    Output → scalar probability [0, 1]

    Dilation schedule
    -----------------
    Block 0: dilation=1  → looks back  2 steps
    Block 1: dilation=2  → looks back  4 steps
    Block 2: dilation=4  → looks back  8 steps
    Block k: dilation=2^k

    With n_blocks=8 and kernel_size=3:
    Total receptive field = Σ(k=0..7) 2×2^k × (kernel_size-1)
                          = 2 × (2^8 - 1) × 2 = 1020 steps
    At 5-min cadence → 1020 × 5 min ≈ 85 hours of context!

    Parameters
    ----------
    n_features   : number of input physics features
    n_filters    : base number of TCN filters (width)
    kernel_size  : conv kernel size
    n_blocks     : number of residual blocks (controls receptive field depth)
    dropout      : dropout probability
    """

    def __init__(
        self,
        n_features:  int,
        n_filters:   int   = 64,
        kernel_size: int   = 3,
        n_blocks:    int   = 6,
        dropout:     float = 0.2,
    ) -> None:
        super().__init__()

        self.receptive_field = self._calc_rf(kernel_size, n_blocks)
        logger.info(
            "TCN receptive field = %d time-steps ≈ %.1f hours (5-min cadence)",
            self.receptive_field, self.receptive_field * 5 / 60,
        )

        # Input projection
        self.input_proj = nn.Conv1d(n_features, n_filters, kernel_size=1)

        # Residual blocks with exponentially increasing dilation
        blocks = []
        for i in range(n_blocks):
            dilation  = 2 ** i
            in_ch     = n_filters
            out_ch    = n_filters
            blocks.append(
                ResidualTCNBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )
        self.tcn_blocks = nn.Sequential(*blocks)

        # Classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # (batch, filters, 1) → global avg
            nn.Flatten(),             # (batch, filters)
            nn.Linear(n_filters, n_filters // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_filters // 2, 1),
            # NOTE: Sigmoid omitted here — BCEWithLogitsLoss is numerically
            # more stable and is used during training. Apply sigmoid at inference.
        )

    @staticmethod
    def _calc_rf(kernel_size: int, n_blocks: int) -> int:
        """Calculate the total temporal receptive field in time-steps."""
        return sum((kernel_size - 1) * (2 ** i) for i in range(n_blocks)) + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, n_features)  [standard ML convention]
        Returns logits: (batch,)
        """
        # PyTorch Conv1d expects (batch, channels, time)
        x = x.transpose(1, 2)          # → (batch, n_features, seq_len)
        x = self.input_proj(x)          # → (batch, n_filters, seq_len)
        x = self.tcn_blocks(x)          # → (batch, n_filters, seq_len)
        logits = self.head(x).squeeze(1)  # → (batch,)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Inference helper: returns probabilities via sigmoid."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


# ---------------------------------------------------------------------------
# SECTION 3: WEIGHTED LOSS & METRICS
# ---------------------------------------------------------------------------

def compute_pos_weight(y_train: np.ndarray) -> float:
    """
    Compute pos_weight = (# negatives) / (# positives).

    This scalar is passed to BCEWithLogitsLoss. For a 95/5 split:
      pos_weight ≈ 19
    → The model now penalises missing a CME 19× more than a false alarm.
    """
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    if n_pos == 0:
        logger.warning("No positive samples in training set! pos_weight set to 1.0")
        return 1.0
    weight = n_neg / n_pos
    logger.info("Class imbalance: %d pos, %d neg → pos_weight=%.2f", int(n_pos), int(n_neg), weight)
    return float(weight)


def evaluate_model(
    model:   CMEDetectorTCN,
    loader:  DataLoader,
    device:  torch.device,
    threshold: float = 0.5,
) -> dict:
    """
    Compute binary classification metrics on a DataLoader.

    Returns dict with: accuracy, precision, recall, f1, roc_auc, avg_precision
    """
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            probs   = torch.sigmoid(model(X_batch)).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())

    probs  = np.array(all_probs)
    labels = np.array(all_labels).astype(int)
    preds  = (probs >= threshold).astype(int)

    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    metrics = {
        "accuracy":      report["accuracy"],
        "precision":     report.get("1", {}).get("precision", 0.0),
        "recall":        report.get("1", {}).get("recall",    0.0),
        "f1":            report.get("1", {}).get("f1-score",  0.0),
        "roc_auc":       roc_auc_score(labels, probs)       if labels.sum() > 0 else 0.0,
        "avg_precision": average_precision_score(labels, probs) if labels.sum() > 0 else 0.0,
    }
    return metrics


# ---------------------------------------------------------------------------
# SECTION 4: TRAINING LOOP
# ---------------------------------------------------------------------------

def train_model(
    splits:        dict,
    # Architecture hyperparams
    n_filters:     int   = 64,
    kernel_size:   int   = 3,
    n_blocks:      int   = 6,
    dropout:       float = 0.2,
    # Training hyperparams
    epochs:        int   = 50,
    batch_size:    int   = 128,
    learning_rate: float = 3e-4,
    weight_decay:  float = 1e-4,
    patience:      int   = 10,    # early stopping patience
    threshold:     float = 0.5,
    # I/O
    checkpoint_dir: str  = "./checkpoints",
    use_wandb:      bool = False,
    wandb_project:  str  = "aditya-l1-cme-v2",
    wandb_run_name: str  = "tcn-run",
) -> tuple[CMEDetectorTCN, dict]:
    """
    Full training loop with early stopping, LR scheduling, and W&B logging.

    Returns
    -------
    (trained_model, best_metrics_dict)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    # --- W&B init ---
    if use_wandb and HAS_WANDB:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "n_filters": n_filters, "kernel_size": kernel_size,
                "n_blocks": n_blocks, "dropout": dropout,
                "epochs": epochs, "batch_size": batch_size,
                "lr": learning_rate, "weight_decay": weight_decay,
                "patience": patience,
            },
            reinit=True,
        )

    # --- Data loaders ---
    def to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32)

    train_ds = TensorDataset(to_tensor(splits["X_train"]), to_tensor(splits["y_train"]))
    val_ds   = TensorDataset(to_tensor(splits["X_val"]),   to_tensor(splits["y_val"]))
    test_ds  = TensorDataset(to_tensor(splits["X_test"]),  to_tensor(splits["y_test"]))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    # --- Model ---
    n_features = splits["feature_dim"]
    model      = CMEDetectorTCN(n_features, n_filters, kernel_size, n_blocks, dropout).to(device)
    n_params   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %s | Receptive field: %d steps", f"{n_params:,}", model.receptive_field)

    # --- Loss, optimiser, scheduler ---
    pos_weight   = compute_pos_weight(splits["y_train"])
    pos_w_tensor = torch.tensor([pos_weight], device=device)
    criterion    = nn.BCEWithLogitsLoss(pos_weight=pos_w_tensor)

    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs, eta_min=1e-6)

    # --- Training loop ---
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_val_f1   = -1.0
    patience_ctr  = 0
    best_metrics  = {}
    history       = {"train_loss": [], "val_loss": [], "val_f1": [], "val_roc_auc": []}

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimiser.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(train_ds)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits   = model(X_batch)
                val_loss += criterion(logits, y_batch).item() * len(X_batch)
        val_loss /= len(val_ds)

        val_metrics = evaluate_model(model, val_loader, device, threshold)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_metrics["f1"])
        history["val_roc_auc"].append(val_metrics["roc_auc"])

        log_str = (
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | ROC-AUC: {val_metrics['roc_auc']:.4f} | "
            f"LR: {current_lr:.2e}"
        )
        logger.info(log_str)

        if use_wandb and HAS_WANDB:
            wandb.log({
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                "val_f1": val_metrics["f1"], "val_roc_auc": val_metrics["roc_auc"],
                "val_precision": val_metrics["precision"], "val_recall": val_metrics["recall"],
                "learning_rate": current_lr,
            })

        # ---- Early stopping & checkpoint ----
        if val_metrics["f1"] > best_val_f1:
            best_val_f1  = val_metrics["f1"]
            best_metrics = val_metrics.copy()
            patience_ctr = 0
            ckpt_path = Path(checkpoint_dir) / "best_model.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "optimiser_state": optimiser.state_dict(),
                        "val_f1": best_val_f1, "config": {
                            "n_features": n_features, "n_filters": n_filters,
                            "kernel_size": kernel_size, "n_blocks": n_blocks,
                            "dropout": dropout,
                        }}, ckpt_path)
            logger.info("  ✓ New best model saved → %s (F1=%.4f)", ckpt_path, best_val_f1)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                logger.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch, patience)
                break

    # ---- Final test evaluation ----
    logger.info("\n=== Loading best checkpoint for test evaluation ===")
    ckpt = torch.load(Path(checkpoint_dir) / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_metrics = evaluate_model(model, test_loader, device, threshold)
    logger.info("TEST RESULTS: %s", test_metrics)

    if use_wandb and HAS_WANDB:
        wandb.log({f"test_{k}": v for k, v in test_metrics.items()})
        # Log model artifact
        artifact = wandb.Artifact("cme-tcn-model", type="model")
        artifact.add_file(str(Path(checkpoint_dir) / "best_model.pt"))
        wandb.log_artifact(artifact)
        wandb.finish()

    return model, {**best_metrics, "test": test_metrics, "history": history}


# ---------------------------------------------------------------------------
# SECTION 5: MODEL LOADING & ONNX EXPORT
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> CMEDetectorTCN:
    """
    Load a saved TCN model from a checkpoint.

    Parameters
    ----------
    checkpoint_path : path to .pt file saved by train_model()
    device          : target device (defaults to CUDA if available)

    Returns
    -------
    CMEDetectorTCN in eval mode
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt["config"]

    model = CMEDetectorTCN(
        n_features  = cfg["n_features"],
        n_filters   = cfg["n_filters"],
        kernel_size = cfg["kernel_size"],
        n_blocks    = cfg["n_blocks"],
        dropout     = cfg["dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info(
        "Loaded CMEDetectorTCN from %s (epoch=%d, val_f1=%.4f)",
        checkpoint_path, ckpt.get("epoch", -1), ckpt.get("val_f1", 0.0),
    )
    return model


def export_onnx(
    model:          CMEDetectorTCN,
    seq_len:        int,
    n_features:     int,
    output_path:    str = "./checkpoints/cme_detector.onnx",
    device:         Optional[torch.device] = None,
) -> str:
    """
    Export model to ONNX for cross-platform / edge deployment.

    Usage after export:
        import onnxruntime as ort
        sess = ort.InferenceSession("cme_detector.onnx")
        out  = sess.run(None, {"input": x_np})[0]
    """
    if device is None:
        device = torch.device("cpu")

    model = model.to(device).eval()
    dummy = torch.randn(1, seq_len, n_features, device=device)

    torch.onnx.export(
        model, dummy, output_path,
        input_names=["input"], output_names=["logit"],
        dynamic_axes={"input": {0: "batch_size"}, "logit": {0: "batch_size"}},
        opset_version=17,
    )
    logger.info("ONNX model exported → %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# SECTION 6: HYPERPARAMETER SWEEP CONFIG FOR W&B
# ---------------------------------------------------------------------------

WANDB_SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val_f1", "goal": "maximize"},
    "parameters": {
        "n_filters":     {"values": [32, 64, 128]},
        "kernel_size":   {"values": [3, 5, 7]},
        "n_blocks":      {"values": [4, 6, 8]},
        "dropout":       {"min": 0.1, "max": 0.4},
        "learning_rate": {"min": 1e-4, "max": 1e-3, "distribution": "log_uniform_values"},
        "batch_size":    {"values": [64, 128, 256]},
    },
}

"""
To run a W&B sweep in Colab:

    import wandb
    from model_factory import WANDB_SWEEP_CONFIG, train_model

    def sweep_train():
        with wandb.init() as run:
            cfg    = wandb.config
            splits = ...  # load from data_pipeline
            train_model(splits, use_wandb=True, **dict(cfg))

    sweep_id = wandb.sweep(WANDB_SWEEP_CONFIG, project="aditya-l1-cme-v2")
    wandb.agent(sweep_id, sweep_train, count=30)
"""


if __name__ == "__main__":
    # Quick architecture sanity check (no real data needed)
    model = CMEDetectorTCN(n_features=6, n_filters=64, kernel_size=3, n_blocks=6)
    x     = torch.randn(8, 128, 6)      # batch=8, seq=128, features=6
    out   = model(x)
    print(f"Input : {x.shape}")
    print(f"Output: {out.shape}")         # should be (8,)
    print(f"Receptive field: {model.receptive_field} steps")
    probs = model.predict_proba(x)
    print(f"Probabilities range: [{probs.min():.4f}, {probs.max():.4f}]")
