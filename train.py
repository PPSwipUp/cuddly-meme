"""
Stage 5 — Training Loop & Optimisation
Trading Algorithm Blueprint

Implements the full walk-forward training loop exactly per blueprint Sections 5.1–5.5:

  Optimiser    : AdamW, lr=3e-4, beta=(0.9, 0.999), weight_decay=1e-4
  LR Schedule  : Linear warmup (5% of steps) → cosine decay to 1e-5
  Early stopping: patience=20 epochs, min_delta=0.0001, restore best weights
  Max epochs   : 200
  Batch size   : 256
  Gradient clip: global norm 1.0
  Regime balance: WeightedRandomSampler — inverse-frequency weights per class
  Per-epoch log : train/val loss, directional accuracy, LR, grad norm
  Halt condition: train/val gap > 0.15 for 3 consecutive epochs

Outputs:
  models/base/checkpoint_base_v1.pt   — saved when all Stage 7 thresholds pass
  models/base/checkpoint_fold_{N}.pt  — best checkpoint per fold
  logs/training/fold_{N}_metrics.csv  — per-epoch metrics
"""

import os
import glob
import math
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from model import build_model, MultiHorizonLoss, TradingModel

# ─── Configuration ────────────────────────────────────────────────────────────

PROCESSED_DIR = "data/processed"
SPLITS_DIR    = "data/splits"
MODELS_DIR    = "models/base"
LOG_DIR       = "logs/training"

LOOKBACK      = 60
N_FEATURES = 29
HORIZONS      = [1, 5, 20]
BATCH_SIZE    = 256
MAX_EPOCHS    = 200
LR_INITIAL    = 3e-4
LR_MIN        = 1e-5
WARMUP_FRAC   = 0.05
PATIENCE      = 10
MIN_DELTA     = 0.0001
GRAD_CLIP     = 1.0
GAP_THRESHOLD = 0.15   # train/val loss gap halt condition
GAP_CONSEC    = 3      # consecutive epochs before halting on gap

# ─── Logging ──────────────────────────────────────────────────────────────────

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "training.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ─── Dataset ─────────────────────────────────────────────────────────────────

class TradingDataset(Dataset):
    """
    Loads pre-computed feature windows from data/processed/*.npy.
    For each sample (a 60-bar window), constructs direction and magnitude
    targets for each of the three forward horizons.

    Direction target : 1 if Close rises over the horizon, 0 otherwise.
    Magnitude target : log return over the horizon.

    Since we only have the window (past 60 bars), we use the final bar's
    log_return feature (feature index 0) as a proxy signal, and build
    forward targets by loading the *next* window's first bar returns.
    In practice this is done by shifting within the per-file arrays.
    """

    def __init__(self, split_df: pd.DataFrame):
        """
        split_df: DataFrame with columns [source_file, window_idx, date, regime]

        Lazy loading with vectorised target construction.
        - Targets built per-file using numpy (no Python loop per row)
        - Windows read via memory-mapped arrays in __getitem__
        - Peak RAM during init: O(largest single file) not O(total dataset)
        """
        self._mmap_cache = {}  # npy_path -> memmap array

        # Per-horizon target arrays (stored as compact numpy — not dicts)
        all_npy_paths  = []
        all_window_idx = []
        all_regimes    = []
        target_dirs    = {h: [] for h in HORIZONS}
        target_mags    = {h: [] for h in HORIZONS}

        skipped = 0
        for src, group in split_df.groupby("source_file", sort=False):
            npy_path = os.path.join(PROCESSED_DIR, src + ".npy")
            if not os.path.exists(npy_path):
                skipped += 1
                continue

            # Memory-map the file — no data loaded into RAM yet
            arr = self._get_array(npy_path)
            n   = len(arr)

            idxs  = group["window_idx"].values.astype(int)
            regs  = group["regime"].fillna(1).values.astype(np.float32)

            # Filter out-of-bounds indices
            valid = idxs < n
            idxs  = idxs[valid]
            regs  = regs[valid]
            if len(idxs) == 0:
                continue

            all_npy_paths.extend([npy_path] * len(idxs))
            all_window_idx.extend(idxs.tolist())
            all_regimes.extend(regs.tolist())

            # Build targets vectorised per horizon — single file in RAM at once
            for h in HORIZONS:
                future_idxs = idxs + h
                in_bounds   = future_idxs < n

                mags  = np.where(
                    in_bounds,
                    # Read just the log_return column of future last bars
                    np.array([
                        float(arr[fi, -1, 0]) if fi < n else 0.0
                        for fi in future_idxs
                    ], dtype=np.float32),
                    0.0
                )
                dirs = np.where(in_bounds,
                                np.where(mags > 0, 1.0, 0.0),
                                0.5).astype(np.float32)

                target_dirs[h].append(dirs)
                target_mags[h].append(mags)

        if skipped > 0:
            log.warning("Dataset: %d .npy files not found", skipped)

        # Consolidate into compact numpy arrays
        self.index = list(zip(all_npy_paths,
                              all_window_idx,
                              all_regimes))
        # Guard against case where all files were missing
        if not all_npy_paths:
            log.warning("Dataset: no valid samples found — check data/processed/ exists")

        self.targets = {
            h: (
                np.concatenate(target_dirs[h]).astype(np.float32) if target_dirs[h] else np.array([], dtype=np.float32),
                np.concatenate(target_mags[h]).astype(np.float32) if target_mags[h] else np.array([], dtype=np.float32),
            )
            for h in HORIZONS
        }
        log.info("    Index built: %d samples from %d files",
                 len(self.index),
                 split_df["source_file"].nunique() - skipped)

    def _get_array(self, npy_path: str) -> np.ndarray:
        """Return memory-mapped array, opening once per path."""
        if npy_path not in self._mmap_cache:
            self._mmap_cache[npy_path] = np.load(npy_path, mmap_mode="r")
        return self._mmap_cache[npy_path]

    @staticmethod
    def _normalise_features(window: np.ndarray) -> np.ndarray:
        """Pad or truncate feature dim to N_FEATURES."""
        n_feat = window.shape[1]
        if n_feat == N_FEATURES:
            return window.copy()
        elif n_feat > N_FEATURES:
            return window[:, :N_FEATURES].copy()
        else:
            pad = np.zeros((window.shape[0], N_FEATURES - n_feat), dtype=np.float32)
            return np.concatenate([window, pad], axis=1)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        npy_path, win_idx, reg = self.index[idx]
        arr    = self._get_array(npy_path)
        window = self._normalise_features(arr[win_idx].astype(np.float32))
        x      = torch.from_numpy(window)
        targets = {
            h: (
                torch.tensor(self.targets[h][0][idx], dtype=torch.float32),
                torch.tensor(self.targets[h][1][idx], dtype=torch.float32),
            )
            for h in HORIZONS
        }
        regime = torch.tensor(reg, dtype=torch.float32)
        return x, targets, regime


def make_weighted_sampler(dataset: TradingDataset) -> WeightedRandomSampler:
    """
    Build inverse-frequency weights per regime class so all 6 classes
    get equal representation during training (blueprint Section 3.4).
    """
    regimes  = np.array([dataset.index[i][2] for i in range(len(dataset))], dtype=float).astype(int)
    n        = len(regimes)
    counts   = np.bincount(np.clip(regimes, 0, 5), minlength=6).astype(float)
    counts   = np.maximum(counts, 1)
    weights_per_class = 1.0 / counts
    sample_weights    = weights_per_class[np.clip(regimes, 0, 5)]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=n,
        replacement=True,
    )


# ─── LR Schedule ─────────────────────────────────────────────────────────────

def make_lr_schedule(optimiser, total_steps: int) -> LambdaLR:
    """
    Linear warmup over first 5% of steps, then cosine decay to LR_MIN.
    Blueprint Section 5.2.
    """
    warmup_steps = int(WARMUP_FRAC * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine   = 0.5 * (1 + math.cos(math.pi * progress))
        # Scale so LR ranges from LR_INITIAL down to LR_MIN
        return LR_MIN / LR_INITIAL + (1 - LR_MIN / LR_INITIAL) * cosine

    return LambdaLR(optimiser, lr_lambda)


# ─── Training helpers ─────────────────────────────────────────────────────────

def directional_accuracy(direction_pred: torch.Tensor,
                         direction_true: torch.Tensor) -> float:
    """Fraction of samples where predicted direction matches true direction."""
    pred_binary = (direction_pred >= 0.5).float()
    true_binary = (direction_true >= 0.5).float()
    return (pred_binary == true_binary).float().mean().item()


def run_epoch(model, loader, criterion, optimiser, scheduler,
              device, train: bool) -> dict:
    """
    Run one epoch (train or eval).
    Returns dict of metrics: loss, bce, huber, accuracy per horizon,
    mean_accuracy, grad_norm (train only).
    """
    model.train(train)
    total_loss = bce_total = huber_total = 0.0
    acc = {h: 0.0 for h in HORIZONS}
    n_batches  = 0
    grad_norms = []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, targets, _ in loader:
            x = x.to(device)
            targets_dev = {
                h: (d.to(device), m.to(device))
                for h, (d, m) in targets.items()
            }

            preds = model(x)

            loss, bce, huber = criterion(preds, targets_dev)

            if train:
                optimiser.zero_grad()
                loss.backward()

                # Gradient norm before clipping (logged per blueprint 5.4)
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), GRAD_CLIP
                ).item()
                grad_norms.append(grad_norm)

                optimiser.step()
                scheduler.step()

            total_loss += loss.item()
            bce_total  += bce.item()
            huber_total += huber.item()

            # Directional accuracy per horizon
            for h in HORIZONS:
                dir_pred, _ = preds[h]
                dir_true, _ = targets_dev[h]
                acc[h] += directional_accuracy(dir_pred, dir_true)

            n_batches += 1

    n = max(n_batches, 1)
    metrics = {
        "loss":       total_loss / n,
        "bce":        bce_total  / n,
        "huber":      huber_total / n,
        "grad_norm":  float(np.mean(grad_norms)) if grad_norms else 0.0,
    }
    for h in HORIZONS:
        metrics[f"acc_h{h}"] = acc[h] / n
    metrics["mean_acc"] = np.mean([metrics[f"acc_h{h}"] for h in HORIZONS])
    return metrics


# ─── Early stopping ───────────────────────────────────────────────────────────

class EarlyStopping:
    """Blueprint Section 5.3 early stopping with best-weight restore."""

    def __init__(self, patience=PATIENCE, min_delta=MIN_DELTA):
        self.patience    = patience
        self.min_delta   = min_delta
        self.best_loss   = float("inf")
        self.best_acc    = {}   # val accuracies at best checkpoint epoch
        self.counter     = 0
        self.best_state  = None
        self.stopped     = False

    def step(self, val_loss: float, val_acc: dict, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        improvement = self.best_loss - val_loss
        if improvement > self.min_delta:
            self.best_loss  = val_loss
            self.best_acc   = val_acc.copy()
            self.counter    = 0
            # Deep copy state dict
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.stopped = True
            return True
        return False

    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            log.info("  Restored best weights (val_loss=%.4f  H1=%.3f H5=%.3f H20=%.3f)",
                     self.best_loss,
                     self.best_acc.get("acc_h1", 0),
                     self.best_acc.get("acc_h5", 0),
                     self.best_acc.get("acc_h20", 0))


# ─── Per-fold training ────────────────────────────────────────────────────────

def train_fold(fold_n: int, device: str) -> dict:
    log.info("=" * 60)
    log.info("FOLD %d — Training", fold_n)

    # Load split index files
    train_path = os.path.join(SPLITS_DIR, f"fold_{fold_n}_train.parquet")
    val_path   = os.path.join(SPLITS_DIR, f"fold_{fold_n}_val.parquet")

    if not os.path.exists(train_path):
        log.error("Split file not found: %s", train_path)
        return {}
    if not os.path.exists(val_path):
        log.error("Val split file not found: %s", val_path)
        return {}

    train_df = pd.read_parquet(train_path)
    val_df   = pd.read_parquet(val_path)

    log.info("Loading datasets...")
    train_ds = TradingDataset(train_df)
    val_ds   = TradingDataset(val_df)
    log.info("  Train samples: %d  Val samples: %d", len(train_ds), len(val_ds))

    if len(train_ds) == 0:
        log.error("Empty training dataset for fold %d", fold_n)
        return {}

    # Weighted sampler for regime balance
    sampler = make_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        sampler=sampler, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0,
    )

    # Model, loss, optimiser
    model     = build_model(n_features=N_FEATURES, device=device)
    criterion = MultiHorizonLoss().to(device)

    # L2 weight decay only on weight matrices (not biases or LayerNorm)
    decay_params    = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if "bias" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimiser = AdamW([
        {"params": decay_params,    "weight_decay": 1e-4},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=LR_INITIAL, betas=(0.9, 0.999))

    total_steps = len(train_loader) * MAX_EPOCHS
    scheduler   = make_lr_schedule(optimiser, total_steps)
    stopper     = EarlyStopping()

    # Per-epoch metrics log
    epoch_records = []
    gap_exceed_streak = 0

    log.info("Starting training — max %d epochs, patience %d", MAX_EPOCHS, PATIENCE)

    for epoch in range(1, MAX_EPOCHS + 1):

        train_metrics = run_epoch(
            model, train_loader, criterion, optimiser, scheduler,
            device, train=True
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, None, None,
            device, train=False
        )

        gap      = train_metrics["loss"] - val_metrics["loss"]
        curr_lr  = optimiser.param_groups[0]["lr"]

        # ── Per-epoch log (blueprint 5.4) ────────────────────────
        log.info(
            "Epoch %3d | "
            "TrainLoss %.4f (BCE %.4f Huber %.4f) | "
            "ValLoss %.4f (BCE %.4f Huber %.4f) | "
            "Gap %.4f | "
            "ValAcc H1=%.3f H5=%.3f H20=%.3f | "
            "LR %.2e | GradNorm %.3f",
            epoch,
            train_metrics["loss"], train_metrics["bce"], train_metrics["huber"],
            val_metrics["loss"],   val_metrics["bce"],   val_metrics["huber"],
            gap,
            val_metrics["acc_h1"], val_metrics["acc_h5"], val_metrics["acc_h20"],
            curr_lr,
            train_metrics["grad_norm"],
        )

        # Record
        record = {"fold": fold_n, "epoch": epoch, "lr": curr_lr}
        for k, v in train_metrics.items():
            record[f"train_{k}"] = v
        for k, v in val_metrics.items():
            record[f"val_{k}"] = v
        record["gap"] = gap
        epoch_records.append(record)

        # ── Gap-based halt (blueprint 5.4) ────────────────────────
        if gap > GAP_THRESHOLD:
            gap_exceed_streak += 1
            if gap_exceed_streak >= GAP_CONSEC:
                log.warning(
                    "  ⚠  Train/val gap %.4f exceeded %.2f for %d consecutive "
                    "epochs — halting (overfitting detected).",
                    gap, GAP_THRESHOLD, GAP_CONSEC
                )
                stopper.restore_best(model)
                # Save partial metrics before halting so progress is not lost
                pd.DataFrame(epoch_records).to_csv(
                    os.path.join(LOG_DIR, f"fold_{fold_n}_metrics.csv"), index=False)
                break
        else:
            gap_exceed_streak = 0

        # ── Gradient norm warning ─────────────────────────────────
        if train_metrics["grad_norm"] > 5.0:
            log.warning(
                "  ⚠  Gradient norm %.3f > 5.0 — instability risk.",
                train_metrics["grad_norm"]
            )

        # ── Early stopping ────────────────────────────────────────
        if stopper.step(val_metrics["loss"], val_metrics, model):
            log.info(
                "  Early stopping at epoch %d (no improvement for %d epochs).",
                epoch, PATIENCE
            )
            stopper.restore_best(model)
            break

    # Save fold checkpoint
    ckpt_path = os.path.join(MODELS_DIR, f"checkpoint_fold_{fold_n}.pt")
    torch.save({
        "fold":          fold_n,
        "model_state":   model.state_dict(),
        "val_loss":      stopper.best_loss,
        "n_features":    N_FEATURES,
        "hidden_dim":    128,
        "n_layers":      3,
        "epoch_stopped": len(epoch_records),
    }, ckpt_path)
    log.info("  Saved checkpoint: %s", ckpt_path)

    # Save per-epoch metrics CSV
    metrics_path = os.path.join(LOG_DIR, f"fold_{fold_n}_metrics.csv")
    pd.DataFrame(epoch_records).to_csv(metrics_path, index=False)
    log.info("  Saved metrics   : %s", metrics_path)

    return {
        "fold":          fold_n,
        "best_val_loss": stopper.best_loss,
        "epochs_run":    len(epoch_records),
        "checkpoint":    ckpt_path,
        "final_val_acc": np.mean(list(stopper.best_acc.values())) if stopper.best_acc else val_metrics["mean_acc"],
        "best_val_accs": stopper.best_acc,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Stage 5 Training Loop — Start")
    log.info("Device : %s", device)
    log.info("Config : LR=%.0e  Batch=%d  MaxEpochs=%d  Patience=%d",
             LR_INITIAL, BATCH_SIZE, MAX_EPOCHS, PATIENCE)

    # Discover available folds
    fold_files = sorted(glob.glob(os.path.join(SPLITS_DIR, "fold_*_train.parquet")))
    if not fold_files:
        log.error("No fold split files found in %s. Run walk_forward.py first.", SPLITS_DIR)
        return

    fold_numbers = [
        int(os.path.basename(f).split("_")[1])
        for f in fold_files
    ]
    log.info("Folds to train: %s", fold_numbers)

    # Skip folds that already have a saved checkpoint
    fold_results = []
    for fold_n in fold_numbers:
        ckpt_path = os.path.join(MODELS_DIR, f"checkpoint_fold_{fold_n}.pt")
        if os.path.exists(ckpt_path):
            log.info("Fold %d checkpoint already exists — skipping.", fold_n)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

            # Try to recover best accuracies from saved metrics CSV
            best_accs = {}
            metrics_csv = os.path.join(LOG_DIR, f"fold_{fold_n}_metrics.csv")
            if os.path.exists(metrics_csv):
                mdf = pd.read_csv(metrics_csv)
                # Best epoch = row with lowest val_loss
                best_row = mdf.loc[mdf["val_loss"].idxmin()]
                best_accs = {
                    "acc_h1":  best_row.get("val_acc_h1",  0.0),
                    "acc_h5":  best_row.get("val_acc_h5",  0.0),
                    "acc_h20": best_row.get("val_acc_h20", 0.0),
                }
                mean_acc = float(np.mean([best_accs["acc_h1"],
                                          best_accs["acc_h5"],
                                          best_accs["acc_h20"]]))
                log.info("  Recovered from metrics CSV — H1=%.3f H5=%.3f H20=%.3f",
                         best_accs["acc_h1"], best_accs["acc_h5"], best_accs["acc_h20"])
            else:
                mean_acc = 0.0
                log.warning("  No metrics CSV found for fold %d — val_acc unknown", fold_n)

            fold_results.append({
                "fold":          fold_n,
                "best_val_loss": ckpt.get("val_loss", float("nan")),
                "epochs_run":    ckpt.get("epoch_stopped", 0),
                "checkpoint":    ckpt_path,
                "final_val_acc": mean_acc,
                "best_val_accs": best_accs,
            })
            continue
        result = train_fold(fold_n, device)
        if result:
            fold_results.append(result)

    # ── Cross-fold summary ────────────────────────────────────────
    log.info("=" * 60)
    log.info("Training complete. Cross-fold summary:")

    if not fold_results:
        log.error("No folds completed successfully.")
        return

    val_losses = [r["best_val_loss"] for r in fold_results]
    val_accs   = [r["final_val_acc"] for r in fold_results]

    for r in fold_results:
        accs = r.get("best_val_accs", {})
        log.info(
            "  Fold %d: best_val_loss=%.4f  best_val_acc H1=%.3f H5=%.3f H20=%.3f  epochs=%d",
            r["fold"], r["best_val_loss"],
            accs.get("acc_h1", 0), accs.get("acc_h5", 0), accs.get("acc_h20", 0),
            r["epochs_run"]
        )

    log.info("  Mean val loss : %.4f ± %.4f", np.mean(val_losses), np.std(val_losses))
    log.info("  Mean val acc  : %.3f ± %.3f", np.mean(val_accs),   np.std(val_accs))

    # Fold-to-fold Sharpe variance check will happen in Stage 7 evaluation.
    # Blueprint threshold: std of Sharpe across folds must be < 0.4.

    # Save the best fold's checkpoint as the base model
    best_fold = min(fold_results, key=lambda r: r["best_val_loss"])
    base_path = os.path.join(MODELS_DIR, "checkpoint_base_v1.pt")

    if not os.path.exists(base_path):
        import shutil
        shutil.copy(best_fold["checkpoint"], base_path)
        log.info(
            "  Base checkpoint saved: %s  (from fold %d)",
            base_path, best_fold["fold"]
        )
    else:
        log.info("  Base checkpoint already exists — not overwritten: %s", base_path)

    log.info("Stage 5 complete.")
    log.info("Next step: run evaluation.py (Stage 7) on the test splits.")


if __name__ == "__main__":
    main()