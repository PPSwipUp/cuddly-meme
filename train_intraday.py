"""
Stage 5 (Intraday) — Model Training
=====================================
Trains a TCN model on 1m/5m/15m intraday data.

Key differences from the daily pipeline (train.py):
  - Single chronological split (70/15/15) instead of walk-forward folds.
    Walk-forward requires years of data; intraday only has 7-60 days.
  - Processes all three resolutions separately — one model per resolution.
  - Saves to models/intraday/ (completely separate from models/base/).
  - Uses the same TCN architecture and loss from model.py — no changes there.
  - Session features (Asian/London/NY overlap) are genuinely populated for
    intraday data, making the calendar feature set more informative than daily.

Retraining frequency:
  1m  → run weekly  (data window is only 7 days)
  5m  → run monthly (60-day window, more stable)
  15m → run monthly (same)

Usage:
  python train_intraday.py                   # train all three resolutions
  python train_intraday.py --resolution 5m   # train 5m model only
  python train_intraday.py --resolution 15m  # train 15m model only

Prerequisites:
  python collect_data_intraday.py
  python feature_engineering.py   # with intraday data in data/raw_intraday/
  (feature_engineering.py reads raw_dir from its config — point it at
   data/raw_intraday/ using the --raw_dir flag added below, or run
   feature_engineering.py --raw_dir data/raw_intraday
                           --output_dir data/processed_intraday)
"""

import os
import glob
import logging
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

from model import build_model, MultiHorizonLoss

# ─── Configuration ─────────────────────────────────────────────────────────────

PROCESSED_DIR = "data/processed_intraday"
MODELS_DIR    = "models/intraday"
LOG_DIR       = "logs/intraday"

N_FEATURES = 44
HORIZONS   = [1, 5, 20]
LOOKBACK   = 60

# Training hyperparameters — similar to daily but with adjustments for
# the higher bar frequency and shorter overall data window.
BATCH_SIZE   = 128    # reduced to limit peak RAM — 86k windows × 60 × 44 is heavy
MAX_EPOCHS   = 60
PATIENCE     = 15
LR           = 5e-5
LABEL_SMOOTH = 0.05

# Split ratios (chronological — never shuffle across time)
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# Test frac  = 1 - 0.70 - 0.15 = 0.15

RESOLUTIONS_TO_TRAIN = ["1m", "5m", "15m"]

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "train_intraday.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ─── Dataset ───────────────────────────────────────────────────────────────────

class IntradayDataset(Dataset):
    """
    Loads pre-built .npy windows for a specific resolution.
    Applies a chronological slice (train / val / test).
    """

    def __init__(self, resolution: str, split: str):
        """
        resolution : "1m", "5m", or "15m"
        split      : "train", "val", or "test"
        """
        pattern = os.path.join(PROCESSED_DIR, f"*_{resolution}_*.npy")
        npy_files = sorted(glob.glob(pattern))

        if not npy_files:
            raise FileNotFoundError(
                f"No processed .npy files found for resolution {resolution} "
                f"in {PROCESSED_DIR}. Run feature_engineering.py first with "
                f"--raw_dir data/raw_intraday --output_dir data/processed_intraday"
            )

        all_X      = []
        all_labels = []   # direction labels derived from next-bar returns

        for npy_path in npy_files:
            X   = np.load(npy_path, mmap_mode="r")
            n   = len(X)
            if n < LOOKBACK + max(HORIZONS) + 5:
                continue

            # Direction label: 1 if close[t+h] > close[t], else 0
            # We extract from the last timestep of each window's log_return
            # (feature index 0 = log_return, the most recent bar's return)
            # For horizon h, shift forward h positions
            labels = {}
            for h in HORIZONS:
                # log_return at position i+h is in window i+h, timestep -1, feature 0
                # We can only compute this for windows where i+h < n
                usable = n - h
                dir_h  = np.zeros(usable, dtype=np.float32)
                for i in range(usable):
                    # last bar of window i+h has the return that happened h bars after window i
                    future_return = float(X[i + h, -1, 0]) if (i + h) < n else 0.0
                    dir_h[i] = 1.0 if future_return > 0 else 0.0
                labels[h] = dir_h

            usable = min(len(v) for v in labels.values())
            X_usable = X[:usable]
            if len(X_usable) == 0:
                continue

            # Cap per-instrument windows to limit total RAM usage.
            # 5m data for 60 days gives ~17k windows per instrument;
            # with 15+ forex instruments that's 250k+ windows → OOM.
            # Cap at MAX_WINDOWS_PER_INST keeps peak RAM manageable.
            MAX_WINDOWS_PER_INST = 3000
            if len(X_usable) > MAX_WINDOWS_PER_INST:
                # Take the most recent windows — most relevant for intraday
                X_usable = X_usable[-MAX_WINDOWS_PER_INST:]
                for h in labels:
                    labels[h] = labels[h][-MAX_WINDOWS_PER_INST:]
                usable = MAX_WINDOWS_PER_INST

            all_X.append(X_usable)
            all_labels.append({h: labels[h][:usable] for h in HORIZONS})

        if not all_X:
            raise ValueError(f"No usable data for resolution {resolution}")

        # Concatenate across instruments then free the per-instrument lists
        X_all   = np.concatenate(all_X,  axis=0)
        lab_all = {h: np.concatenate([l[h] for l in all_labels], axis=0)
                   for h in HORIZONS}
        del all_X, all_labels   # free RAM before slicing and converting to tensors

        n_total = len(X_all)
        n_train = int(n_total * TRAIN_FRAC)
        n_val   = int(n_total * VAL_FRAC)

        if split == "train":
            sl = slice(0, n_train)
        elif split == "val":
            sl = slice(n_train, n_train + n_val)
        else:
            sl = slice(n_train + n_val, n_total)

        # Store as float16 to halve RAM. Cast to float32 in __getitem__.
        self.X      = torch.from_numpy(self._norm(X_all[sl])).half()
        self.labels = {h: torch.from_numpy(lab_all[h][sl]) for h in HORIZONS}
        del X_all, lab_all   # free the full arrays once sliced
        log.info("  %s %s: %d windows  (RAM: ~%.0fMB)",
                 resolution, split, len(self.X),
                 len(self.X) * 60 * 44 * 2 / 1024**2)

    def _norm(self, X: np.ndarray) -> np.ndarray:
        """Pad or truncate to N_FEATURES."""
        n = X.shape[2]
        if n == N_FEATURES:
            return X.astype(np.float32)
        if n > N_FEATURES:
            return X[:, :, :N_FEATURES].astype(np.float32)
        pad = np.zeros((X.shape[0], X.shape[1], N_FEATURES - n), dtype=np.float32)
        return np.concatenate([X.astype(np.float32), pad], axis=2)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].float()   # cast from float16 storage to float32 for model
        t = {h: (self.labels[h][idx], torch.tensor(0.0)) for h in HORIZONS}
        return x, t


# ─── Training loop ─────────────────────────────────────────────────────────────

def train_resolution(resolution: str, device: str,
                     _force_retrain: bool = False) -> dict:
    """
    Train one model for one resolution.
    Returns dict with training results.
    """
    log.info("=" * 60)
    log.info("Training intraday model — resolution: %s", resolution)
    log.info("=" * 60)

    # Skip if checkpoint already exists — pass --force to override
    ckpt_exists = os.path.join(MODELS_DIR, f"checkpoint_intraday_{resolution}_v1.pt")
    if os.path.exists(ckpt_exists) and not _force_retrain:
        log.info("  [SKIP] Checkpoint already exists: %s", os.path.basename(ckpt_exists))
        log.info("  Pass --force to retrain this resolution.")
        # Return the existing checkpoint's metadata so the summary still shows it
        try:
            import torch as _torch
            existing = _torch.load(ckpt_exists, map_location="cpu", weights_only=True)
            return {
                "resolution": resolution,
                "val_loss":   existing.get("val_loss", 0),
                "test_h1":    existing.get("test_acc", {}).get(1, 0),
                "test_h5":    existing.get("test_acc", {}).get(5, 0),
                "test_h20":   existing.get("test_acc", {}).get(20, 0),
                "checkpoint": ckpt_exists,
                "skipped":    True,
            }
        except Exception:
            return {"resolution": resolution, "skipped": True}

    try:
        train_ds = IntradayDataset(resolution, "train")
        val_ds   = IntradayDataset(resolution, "val")
        test_ds  = IntradayDataset(resolution, "test")
    except (FileNotFoundError, ValueError) as e:
        log.error("Cannot train %s: %s", resolution, e)
        return {}

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    model     = build_model(n_features=N_FEATURES, device=device)
    criterion = MultiHorizonLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_val_loss = float("inf")
    best_state    = None
    patience_ctr  = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for x, targets in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            preds = model(x)
            # MultiHorizonLoss.forward(predictions, targets)
            # predictions: {h: (dir_pred, mag_pred)}
            # targets:     {h: (dir_true, mag_true)}
            # Magnitude targets are zeros — we only supervise direction here.
            _preds   = {h: preds[h] for h in HORIZONS}
            _targets = {h: (targets[h][0].to(device),
                            torch.zeros_like(targets[h][0]).to(device))
                        for h in HORIZONS}
            loss, _, _ = criterion(_preds, _targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── Validate ────────────────────────────────────────────────────────
        model.eval()
        val_loss  = 0.0
        val_corr  = {h: 0 for h in HORIZONS}
        val_total = 0

        with torch.no_grad():
            for x, targets in val_loader:
                x = x.to(device)
                preds = model(x)
                _preds   = {h: preds[h] for h in HORIZONS}
                _targets = {h: (targets[h][0].to(device),
                                torch.zeros_like(targets[h][0]).to(device))
                            for h in HORIZONS}
                loss, _, _ = criterion(_preds, _targets)
                val_loss  += loss.item()
                for h in HORIZONS:
                    pred_dir = (preds[h][0].cpu() >= 0.5).float()
                    true_dir = (targets[h][0] >= 0.5).float()
                    val_corr[h]  += (pred_dir == true_dir).sum().item()
                val_total += len(x)

        val_loss /= len(val_loader)
        acc = {h: val_corr[h] / val_total for h in HORIZONS}

        scheduler.step(val_loss)

        log.info("  Epoch %3d | train=%.4f | val=%.4f | "
                 "H1=%.3f H5=%.3f H20=%.3f",
                 epoch, train_loss, val_loss,
                 acc[1], acc[5], acc[20])

        # ── Early stopping ─────────────────────────────────────────────────
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                log.info("  Early stopping at epoch %d", epoch)
                break

    # ── Evaluate on test set ───────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    test_corr  = {h: 0 for h in HORIZONS}
    test_total = 0

    with torch.no_grad():
        for x, targets in test_loader:
            x = x.to(device)
            preds = model(x)
            for h in HORIZONS:
                pred_dir = (preds[h][0].cpu() >= 0.5).float()
                true_dir = (targets[h][0] >= 0.5).float()
                test_corr[h]  += (pred_dir == true_dir).sum().item()
            test_total += len(x)

    test_acc = {h: test_corr[h] / test_total for h in HORIZONS}

    log.info("─" * 50)
    log.info("Test accuracy — H1=%.3f  H5=%.3f  H20=%.3f",
             test_acc[1], test_acc[5], test_acc[20])
    log.info("Best val loss: %.4f", best_val_loss)

    # ── Save checkpoint ────────────────────────────────────────────────────
    ckpt_path = os.path.join(MODELS_DIR, f"checkpoint_intraday_{resolution}_v1.pt")
    torch.save({
        "resolution":  resolution,
        "n_features":  N_FEATURES,
        "val_loss":    best_val_loss,
        "test_acc":    test_acc,
        "model_state": best_state if best_state else model.state_dict(),
    }, ckpt_path)
    log.info("Checkpoint saved: %s", ckpt_path)

    return {
        "resolution": resolution,
        "val_loss":   best_val_loss,
        "test_h1":    test_acc[1],
        "test_h5":    test_acc[5],
        "test_h20":   test_acc[20],
        "checkpoint": ckpt_path,
    }


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage 5 (Intraday) — Train TCN models on 1m/5m/15m data"
    )
    parser.add_argument(
        "--resolution", default=None, choices=["1m", "5m", "15m"],
        help="Train only this resolution (default: all three)."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Retrain even if a checkpoint already exists for that resolution."
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Intraday Training — device=%s", device)

    resolutions = [args.resolution] if args.resolution else RESOLUTIONS_TO_TRAIN

    results = []
    for res in resolutions:
        result = train_resolution(res, device, _force_retrain=args.force)
        if result:
            results.append(result)

    if results:
        log.info("=" * 60)
        log.info("INTRADAY TRAINING SUMMARY")
        log.info("%-6s  %-10s  %-6s  %-6s  %-6s",
                 "Res", "Val Loss", "H1", "H5", "H20")
        log.info("─" * 50)
        for r in results:
            if r.get("skipped"):
                log.info("%-6s  %-10s  %-6s  %-6s  %-6s  (skipped — checkpoint exists)",
                         r["resolution"], "—", "—", "—", "—")
            else:
                log.info("%-6s  %-10.4f  %-6.3f  %-6.3f  %-6.3f",
                         r["resolution"], r["val_loss"],
                         r["test_h1"], r["test_h5"], r["test_h20"])
        log.info("=" * 60)

        # Retraining reminder
        log.warning(
            "RETRAINING REMINDERS:\n"
            "  1m model  → retrain weekly  (data window is only 7 days)\n"
            "  5m model  → retrain monthly (60-day data window)\n"
            "  15m model → retrain monthly (60-day data window)\n"
            "  Command: python collect_data_intraday.py --refresh && "
            "python feature_engineering.py "
            "--raw_dir data/raw_intraday --output_dir data/processed_intraday && "
            "python train_intraday.py"
        )


if __name__ == "__main__":
    main()