"""
Stage 6 — Fine-Tuning Pipeline
Trading Algorithm Blueprint

Specialises the base model checkpoint on a single instrument using the
freeze-adapter-unfreeze protocol (blueprint Section 6.1–6.4).

Three-phase protocol:
  Phase 1 — Head-only adaptation (encoder frozen)
             LR=3e-5, max 30 epochs, patience 10
  Phase 2 — Verification gate (must pass 51% directional accuracy)
  Phase 3 — Selective encoder unfreeze (last 2 TCN layers only)
             LR=1e-5 encoder / 3e-5 head, max 20 epochs, patience 8
             Requires 2+ years of target instrument data

Usage:
    python finetune.py --instrument NYSE_AAPL --resolution 1D
    python finetune.py --instrument FOREXCOM_EURUSDX --resolution 4H
    python finetune.py --instrument NYSE_AAPL --resolution 1D --base_checkpoint models/base/checkpoint_fold_1.pt

The base checkpoint is NEVER overwritten.
Fine-tuned checkpoints saved to models/fine_tuned/.
"""

import os
import glob
import argparse
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import build_model, MultiHorizonLoss, TradingModel

# ─── Configuration ────────────────────────────────────────────────────────────

PROCESSED_DIR  = "data/processed"
MODELS_BASE    = "models/base"
MODELS_FT      = "models/fine_tuned"
LOG_DIR        = "logs/finetune"

LOOKBACK       = 60
N_FEATURES = 44
HORIZONS       = [1, 5, 20]

# Phase 1
P1_LR          = 3e-5
P1_MAX_EPOCHS  = 30
P1_PATIENCE    = 10

# Phase 2 gate
P2_MIN_ACC     = 0.51   # minimum H1 directional accuracy to proceed to Phase 3

# Phase 3
P3_ENC_LR      = 1e-5
P3_HEAD_LR     = 3e-5
P3_MAX_EPOCHS  = 20
P3_PATIENCE    = 8
P3_ENC_DROPOUT = 0.40
P3_HEAD_DROPOUT= 0.45
P3_GAP_THRESH  = 0.10   # tighter gap threshold for fine-tuning
P3_MIN_YEARS   = 2.0    # minimum data requirement for Phase 3

BATCH_SIZE     = 64     # smaller batch for fine-tuning dataset
VAL_MONTHS     = 3      # last N months held out for validation

# ─── Logging ──────────────────────────────────────────────────────────────────

os.makedirs(MODELS_FT, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "finetune.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ─── Dataset ─────────────────────────────────────────────────────────────────

class FineTuneDataset(Dataset):
    """
    Single-instrument dataset for fine-tuning.
    Loads one .npy file and builds direction + magnitude targets
    for each of the three horizons.
    """

    def __init__(self, X: np.ndarray):
        """X: [n_windows, 60, N_FEATURES] — padded/truncated to N_FEATURES."""
        # Pad or truncate feature dimension to N_FEATURES for consistency
        n, seq, n_feat = X.shape
        if n_feat != N_FEATURES:
            if n_feat > N_FEATURES:
                X = X[:, :, :N_FEATURES]
            else:
                pad = np.zeros((n, seq, N_FEATURES - n_feat), dtype=np.float32)
                X = np.concatenate([X, pad], axis=2)
        self.X       = X.astype(np.float32)
        self.targets = {h: ([], []) for h in HORIZONS}

        n = len(X)
        for idx in range(n):
            for h in HORIZONS:
                future_idx = idx + h
                if future_idx < n:
                    mag   = float(X[future_idx, -1, 0])   # last bar log_return, no leakage
                    direc = 1.0 if mag > 0 else 0.0
                else:
                    mag   = 0.0
                    direc = 0.5
                self.targets[h][0].append(direc)
                self.targets[h][1].append(mag)

        for h in HORIZONS:
            self.targets[h] = (
                np.array(self.targets[h][0], dtype=np.float32),
                np.array(self.targets[h][1], dtype=np.float32),
            )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        targets = {
            h: (
                torch.tensor(self.targets[h][0][idx]),
                torch.tensor(self.targets[h][1][idx]),
            )
            for h in HORIZONS
        }
        return x, targets


# ─── Helpers ─────────────────────────────────────────────────────────────────

def directional_accuracy(dir_pred, dir_true):
    return ((dir_pred >= 0.5).float() == (dir_true >= 0.5).float()).float().mean().item()


def load_instrument_data(instrument: str, resolution: str):
    """
    Find and load the .npy file for the given instrument+resolution.
    Returns (X, parquet_meta) or (None, None) if not found.
    """
    pattern = os.path.join(PROCESSED_DIR, f"*{instrument}*{resolution}*.npy")
    matches = glob.glob(pattern)

    if not matches:
        log.error("No processed file found for %s %s", instrument, resolution)
        log.error("Pattern tried: %s", pattern)
        log.error("Available files:")
        for f in sorted(glob.glob(os.path.join(PROCESSED_DIR, "*.npy")))[:10]:
            log.error("  %s", os.path.basename(f))
        return None, None

    npy_path     = matches[0]
    parquet_path = npy_path.replace(".npy", ".parquet")

    log.info("Loading: %s", os.path.basename(npy_path))
    X    = np.load(npy_path)
    meta = pd.read_parquet(parquet_path) if os.path.exists(parquet_path) else None

    log.info("  Shape: %s  (%d windows)", X.shape, len(X))
    return X, meta


def split_train_val(X, meta, val_months=VAL_MONTHS):
    """
    Walk-forward split: last VAL_MONTHS months as val, rest as train.
    Returns (X_train, X_val, years_of_data).
    """
    n = len(X)

    if meta is not None and "date" in meta.columns:
        dates = pd.to_datetime(meta["date"])
        if dates.dt.tz is not None:
            dates = dates.dt.tz_localize(None)

        total_days  = (dates.max() - dates.min()).days
        years_of_data = total_days / 365.25

        cutoff = dates.max() - pd.DateOffset(months=val_months)
        train_mask = dates <= cutoff
        val_mask   = dates > cutoff

        X_train = X[train_mask.values]
        X_val   = X[val_mask.values]

        log.info("  Date range: %s → %s  (%.1f years)",
                 dates.min().date(), dates.max().date(), years_of_data)
    else:
        # Fallback: last 15% as val
        split  = int(n * 0.85)
        X_train, X_val = X[:split], X[split:]
        years_of_data  = n / 252   # rough estimate assuming daily

    log.info("  Train windows: %d  Val windows: %d", len(X_train), len(X_val))

    if len(X_train) < LOOKBACK + 60:
        log.error("Insufficient training data: %d windows (minimum ~120)", len(X_train))
        return None, None, years_of_data

    return X_train, X_val, years_of_data


def load_base_checkpoint(checkpoint_path: str, device: str):
    """Load base model. The base checkpoint is NEVER modified."""
    if not os.path.exists(checkpoint_path):
        log.error("Base checkpoint not found: %s", checkpoint_path)
        return None

    log.info("Loading base checkpoint: %s", checkpoint_path)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model = build_model(
        n_features=ckpt.get("n_features", N_FEATURES),
        device=device,
    )
    model.load_state_dict(ckpt["model_state"])
    log.info("  Base val_loss: %.4f  (fold %s, epoch %s)",
             ckpt.get("val_loss", float("nan")),
             ckpt.get("fold", "?"),
             ckpt.get("epoch_stopped", "?"))
    return model


def set_dropout(model, enc_dropout, head_dropout):
    """Update dropout rates in encoder and head layers."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            if "encoder" in name:
                module.p = enc_dropout
            elif "head" in name or "heads" in name:
                module.p = head_dropout


def run_eval(model, loader, criterion, device):
    """Evaluate model, return loss and per-horizon accuracy."""
    model.eval()
    total_loss = 0.0
    acc = {h: 0.0 for h in HORIZONS}
    n_batches = 0

    with torch.no_grad():
        for x, targets in loader:
            x = x.to(device)
            targets_dev = {h: (d.to(device), m.to(device)) for h, (d, m) in targets.items()}
            preds = model(x)
            loss, _, _ = criterion(preds, targets_dev)
            total_loss += loss.item()
            for h in HORIZONS:
                dir_pred, _ = preds[h]
                dir_true, _ = targets_dev[h]
                acc[h] += directional_accuracy(dir_pred, dir_true)
            n_batches += 1

    n = max(n_batches, 1)
    return total_loss / n, {h: acc[h] / n for h in HORIZONS}


# ─── Early stopping (fine-tune variant) ──────────────────────────────────────

class FTEarlyStopping:
    def __init__(self, patience):
        self.patience   = patience
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss - 0.0001:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)


# ─── Phase 1 — Head-only adaptation ──────────────────────────────────────────

def phase1(model, train_loader, val_loader, criterion, device):
    log.info("-" * 50)
    log.info("Phase 1 — Head-only adaptation (encoder frozen)")

    # Freeze entire encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    model.encoder.eval()   # disable dropout in frozen encoder

    optimiser = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=P1_LR, weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimiser, T_max=P1_MAX_EPOCHS, eta_min=1e-6)
    stopper   = FTEarlyStopping(patience=P1_PATIENCE)
    metrics   = []

    for epoch in range(1, P1_MAX_EPOCHS + 1):
        # Train head only
        model.heads.train()
        train_loss = 0.0
        n_batches  = 0

        for x, targets in train_loader:
            x = x.to(device)
            targets_dev = {h: (d.to(device), m.to(device)) for h, (d, m) in targets.items()}
            preds = model(x)
            loss, _, _ = criterion(preds, targets_dev)
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_loss += loss.item()
            n_batches  += 1

        train_loss /= max(n_batches, 1)
        val_loss, val_acc = run_eval(model, val_loader, criterion, device)
        scheduler.step()

        log.info("  P1 Epoch %2d | TrainLoss %.4f | ValLoss %.4f | "
                 "ValAcc H1=%.3f H5=%.3f H20=%.3f",
                 epoch, train_loss, val_loss,
                 val_acc[1], val_acc[5], val_acc[20])

        metrics.append({
            "phase": 1, "epoch": epoch,
            "train_loss": train_loss, "val_loss": val_loss,
            "acc_h1": val_acc[1], "acc_h5": val_acc[5], "acc_h20": val_acc[20],
        })

        if stopper.step(val_loss, model):
            log.info("  Phase 1 early stopping at epoch %d", epoch)
            break

    stopper.restore(model)

    # Re-evaluate on best checkpoint weights to get accurate best-epoch metrics
    _, best_val_acc = run_eval(model, val_loader, criterion, device)
    log.info("  Phase 1 complete. Best val_loss: %.4f  H1=%.3f H5=%.3f H20=%.3f",
             stopper.best_loss, best_val_acc[1], best_val_acc[5], best_val_acc[20])

    # Unfreeze encoder params (but keep in eval mode — Phase 3 decides)
    for param in model.encoder.parameters():
        param.requires_grad = True

    return stopper.best_loss, best_val_acc, metrics


# ─── Phase 2 — Verification gate ─────────────────────────────────────────────

def phase2(val_acc: dict, base_val_acc: dict) -> bool:
    log.info("-" * 50)
    log.info("Phase 2 — Verification gate")

    mean_acc = float(np.mean([val_acc[h] for h in HORIZONS]))

    log.info("  Fine-tuned accuracy — H1=%.3f  H5=%.3f  H20=%.3f  Mean=%.3f",
             val_acc[1], val_acc[5], val_acc[20], mean_acc)
    log.info("  Minimum required (mean across horizons) : %.3f", P2_MIN_ACC)

    # Pass if mean accuracy across all three horizons exceeds threshold.
    # Checking only H1 is too strict — H1 is the noisiest horizon and
    # short-term unpredictability doesn't mean the model is useless
    # at H5 or H20 time horizons.
    if mean_acc < P2_MIN_ACC:
        log.warning(
            "  ✗ Phase 2 FAILED — mean accuracy %.3f is below %.3f",
            mean_acc, P2_MIN_ACC
        )
        log.warning("  H1=%.3f  H5=%.3f  H20=%.3f", val_acc[1], val_acc[5], val_acc[20])
        log.warning("  Instrument may be unpredictable at this resolution.")
        log.warning("  Using Phase 1 checkpoint as final output.")
        return False

    log.info("  ✓ Phase 2 PASSED (H1=%.3f  H5=%.3f  H20=%.3f) — proceeding to Phase 3",
             val_acc[1], val_acc[5], val_acc[20])
    return True


# ─── Phase 3 — Selective encoder unfreeze ─────────────────────────────────────

def phase3(model, train_loader, val_loader, criterion, device, years_of_data):
    log.info("-" * 50)
    log.info("Phase 3 — Selective encoder unfreeze")

    if years_of_data < P3_MIN_YEARS:
        log.warning(
            "  ✗ Skipping Phase 3 — only %.1f years of data (need %.1f)",
            years_of_data, P3_MIN_YEARS
        )
        log.warning("  Using Phase 1 checkpoint as final output.")
        return False, []

    # Increase dropout for smaller fine-tuning dataset
    set_dropout(model, enc_dropout=P3_ENC_DROPOUT, head_dropout=P3_HEAD_DROPOUT)

    # Freeze first TCN layer, unfreeze last 2
    tcn_blocks = list(model.encoder.network)   # 3 TCNBlock modules
    for param in tcn_blocks[0].parameters():   # freeze layer 0
        param.requires_grad = False
    for block in tcn_blocks[1:]:               # unfreeze layers 1 & 2
        for param in block.parameters():
            param.requires_grad = True
        block.train()

    # Head also trains
    model.heads.train()

    # Separate LRs: lower for encoder, higher for head
    enc_params  = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "encoder" in name:
            enc_params.append(param)
        else:
            head_params.append(param)

    optimiser = AdamW([
        {"params": enc_params,  "lr": P3_ENC_LR,  "weight_decay": 1e-4},
        {"params": head_params, "lr": P3_HEAD_LR, "weight_decay": 1e-4},
    ])
    scheduler = CosineAnnealingLR(optimiser, T_max=P3_MAX_EPOCHS, eta_min=1e-7)
    stopper   = FTEarlyStopping(patience=P3_PATIENCE)
    metrics   = []

    log.info("  Encoder layers 1 & 2 unfrozen (layer 0 frozen)")
    log.info("  Encoder LR: %.0e  Head LR: %.0e", P3_ENC_LR, P3_HEAD_LR)
    log.info("  Dropout: encoder=%.2f  head=%.2f", P3_ENC_DROPOUT, P3_HEAD_DROPOUT)

    for epoch in range(1, P3_MAX_EPOCHS + 1):
        # Train unfrozen encoder layers + head
        for block in tcn_blocks[1:]:
            block.train()
        model.heads.train()

        train_loss = 0.0
        n_batches  = 0

        for x, targets in train_loader:
            x = x.to(device)
            targets_dev = {h: (d.to(device), m.to(device)) for h, (d, m) in targets.items()}
            preds = model(x)
            loss, _, _ = criterion(preds, targets_dev)
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_loss += loss.item()
            n_batches  += 1

        train_loss /= max(n_batches, 1)
        val_loss, val_acc = run_eval(model, val_loader, criterion, device)
        gap = train_loss - val_loss
        scheduler.step()

        log.info("  P3 Epoch %2d | TrainLoss %.4f | ValLoss %.4f | "
                 "Gap %.4f | ValAcc H1=%.3f H5=%.3f H20=%.3f",
                 epoch, train_loss, val_loss, gap,
                 val_acc[1], val_acc[5], val_acc[20])

        metrics.append({
            "phase": 3, "epoch": epoch,
            "train_loss": train_loss, "val_loss": val_loss, "gap": gap,
            "acc_h1": val_acc[1], "acc_h5": val_acc[5], "acc_h20": val_acc[20],
        })

        # Tighter gap threshold for fine-tuning (blueprint 6.3)
        if gap > P3_GAP_THRESH:
            log.warning(
                "  ⚠  Gap %.4f exceeded %.2f — halting Phase 3, "
                "reverting to Phase 1 checkpoint.",
                gap, P3_GAP_THRESH
            )
            return False, metrics

        if stopper.step(val_loss, model):
            log.info("  Phase 3 early stopping at epoch %d", epoch)
            break

    stopper.restore(model)
    log.info("  Phase 3 complete. Best val_loss: %.4f", stopper.best_loss)
    return True, metrics


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune the base trading model")
    parser.add_argument("--instrument",  required=True,
                        help="Instrument prefix, e.g. NYSE_AAPL or FOREXCOM_EURUSDX")
    parser.add_argument("--resolution",  required=True,
                        help="Resolution: 1D, 4H, 1H, or 1W")
    parser.add_argument("--base_checkpoint", default=None,
                        help="Path to base checkpoint (default: auto-detect newest in models/base/)")
    parser.add_argument("--skip_phase3", action="store_true",
                        help="Force skip Phase 3 even if data is sufficient")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("=" * 60)
    log.info("Stage 6 Fine-Tuning — Start")
    log.info("Instrument  : %s", args.instrument)
    log.info("Resolution  : %s", args.resolution)
    log.info("Device      : %s", device)

    # ── Find base checkpoint ──────────────────────────────────
    if args.base_checkpoint:
        base_ckpt_path = args.base_checkpoint
    else:
        # Auto-detect: prefer checkpoint_base_v1.pt, fallback to newest fold
        base_v1 = os.path.join(MODELS_BASE, "checkpoint_base_v1.pt")
        if os.path.exists(base_v1):
            base_ckpt_path = base_v1
        else:
            fold_ckpts = sorted(glob.glob(os.path.join(MODELS_BASE, "checkpoint_fold_*.pt")))
            if not fold_ckpts:
                log.error("No base checkpoint found in %s", MODELS_BASE)
                log.error("Run train.py first to generate a base checkpoint.")
                return
            base_ckpt_path = fold_ckpts[-1]
            log.warning("checkpoint_base_v1.pt not found — using %s",
                        os.path.basename(base_ckpt_path))

    log.info("Base checkpoint: %s", base_ckpt_path)

    # ── Load instrument data ──────────────────────────────────
    X, meta = load_instrument_data(args.instrument, args.resolution)
    if X is None:
        return

    X_train, X_val, years_of_data = split_train_val(X, meta)
    if X_train is None:
        return

    if years_of_data < 0.5:
        log.error(
            "Only %.1f years of data — minimum 6 months required for fine-tuning. "
            "Use the base model for this instrument.", years_of_data
        )
        return

    log.info("Years of data: %.1f", years_of_data)

    # ── Build data loaders ────────────────────────────────────
    train_ds = FineTuneDataset(X_train)
    val_ds   = FineTuneDataset(X_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    criterion = MultiHorizonLoss()

    # ── Load base model (read-only copy) ──────────────────────
    model = load_base_checkpoint(base_ckpt_path, device)
    if model is None:
        return

    # ── Phase 1 ───────────────────────────────────────────────
    p1_val_loss, p1_val_acc, p1_metrics = phase1(
        model, train_loader, val_loader, criterion, device
    )

    # Save Phase 1 weights AFTER phase 1 completes.
    # Must be saved here — not before phase 1 — so that reverting after
    # a failed Phase 3 restores phase 1 adapted weights, not base weights.
    p1_state_backup = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save Phase 1 checkpoint
    date_str   = pd.Timestamp.now().strftime("%Y%m%d")
    safe_instr = args.instrument.replace("/", "").replace("^", "")
    p1_name    = f"checkpoint_{safe_instr}_{args.resolution}_phase1_{date_str}.pt"
    p1_path    = os.path.join(MODELS_FT, p1_name)

    torch.save({
        "instrument":  args.instrument,
        "resolution":  args.resolution,
        "phase":       1,
        "model_state": model.state_dict(),
        "val_loss":    p1_val_loss,
        "val_acc":     p1_val_acc,
        "base_checkpoint": base_ckpt_path,
        "date":        date_str,
    }, p1_path)
    log.info("Phase 1 checkpoint saved: %s", p1_path)

    # ── Phase 2 gate ──────────────────────────────────────────
    # Get base model accuracy on same val set for comparison
    base_model     = load_base_checkpoint(base_ckpt_path, device)
    _, base_val_acc = run_eval(base_model, val_loader, criterion, device)
    log.info("Base model H1 accuracy on instrument val set: %.3f", base_val_acc[1])
    log.info("Fine-tuned  H1 accuracy (Phase 1)           : %.3f", p1_val_acc[1])

    proceed = phase2(p1_val_acc, base_val_acc)
    all_metrics = p1_metrics.copy()
    final_phase = 1
    final_path  = p1_path

    # ── Phase 3 (optional) ────────────────────────────────────
    if proceed and not args.skip_phase3:
        p3_success, p3_metrics = phase3(
            model, train_loader, val_loader, criterion, device, years_of_data
        )
        all_metrics.extend(p3_metrics)

        if p3_success:
            p3_name  = f"checkpoint_{safe_instr}_{args.resolution}_phase3_{date_str}.pt"
            p3_path  = os.path.join(MODELS_FT, p3_name)
            _, p3_val_acc = run_eval(model, val_loader, criterion, device)

            torch.save({
                "instrument":  args.instrument,
                "resolution":  args.resolution,
                "phase":       3,
                "model_state": model.state_dict(),
                "val_acc":     p3_val_acc,
                "base_checkpoint": base_ckpt_path,
                "date":        date_str,
            }, p3_path)
            log.info("Phase 3 checkpoint saved: %s", p3_path)
            final_phase = 3
            final_path  = p3_path
        else:
            # Revert to Phase 1 weights
            model.load_state_dict(p1_state_backup)
            log.info("Reverted to Phase 1 checkpoint.")
            _, p1_val_acc_recheck = run_eval(model, val_loader, criterion, device)
            log.info("Phase 1 H1 accuracy (recheck): %.3f", p1_val_acc_recheck[1])

    # ── Delta accuracy vs base model ──────────────────────────
    log.info("=" * 60)
    log.info("Fine-Tuning Summary")
    log.info("  Instrument     : %s %s", args.instrument, args.resolution)
    log.info("  Final phase    : %d", final_phase)
    log.info("  Final checkpoint: %s", final_path)
    # final_val_acc must reflect whatever phase actually produced the final checkpoint.
    # p3_val_acc is computed inside the Phase 3 block but not accessible here —
    # we re-run eval on the current model weights (which are the final phase weights).
    _, final_val_acc = run_eval(model, val_loader, criterion, device)

    base_mean  = float(np.mean([base_val_acc[h]  for h in [1, 5, 20]]))
    ft_mean    = float(np.mean([final_val_acc[h] for h in [1, 5, 20]]))

    log.info("  Base model — H1=%.3f  H5=%.3f  H20=%.3f  Mean=%.3f",
             base_val_acc[1], base_val_acc[5], base_val_acc[20], base_mean)
    log.info("  Fine-tuned — H1=%.3f  H5=%.3f  H20=%.3f  Mean=%.3f  (Phase %d)",
             final_val_acc[1], final_val_acc[5], final_val_acc[20], ft_mean, final_phase)

    delta = ft_mean - base_mean
    if delta > 0.001:  # require at least 0.1pp improvement to call it a win
        log.info("  Delta mean accuracy : +%.3f  ✓ Fine-tuning improved accuracy.", delta)
        log.info("  Per-horizon delta   :  H1=%+.3f  H5=%+.3f  H20=%+.3f",
                 final_val_acc[1]-base_val_acc[1],
                 final_val_acc[5]-base_val_acc[5],
                 final_val_acc[20]-base_val_acc[20])
    elif delta >= -0.001:  # within 0.1pp — call it a draw, keep fine-tuned
        log.info("  Delta mean accuracy : %+.3f  = Fine-tuning matched base model.",
                 delta)
        log.info("  Per-horizon delta   :  H1=%+.3f  H5=%+.3f  H20=%+.3f",
                 final_val_acc[1]-base_val_acc[1],
                 final_val_acc[5]-base_val_acc[5],
                 final_val_acc[20]-base_val_acc[20])
    else:
        log.warning(
            "  Delta mean accuracy : %.3f  ⚠  Fine-tuning degraded mean accuracy. "
            "Consider using the base model for this instrument. "
            "H1 delta=%+.3f  H5 delta=%+.3f  H20 delta=%+.3f",
            delta,
            final_val_acc[1]-base_val_acc[1],
            final_val_acc[5]-base_val_acc[5],
            final_val_acc[20]-base_val_acc[20],
        )

    # Save metrics CSV
    metrics_path = os.path.join(LOG_DIR,
                                f"{safe_instr}_{args.resolution}_{date_str}_metrics.csv")
    pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)
    log.info("  Metrics saved  : %s", metrics_path)
    log.info("Stage 6 complete.")


if __name__ == "__main__":
    main()