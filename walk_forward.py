"""
Stage 3 — Walk-Forward Validation Split Generator
Trading Algorithm Blueprint

Reads all processed .parquet metadata files from data/processed/,
builds chronologically strict train/val/test split index files,
and saves them to data/splits/.

Split structure per fold (blueprint Section 3.2):
  Training window : 6 years (fixed-length sliding)
  Purge gap       : 20 bars
  Validation      : 1 year
  Test            : 1 year
  Slide increment : 1 year
  Minimum folds   : 4

Outputs per fold:
  data/splits/fold_{N}_train.parquet
  data/splits/fold_{N}_val.parquet
  data/splits/fold_{N}_test.parquet

Each output contains columns: [source_file, window_idx, date, regime]
"""

import os
import glob
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ─── Configuration ────────────────────────────────────────────────────────────

PROCESSED_DIR = "data/processed"
SPLITS_DIR    = "data/splits"
LOG_DIR       = "logs"

TRAIN_YEARS  = 6
VAL_YEARS    = 1
TEST_YEARS   = 1
SLIDE_YEARS  = 1
PURGE_BARS   = 20          # gap between train end and val start
MIN_FOLDS    = 4
MAX_INST_PCT = 0.15        # no single instrument > 15% of training samples

# ─── Logging ──────────────────────────────────────────────────────────────────

os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "walk_forward.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ─── Load all metadata ────────────────────────────────────────────────────────

def load_all_metadata():
    """
    Load every .parquet metadata file from data/processed/.
    Each file records (date, regime) for every window in that instrument/resolution.
    We add window_idx (position within the file) and source_file columns.
    """
    parquet_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*.parquet")))
    if not parquet_files:
        log.error("No .parquet files found in %s", PROCESSED_DIR)
        return None

    frames = []
    for fpath in parquet_files:
        fname = os.path.basename(fpath)
        try:
            df = pd.read_parquet(fpath)
            df["date"] = pd.to_datetime(df["date"])
            # Strip tz if present
            if df["date"].dt.tz is not None:
                df["date"] = df["date"].dt.tz_localize(None)
            df["source_file"] = fname.replace(".parquet", "")
            df["window_idx"]  = np.arange(len(df))
            frames.append(df)
        except Exception as e:
            log.warning("Could not load %s: %s", fname, e)

    if not frames:
        return None

    meta = pd.concat(frames, ignore_index=True)
    meta = meta.sort_values("date").reset_index(drop=True)
    log.info("Loaded %d total windows from %d files", len(meta), len(frames))
    return meta

# ─── Date boundary helpers ────────────────────────────────────────────────────

def years_offset(dt, years):
    try:
        return dt.replace(year=dt.year + years)
    except ValueError:
        # Feb 29 edge case
        return dt.replace(year=dt.year + years, day=28)


def build_fold_boundaries(global_start, global_end):
    """
    Slide the window forward in SLIDE_YEARS increments.
    Returns list of (train_start, train_end, val_start, val_end, test_start, test_end).
    """
    folds = []
    train_start = global_start

    while True:
        train_end  = years_offset(train_start, TRAIN_YEARS)
        val_start  = train_end      # purge applied at sample level, not date level
        val_end    = years_offset(val_start, VAL_YEARS)
        test_start = val_end
        test_end   = years_offset(test_start, TEST_YEARS)

        if test_end > global_end:
            break

        folds.append((train_start, train_end, val_start, val_end, test_start, test_end))
        train_start = years_offset(train_start, SLIDE_YEARS)

    return folds


# ─── Per-instrument cap ───────────────────────────────────────────────────────

def apply_instrument_cap(df, max_pct=MAX_INST_PCT):
    """
    Downsample any single instrument that contributes more than max_pct
    of total samples (blueprint Section 3.3).
    """
    total   = len(df)
    cap     = int(total * max_pct)
    counts  = df["source_file"].value_counts()
    over    = counts[counts > cap].index.tolist()

    if not over:
        return df

    keep_parts = []
    for src in df["source_file"].unique():
        subset = df[df["source_file"] == src]
        if src in over:
            subset = subset.sample(n=cap, random_state=42)
            log.info("  Downsampled %s from %d → %d samples", src, len(df[df["source_file"]==src]), cap)
        keep_parts.append(subset)

    return pd.concat(keep_parts, ignore_index=True)


# ─── Regime balance check ─────────────────────────────────────────────────────

def check_regime_balance(df, fold_n, split_name):
    """Log regime class distribution and flag any class below 5%."""
    valid = df["regime"].dropna()
    total = len(valid)
    if total == 0:
        return

    log.info("  Regime distribution [Fold %d %s]:", fold_n, split_name)
    for cls in range(6):
        cnt = (valid == cls).sum()
        pct = 100 * cnt / total
        flag = "  ⚠  BELOW 5%" if pct < 5 else ""
        log.info("    Class %d: %5d  (%5.1f%%)%s", cls, cnt, pct, flag)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    log.info("Stage 3 Walk-Forward Split Generation — Start")

    meta = load_all_metadata()
    if meta is None:
        return

    global_start = meta["date"].min().to_pydatetime()
    global_end   = meta["date"].max().to_pydatetime()
    log.info("Data range: %s → %s", global_start.date(), global_end.date())

    folds = build_fold_boundaries(global_start, global_end)
    log.info("Folds generated: %d", len(folds))

    if len(folds) < MIN_FOLDS:
        log.warning(
            "Only %d folds generated — blueprint requires minimum %d. "
            "Consider extending data history or reducing TRAIN_YEARS.",
            len(folds), MIN_FOLDS
        )

    fold_summaries = []

    for fold_n, (tr_s, tr_e, va_s, va_e, te_s, te_e) in enumerate(folds, start=1):
        log.info("=" * 60)
        log.info("Fold %d:", fold_n)
        log.info("  Train : %s → %s", tr_s.date(), tr_e.date())
        log.info("  Val   : %s → %s", va_s.date(), va_e.date())
        log.info("  Test  : %s → %s", te_s.date(), te_e.date())

        # Strict chronological split — no overlap
        train_df = meta[(meta["date"] >= tr_s) & (meta["date"] < tr_e)].copy()
        val_df   = meta[(meta["date"] >= va_s) & (meta["date"] < va_e)].copy()
        test_df  = meta[(meta["date"] >= te_s) & (meta["date"] < te_e)].copy()

        # Apply purge gap: drop first PURGE_BARS rows of val (per instrument)
        # to prevent lookback-window feature overlap with train data
        purged_val_parts = []
        n_purged_empty = 0
        for src in val_df["source_file"].unique():
            subset = val_df[val_df["source_file"] == src].sort_values("date")
            purged = subset.iloc[PURGE_BARS:]
            if len(purged) == 0:
                n_purged_empty += 1  # instrument had fewer rows than purge gap
            else:
                purged_val_parts.append(purged)
        if n_purged_empty > 0:
            log.warning("  ⚠  %d instruments had fewer than %d val rows "
                        "— fully removed by purge gap.", n_purged_empty, PURGE_BARS)
        val_df = pd.concat(purged_val_parts, ignore_index=True) if purged_val_parts else val_df

        # Verify no temporal overlap between splits
        if len(train_df) > 0 and len(val_df) > 0:
            assert train_df["date"].max() <= val_df["date"].min(), \
                f"Fold {fold_n}: temporal overlap between train and val!"
        if len(val_df) > 0 and len(test_df) > 0:
            assert val_df["date"].max() <= test_df["date"].min(), \
                f"Fold {fold_n}: temporal overlap between val and test!"

        # Apply per-instrument cap to training set only
        train_df = apply_instrument_cap(train_df)

        # Shuffle training set at sample level (not instrument level)
        train_df = train_df.sample(frac=1, random_state=fold_n * 42).reset_index(drop=True)

        log.info("  Samples — Train: %d  Val: %d  Test: %d",
                 len(train_df), len(val_df), len(test_df))

        # Regime balance checks
        check_regime_balance(train_df, fold_n, "TRAIN")
        check_regime_balance(val_df,   fold_n, "VAL")

        # Save split index files
        train_path = os.path.join(SPLITS_DIR, f"fold_{fold_n}_train.parquet")
        val_path   = os.path.join(SPLITS_DIR, f"fold_{fold_n}_val.parquet")
        test_path  = os.path.join(SPLITS_DIR, f"fold_{fold_n}_test.parquet")

        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path,     index=False)
        test_df.to_parquet(test_path,   index=False)

        log.info("  Saved: %s", train_path)
        log.info("  Saved: %s", val_path)
        log.info("  Saved: %s", test_path)

        fold_summaries.append({
            "fold":       fold_n,
            "train_start": tr_s.date(),
            "train_end":   tr_e.date(),
            "val_start":   va_s.date(),
            "val_end":     va_e.date(),
            "test_start":  te_s.date(),
            "test_end":    te_e.date(),
            "n_train":     len(train_df),
            "n_val":       len(val_df),
            "n_test":      len(test_df),
        })

    # ── Summary table ─────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Walk-Forward Split Summary:")
    summary_df = pd.DataFrame(fold_summaries)
    log.info("\n%s", summary_df.to_string(index=False))

    summary_df.to_parquet(os.path.join(SPLITS_DIR, "fold_summary.parquet"), index=False)
    summary_df.to_csv(os.path.join(SPLITS_DIR, "fold_summary.csv"), index=False)
    log.info("\nSummary saved to data/splits/fold_summary.csv")
    log.info("Stage 3 complete — %d folds ready for training.", len(folds))


if __name__ == "__main__":
    main()