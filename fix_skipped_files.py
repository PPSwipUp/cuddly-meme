"""
fix_skipped_files.py — Diagnoses and fixes the 47 skipped files.

The most common causes when a file has enough rows but still fails:
  1. yfinance wrote a MultiIndex header (two header rows)
  2. Column names are lowercase (open/high/low/close/volume)
  3. Column names include the ticker (Open_AAPL etc.)
  4. Datetime column is named 'Date', 'date', 'index', or 'timestamp'
  5. NaN columns from Volume=0 rows (e.g. weekly crypto)

This script:
  - Audits every skipped file and prints exactly what's wrong
  - Standardises column names to: Datetime, Open, High, Low, Close, Volume
  - Saves fixed copies back to data/raw/ (overwrites in place)
  - Re-runs feature engineering on fixed files only

Run from your project folder:
    python fix_skipped_files.py
"""

import os
import glob
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
LOG_DIR       = "logs"

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "fix_skipped.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ─── Which files to fix ────────────────────────────────────────────────────────

SKIPPED = [
    'ASX_ANZAX_1H_20240330_20260320.csv',    'ASX_ANZAX_1W_20150101_20260320.csv',
    'ASX_BHPAX_1H_20240330_20260320.csv',    'ASX_BHPAX_1W_20150101_20260320.csv',
    'ASX_CBAAX_1H_20240330_20260320.csv',    'ASX_CBAAX_1W_20150101_20260320.csv',
    'ASX_CSLAX_1H_20240330_20260320.csv',    'ASX_CSLAX_1W_20150101_20260320.csv',
    'ASX_WBCAX_1H_20240330_20260320.csv',    'ASX_WBCAX_1W_20150101_20260320.csv',
    'CRYPTO_BTC-USD_1H_20240330_20260320.csv','CRYPTO_ETH-USD_1H_20240330_20260320.csv',
    'CRYPTO_ETH-USD_1W_20150101_20260320.csv','DAX_GDAXI_1H_20240330_20260320.csv',
    'DAX_GDAXI_1W_20150101_20260320.csv',    'EURONEXT_ASMLAS_1H_20240330_20260320.csv',
    'EURONEXT_ASMLAS_1W_20150101_20260320.csv','EURONEXT_MCPA_1H_20240330_20260320.csv',
    'EURONEXT_MCPA_1W_20150101_20260320.csv', 'EURONEXT_SAPDE_1H_20240330_20260320.csv',
    'EURONEXT_SAPDE_1W_20150101_20260320.csv','EURONEXT_SIEDE_1H_20240330_20260320.csv',
    'EURONEXT_SIEDE_1W_20150101_20260320.csv','EURONEXT_TTEPA_1H_20240330_20260320.csv',
    'EURONEXT_TTEPA_1W_20150101_20260320.csv','FOREXCOM_AUDNZDX_1H_20240330_20260320.csv',
    'FOREXCOM_AUDUSDX_1H_20240330_20260320.csv','FOREXCOM_EURCHFX_1H_20240330_20260320.csv',
    'FOREXCOM_EURGBPX_1H_20240330_20260320.csv','FOREXCOM_EURUSDX_1H_20240330_20260320.csv',
    'FOREXCOM_GBPJPYX_1H_20240330_20260320.csv','FOREXCOM_GBPUSDX_1H_20240330_20260320.csv',
    'FOREXCOM_USDCADX_1H_20240330_20260320.csv','FOREXCOM_USDCHFX_1H_20240330_20260320.csv',
    'FOREXCOM_USDJPYX_1H_20240330_20260320.csv','FTSE_FTSE_1H_20240330_20260320.csv',
    'HSI_HSI_1H_20240330_20260320.csv',      'HSI_HSI_1W_20150101_20260320.csv',
    'HSI_HSI_4H_20240330_20260320.csv',      'LSE_AZNL_1H_20240330_20260320.csv',
    'LSE_BPL_1H_20240330_20260320.csv',      'LSE_GSKL_1H_20240330_20260320.csv',
    'LSE_HSBAL_1H_20240330_20260320.csv',    'LSE_SHELL_1H_20240330_20260320.csv',
    'NIKKEI_N225_1H_20240330_20260320.csv',  'NIKKEI_N225_1W_20150101_20260320.csv',
    'NIKKEI_N225_4H_20240330_20260320.csv',
]

# ─── Column standardiser ──────────────────────────────────────────────────────

DATETIME_ALIASES = {
    'datetime', 'date', 'timestamp', 'time', 'index',
    'datetime (utc)', 'date (utc)',
}
OHLCV_ALIASES = {
    'open':   'Open',
    'high':   'High',
    'low':    'Low',
    'close':  'Close',
    'adj close': None,      # drop adjusted close
    'volume': 'Volume',
}


def standardise_columns(df: pd.DataFrame, filepath: str) -> pd.DataFrame | None:
    """
    Detect and fix column naming issues. Returns standardised DataFrame or None on failure.
    """
    original_cols = list(df.columns)

    # ── Handle MultiIndex (two header rows from yfinance) ──────────────────
    if isinstance(df.columns, pd.MultiIndex):
        log.info("    MultiIndex detected — flattening")
        df.columns = [
            c[0] if c[1] == '' else f"{c[0]}_{c[1]}"
            for c in df.columns
        ]

    # ── Strip whitespace from column names ─────────────────────────────────
    df.columns = [str(c).strip() for c in df.columns]

    # ── Find datetime column ───────────────────────────────────────────────
    dt_col = None
    for col in df.columns:
        if col.lower() in DATETIME_ALIASES or col == df.columns[0]:
            dt_col = col
            break

    if dt_col is None:
        # Try index
        if df.index.name and df.index.name.lower() in DATETIME_ALIASES:
            df = df.reset_index()
            dt_col = df.columns[0]
        else:
            log.error("    Cannot find datetime column. Columns: %s", list(df.columns))
            return None

    # ── Rename datetime column ─────────────────────────────────────────────
    df = df.rename(columns={dt_col: 'Datetime'})

    # ── Find and rename OHLCV columns ─────────────────────────────────────
    rename_map = {}
    cols_lower = {c.lower().split('_')[0]: c for c in df.columns}  # handle "Open_AAPL"

    for alias, standard in OHLCV_ALIASES.items():
        for col in df.columns:
            col_clean = col.lower().split('_')[0].strip()
            if col_clean == alias:
                if standard is None:
                    df = df.drop(columns=[col], errors='ignore')
                else:
                    rename_map[col] = standard
                break

    df = df.rename(columns=rename_map)

    # ── Keep only needed columns ───────────────────────────────────────────
    needed = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        log.error("    Missing required columns %s. Have: %s", missing, list(df.columns))
        return None

    df = df[needed].copy()

    # ── Parse Datetime ─────────────────────────────────────────────────────
    try:
        df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True).dt.tz_localize(None)
    except Exception:
        try:
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            if df['Datetime'].dt.tz is not None:
                df['Datetime'] = df['Datetime'].dt.tz_localize(None)
        except Exception as e:
            log.error("    Cannot parse Datetime column: %s", e)
            return None

    # ── Coerce OHLCV to numeric ────────────────────────────────────────────
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # ── Drop rows where Close is null/zero ────────────────────────────────
    before = len(df)
    df = df[(df['Close'] > 0)].dropna(subset=['Open', 'High', 'Low', 'Close'])
    after = len(df)
    if before - after > 0:
        log.info("    Dropped %d null/zero-close rows (%d → %d)", before - after, before, after)

    # ── Fill zero Volume with NaN then forward-fill ────────────────────────
    # (some instruments report 0 volume on non-trading bars)
    zero_vol = (df['Volume'] == 0).sum()
    if zero_vol > 0:
        log.info("    Replacing %d zero-volume rows with forward-filled values", zero_vol)
        df['Volume'] = df['Volume'].replace(0, np.nan).ffill().fillna(1)

    df = df.sort_values('Datetime').reset_index(drop=True)

    if len(df) < 130:
        log.error("    Only %d clean rows after fix — still too few", len(df))
        return None

    log.info("    Fixed columns: %s → %s", original_cols[:6], list(df.columns))
    return df


# ─── Full audit ───────────────────────────────────────────────────────────────

def audit_file(fname: str) -> dict:
    """Peek at a file and report what's actually in it."""
    fpath = os.path.join(RAW_DIR, fname)
    result = {"fname": fname, "exists": False, "rows": 0, "columns": [], "issue": None}

    if not os.path.exists(fpath):
        result["issue"] = "FILE MISSING"
        return result

    result["exists"] = True

    # Try reading with 1 header row first
    try:
        df = pd.read_csv(fpath, nrows=5)
        result["columns"] = list(df.columns)
        result["rows"] = sum(1 for _ in open(fpath)) - 1  # fast row count
    except Exception as e:
        result["issue"] = f"READ ERROR: {e}"
        return result

    # Check for MultiIndex (two header rows)
    try:
        df2 = pd.read_csv(fpath, header=[0, 1], nrows=3)
        if isinstance(df2.columns, pd.MultiIndex):
            result["issue"] = "MULTIINDEX_HEADER"
            result["multi_columns"] = list(df2.columns)
    except Exception:
        pass

    # Check column names
    cols_lower = [c.lower() for c in result["columns"]]
    if 'datetime' not in cols_lower and 'date' not in cols_lower:
        result["issue"] = f"NO_DATETIME_COL — cols: {result['columns']}"

    return result


def fix_and_save(fname: str) -> bool:
    """Load, fix, and overwrite a file. Returns True on success."""
    fpath = os.path.join(RAW_DIR, fname)

    # Try standard read first
    df = None
    for header in [0, [0, 1]]:
        try:
            df_try = pd.read_csv(fpath, header=header)
            if isinstance(df_try.columns, pd.MultiIndex) or len(df_try.columns) >= 4:
                df = df_try
                break
        except Exception:
            continue

    if df is None:
        log.error("  [FAIL] Could not read %s", fname)
        return False

    fixed = standardise_columns(df, fpath)
    if fixed is None:
        return False

    fixed.to_csv(fpath, index=False)
    log.info("  [FIXED] %s — %d rows saved", fname, len(fixed))
    return True


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Phase 1: Auditing %d skipped files", len(SKIPPED))
    log.info("=" * 60)

    # Group issues
    issues = {}
    for fname in SKIPPED:
        info = audit_file(fname)
        issue = info.get("issue", "UNKNOWN")
        issues.setdefault(issue, []).append((fname, info))

    for issue, items in sorted(issues.items()):
        log.info("\nIssue: %s — %d files", issue, len(items))
        for fname, info in items[:3]:
            log.info("  %s", fname)
            log.info("    cols=%s  rows=%s", info.get("columns", [])[:6], info.get("rows", "?"))

    log.info("\n" + "=" * 60)
    log.info("Phase 2: Fixing files")
    log.info("=" * 60)

    fixed_ok  = []
    fixed_fail = []
    missing   = []

    for fname in SKIPPED:
        fpath = os.path.join(RAW_DIR, fname)
        if not os.path.exists(fpath):
            log.warning("  [MISS]  %s — not found in data/raw/", fname)
            missing.append(fname)
            continue

        log.info("\nFixing: %s", fname)
        ok = fix_and_save(fname)
        if ok:
            fixed_ok.append(fname)
        else:
            fixed_fail.append(fname)

    log.info("\n" + "=" * 60)
    log.info("Fix summary")
    log.info("=" * 60)
    log.info("  Fixed successfully : %d", len(fixed_ok))
    log.info("  Failed to fix      : %d", len(fixed_fail))
    log.info("  Missing files      : %d", len(missing))

    if fixed_fail:
        log.warning("  Still failing: %s", fixed_fail)

    if not fixed_ok:
        log.info("No files were fixed — nothing to reprocess.")
        return

    log.info("\n" + "=" * 60)
    log.info("Phase 3: Re-running feature engineering on fixed files")
    log.info("=" * 60)

    # Dynamically import and run feature engineering on fixed files only
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "feature_engineering",
        os.path.join(os.path.dirname(__file__), "feature_engineering.py")
    )
    fe = importlib.util.load_from_spec(spec) if hasattr(importlib.util, 'load_from_spec') else None

    # Fallback: patch RAW_DIR via a direct targeted loop
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import feature_engineering as fe_mod

        index_dfs = fe_mod.load_index_refs()
        processed = 0
        for fname in fixed_ok:
            fpath  = os.path.join(RAW_DIR, fname)
            result = fe_mod.process_file(fpath, index_dfs)
            if result is None:
                log.warning("  Still no output for %s after fix", fname)
                continue

            out_name = fname.replace(".csv", ".parquet")
            out_path = os.path.join(PROCESSED_DIR, out_name)
            meta_df  = pd.DataFrame({"date": result["dates"], "regime": result["regimes"]})
            np.save(out_path.replace(".parquet", ".npy"), result["X"])
            meta_df.to_parquet(out_path, index=False)
            log.info("  ✓  %s → %d windows × %d features",
                     fname, result["X"].shape[0], result["X"].shape[2])
            processed += 1

        log.info("\nRe-processed %d / %d fixed files", processed, len(fixed_ok))
        total_new_windows = processed  # rough count already logged per file

    except Exception as e:
        log.error("Could not import feature_engineering.py: %s", e)
        log.info("Please run feature_engineering.py separately after this fix.")


if __name__ == "__main__":
    main()