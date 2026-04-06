"""
Stage 1 (Intraday) — Data Collection
=====================================
Downloads 1-minute, 5-minute, and 15-minute OHLCV data for forex, crypto,
and major index instruments using yfinance.

Kept COMPLETELY SEPARATE from the daily pipeline:
  Output dir : data/raw_intraday/   (NOT data/raw/)
  Log dir    : logs/intraday/

Yahoo Finance intraday data limits:
  1m  → last 7 days only  (retrain model weekly)
  5m  → last 60 days      (retrain model monthly)
  15m → last 60 days      (retrain model monthly)

Instruments: forex, crypto, major indices only.
Individual equities are excluded — intraday equity data has gaps,
earnings moves, and thin liquidity that the model cannot handle well.

Usage:
  python collect_data_intraday.py            # download all (skip existing)
  python collect_data_intraday.py --refresh  # delete existing and re-download
  python collect_data_intraday.py --resolution 5m   # only 5m data
"""

import os
import glob
import logging
import argparse
from datetime import datetime

import pandas as pd
import yfinance as yf

# ─── Configuration ─────────────────────────────────────────────────────────────

OUTPUT_DIR = "data/raw_intraday"
LOG_DIR    = "logs/intraday"

# yfinance interval → (period_string, label_used_in_filename)
RESOLUTIONS = {
    "1m":  ("7d",  "1m"),
    "5m":  ("60d", "5m"),
    "15m": ("60d", "15m"),
}

# Instruments: (exchange_label, yfinance_ticker)
# Focused on liquid instruments with clean intraday data.
# Forex is the primary focus — 24/5 trading, no gaps, consistent behaviour.

FOREX = [
    ("FOREXCOM", "EURUSD=X"),   # EUR/USD
    ("FOREXCOM", "GBPUSD=X"),   # GBP/USD
    ("FOREXCOM", "USDJPY=X"),   # USD/JPY
    ("FOREXCOM", "USDCHF=X"),   # USD/CHF
    ("FOREXCOM", "AUDUSD=X"),   # AUD/USD
    ("FOREXCOM", "USDCAD=X"),   # USD/CAD
    ("FOREXCOM", "EURGBP=X"),   # EUR/GBP
    ("FOREXCOM", "EURCHF=X"),   # EUR/CHF
    ("FOREXCOM", "GBPJPY=X"),   # GBP/JPY
    ("FOREXCOM", "EURJPY=X"),   # EUR/JPY
]

CRYPTO = [
    ("CRYPTO", "BTC-USD"),      # Bitcoin
    ("CRYPTO", "ETH-USD"),      # Ethereum
]

INDICES = [
    ("DAX",    "^GDAXI"),       # German DAX
    ("FTSE",   "^FTSE"),        # UK FTSE 100
    ("SP500",  "^GSPC"),        # S&P 500 (also used as index reference)
    ("NIKKEI", "^N225"),        # Nikkei 225
]

ALL_INSTRUMENTS = FOREX + CRYPTO + INDICES

# S&P 500 is used as the correlation reference — always download it
INDEX_REF_TICKER = "^GSPC"
INDEX_REF_LABEL  = "SP500_GSPC"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "collect_intraday.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ─── Helpers ───────────────────────────────────────────────────────────────────

def clean_ticker(ticker: str) -> str:
    """Make ticker safe for use in a filename."""
    return ticker.replace("=X", "X").replace("^", "").replace("-", "").replace(".", "")


def download(exchange: str, ticker: str, resolution: str,
             period: str, refresh: bool) -> str | None:
    """
    Download one instrument at one resolution and save to CSV.
    Returns the filepath on success, None on failure.
    Skips if the file already exists and --refresh was not passed.
    """
    safe    = clean_ticker(ticker)
    today   = datetime.today().strftime("%Y%m%d")
    fname   = f"{exchange}_{safe}_{resolution}_{today}.csv"
    fpath   = os.path.join(OUTPUT_DIR, fname)

    # Incremental skip — check for ANY existing file for this instrument+resolution
    # (the date in the filename changes daily so we match on prefix)
    existing = glob.glob(os.path.join(OUTPUT_DIR, f"{exchange}_{safe}_{resolution}_*.csv"))
    if existing and not refresh:
        log.info("[SKIP] %s %s — already exists", f"{exchange}_{safe}", resolution)
        return existing[0]

    # Delete old dated files before downloading fresh
    for old in existing:
        os.remove(old)

    log.info("[DL]   %s %s %s", ticker, resolution, period)

    try:
        df = yf.download(
            ticker,
            period=period,
            interval=resolution,
            auto_adjust=False,
            progress=False,
        )
    except Exception as e:
        log.error("  ✗ Download failed for %s %s: %s", ticker, resolution, e)
        return None

    if df is None or df.empty:
        log.warning("  ✗ No data for %s %s", ticker, resolution)
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df   = df[keep].copy()
    df.index.name = "Datetime"
    df.reset_index(inplace=True)

    if df.empty:
        log.warning("  ✗ Empty after processing: %s %s", ticker, resolution)
        return None

    df.to_csv(fpath, index=False)
    log.info("  ✓ Saved: %s  (%d bars)", fname, len(df))
    return fpath


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 (Intraday) — Download 1m/5m/15m OHLCV data"
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Delete all existing intraday CSVs and re-download everything."
    )
    parser.add_argument(
        "--resolution", default=None, choices=["1m", "5m", "15m"],
        help="Only download this resolution (default: all three)."
    )
    args = parser.parse_args()

    if args.refresh:
        existing = glob.glob(os.path.join(OUTPUT_DIR, "*.csv"))
        if existing:
            log.info("--refresh: deleting %d existing files", len(existing))
            for f in existing:
                os.remove(f)
        else:
            log.info("--refresh: no existing files found")

    resolutions = {args.resolution: RESOLUTIONS[args.resolution]} \
                  if args.resolution else RESOLUTIONS

    log.info("Stage 1 (Intraday) — Start")
    log.info("Output dir  : %s", os.path.abspath(OUTPUT_DIR))
    log.info("Resolutions : %s", list(resolutions.keys()))
    log.info("Instruments : %d", len(ALL_INSTRUMENTS))

    passed = 0
    failed = 0
    skipped = 0

    for resolution, (period, _) in resolutions.items():
        log.info("─── Resolution: %s (period=%s) ───", resolution, period)

        # Always download S&P 500 as index reference first
        result = download(INDEX_REF_LABEL, INDEX_REF_TICKER,
                          resolution, period, args.refresh)
        if result:
            passed += 1
        else:
            failed += 1

        for exchange, ticker in ALL_INSTRUMENTS:
            # Skip index ref if it's already in the list
            if ticker == INDEX_REF_TICKER:
                continue

            result = download(exchange, ticker, resolution, period, args.refresh)
            if result is None:
                failed += 1
            elif result and "SKIP" in str(result):
                skipped += 1
            else:
                passed += 1

    log.info("─" * 50)
    log.info("Complete — passed=%d  failed=%d  skipped=%d", passed, failed, skipped)

    # Summary of what's in the output directory
    all_files = glob.glob(os.path.join(OUTPUT_DIR, "*.csv"))
    log.info("Total files in %s: %d", OUTPUT_DIR, len(all_files))

    # Warn about 1m data freshness
    if "1m" in resolutions:
        log.warning(
            "⚠  1m data covers only the last 7 days (Yahoo Finance limit). "
            "Re-run collect_data_intraday.py --refresh weekly to keep the "
            "1m model training data current. The 5m and 15m models only "
            "need refreshing monthly."
        )


if __name__ == "__main__":
    main()