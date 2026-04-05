"""
Stage 1 — Data Collection Script
Trading Algorithm Blueprint

Downloads OHLCV data for all instruments specified in the blueprint using yfinance.
Saves CSVs to data/raw/ with the standardised naming convention:
  {EXCHANGE}_{TICKER}_{RESOLUTION}_{STARTDATE}_{ENDDATE}.csv

NOTE on intraday (1H / 4H) limits:
  Yahoo Finance only provides 1H data for the last ~730 days.
  4H is synthesised by resampling from 1H data.
  Daily (1D) and Weekly (1W) data goes back 10+ years with no restriction.

Usage:
  pip install yfinance pandas
  python collect_data.py
"""

import os
import time
import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

# ─── Configuration ────────────────────────────────────────────────────────────

OUTPUT_DIR = "data/raw"
LOG_DIR    = "logs"

START_DATE_LONG  = "2015-01-01"   # for 1D and 1W (8+ years)
END_DATE         = datetime.today().strftime("%Y-%m-%d")

# Intraday history is capped by Yahoo Finance at ~730 days
START_DATE_INTRA = (datetime.today() - timedelta(days=720)).strftime("%Y-%m-%d")

RESOLUTIONS = {
    "1D": "1d",
    "1W": "1wk",
    "1H": "1h",   # resampled to 4H below
}

# ─── Instrument Universe ──────────────────────────────────────────────────────
# Format: (exchange_label, yfinance_ticker)

EQUITIES = [
    # US — NYSE / NASDAQ
    ("NYSE",   "AAPL"),   # Apple
    ("NYSE",   "MSFT"),   # Microsoft
    ("NYSE",   "JPM"),    # JPMorgan
    ("NYSE",   "XOM"),    # ExxonMobil
    ("NYSE",   "JNJ"),    # Johnson & Johnson
    ("NYSE",   "PG"),     # Procter & Gamble
    ("NYSE",   "UNH"),    # UnitedHealth
    ("NYSE",   "BAC"),    # Bank of America
    ("NYSE",   "CVX"),    # Chevron
    ("NYSE",   "AMZN"),   # Amazon
    ("NYSE",   "GOOGL"),  # Alphabet
    ("NYSE",   "META"),   # Meta
    ("NYSE",   "TSLA"),   # Tesla
    ("NYSE",   "NVDA"),   # Nvidia
    ("NYSE",   "V"),      # Visa
    # UK — LSE  (suffix .L for yfinance)
    ("LSE",    "SHEL.L"), # Shell
    ("LSE",    "AZN.L"),  # AstraZeneca
    ("LSE",    "HSBA.L"), # HSBC
    ("LSE",    "BP.L"),   # BP
    ("LSE",    "GSK.L"),  # GSK
    # Australia — ASX (suffix .AX)
    ("ASX",    "BHP.AX"), # BHP
    ("ASX",    "CBA.AX"), # Commonwealth Bank
    ("ASX",    "CSL.AX"), # CSL
    ("ASX",    "WBC.AX"), # Westpac
    ("ASX",    "ANZ.AX"), # ANZ
    # Canada — TSX (suffix .TO)
    ("TSX",    "RY.TO"),  # Royal Bank of Canada
    ("TSX",    "TD.TO"),  # TD Bank
    ("TSX",    "CNR.TO"), # Canadian National Railway
    ("TSX",    "SU.TO"),  # Suncor Energy
    ("TSX",    "MFC.TO"), # Manulife
    # Europe — Euronext / XETRA (various suffixes)
    ("EURONEXT", "ASML.AS"),  # ASML
    ("EURONEXT", "MC.PA"),    # LVMH
    ("EURONEXT", "SAP.DE"),   # SAP
    ("EURONEXT", "SIE.DE"),   # Siemens
    ("EURONEXT", "TTE.PA"),   # TotalEnergies
]

FOREX = [
    # Majors
    ("FOREXCOM", "EURUSD=X"),
    ("FOREXCOM", "GBPUSD=X"),
    ("FOREXCOM", "USDJPY=X"),
    ("FOREXCOM", "AUDUSD=X"),
    # Minors
    ("FOREXCOM", "EURGBP=X"),
    ("FOREXCOM", "GBPJPY=X"),
    ("FOREXCOM", "AUDNZD=X"),
    ("FOREXCOM", "EURCHF=X"),
    ("FOREXCOM", "USDCAD=X"),
    ("FOREXCOM", "USDCHF=X"),
]

COMMODITIES = [
    ("COMEX",  "GC=F"),   # Gold
    ("NYMEX",  "CL=F"),   # Crude Oil WTI
    ("ICE",    "BZ=F"),   # Brent Crude
    ("NYMEX",  "NG=F"),   # Natural Gas
    ("CBOT",   "ZW=F"),   # Wheat
    ("COMEX",  "HG=F"),   # Copper
    ("COMEX",  "SI=F"),   # Silver
]

INDICES = [
    ("SP500",  "^GSPC"),  # S&P 500
    ("DAX",    "^GDAXI"), # DAX
    ("NIKKEI", "^N225"),  # Nikkei 225
    ("FTSE",   "^FTSE"),  # FTSE 100
    ("HSI",    "^HSI"),   # Hang Seng
]

CRYPTO = [
    ("CRYPTO", "BTC-USD"),
    ("CRYPTO", "ETH-USD"),
]

ALL_INSTRUMENTS = EQUITIES + FOREX + COMMODITIES + INDICES + CRYPTO

# ─── Logging ──────────────────────────────────────────────────────────────────

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "data_collection.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def clean_ticker_for_filename(ticker: str) -> str:
    """Remove / ^ = . characters so the ticker is safe in a filename."""
    return ticker.replace("/", "").replace("^", "").replace("=", "").replace(".", "")


def resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Resample 1H OHLCV dataframe to 4H bars."""
    if df_1h.empty:
        return df_1h
    ohlc_dict = {
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    }
    df_4h = df_1h.resample("4h").agg(ohlc_dict).dropna(subset=["Open", "Close"])
    return df_4h


def validate_row_count(filepath: str, resolution: str) -> bool:
    """
    Warn if a daily file looks sparse.
    ~252 trading days/year → 8 years ≈ 2016 rows minimum.
    Intraday (1H/4H) is capped at ~720 days so thresholds are lower.
    """
    df = pd.read_csv(filepath)
    n = len(df)
    thresholds = {
        "1D": 1800,
        "1W": 350,
        "1H": 3000,   # ~720 days × ~6.5h average
        "4H": 700,
    }
    minimum = thresholds.get(resolution, 100)
    if n < minimum:
        log.warning(
            "  ⚠  %s has only %d rows (expected ≥%d for %s) — check for gaps.",
            filepath, n, minimum, resolution,
        )
        return False
    log.info("  ✓  %d rows — OK", n)
    return True


def download_and_save(
    exchange: str,
    ticker: str,
    resolution_label: str,      # "1D", "1W", "1H", "4H"
    yf_interval: str,           # yfinance interval string
    start: str,
    end: str,
) -> str | None:
    """Download OHLCV from Yahoo Finance and save to CSV. Returns filepath or None."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    safe_ticker = clean_ticker_for_filename(ticker)
    start_compact = start.replace("-", "")
    end_compact   = end.replace("-", "")
    filename = f"{exchange}_{safe_ticker}_{resolution_label}_{start_compact}_{end_compact}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(filepath):
        log.info("[SKIP] %s already exists.", filename)
        return filepath

    log.info("[DL]   %s — %s — %s", ticker, resolution_label, yf_interval)

    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=yf_interval,
            auto_adjust=False,   # keep unadjusted prices (blueprint requirement)
            progress=False,
        )
    except Exception as exc:
        log.error("  ✗  Download failed for %s: %s", ticker, exc)
        return None

    if df is None or df.empty:
        log.warning("  ✗  No data returned for %s (%s).", ticker, resolution_label)
        return None

    # Flatten MultiIndex columns if present (yfinance ≥0.2 behaviour)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only OHLCV columns
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].copy()
    df.index.name = "Datetime"
    df.reset_index(inplace=True)

    df.to_csv(filepath, index=False)
    log.info("  → Saved: %s", filepath)
    validate_row_count(filepath, resolution_label)
    return filepath


def download_instrument(exchange: str, ticker: str) -> None:
    """Download all four resolutions for a single instrument."""

    log.info("=" * 60)
    log.info("Instrument: %s  [%s]", ticker, exchange)

    # ── Daily & Weekly (long history) ──────────────────────────────
    for label, yf_int in [("1D", "1d"), ("1W", "1wk")]:
        download_and_save(exchange, ticker, label, yf_int, START_DATE_LONG, END_DATE)
        time.sleep(0.5)   # polite rate-limiting

    # ── 1H (capped at ~720 days) ───────────────────────────────────
    path_1h = download_and_save(
        exchange, ticker, "1H", "1h", START_DATE_INTRA, END_DATE
    )
    time.sleep(0.5)

    # ── 4H (resampled from 1H) ─────────────────────────────────────
    if path_1h and os.path.exists(path_1h):
        safe_ticker   = clean_ticker_for_filename(ticker)
        start_compact = START_DATE_INTRA.replace("-", "")
        end_compact   = END_DATE.replace("-", "")
        fname_4h      = f"{exchange}_{safe_ticker}_4H_{start_compact}_{end_compact}.csv"
        fpath_4h      = os.path.join(OUTPUT_DIR, fname_4h)

        if os.path.exists(fpath_4h):
            log.info("[SKIP] %s already exists.", fname_4h)
        else:
            try:
                df_1h = pd.read_csv(path_1h, index_col="Datetime", parse_dates=True)
                df_4h = resample_to_4h(df_1h)
                df_4h.index.name = "Datetime"
                df_4h.reset_index(inplace=True)
                df_4h.to_csv(fpath_4h, index=False)
                log.info("[4H]  Resampled from 1H → %s", fpath_4h)
                validate_row_count(fpath_4h, "4H")
            except Exception as exc:
                log.error("  ✗  4H resample failed for %s: %s", ticker, exc)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Stage 1 Data Collection")
    parser.add_argument(
        "--refresh", action="store_true",
        help="Delete all existing CSVs in data/raw/ before downloading. "
             "Use this to pull fresh data up to today's date."
    )
    args = parser.parse_args()

    if args.refresh:
        import glob as _glob
        existing = _glob.glob(os.path.join(OUTPUT_DIR, "*.csv"))
        if existing:
            log.info("--refresh: deleting %d existing CSVs from %s",
                     len(existing), OUTPUT_DIR)
            for f in existing:
                os.remove(f)
            log.info("--refresh: all old CSVs removed — downloading fresh data")
        else:
            log.info("--refresh: no existing CSVs found — clean download")

    log.info("Stage 1 Data Collection — Start")
    log.info("Output directory : %s", os.path.abspath(OUTPUT_DIR))
    log.info("Long history from: %s", START_DATE_LONG)
    log.info("Intraday from    : %s  (Yahoo Finance cap)", START_DATE_INTRA)
    log.info("End date         : %s", END_DATE)
    log.info("Total instruments: %d", len(ALL_INSTRUMENTS))

    success = 0
    failed  = []

    for exchange, ticker in ALL_INSTRUMENTS:
        try:
            download_instrument(exchange, ticker)
            success += 1
        except Exception as exc:
            log.error("Unhandled error for %s: %s", ticker, exc)
            failed.append(ticker)
        time.sleep(1.0)   # ~1 second between instruments

    log.info("=" * 60)
    log.info("Collection complete.")
    log.info("  Instruments attempted : %d", len(ALL_INSTRUMENTS))
    log.info("  Successful            : %d", success)
    log.info("  Failed / skipped      : %s", failed if failed else "none")
    log.info("CSVs saved to           : %s", os.path.abspath(OUTPUT_DIR))

    # ── Summary report ────────────────────────────────────────────
    files = os.listdir(OUTPUT_DIR)
    resolution_counts = {}
    for f in files:
        parts = f.split("_")
        if len(parts) >= 3:
            res = parts[2]
            resolution_counts[res] = resolution_counts.get(res, 0) + 1

    log.info("\nFiles by resolution:")
    for res, count in sorted(resolution_counts.items()):
        log.info("  %-4s : %d files", res, count)


if __name__ == "__main__":
    main()