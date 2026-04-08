"""
Paper Trading Engine
====================
Runs the trained models in paper-trading mode — no real money, full simulation.

Each run (call once per day for 1D instruments, or schedule for other resolutions):
  1. Downloads latest bars via yfinance
  2. Builds features using the same pipeline as training
  3. Runs the best available model (fine-tuned > base)
  4. Generates predictions: direction, confidence, price target
  5. Opens / holds / closes paper positions
  6. Sends email alerts on trade events
  7. At day end, sends a full portfolio summary email

Setup:
  Set environment variables before running:
    export PAPER_EMAIL_ADDRESS="your@outlook.com"
    export PAPER_EMAIL_PASSWORD="your-app-password"
    export PAPER_EMAIL_TO="recipient@email.com"      # can be same as FROM

  Or fill in the CONFIG block below directly (less secure).

Usage:
  python paper_trading.py                  # run trading cycle for all instruments
  python paper_trading.py --summary        # force send daily summary email now
  python paper_trading.py --instrument FOREXCOM_EURUSDX --resolution 1D
                                           # run for one instrument only

Schedule (Linux cron — daily at 9am):
  0 9 * * 1-5 cd /workspaces/cuddly-meme && python paper_trading.py

Schedule (Windows Task Scheduler):
  Action: python paper_trading.py
  Start in: C:\\path\\to\\cuddly-meme
  Trigger: Daily at 09:00
"""

import os
import sys
import glob
import json
import argparse
import logging
import smtplib
import warnings
import traceback
from datetime import datetime, date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import yfinance as yf

# ─── Add project root to path so we can import local modules ─────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import build_model
from feature_engineering import (
    compute_ohlc_features,
    compute_technical_features,
    compute_calendar_features,
    add_index_correlation,
    apply_rolling_zscore_pipeline,
)
from position_sizing import get_leverage_for_source

# ─── Configuration ────────────────────────────────────────────────────────────
# Fill these in or set as environment variables.

CONFIG = {
    # ── Email ─────────────────────────────────────────────────────────────────
    # Set credentials in one of two ways:
    #   A) GitHub Actions: add PAPER_EMAIL_ADDRESS / PAPER_EMAIL_PASSWORD /
    #      PAPER_EMAIL_TO as repository secrets (Settings → Secrets → Actions)
    #   B) Local / codespace: fill in the defaults below directly
    "email_from":     os.getenv("PAPER_EMAIL_ADDRESS",  "ppsupton08@gmail.com"),
    "email_password": os.getenv("PAPER_EMAIL_PASSWORD", "moxo dkhu qphw olow"),
    "email_to":       os.getenv("PAPER_EMAIL_TO",       "ppsupton08@gmail.com"),
    "smtp_host":      "smtp.gmail.com",
    "smtp_port":      587,

    # ── Capital & risk ────────────────────────────────────────────────────────
    "starting_capital":    10_000.0,   # £ starting paper capital
    "kelly_fraction":      1.0,        # full Kelly
    "max_risk_per_trade":  0.08,       # 8% max risk per trade
    "min_prob":            0.55,       # minimum confidence to open a trade
    "leverage_min_prob":   0.56,
    "leverage_max_prob":   0.63,

    # ── Model paths ───────────────────────────────────────────────────────────
    "base_checkpoint":  "models/base/checkpoint_base_v1.pt",
    "ft_dir":           "models/fine_tuned",
    "n_features":       44,
    "lookback":         60,

    # ── Data ─────────────────────────────────────────────────────────────────
    "raw_dir":          "data/raw",
    "state_file":       "logs/paper_trading/paper_state.json",
    "trade_log":        "logs/paper_trading/trade_log.csv",
    "log_dir":          "logs/paper_trading",

    # ── Instruments to trade ─────────────────────────────────────────────────
    # Format: ("INSTRUMENT_PREFIX", "RESOLUTION", "YFINANCE_TICKER")
    # Add or remove instruments here.
    "instruments": [
        # ── Forex ────────────────────────────────────────────────────────────
        ("FOREXCOM_EURUSDX",  "1D", "EURUSD=X"),
        ("FOREXCOM_GBPUSDX",  "1D", "GBPUSD=X"),
        ("FOREXCOM_USDJPYX",  "1D", "USDJPY=X"),
        ("FOREXCOM_USDCADX",  "1D", "USDCAD=X"),
        ("FOREXCOM_USDCHFX",  "1D", "USDCHF=X"),
        ("FOREXCOM_EURGBPX",  "1D", "EURGBP=X"),
        ("FOREXCOM_EURCHFX",  "1D", "EURCHF=X"),
        ("FOREXCOM_AUDUSDX",  "1D", "AUDUSD=X"),
        ("FOREXCOM_GBPJPYX",  "1D", "GBPJPY=X"),
        ("FOREXCOM_AUDNZDX",  "1D", "AUDNZD=X"),
        # ── Indices ──────────────────────────────────────────────────────────
        ("DAX_GDAXI",         "1D", "^GDAXI"),
        ("FTSE_FTSE",         "1D", "^FTSE"),
        ("HSI_HSI",           "1D", "^HSI"),
        ("NIKKEI_N225",       "1D", "^N225"),
        ("SP500_GSPC",        "1D", "^GSPC"),
        # ── Commodities ──────────────────────────────────────────────────────
        ("COMEX_GCF",         "1D", "GC=F"),    # Gold
        ("COMEX_HGF",         "1D", "HG=F"),    # Copper
        ("COMEX_SIF",         "1D", "SI=F"),    # Silver
        ("NYMEX_CLF",         "1D", "CL=F"),    # Crude Oil
        ("NYMEX_NGF",         "1D", "NG=F"),    # Natural Gas
        ("ICE_BZF",           "1D", "BZ=F"),    # Brent Oil
        ("CBOT_ZWF",          "1D", "ZW=F"),    # Wheat
        # ── Crypto ───────────────────────────────────────────────────────────
        ("CRYPTO_BTC-USD",    "1D", "BTC-USD"),
        ("CRYPTO_ETH-USD",    "1D", "ETH-USD"),
        # ── UK Equities ──────────────────────────────────────────────────────
        ("LSE_BPL",           "1D", "BP.L"),
        ("LSE_GSKL",          "1D", "GSK.L"),
        ("LSE_AZNL",          "1D", "AZN.L"),
        ("LSE_HSBAL",         "1D", "HSBA.L"),
        ("LSE_SHELL",         "1D", "SHEL.L"),
        # ── US Equities ──────────────────────────────────────────────────────
        ("NYSE_AAPL",         "1D", "AAPL"),
        ("NYSE_MSFT",         "1D", "MSFT"),
        ("NYSE_JPM",          "1D", "JPM"),
        ("NYSE_BAC",          "1D", "BAC"),
        ("NYSE_AMZN",         "1D", "AMZN"),
        ("NYSE_GOOGL",        "1D", "GOOGL"),
        ("NYSE_NVDA",         "1D", "NVDA"),
        ("NYSE_TSLA",         "1D", "TSLA"),
        ("NYSE_V",            "1D", "V"),
        ("NYSE_META",         "1D", "META"),
        ("NYSE_XOM",          "1D", "XOM"),
        ("NYSE_CVX",          "1D", "CVX"),
        ("NYSE_JNJ",          "1D", "JNJ"),
        ("NYSE_UNH",          "1D", "UNH"),
        ("NYSE_PG",           "1D", "PG"),
        # ── Canadian Equities ─────────────────────────────────────────────────
        ("TSX_TDTO",          "1D", "TD.TO"),
        ("TSX_RYTO",          "1D", "RY.TO"),
        ("TSX_CNRTO",         "1D", "CNR.TO"),
        ("TSX_MFCTO",         "1D", "MFC.TO"),
        # ── Australian Equities ───────────────────────────────────────────────
        ("ASX_ANZAX",         "1D", "ANZ.AX"),
        ("ASX_BHPAX",         "1D", "BHP.AX"),
        ("ASX_CBAAX",         "1D", "CBA.AX"),
        ("ASX_CSLAX",         "1D", "CSL.AX"),
        # ── European Equities ─────────────────────────────────────────────────
        ("EURONEXT_SAPDE",    "1D", "SAP.DE"),
        ("EURONEXT_SIEDE",    "1D", "SIE.DE"),
        ("EURONEXT_ASMLAS",   "1D", "ASML.AS"),
        ("EURONEXT_MCPA",     "1D", "MC.PA"),
        ("EURONEXT_TTEPA",    "1D", "TTE.PA"),
    ],

    # ── Index reference for correlation feature ───────────────────────────────
    "index_ticker": "^GSPC",    # S&P 500
    "index_bars":   300,        # bars to download for index (rolling correlation)

    # ── How many bars to download per instrument ──────────────────────────────
    # Must be > LOOKBACK (60) + Z_WIN (60) + buffer. 250 = ~1 year daily.
    "bars_to_download": 250,

    # ── Daily summary time ────────────────────────────────────────────────────
    # The script checks if it's past this hour and sends a summary if not sent today.
    "summary_hour": 17,    # 5pm
}

# ─── Logging ──────────────────────────────────────────────────────────────────
os.makedirs(CONFIG["log_dir"], exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(CONFIG["log_dir"], "paper_trading.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

HORIZONS = [1, 5, 20]

# ─── Confidence classification ────────────────────────────────────────────────
def classify_confidence(prob: float) -> str:
    edge = abs(prob - 0.5)
    if edge >= 0.15: return "Very High"
    if edge >= 0.10: return "High"
    if edge >= 0.06: return "Moderate"
    if edge >= 0.03: return "Low"
    return "Very Low"

def confidence_pct(prob: float) -> float:
    """Convert probability to confidence percentage (0-100%)."""
    return round(abs(prob - 0.5) * 200, 1)   # 0.5→0%, 1.0→100%


# ─── State management ─────────────────────────────────────────────────────────
def load_state() -> dict:
    path = CONFIG["state_file"]
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "positions":     {},
        "capital":       CONFIG["starting_capital"],
        "realised_pnl":  0.0,
        "trade_count":   0,
        "last_run":      None,
        "last_summary":  None,
    }


def save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(CONFIG["state_file"]), exist_ok=True)
    with open(CONFIG["state_file"], "w") as f:
        json.dump(state, f, indent=2, default=str)


def log_trade(row: dict) -> None:
    path = CONFIG["trade_log"]
    df   = pd.DataFrame([row])
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


# ─── Data fetching ─────────────────────────────────────────────────────────────
def fetch_bars(ticker: str, resolution: str, n_bars: int) -> pd.DataFrame | None:
    """Download OHLCV bars from Yahoo Finance."""
    interval_map = {"1D": "1d", "1W": "1wk", "4H": "1h", "1H": "1h"}
    yf_interval  = interval_map.get(resolution, "1d")

    # For intraday we download more to resample
    period = "2y" if resolution in ("1D", "1W") else "60d"
    try:
        df = yf.download(ticker, period=period, interval=yf_interval,
                         auto_adjust=False, progress=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index.name = "Datetime"
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Resample 1H → 4H
        if resolution == "4H":
            df = df.resample("4h").agg({
                "Open": "first", "High": "max",
                "Low":  "min",   "Close": "last",
                "Volume": "sum",
            }).dropna()

        return df.tail(n_bars)
    except Exception as e:
        log.warning("Download failed for %s: %s", ticker, e)
        return None


# ─── Feature engineering (reuses feature_engineering.py functions) ────────────
NON_ZSCORE_COLS = {
    "bb_position", "rsi14", "rsi28", "range_percentile",
    "donchian_position", "stoch_k_14", "williams_r_14", "consec_bars",
    "dow_sin", "dow_cos", "month_sin", "month_cos", "quarter_end_flag",
    "hour_sin", "hour_cos", "session_asian", "session_london",
    "session_newyork", "session_overlap",
}

Z_WIN  = 60
Z_CLIP = 3.0


def build_feature_window(df: pd.DataFrame, resolution: str,
                         index_df: pd.DataFrame | None) -> np.ndarray | None:
    """
    Build a single feature window from a raw OHLCV DataFrame.
    Returns shape [1, LOOKBACK, N_FEATURES] or None on failure.
    """
    LOOKBACK   = CONFIG["lookback"]
    N_FEATURES = CONFIG["n_features"]
    MIN_ROWS   = LOOKBACK + Z_WIN + 10

    if len(df) < MIN_ROWS:
        log.debug("Not enough bars: %d (need %d)", len(df), MIN_ROWS)
        return None

    df = df[df["Close"] > 0].dropna(subset=["Open", "High", "Low", "Close"])

    try:
        ohlc  = compute_ohlc_features(df)
        tech  = compute_technical_features(df)
        cal   = compute_calendar_features(df, resolution)
        corr  = add_index_correlation(df, index_df, resolution).to_frame()

        all_feats = pd.concat([ohlc, tech, corr, cal], axis=1)
        all_feats = all_feats.loc[df.index]

        cal_cols   = list(cal.columns)
        normalised = apply_rolling_zscore_pipeline(all_feats,
                                                   non_zscore_cols=set(cal_cols))
        normalised = normalised.dropna()

        if len(normalised) < LOOKBACK + 5:
            return None

        # Take last LOOKBACK bars as one window
        window = normalised.iloc[-LOOKBACK:].values.astype(np.float32)
        n_feat = window.shape[1]
        if n_feat < N_FEATURES:
            window = np.pad(window, ((0, 0), (0, N_FEATURES - n_feat)))
        elif n_feat > N_FEATURES:
            window = window[:, :N_FEATURES]

        return window[np.newaxis, :, :]     # [1, LOOKBACK, N_FEATURES]

    except Exception as e:
        log.warning("Feature build failed: %s", e)
        return None


# ─── Model loading ─────────────────────────────────────────────────────────────
_model_cache: dict = {}


# Cache for finetune_summary recommendations — loaded once on first call
_ft_summary_cache: dict | None = None


def _load_ft_summary() -> dict:
    """
    Load finetune_summary.csv and return a dict mapping
    "INSTRUMENT_RESOLUTION" → "fine_tuned" | "base"

    The summary CSV lives at logs/evaluation/finetune_summary.csv and is
    produced by run_eval_finetuned.sh. Columns include:
      instrument, resolution, base_h1, ft_h1 (and h5/h20 variants)

    Decision: use fine-tuned if ft_h1 > base_h1, else use base.
    If the CSV is missing, default to fine-tuned when checkpoint exists.
    """
    global _ft_summary_cache
    if _ft_summary_cache is not None:
        return _ft_summary_cache

    summary_path = os.path.join("logs", "evaluation", "finetune_summary.csv")
    rec = {}

    if os.path.exists(summary_path):
        try:
            df = pd.read_csv(summary_path)
            for _, row in df.iterrows():
                key = f"{row['instrument']}_{row['resolution']}"
                # Use fine-tuned only if it outperforms base on H1
                ft_h1   = float(row.get("ft_h1",   0) or 0)
                base_h1 = float(row.get("base_h1", 0) or 0)
                rec[key] = "fine_tuned" if ft_h1 > base_h1 else "base"
            log.info("Loaded finetune_summary.csv — %d recommendations", len(rec))
        except Exception as e:
            log.warning("Could not load finetune_summary.csv: %s — "
                        "defaulting to fine-tuned where available", e)
    else:
        log.debug("finetune_summary.csv not found — "
                  "defaulting to fine-tuned where available")

    _ft_summary_cache = rec
    return rec


def load_model(instrument: str, resolution: str, device: str):
    """
    Load the best available model for this instrument+resolution.

    Priority:
      1. Check finetune_summary.csv recommendation for this instrument.
      2. If recommendation is 'fine_tuned' AND a checkpoint exists → use it.
      3. If recommendation is 'base' OR no fine-tuned checkpoint exists → use base.
      4. If base checkpoint also missing → return None, None.
    """
    key = f"{instrument}_{resolution}"
    if key in _model_cache:
        return _model_cache[key]

    recommendations = _load_ft_summary()
    recommendation  = recommendations.get(key, "fine_tuned")  # default: try ft

    ckpt_path = None
    source    = "base"

    if recommendation == "fine_tuned":
        # Try fine-tuned phase3 first, then phase1
        for phase in ("phase3", "phase1"):
            pattern = os.path.join(
                CONFIG["ft_dir"],
                f"checkpoint_{instrument}_{resolution}_{phase}_*.pt"
            )
            matches = sorted(glob.glob(pattern))
            if matches:
                ckpt_path = matches[-1]
                source    = f"fine-tuned ({phase})"
                break

    if ckpt_path is None or not os.path.exists(ckpt_path):
        # Fall back to base
        ckpt_path = CONFIG["base_checkpoint"]
        source    = "base"
        if recommendation == "fine_tuned" and key in recommendations:
            log.debug("%s: ft checkpoint not found — falling back to base", key)
        elif recommendation == "base":
            log.debug("%s: finetune_summary recommends base model", key)

    if not os.path.exists(ckpt_path):
        log.error("No checkpoint found for %s (tried ft and base)", key)
        return None, None

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = build_model(n_features=CONFIG["n_features"], device=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _model_cache[key] = (model, source)
    return model, source


# ─── Inference ────────────────────────────────────────────────────────────────
def predict(model, window: np.ndarray, device: str) -> dict:
    """
    Run inference on one window.
    Returns dict with H1/H5/H20 probabilities and magnitude predictions.
    """
    x = torch.from_numpy(window).to(device)
    with torch.no_grad():
        preds = model(x)

    result = {}
    for h in HORIZONS:
        dir_prob, mag = preds[h]
        result[f"h{h}_prob"] = float(dir_prob[0].cpu())
        result[f"h{h}_mag"]  = float(mag[0].cpu())

    return result


def interpret_prediction(pred: dict, current_price: float,
                          instrument: str) -> dict:
    """
    Convert raw model outputs into human-readable signals.

    Direction: UP if prob > 0.5, DOWN if prob < 0.5
    Confidence: |prob - 0.5| as percentage and label

    Signal%: a combined conviction score above the coin-flip baseline.
      Computed as: sign × |magnitude_z_score| × asset_class_sigma × 100
      A higher Signal% means the model has stronger conviction on BOTH
      the direction AND the magnitude of the predicted move.
      This is NOT a price % forecast — it is a dimensionless signal
      strength indicator. 0% = no conviction, ~30% = high conviction.
    """
    asset_sigma = _asset_sigma(instrument)

    signals = {}
    for h in HORIZONS:
        prob = pred[f"h{h}_prob"]
        mag  = pred[f"h{h}_mag"]

        direction  = "UP" if prob >= 0.5 else "DOWN"
        conf_label = classify_confidence(prob)
        conf_pct   = confidence_pct(prob)

        # Estimated % price move (sign follows predicted direction)
        sign       = 1 if prob >= 0.5 else -1
        est_move   = sign * abs(mag) * asset_sigma * 100
        # Signal%: model's conviction in the move direction above a coin flip.
        # NOT a price % forecast. 0%=no edge, 100%=max conviction.
        # Larger values = model has stronger conviction on magnitude of move.

        signals[f"h{h}"] = {
            "prob":      round(prob, 4),
            "direction": direction,
            "conf_label": conf_label,
            "conf_pct":   conf_pct,
            "est_move":   round(est_move, 3),
        }

    return signals


def _asset_sigma(instrument: str) -> float:
    """Approximate 1-sigma daily % move for the instrument's asset class."""
    if "FOREXCOM" in instrument: return 0.5
    if any(x in instrument for x in ["COMEX", "NYMEX", "ICE", "CBOT"]): return 0.9
    if any(x in instrument for x in ["DAX", "FTSE", "NIKKEI", "HSI", "SP500"]): return 0.8
    if "CRYPTO" in instrument: return 3.5
    return 1.2   # equities default


# ─── Position management ──────────────────────────────────────────────────────
def compute_position_size(prob: float, capital: float, instrument: str,
                           resolution: str) -> tuple[float, float]:
    """
    Compute GBP position size and leverage using Kelly + asset-class limits.
    Returns (size_gbp, leverage).
    """
    max_lev = get_leverage_for_source(f"{instrument}_{resolution}")

    # Linear leverage ramp between min and max prob
    lev_min = CONFIG["leverage_min_prob"]
    lev_max = CONFIG["leverage_max_prob"]
    if prob < lev_min:
        leverage = 1.0
    elif prob > lev_max:
        leverage = max_lev
    else:
        t        = (prob - lev_min) / (lev_max - lev_min)
        leverage = 1.0 + t * (max_lev - 1.0)

    leverage = min(leverage, max_lev)

    # Kelly fraction of capital to deploy as margin.
    # MAX_RISK caps the MARGIN (capital at risk), NOT margin × leverage.
    # Leverage amplifies P&L but does not change how much capital you deploy.
    # e.g. 8% MAX_RISK = max £800 margin regardless of whether leverage is 1x or 5x.
    edge     = prob - 0.5
    fraction = CONFIG["kelly_fraction"] * (2 * edge)
    size_gbp = min(capital * fraction,
                   capital * CONFIG["max_risk_per_trade"])
    size_gbp = max(size_gbp, 0.0)

    return round(size_gbp, 2), round(leverage, 2)


def current_price_from_df(df: pd.DataFrame) -> float | None:
    """
    Return the most recent valid (non-NaN, positive) close price.
    yfinance sometimes returns NaN for the latest bar of indices like ^GDAXI
    when the market hasn't closed yet or data is delayed.
    Walk back through recent bars to find the last valid price.
    """
    import math
    closes = df["Close"].dropna()
    closes = closes[closes > 0]
    if closes.empty:
        return None
    return float(closes.iloc[-1])


# ─── Email ────────────────────────────────────────────────────────────────────
def _email_cfg_ok() -> bool:
    return bool(CONFIG["email_from"] and CONFIG["email_password"] and CONFIG["email_to"])


def send_email(subject: str, html_body: str) -> None:
    if not _email_cfg_ok():
        log.warning("Email not configured — skipping send.")
        log.warning("  email_from    : %s", CONFIG["email_from"]     or "*** EMPTY ***")
        log.warning("  email_password: %s", "***set***" if CONFIG["email_password"] else "*** EMPTY ***")
        log.warning("  email_to      : %s", CONFIG["email_to"]       or "*** EMPTY ***")
        log.warning("  Edit the CONFIG block at the top of paper_trading.py and fill in your Gmail details.")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = CONFIG["email_from"]
    msg["To"]      = CONFIG["email_to"]
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(CONFIG["smtp_host"], CONFIG["smtp_port"]) as server:
            server.ehlo()
            server.starttls()
            server.login(CONFIG["email_from"], CONFIG["email_password"])
            server.sendmail(CONFIG["email_from"], CONFIG["email_to"], msg.as_string())
        log.info("Email sent: %s", subject)
    except Exception as e:
        log.error("Email failed: %s", e)


# ─── HTML helpers ─────────────────────────────────────────────────────────────
_CSS = """
  body { font-family: Arial, sans-serif; font-size: 14px; color: #222; }
  h2   { color: #1a3a5c; border-bottom: 2px solid #1a3a5c; padding-bottom: 6px; }
  h3   { color: #2c5f8a; margin-top: 20px; }
  table { border-collapse: collapse; width: 100%; margin: 10px 0; }
  th { background: #1a3a5c; color: #fff; padding: 8px 10px; text-align: left; }
  td { padding: 7px 10px; border-bottom: 1px solid #ddd; }
  tr:nth-child(even) { background: #f5f8fc; }
  .up   { color: #1a7a1a; font-weight: bold; }
  .down { color: #c0392b; font-weight: bold; }
  .tag-vh { background:#1a7a1a; color:#fff; border-radius:4px; padding:2px 6px; font-size:12px; }
  .tag-h  { background:#2ecc71; color:#fff; border-radius:4px; padding:2px 6px; font-size:12px; }
  .tag-m  { background:#f39c12; color:#fff; border-radius:4px; padding:2px 6px; font-size:12px; }
  .tag-l  { background:#e67e22; color:#fff; border-radius:4px; padding:2px 6px; font-size:12px; }
  .tag-vl { background:#c0392b; color:#fff; border-radius:4px; padding:2px 6px; font-size:12px; }
  .pnl-pos { color: #1a7a1a; }
  .pnl-neg { color: #c0392b; }
  .footer  { color: #888; font-size: 12px; margin-top: 20px; }
"""


def _conf_tag(label: str) -> str:
    cls = {"Very High": "vh", "High": "h", "Moderate": "m",
           "Low": "l", "Very Low": "vl"}.get(label, "vl")
    return f'<span class="tag-{cls}">{label}</span>'


def _dir_html(direction: str) -> str:
    arrow = "↑ UP" if direction == "UP" else "↓ DOWN"
    cls   = "up" if direction == "UP" else "down"
    return f'<span class="{cls}">{arrow}</span>'


def _pnl_html(pnl: float) -> str:
    cls = "pnl-pos" if pnl >= 0 else "pnl-neg"
    return f'<span class="{cls}">{pnl:+.2f}</span>'


def prediction_table_html(predictions: list[dict]) -> str:
    """Build HTML table of all predictions."""
    rows = ""
    for p in predictions:
        sig   = p["signals"]["h1"]
        sig5  = p["signals"]["h5"]
        sig20 = p["signals"]["h20"]
        rows += f"""
        <tr>
          <td><b>{p['instrument']}</b></td>
          <td>{p['resolution']}</td>
          <td>{p['current_price']:.5g}</td>
          <td>{_dir_html(sig['direction'])}</td>
          <td>{sig['prob']:.3f}</td>
          <td>{sig5['prob']:.3f}</td>
          <td>{sig20['prob']:.3f}</td>
          <td>{_conf_tag(sig['conf_label'])} {sig['conf_pct']:.0f}%</td>
          <td>{sig['est_move']:+.2f}%</td>
          <td><i>{p['model_source']}</i></td>
        </tr>"""

    return f"""
    <h3>📊 Market Predictions (H1 = next 1 bar)</h3>
    <table>
      <tr>
        <th>Instrument</th><th>Res</th><th>Price</th>
        <th>Direction</th><th>H1 Prob</th><th>H5 Prob</th><th>H20 Prob</th>
        <th>Confidence</th><th>Signal%</th><th>Model</th>
      </tr>
      {rows}
    </table>
    <p style="font-size:12px;color:#888;">
      Probabilities: H1=next bar, H5=next 5 bars, H20=next 20 bars.<br>
      Confidence = how far the probability deviates from 50% (coin flip).<br>
      Signal% = model's conviction above a coin flip (0% = no edge, ~30% = high conviction).<br>
      &nbsp;&nbsp;This is NOT a price % forecast — it reflects how strongly the model believes in the direction.<br>
    </p>"""


def trade_alert_html(action: str, instrument: str, resolution: str,
                      direction: str, price: float, size_gbp: float,
                      leverage: float, prob: float, signals: dict,
                      pnl_gbp: float | None = None,
                      entry_price: float | None = None) -> str:
    action_colour = {"OPEN": "#1a7a1a", "CLOSE": "#c0392b", "HOLD": "#2c5f8a"}.get(action, "#222")
    action_label  = {"OPEN": "🟢 TRADE OPENED", "CLOSE": "🔴 TRADE CLOSED",
                     "HOLD": "🔵 POSITION HELD"}.get(action, action)

    pnl_row = ""
    if pnl_gbp is not None and entry_price is not None:
        pnl_pct = (price / entry_price - 1) * (1 if direction == "UP" else -1) * 100
        pnl_row = f"""
        <tr><td><b>Realised P&amp;L</b></td>
            <td>{_pnl_html(pnl_gbp)} ({pnl_pct:+.2f}%)</td></tr>
        <tr><td><b>Entry Price</b></td><td>{entry_price:.5g}</td></tr>"""

    sig  = signals.get("h1", {})
    sig5 = signals.get("h5", {})
    sig20= signals.get("h20", {})

    return f"""
    <style>{_CSS}</style>
    <h2 style="color:{action_colour}">{action_label}</h2>
    <table>
      <tr><td><b>Instrument</b></td><td>{instrument} {resolution}</td></tr>
      <tr><td><b>Direction</b></td><td>{_dir_html(direction)}</td></tr>
      <tr><td><b>Price</b></td><td>{price:.5g}</td></tr>
      <tr><td><b>Size (paper)</b></td><td>£{size_gbp:.2f}</td></tr>
      <tr><td><b>Leverage</b></td><td>{leverage:.1f}×</td></tr>
      <tr><td><b>H1 Confidence</b></td>
          <td>{_conf_tag(sig.get('conf_label','?'))} {sig.get('conf_pct',0):.0f}% edge<br>
          <small style="color:#555">Raw model prob: {prob:.3f} &rarr;
          {'DOWN' if prob < 0.5 else 'UP'} probability:
          {(1-prob) if prob < 0.5 else prob:.1%}</small></td></tr>
      {pnl_row}
    </table>

    <h3>Model Predictions</h3>
    <table>
      <tr><th>Horizon</th><th>Direction</th><th>Raw Prob</th>
          <th>Dir. Prob</th><th>Confidence</th><th>Signal%</th></tr>
      <tr>
        <td>H1 (next bar)</td>
        <td>{_dir_html(sig.get('direction','?'))}</td>
        <td>{sig.get('prob',0):.3f}</td>
        <td>{(1-sig.get('prob',0.5)) if sig.get('direction')=='DOWN' else sig.get('prob',0.5):.1%}</td>
        <td>{_conf_tag(sig.get('conf_label','?'))} {sig.get('conf_pct',0):.0f}% edge</td>
        <td>{sig.get('est_move',0):+.2f}%</td>
      </tr>
      <tr>
        <td>H5 (5 bars)</td>
        <td>{_dir_html(sig5.get('direction','?'))}</td>
        <td>{sig5.get('prob',0):.3f}</td>
        <td>{(1-sig5.get('prob',0.5)) if sig5.get('direction')=='DOWN' else sig5.get('prob',0.5):.1%}</td>
        <td>{_conf_tag(sig5.get('conf_label','?'))} {sig5.get('conf_pct',0):.0f}% edge</td>
        <td>{sig5.get('est_move',0):+.2f}%</td>
      </tr>
      <tr>
        <td>H20 (20 bars)</td>
        <td>{_dir_html(sig20.get('direction','?'))}</td>
        <td>{sig20.get('prob',0):.3f}</td>
        <td>{(1-sig20.get('prob',0.5)) if sig20.get('direction')=='DOWN' else sig20.get('prob',0.5):.1%}</td>
        <td>{_conf_tag(sig20.get('conf_label','?'))} {sig20.get('conf_pct',0):.0f}% edge</td>
        <td>{sig20.get('est_move',0):+.2f}%</td>
      </tr>
    </table>
    <p class="footer">Paper trading — simulated positions only. No real money involved.</p>"""


def daily_summary_html(state: dict, all_predictions: list[dict],
                        days_running: int) -> str:
    positions = state.get("positions", {})
    capital   = state["capital"]
    realised  = state.get("realised_pnl", 0.0)
    today     = date.today().strftime("%A %d %B %Y")

    # Open positions table
    pos_rows = ""
    unrealised = 0.0
    for key, pos in positions.items():
        inst, res = key.rsplit("_", 1)
        ep    = pos["entry_price"]
        cp    = pos.get("current_price", ep)
        dirn  = pos["direction"]
        size  = pos["size_gbp"]
        lev   = pos.get("leverage", 1.0)
        pnl   = (cp / ep - 1) * dirn * size * lev
        unrealised += pnl
        days  = (date.today() - date.fromisoformat(pos["entry_date"])).days

        pos_rows += f"""
        <tr>
          <td>{inst}</td><td>{res}</td>
          <td>{_dir_html("UP" if dirn == 1 else "DOWN")}</td>
          <td>{ep:.5g}</td><td>{cp:.5g}</td>
          <td>{lev:.1f}×</td><td>£{size:.2f}</td>
          <td>{_pnl_html(pnl)} ({pnl/size*100:+.1f}%)</td>
          <td>{days}d</td>
        </tr>"""

    if not pos_rows:
        pos_rows = '<tr><td colspan="9" style="text-align:center;color:#888">No open positions</td></tr>'

    total_pnl = realised + unrealised
    total_pct = total_pnl / CONFIG["starting_capital"] * 100

    return f"""
    <style>{_CSS}</style>
    <h2>📈 Daily Paper Trading Summary — {today}</h2>

    <h3>Portfolio Overview</h3>
    <table>
      <tr><td><b>Starting Capital</b></td><td>£{CONFIG['starting_capital']:,.2f}</td></tr>
      <tr><td><b>Current Capital</b></td><td>£{capital:,.2f}</td></tr>
      <tr><td><b>Realised P&amp;L</b></td><td>{_pnl_html(realised)}</td></tr>
      <tr><td><b>Unrealised P&amp;L</b></td><td>{_pnl_html(unrealised)}</td></tr>
      <tr><td><b>Total P&amp;L</b></td>
          <td>{_pnl_html(total_pnl)} ({total_pct:+.1f}%)</td></tr>
      <tr><td><b>Open Positions</b></td><td>{len(positions)}</td></tr>
      <tr><td><b>Total Trades</b></td><td>{state.get('trade_count', 0)}</td></tr>
      <tr><td><b>Days Running</b></td><td>{days_running}</td></tr>
    </table>

    <h3>Open Positions</h3>
    <table>
      <tr>
        <th>Instrument</th><th>Res</th><th>Direction</th>
        <th>Entry</th><th>Current</th><th>Leverage</th>
        <th>Size</th><th>Unrealised P&amp;L</th><th>Held</th>
      </tr>
      {pos_rows}
    </table>

    {prediction_table_html(all_predictions)}

    <p class="footer">
      Paper trading — simulated positions only. No real money involved.<br>
      Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </p>"""


# ─── Core trading cycle ───────────────────────────────────────────────────────

def predict_only(instrument: str, resolution: str, yf_ticker: str,
                 device: str, index_dfs: dict) -> dict | None:
    """
    Run inference for one instrument WITHOUT making any trading decisions.
    Used by --summary so it never accidentally opens/closes positions.
    Returns the same prediction dict shape as run_instrument() but with
    action=None and no state modifications.
    """
    log.info("  [predict] %s %s", instrument, resolution)

    df = fetch_bars(yf_ticker, resolution, CONFIG["bars_to_download"])
    if df is None or len(df) < 130:
        return None

    current_price = current_price_from_df(df)
    if current_price is None or __import__("math").isnan(current_price):
        return None

    idx_df = index_dfs.get(resolution)
    window = build_feature_window(df, resolution, idx_df)
    if window is None:
        return None

    model, source = load_model(instrument, resolution, device)
    if model is None:
        return None

    pred    = predict(model, window, device)
    signals = interpret_prediction(pred, current_price, instrument)

    return {
        "instrument":    instrument,
        "resolution":    resolution,
        "current_price": round(current_price, 5),
        "signals":       signals,
        "model_source":  source,
        "action":        None,
        "h1_prob":       pred["h1_prob"],
    }


def run_instrument(instrument: str, resolution: str, yf_ticker: str,
                   state: dict, device: str,
                   index_dfs: dict) -> dict | None:
    """
    Run one full cycle for one instrument.
    Returns prediction dict for the summary email, or None on failure.
    """
    key = f"{instrument}_{resolution}"
    log.info("Processing: %s %s", instrument, resolution)

    # ── Fetch data ────────────────────────────────────────────────────────────
    df = fetch_bars(yf_ticker, resolution, CONFIG["bars_to_download"])
    if df is None or len(df) < 130:
        log.warning("  Insufficient data for %s", key)
        return None

    current_price = current_price_from_df(df)
    if current_price is None or (hasattr(current_price, '__float__') and
                                  __import__('math').isnan(current_price)):
        log.warning("  No valid price for %s — skipping this run", key)
        return None

    # ── Feature engineering ───────────────────────────────────────────────────
    idx_df = index_dfs.get(resolution)
    window = build_feature_window(df, resolution, idx_df)
    if window is None:
        log.warning("  Feature build failed for %s", key)
        return None

    # ── Model inference ───────────────────────────────────────────────────────
    model, source = load_model(instrument, resolution, device)
    if model is None:
        return None

    pred    = predict(model, window, device)
    signals = interpret_prediction(pred, current_price, instrument)

    h1_prob = pred["h1_prob"]
    h1_dir  = 1 if h1_prob >= 0.5 else -1
    h1_sig  = signals["h1"]

    # ── Decision logic ────────────────────────────────────────────────────────
    existing = state["positions"].get(key)
    action   = None
    pnl_gbp  = None
    entry_price_for_alert = None

    if existing:
        # Only update current_price if the new value is valid
        if current_price and not __import__('math').isnan(current_price):
            existing["current_price"] = current_price

        # Close if model flips direction or confidence drops below threshold
        if h1_prob < CONFIG["min_prob"]:
            action = "CLOSE"
            reason = f"prob={h1_prob:.3f} below threshold {CONFIG['min_prob']}"
        elif existing["direction"] != h1_dir:
            action = "CLOSE"
            reason = f"direction flip (was {'long' if existing['direction']==1 else 'short'}, now {'long' if h1_dir==1 else 'short'})"
        else:
            action = "HOLD"
            reason = "signal unchanged"

        if action == "CLOSE":
            ep      = existing["entry_price"]
            size    = existing["size_gbp"]
            lev     = existing.get("leverage", 1.0)
            pnl_gbp = (current_price / ep - 1) * existing["direction"] * size * lev

            state["capital"]      += pnl_gbp
            state["realised_pnl"]  = state.get("realised_pnl", 0.0) + pnl_gbp
            state["trade_count"]   = state.get("trade_count", 0) + 1
            entry_price_for_alert  = ep

            log_trade({
                "date":        datetime.now().isoformat(),
                "instrument":  instrument,
                "resolution":  resolution,
                "action":      "CLOSE",
                "direction":   "long" if existing["direction"] == 1 else "short",
                "entry_price": ep,
                "exit_price":  current_price,
                "size_gbp":    size,
                "leverage":    lev,
                "pnl_gbp":     round(pnl_gbp, 2),
                "pnl_pct":     round(pnl_gbp / size * 100, 3),
                "h1_prob":     h1_prob,
                "confidence":  h1_sig["conf_label"],
                "reason":      reason,
            })
            del state["positions"][key]

            # Immediately open in new direction if confident enough
            if h1_prob >= CONFIG["min_prob"] or (1 - h1_prob) >= CONFIG["min_prob"]:
                action = "OPEN"
                reason = "flip — reopen in new direction"

    # Open new position
    if existing is None or action == "OPEN":
        action_prob = h1_prob if h1_dir == 1 else (1 - h1_prob)
        if action_prob >= CONFIG["min_prob"]:
            size_gbp, leverage = compute_position_size(
                action_prob, state["capital"], instrument, resolution
            )
            if size_gbp > 0:
                action = "OPEN"
                reason = f"new signal: {'long' if h1_dir==1 else 'short'} prob={action_prob:.3f}"
                state["positions"][key] = {
                    "direction":    h1_dir,
                    "entry_price":  current_price,
                    "current_price": current_price,
                    "entry_date":   date.today().isoformat(),
                    "size_gbp":     size_gbp,
                    "leverage":     leverage,
                }
                state["trade_count"] = state.get("trade_count", 0) + 1
                log_trade({
                    "date":        datetime.now().isoformat(),
                    "instrument":  instrument,
                    "resolution":  resolution,
                    "action":      "OPEN",
                    "direction":   "long" if h1_dir == 1 else "short",
                    "entry_price": current_price,
                    "exit_price":  "",
                    "size_gbp":    size_gbp,
                    "leverage":    leverage,
                    "pnl_gbp":     "",
                    "pnl_pct":     "",
                    "h1_prob":     h1_prob,
                    "confidence":  h1_sig["conf_label"],
                    "reason":      reason,
                })

    # ── Send trade alert email ────────────────────────────────────────────────
    if action in ("OPEN", "CLOSE"):
        pos      = state["positions"].get(key, {})
        size_gbp = pos.get("size_gbp", existing.get("size_gbp", 0) if existing else 0)
        leverage = pos.get("leverage", existing.get("leverage", 1.0) if existing else 1.0)
        dirn_str = "UP" if h1_dir == 1 else "DOWN"

        html = trade_alert_html(
            action       = action,
            instrument   = instrument,
            resolution   = resolution,
            direction    = dirn_str,
            price        = current_price,
            size_gbp     = size_gbp,
            leverage     = leverage,
            prob         = h1_prob,
            signals      = signals,
            pnl_gbp      = pnl_gbp,
            entry_price  = entry_price_for_alert,
        )
        emoji  = "🟢" if action == "OPEN" else "🔴"
        dirn_s = "↑ Long" if h1_dir == 1 else "↓ Short"
        send_email(
            f"{emoji} Paper Trade {action}: {instrument} {resolution} {dirn_s}",
            html
        )

    log.info("  %s %s | price=%.5g | h1=%.3f %s | conf=%s | action=%s",
             instrument, resolution, current_price, h1_prob,
             "↑" if h1_dir == 1 else "↓",
             h1_sig["conf_label"], action or "no-trade")

    return {
        "instrument":    instrument,
        "resolution":    resolution,
        "current_price": round(current_price, 5),
        "signals":       signals,
        "model_source":  source,
        "action":        action,
        "h1_prob":       h1_prob,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def _run_cycle(device: str) -> None:
    """Execute one full trading cycle. Called by daemon mode and directly by main()."""
    state = load_state()

    # Load S&P 500 index
    index_dfs = {}
    for res, period in [("1D", "2y"), ("1W", "5y"), ("4H", "60d"), ("1H", "60d")]:
        yf_int = {"1D": "1d", "1W": "1wk", "4H": "1h", "1H": "1h"}[res]
        idx = yf.download(CONFIG["index_ticker"], period=period,
                          interval=yf_int, auto_adjust=False, progress=False)
        if idx is not None and not idx.empty:
            if isinstance(idx.columns, pd.MultiIndex):
                idx.columns = idx.columns.get_level_values(0)
            idx.index.name = "Datetime"
            if idx.index.tz is not None:
                idx.index = idx.index.tz_localize(None)
            if res == "4H":
                idx = idx.resample("4h").agg({
                    "Open": "first", "High": "max", "Low": "min",
                    "Close": "last", "Volume": "sum"}).dropna()
            index_dfs[res] = idx

    if "start_date" not in state:
        state["start_date"] = date.today().isoformat()

    all_predictions = []
    for inst, res, ticker in CONFIG["instruments"]:
        try:
            pred = run_instrument(inst, res, ticker, state, device, index_dfs)
            if pred:
                all_predictions.append(pred)
        except Exception as e:
            log.error("Error on %s %s: %s", inst, res, e)

    state["last_run"] = datetime.now().isoformat()

    last_summary = state.get("last_summary")
    today_str    = date.today().isoformat()
    now_hour     = datetime.now().hour

    if (now_hour >= CONFIG["summary_hour"] and
            (last_summary is None or not last_summary.startswith(today_str))):
        log.info("Sending daily summary email...")
        start_date = state.get("start_date", today_str)
        days = (date.today() - date.fromisoformat(start_date)).days + 1
        html = daily_summary_html(state, all_predictions, days)
        send_email(
            f"📊 Paper Trading Daily Summary — {date.today().strftime('%d %b %Y')}",
            html
        )
        state["last_summary"] = datetime.now().isoformat()

    save_state(state)

    if all_predictions:
        print(f"\n{'─'*100}")
        print(f"{'Instrument':<28} {'Res':<4} {'Price':>9}  {'H1':>6} {'H5':>6} {'H20':>6}  "
              f"{'Dir':<5} {'Conf':<10} {'Signal%':>8}  Action")
        print(f"{'─'*100}")
        for p in all_predictions:
            s   = p["signals"]["h1"]
            s5  = p["signals"]["h5"]
            s20 = p["signals"]["h20"]
            arr = "↑" if s["direction"] == "UP" else "↓"
            act = p.get("action") or "hold"
            print(f"{p['instrument']:<28} {p['resolution']:<4} "
                  f"{p['current_price']:>9.5g}  "
                  f"{s['prob']:>6.3f} {s5['prob']:>6.3f} {s20['prob']:>6.3f}  "
                  f"{arr} {s['direction']:<4} "
                  f"{s['conf_label']:<10} {s['est_move']:>+8.2f}%  {act}")
        print(f"{'─'*100}\n")

    log.info("Cycle complete. Capital=£%.2f  Open positions=%d",
             state["capital"], len(state.get("positions", {})))


def main():
    parser = argparse.ArgumentParser(description="Paper Trading Engine")
    parser.add_argument("--summary",    action="store_true",
                        help="Force send daily summary email now and exit.")
    parser.add_argument("--instrument", default=None,
                        help="Run for a single instrument only.")
    parser.add_argument("--resolution", default="1D",
                        help="Resolution for --instrument (default: 1D).")
    parser.add_argument("--daemon", action="store_true",
                        help="Run continuously, executing the trading cycle "
                             "each weekday at the configured run_hour. "
                             "Leave this running in a terminal or screen session.")
    parser.add_argument("--run-hour", type=int, default=9,
                        help="Hour (0-23 UTC) to run the trading cycle in "
                             "daemon mode (default: 9 = 9am UTC).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Paper Trading Engine — device=%s", device)

    # ── Daemon mode ───────────────────────────────────────────────────────────
    if args.daemon:
        import time as _time
        run_hour = args.run_hour
        log.info("Daemon mode started — will run trading cycle at %02d:00 UTC "
                 "on weekdays (Mon-Fri). Keep this terminal open.", run_hour)
        log.info("Press Ctrl+C to stop.")

        last_run_date = None

        while True:
            now = datetime.utcnow()
            is_weekday  = now.weekday() < 5          # Mon=0 … Fri=4
            is_run_hour = now.hour == run_hour
            already_ran = (last_run_date == now.date())

            if is_weekday and is_run_hour and not already_ran:
                log.info("Daemon: running trading cycle (%s)", now.strftime("%Y-%m-%d %H:%M UTC"))
                try:
                    # Re-import state fresh each run so manual edits take effect
                    _run_cycle(device)
                    last_run_date = now.date()
                except Exception as e:
                    log.error("Daemon cycle failed: %s", e)
                # Sleep 65 minutes after running so we don't double-fire within the same hour
                log.info("Daemon: cycle complete. Sleeping 65 minutes.")
                _time.sleep(65 * 60)
            else:
                # Sleep until closer to run time — check every 5 minutes
                next_check = 5 * 60
                if is_weekday and not already_ran:
                    # How many seconds until run_hour?
                    target = now.replace(hour=run_hour, minute=0, second=0, microsecond=0)
                    if target < now:
                        target = target.replace(day=target.day + 1)
                    secs_until = (target - now).total_seconds()
                    if secs_until < 300:
                        next_check = max(int(secs_until), 10)
                _time.sleep(next_check)
        return   # never reached but makes intent clear

    state = load_state()

    # ── Load S&P 500 index for correlation feature ────────────────────────────
    log.info("Loading index reference (S&P 500)...")
    index_dfs = {}
    for res, period in [("1D", "2y"), ("1W", "5y"), ("4H", "60d"), ("1H", "60d")]:
        yf_int = {"1D": "1d", "1W": "1wk", "4H": "1h", "1H": "1h"}[res]
        idx = yf.download(CONFIG["index_ticker"], period=period,
                          interval=yf_int, auto_adjust=False, progress=False)
        if idx is not None and not idx.empty:
            if isinstance(idx.columns, pd.MultiIndex):
                idx.columns = idx.columns.get_level_values(0)
            idx.index.name = "Datetime"
            if idx.index.tz is not None:
                idx.index = idx.index.tz_localize(None)
            if res == "4H":
                idx = idx.resample("4h").agg({
                    "Open":"first","High":"max","Low":"min",
                    "Close":"last","Volume":"sum"}).dropna()
            index_dfs[res] = idx

    # ── Force summary mode ────────────────────────────────────────────────────
    if args.summary:
        log.info("Sending forced daily summary (prediction-only, no trading decisions)...")
        preds = []
        instruments = CONFIG["instruments"]
        if args.instrument:
            instruments = [(i, r, t) for i, r, t in instruments
                           if i == args.instrument and r == args.resolution]
        for inst, res, ticker in instruments:
            p = predict_only(inst, res, ticker, device, index_dfs)
            if p: preds.append(p)

        # Also update current prices in state without making trading decisions
        for p in preds:
            key = f"{p['instrument']}_{p['resolution']}"
            if key in state["positions"] and p.get("current_price"):
                import math
                if not math.isnan(p["current_price"]):
                    state["positions"][key]["current_price"] = p["current_price"]

        start_date = state.get("start_date", date.today().isoformat())
        days = (date.today() - date.fromisoformat(start_date)).days + 1
        html = daily_summary_html(state, preds, days)
        send_email(f"📊 Paper Trading Daily Summary — {date.today().strftime('%d %b %Y')}", html)
        save_state(state)
        return

    # ── Normal trading cycle ──────────────────────────────────────────────────
    if "start_date" not in state:
        state["start_date"] = date.today().isoformat()

    instruments = CONFIG["instruments"]
    if args.instrument:
        instruments = [(i, r, t) for i, r, t in instruments
                       if i == args.instrument and r == args.resolution]
        if not instruments:
            log.error("Instrument %s %s not found in config.",
                      args.instrument, args.resolution)
            return

    all_predictions = []
    for inst, res, ticker in instruments:
        try:
            pred = run_instrument(inst, res, ticker, state, device, index_dfs)
            if pred:
                all_predictions.append(pred)
        except Exception as e:
            log.error("Error on %s %s: %s", inst, res, traceback.format_exc())

    state["last_run"] = datetime.now().isoformat()

    # ── Send daily summary if past summary_hour and not already sent today ────
    last_summary = state.get("last_summary")
    today_str    = date.today().isoformat()
    now_hour     = datetime.now().hour

    if (now_hour >= CONFIG["summary_hour"] and
            (last_summary is None or not last_summary.startswith(today_str))):
        log.info("Sending daily summary email...")
        start_date = state.get("start_date", today_str)
        days = (date.today() - date.fromisoformat(start_date)).days + 1
        html = daily_summary_html(state, all_predictions, days)
        send_email(
            f"📊 Paper Trading Daily Summary — {date.today().strftime('%d %b %Y')}",
            html
        )
        state["last_summary"] = datetime.now().isoformat()

    save_state(state)

    # ── Print prediction table to terminal ───────────────────────────────────
    if all_predictions:
        print(f"\n{'─'*95}")
        print(f"{'Instrument':<28} {'Res':<4} {'Price':>9}  {'H1':>6} {'H5':>6} {'H20':>6}  "
              f"{'Dir':<5} {'Conf':<10} {'Signal%':>8}  Action")
        print(f"{'─'*95}")
        for p in all_predictions:
            s    = p["signals"]["h1"]
            s5   = p["signals"]["h5"]
            s20  = p["signals"]["h20"]
            arr  = "↑" if s["direction"] == "UP" else "↓"
            act  = p.get("action") or "hold"
            print(f"{p['instrument']:<28} {p['resolution']:<4} "
                  f"{p['current_price']:>9.5g}  "
                  f"{s['prob']:>6.3f} {s5['prob']:>6.3f} {s20['prob']:>6.3f}  "
                  f"{arr} {s['direction']:<4} "
                  f"{s['conf_label']:<10} {s['est_move']:>+6.2f}%  {act}")
        print(f"{'─'*95}\n")

    log.info("Cycle complete. Capital=£%.2f  Open positions=%d",
             state["capital"], len(state.get("positions", {})))


if __name__ == "__main__":
    main()