"""
Paper Trading Engine — Intraday (1m / 5m / 15m)
================================================
Runs intraday paper trading using models trained on short-timeframe data.
Completely separate from the daily paper trading engine:

  State file : logs/paper_trading_intraday/paper_state_intraday.json
  Trade log  : logs/paper_trading_intraday/trade_log_intraday.csv
  Models     : models/intraday/checkpoint_intraday_{resolution}_v1.pt

Designed to run every 30 minutes via GitHub Actions, covering all major
forex sessions. Each run:
  1. Downloads the latest bars for each instrument+resolution
  2. Builds features using the same pipeline as training
  3. Generates H1/H5/H20 predictions with direction confidence
  4. Opens/holds/closes paper positions
  5. Sends email alerts on trade events
  6. Sends a daily summary at the configured hour

Instruments: forex, crypto, major indices only.
Individual equities excluded (gaps, thin intraday liquidity).

Setup:
  Set email credentials as environment variables or fill in CONFIG:
    export PAPER_EMAIL_ADDRESS="your@gmail.com"
    export PAPER_EMAIL_PASSWORD="your-app-password"
    export PAPER_EMAIL_TO="your@gmail.com"

Usage:
  python paper_trading_intraday.py             # run one cycle
  python paper_trading_intraday.py --summary   # force daily summary email
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

CONFIG = {
    # ── Email ─────────────────────────────────────────────────────────────────
    "email_from":     os.getenv("PAPER_EMAIL_ADDRESS",  "ppsupton08@gmail.com"),
    "email_password": os.getenv("PAPER_EMAIL_PASSWORD", "moxo dkhu qphw olow"),
    "email_to":       os.getenv("PAPER_EMAIL_TO",       "ppsupton08@gmail.com"),
    "smtp_host":      "smtp.gmail.com",
    "smtp_port":      587,

    # ── Capital & risk (separate from daily account) ─────────────────────────
    "starting_capital":    5_000.0,    # £5,000 separate paper account for intraday
    "kelly_fraction":      0.5,        # half-Kelly for intraday (higher noise)
    "max_risk_per_trade":  0.02,       # 2% max risk — tighter for intraday
    "min_prob":            0.56,       # slightly higher threshold than daily
    "leverage_min_prob":   0.58,
    "leverage_max_prob":   0.65,

    # ── Model paths ───────────────────────────────────────────────────────────
    "models_dir":   "models/intraday",
    "n_features":   44,
    "lookback":     60,

    # ── Paths (all separate from daily pipeline) ──────────────────────────────
    "state_file":   "logs/paper_trading_intraday/paper_state_intraday.json",
    "trade_log":    "logs/paper_trading_intraday/trade_log_intraday.csv",
    "log_dir":      "logs/paper_trading_intraday",

    # ── Index reference for correlation feature ───────────────────────────────
    "index_ticker": "^GSPC",

    # ── How many bars to download per instrument per run ─────────────────────
    # Must be > LOOKBACK (60) + Z_WIN (30) + buffer = 100 minimum
    # 200 gives a comfortable buffer without being excessive.
    "bars_to_download": 200,

    # ── Daily summary hour (UTC) ──────────────────────────────────────────────
    "summary_hour": 22,    # 10pm UTC — after all major sessions close

    # ── Instruments to trade ─────────────────────────────────────────────────
    # Format: ("INSTRUMENT_PREFIX", "RESOLUTION", "YFINANCE_TICKER")
    # Each resolution uses its own trained model.
    # Remove any resolution you haven't trained a model for yet.
    "instruments": [
        # ── 15m (train monthly) ────────────────────────────────────────────
        ("FOREXCOM_EURUSDX", "15m", "EURUSD=X"),
        ("FOREXCOM_GBPUSDX", "15m", "GBPUSD=X"),
        ("FOREXCOM_USDJPYX", "15m", "USDJPY=X"),
        ("FOREXCOM_USDCADX", "15m", "USDCAD=X"),
        ("FOREXCOM_EURGBPX", "15m", "EURGBP=X"),
        ("FOREXCOM_EURCHFX", "15m", "EURCHF=X"),
        ("CRYPTO_BTC-USD",   "15m", "BTC-USD"),
        ("CRYPTO_ETH-USD",   "15m", "ETH-USD"),
        ("DAX_GDAXI",        "15m", "^GDAXI"),

        # ── 5m (train monthly) ─────────────────────────────────────────────
        ("FOREXCOM_EURUSDX", "5m",  "EURUSD=X"),
        ("FOREXCOM_GBPUSDX", "5m",  "GBPUSD=X"),
        ("FOREXCOM_USDJPYX", "5m",  "USDJPY=X"),
        ("CRYPTO_BTC-USD",   "5m",  "BTC-USD"),

        # ── 1m (train weekly) ──────────────────────────────────────────────
        ("FOREXCOM_EURUSDX", "1m",  "EURUSD=X"),
        ("FOREXCOM_GBPUSDX", "1m",  "GBPUSD=X"),
        ("CRYPTO_BTC-USD",   "1m",  "BTC-USD"),
    ],
}

# ─── Logging ──────────────────────────────────────────────────────────────────
os.makedirs(CONFIG["log_dir"], exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(CONFIG["log_dir"], "paper_trading_intraday.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

HORIZONS = [1, 5, 20]
Z_WIN    = 30     # shorter z-score window for intraday (less data available)


# ─── Confidence helpers ────────────────────────────────────────────────────────

def classify_confidence(prob: float) -> str:
    edge = abs(prob - 0.5)
    if edge >= 0.15: return "Very High"
    if edge >= 0.10: return "High"
    if edge >= 0.06: return "Moderate"
    if edge >= 0.03: return "Low"
    return "Very Low"

def confidence_pct(prob: float) -> float:
    return round(abs(prob - 0.5) * 200, 1)


# ─── State management ─────────────────────────────────────────────────────────

def load_state() -> dict:
    path = CONFIG["state_file"]
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "positions":    {},
        "capital":      CONFIG["starting_capital"],
        "realised_pnl": 0.0,
        "trade_count":  0,
        "last_run":     None,
        "last_summary": None,
    }

def save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(CONFIG["state_file"]), exist_ok=True)
    with open(CONFIG["state_file"], "w") as f:
        json.dump(state, f, indent=2, default=str)

def log_trade(row: dict) -> None:
    path   = CONFIG["trade_log"]
    df     = pd.DataFrame([row])
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


# ─── Data fetching ─────────────────────────────────────────────────────────────

def fetch_bars(ticker: str, resolution: str, n_bars: int) -> pd.DataFrame | None:
    """Download recent OHLCV bars from Yahoo Finance."""
    period_map   = {"1m": "7d", "5m": "60d", "15m": "60d"}
    period       = period_map.get(resolution, "60d")

    try:
        df = yf.download(ticker, period=period, interval=resolution,
                         auto_adjust=False, progress=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index.name = "Datetime"
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df.tail(n_bars)
    except Exception as e:
        log.warning("Download failed %s %s: %s", ticker, resolution, e)
        return None


# ─── Feature engineering ──────────────────────────────────────────────────────

NON_ZSCORE_COLS = {
    "bb_position", "rsi14", "rsi28", "range_percentile",
    "donchian_position", "stoch_k_14", "williams_r_14", "consec_bars",
    "dow_sin", "dow_cos", "month_sin", "month_cos", "quarter_end_flag",
    "hour_sin", "hour_cos", "session_asian", "session_london",
    "session_newyork", "session_overlap",
}


def build_feature_window(df: pd.DataFrame, resolution: str,
                          index_df: pd.DataFrame | None) -> np.ndarray | None:
    """Build one feature window [1, LOOKBACK, N_FEATURES] from OHLCV data."""
    LOOKBACK   = CONFIG["lookback"]
    N_FEATURES = CONFIG["n_features"]
    MIN_ROWS   = LOOKBACK + Z_WIN + 10

    if len(df) < MIN_ROWS:
        return None

    df = df[df["Close"] > 0].dropna(subset=["Open", "High", "Low", "Close"])

    try:
        ohlc  = compute_ohlc_features(df)
        tech  = compute_technical_features(df)
        cal   = compute_calendar_features(df, resolution)
        corr  = add_index_correlation(df, index_df, resolution).to_frame()

        all_feats = pd.concat([ohlc, tech, corr, cal], axis=1).loc[df.index]
        cal_cols  = list(cal.columns)
        normalised = apply_rolling_zscore_pipeline(
            all_feats, non_zscore_cols=set(cal_cols)
        ).dropna()

        if len(normalised) < LOOKBACK + 5:
            return None

        window = normalised.iloc[-LOOKBACK:].values.astype(np.float32)
        n = window.shape[1]
        if n < N_FEATURES:
            window = np.pad(window, ((0, 0), (0, N_FEATURES - n)))
        elif n > N_FEATURES:
            window = window[:, :N_FEATURES]

        return window[np.newaxis, :, :]
    except Exception as e:
        log.debug("Feature build failed: %s", e)
        return None


# ─── Model loading ─────────────────────────────────────────────────────────────

_model_cache: dict = {}

def load_model(resolution: str, device: str):
    """Load the intraday model for this resolution. Returns (model, source_label)."""
    if resolution in _model_cache:
        return _model_cache[resolution]

    ckpt_path = os.path.join(
        CONFIG["models_dir"],
        f"checkpoint_intraday_{resolution}_v1.pt"
    )
    if not os.path.exists(ckpt_path):
        log.warning("No intraday model found for %s at %s", resolution, ckpt_path)
        log.warning("Run: python train_intraday.py --resolution %s", resolution)
        return None, None

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = build_model(n_features=CONFIG["n_features"], device=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    source = f"intraday_{resolution}"
    _model_cache[resolution] = (model, source)
    log.debug("Loaded model: %s  val_loss=%.4f", ckpt_path, ckpt.get("val_loss", 0))
    return model, source


# ─── Inference & signal interpretation ────────────────────────────────────────

def predict(model, window: np.ndarray, device: str) -> dict:
    x = torch.from_numpy(window).to(device)
    with torch.no_grad():
        preds = model(x)
    return {
        f"h{h}_prob": float(preds[h][0][0].cpu())
        for h in HORIZONS
    }

def _asset_sigma(instrument: str) -> float:
    """Approximate 1-sigma % move at intraday scale (smaller than daily)."""
    if "FOREXCOM" in instrument: return 0.08   # ~0.08% per 15m bar
    if "CRYPTO"   in instrument: return 0.50   # ~0.5% per 15m bar
    if any(x in instrument for x in ["DAX", "FTSE", "SP500"]): return 0.15
    return 0.12

def interpret(pred: dict, current_price: float, instrument: str) -> dict:
    sigma = _asset_sigma(instrument)
    signals = {}
    for h in HORIZONS:
        prob      = pred[f"h{h}_prob"]
        direction = "UP" if prob >= 0.5 else "DOWN"
        sign      = 1 if prob >= 0.5 else -1
        signals[f"h{h}"] = {
            "prob":       round(prob, 4),
            "direction":  direction,
            "conf_label": classify_confidence(prob),
            "conf_pct":   confidence_pct(prob),
            "est_move":   round(sign * abs(prob - 0.5) * 2 * sigma * 100, 3),
        }
    return signals


# ─── Position sizing ──────────────────────────────────────────────────────────

def compute_position_size(prob: float, capital: float,
                           instrument: str, resolution: str):
    max_lev  = get_leverage_for_source(f"{instrument}_{resolution}")
    lev_min  = CONFIG["leverage_min_prob"]
    lev_max  = CONFIG["leverage_max_prob"]

    if prob < lev_min:
        leverage = 1.0
    elif prob > lev_max:
        leverage = max_lev
    else:
        t        = (prob - lev_min) / (lev_max - lev_min)
        leverage = 1.0 + t * (max_lev - 1.0)
    leverage = min(leverage, max_lev)

    edge     = abs(prob - 0.5)
    fraction = CONFIG["kelly_fraction"] * (2 * edge)
    size_gbp = min(
        capital * fraction * leverage,
        capital * CONFIG["max_risk_per_trade"] * leverage
    )
    return round(max(size_gbp, 0.0), 2), round(leverage, 2)


# ─── Email ────────────────────────────────────────────────────────────────────

_CSS = """
  body { font-family: Arial, sans-serif; font-size: 13px; color: #222; }
  h2   { color: #1a3a5c; border-bottom: 2px solid #1a3a5c; padding-bottom:5px; }
  h3   { color: #2c5f8a; margin-top:18px; }
  table { border-collapse: collapse; width: 100%; margin: 8px 0; }
  th { background: #1a3a5c; color: #fff; padding: 7px 9px; text-align: left; }
  td { padding: 6px 9px; border-bottom: 1px solid #ddd; }
  tr:nth-child(even) { background: #f5f8fc; }
  .up   { color: #1a7a1a; font-weight: bold; }
  .down { color: #c0392b; font-weight: bold; }
  .pnl-pos { color: #1a7a1a; }
  .pnl-neg { color: #c0392b; }
  .tag-vh { background:#1a7a1a; color:#fff; border-radius:3px; padding:1px 5px; font-size:11px; }
  .tag-h  { background:#2ecc71; color:#fff; border-radius:3px; padding:1px 5px; font-size:11px; }
  .tag-m  { background:#f39c12; color:#fff; border-radius:3px; padding:1px 5px; font-size:11px; }
  .tag-l  { background:#e67e22; color:#fff; border-radius:3px; padding:1px 5px; font-size:11px; }
  .tag-vl { background:#95a5a6; color:#fff; border-radius:3px; padding:1px 5px; font-size:11px; }
  .footer { color:#888; font-size:11px; margin-top:16px; }
"""

def _conf_tag(label):
    cls = {"Very High":"vh","High":"h","Moderate":"m","Low":"l","Very Low":"vl"}.get(label,"vl")
    return f'<span class="tag-{cls}">{label}</span>'

def _dir_html(d):
    return f'<span class="{"up" if d=="UP" else "down"}">{"↑ UP" if d=="UP" else "↓ DOWN"}</span>'

def _pnl_html(v):
    return f'<span class="{"pnl-pos" if v>=0 else "pnl-neg"}">{v:+.2f}</span>'

def _email_ok():
    return bool(CONFIG["email_from"] and CONFIG["email_password"] and CONFIG["email_to"])

def send_email(subject: str, html: str) -> None:
    if not _email_ok():
        log.warning("Email not configured — skipping. Fill in CONFIG email fields.")
        return
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[Intraday] {subject}"
    msg["From"]    = CONFIG["email_from"]
    msg["To"]      = CONFIG["email_to"]
    msg.attach(MIMEText(html, "html"))
    try:
        with smtplib.SMTP(CONFIG["smtp_host"], CONFIG["smtp_port"]) as s:
            s.ehlo(); s.starttls()
            s.login(CONFIG["email_from"], CONFIG["email_password"])
            s.sendmail(CONFIG["email_from"], CONFIG["email_to"], msg.as_string())
        log.info("Email sent: %s", subject)
    except Exception as e:
        log.error("Email failed: %s", e)

def trade_alert_html(action, instrument, resolution, direction, price,
                      size_gbp, leverage, prob, signals,
                      pnl_gbp=None, entry_price=None):
    colours = {"OPEN":"#1a7a1a","CLOSE":"#c0392b"}
    labels  = {"OPEN":"🟢 TRADE OPENED","CLOSE":"🔴 TRADE CLOSED"}
    sig     = signals.get("h1", {})

    pnl_row = ""
    if pnl_gbp is not None and entry_price is not None:
        pnl_pct = (price/entry_price - 1) * (1 if direction=="UP" else -1) * 100
        pnl_row = (f"<tr><td><b>Realised P&L</b></td>"
                   f"<td>{_pnl_html(pnl_gbp)} ({pnl_pct:+.2f}%)</td></tr>"
                   f"<tr><td><b>Entry Price</b></td><td>{entry_price:.5g}</td></tr>")

    dir_prob = (1-prob) if direction=="DOWN" else prob

    return f"""
    <style>{_CSS}</style>
    <h2 style="color:{colours.get(action,'#222')}">{labels.get(action,action)}</h2>
    <p style="color:#888;font-size:12px;">Intraday model — {resolution} resolution</p>
    <table>
      <tr><td><b>Instrument</b></td><td>{instrument} {resolution}</td></tr>
      <tr><td><b>Direction</b></td><td>{_dir_html(direction)}</td></tr>
      <tr><td><b>Price</b></td><td>{price:.5g}</td></tr>
      <tr><td><b>Size (paper)</b></td><td>£{size_gbp:.2f}</td></tr>
      <tr><td><b>Leverage</b></td><td>{leverage:.1f}×</td></tr>
      <tr><td><b>H1 Confidence</b></td>
          <td>{_conf_tag(sig.get('conf_label','?'))} {sig.get('conf_pct',0):.0f}% edge<br>
          <small style="color:#555">Raw prob={prob:.3f} → {direction} prob={dir_prob:.1%}</small>
          </td></tr>
      {pnl_row}
    </table>
    <h3>Predictions</h3>
    <table>
      <tr><th>Horizon</th><th>Direction</th><th>Raw Prob</th>
          <th>Dir. Prob</th><th>Confidence</th><th>Signal%</th></tr>
      {"".join(
          f"<tr><td>{lbl}</td>"
          f"<td>{_dir_html(signals.get(k,{}).get('direction','?'))}</td>"
          f"<td>{signals.get(k,{}).get('prob',0):.3f}</td>"
          f"<td>{(1-signals.get(k,{}).get('prob',0.5)) if signals.get(k,{}).get('direction')=='DOWN' else signals.get(k,{}).get('prob',0.5):.1%}</td>"
          f"<td>{_conf_tag(signals.get(k,{}).get('conf_label','?'))} {signals.get(k,{}).get('conf_pct',0):.0f}% edge</td>"
          f"<td>{signals.get(k,{}).get('est_move',0):+.2f}%</td></tr>"
          for k, lbl in [("h1","H1 (next bar)"),("h5","H5 (5 bars)"),("h20","H20 (20 bars)")]
      )}
    </table>
    <p class="footer">Intraday paper trading — simulated positions only. No real money involved.</p>"""

def daily_summary_html(state, all_predictions, days_running):
    positions  = state.get("positions", {})
    capital    = state["capital"]
    realised   = state.get("realised_pnl", 0.0)
    today      = date.today().strftime("%A %d %B %Y")
    unrealised = 0.0

    pos_rows = ""
    for key, pos in positions.items():
        ep   = pos["entry_price"]
        cp   = pos.get("current_price", ep)
        dirn = pos["direction"]
        size = pos["size_gbp"]
        lev  = pos.get("leverage", 1.0)
        pnl  = (cp / ep - 1) * dirn * size * lev
        unrealised += pnl
        inst, res = key.rsplit("_", 1)
        mins_held = pos.get("bars_held", 0)
        pos_rows += (
            f"<tr><td>{inst}</td><td>{res}</td>"
            f"<td>{_dir_html('UP' if dirn==1 else 'DOWN')}</td>"
            f"<td>{ep:.5g}</td><td>{cp:.5g}</td>"
            f"<td>{lev:.1f}×</td><td>£{size:.2f}</td>"
            f"<td>{_pnl_html(pnl)}</td>"
            f"<td>{mins_held} bars</td></tr>"
        )

    if not pos_rows:
        pos_rows = '<tr><td colspan="9" style="text-align:center;color:#888">No open positions</td></tr>'

    total_pnl = realised + unrealised
    total_pct = total_pnl / CONFIG["starting_capital"] * 100

    # Build prediction table
    pred_rows = ""
    for p in all_predictions:
        s   = p["signals"]["h1"]
        s5  = p["signals"]["h5"]
        s20 = p["signals"]["h20"]
        pred_rows += (
            f"<tr><td><b>{p['instrument']}</b></td><td>{p['resolution']}</td>"
            f"<td>{p['current_price']:.5g}</td>"
            f"<td>{_dir_html(s['direction'])}</td>"
            f"<td>{s['prob']:.3f}</td><td>{s5['prob']:.3f}</td><td>{s20['prob']:.3f}</td>"
            f"<td>{_conf_tag(s['conf_label'])} {s['conf_pct']:.0f}% edge</td>"
            f"<td>{s['est_move']:+.2f}%</td></tr>"
        )

    return f"""
    <style>{_CSS}</style>
    <h2>📈 Intraday Paper Trading Summary — {today}</h2>
    <h3>Portfolio</h3>
    <table>
      <tr><td><b>Starting Capital</b></td><td>£{CONFIG['starting_capital']:,.2f}</td></tr>
      <tr><td><b>Current Capital</b></td><td>£{capital:,.2f}</td></tr>
      <tr><td><b>Realised P&L</b></td><td>{_pnl_html(realised)}</td></tr>
      <tr><td><b>Unrealised P&L</b></td><td>{_pnl_html(unrealised)}</td></tr>
      <tr><td><b>Total P&L</b></td><td>{_pnl_html(total_pnl)} ({total_pct:+.1f}%)</td></tr>
      <tr><td><b>Open Positions</b></td><td>{len(positions)}</td></tr>
      <tr><td><b>Total Trades</b></td><td>{state.get('trade_count',0)}</td></tr>
      <tr><td><b>Days Running</b></td><td>{days_running}</td></tr>
    </table>
    <h3>Open Positions</h3>
    <table>
      <tr><th>Instrument</th><th>Res</th><th>Dir</th><th>Entry</th><th>Current</th>
          <th>Lev</th><th>Size</th><th>P&L</th><th>Held</th></tr>
      {pos_rows}
    </table>
    <h3>Latest Predictions</h3>
    <table>
      <tr><th>Instrument</th><th>Res</th><th>Price</th><th>Dir</th>
          <th>H1</th><th>H5</th><th>H20</th><th>Confidence</th><th>Signal%</th></tr>
      {pred_rows}
    </table>
    <p class="footer">
      Intraday paper trading — simulated positions only.<br>
      Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
    </p>"""


# ─── Core trading cycle ───────────────────────────────────────────────────────

def run_instrument(instrument, resolution, yf_ticker, state, device, index_dfs):
    key = f"{instrument}_{resolution}"
    log.info("  %s %s", instrument, resolution)

    df = fetch_bars(yf_ticker, resolution, CONFIG["bars_to_download"])
    if df is None or len(df) < CONFIG["lookback"] + Z_WIN + 10:
        log.debug("  Insufficient data for %s", key)
        return None

    current_price = float(df["Close"].iloc[-1])
    idx_df  = index_dfs.get(resolution)
    window  = build_feature_window(df, resolution, idx_df)
    if window is None:
        return None

    model, source = load_model(resolution, device)
    if model is None:
        return None

    pred    = predict(model, window, device)
    signals = interpret(pred, current_price, instrument)

    h1_prob = pred["h1_prob"]
    h1_dir  = 1 if h1_prob >= 0.5 else -1
    h1_sig  = signals["h1"]

    existing = state["positions"].get(key)
    action   = None
    pnl_gbp  = None
    ep_alert = None

    if existing:
        existing["current_price"] = current_price
        existing["bars_held"]     = existing.get("bars_held", 0) + 1

        if h1_prob < CONFIG["min_prob"] and (1 - h1_prob) < CONFIG["min_prob"]:
            action = "CLOSE"
            reason = f"prob={h1_prob:.3f} — no directional conviction"
        elif existing["direction"] != h1_dir:
            action = "CLOSE"
            reason = "direction flip"
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
            ep_alert               = ep

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
                "bars_held":   existing.get("bars_held", 0),
                "pnl_gbp":     round(pnl_gbp, 4),
                "h1_prob":     h1_prob,
                "confidence":  h1_sig["conf_label"],
                "reason":      reason,
            })
            del state["positions"][key]

    if existing is None or action == "OPEN":
        action_prob = h1_prob if h1_dir == 1 else (1 - h1_prob)
        if action_prob >= CONFIG["min_prob"]:
            size_gbp, leverage = compute_position_size(
                action_prob, state["capital"], instrument, resolution
            )
            if size_gbp > 0:
                action = "OPEN"
                state["positions"][key] = {
                    "direction":     h1_dir,
                    "entry_price":   current_price,
                    "current_price": current_price,
                    "entry_date":    datetime.now().isoformat(),
                    "size_gbp":      size_gbp,
                    "leverage":      leverage,
                    "bars_held":     0,
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
                    "bars_held":   0,
                    "pnl_gbp":     "",
                    "h1_prob":     h1_prob,
                    "confidence":  h1_sig["conf_label"],
                    "reason":      f"signal: {'long' if h1_dir==1 else 'short'} prob={action_prob:.3f}",
                })

    if action in ("OPEN", "CLOSE"):
        pos      = state["positions"].get(key, {})
        size_gbp = pos.get("size_gbp", existing.get("size_gbp", 0) if existing else 0)
        leverage = pos.get("leverage", existing.get("leverage", 1.0) if existing else 1.0)
        html = trade_alert_html(
            action, instrument, resolution,
            "UP" if h1_dir == 1 else "DOWN",
            current_price, size_gbp, leverage, h1_prob, signals,
            pnl_gbp, ep_alert
        )
        emoji = "🟢" if action == "OPEN" else "🔴"
        dirn  = "↑ Long" if h1_dir == 1 else "↓ Short"
        send_email(f"{emoji} Intraday {action}: {instrument} {resolution} {dirn}", html)

    log.info("    price=%.5g  h1=%.3f %s  %s  action=%s",
             current_price, h1_prob,
             "↑" if h1_dir == 1 else "↓",
             h1_sig["conf_label"], action or "hold")

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

def main():
    parser = argparse.ArgumentParser(description="Intraday Paper Trading Engine")
    parser.add_argument("--summary",    action="store_true",
                        help="Force send daily summary email now.")
    parser.add_argument("--instrument", default=None)
    parser.add_argument("--resolution", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Intraday Paper Trading — device=%s  %s UTC",
             device, datetime.utcnow().strftime("%Y-%m-%d %H:%M"))

    state = load_state()

    # Load S&P 500 index reference for correlation feature
    log.info("Loading index reference...")
    index_dfs = {}
    for res, period in [("1m", "7d"), ("5m", "60d"), ("15m", "60d")]:
        idx = yf.download(CONFIG["index_ticker"], period=period,
                          interval=res, auto_adjust=False, progress=False)
        if idx is not None and not idx.empty:
            if isinstance(idx.columns, pd.MultiIndex):
                idx.columns = idx.columns.get_level_values(0)
            idx.index.name = "Datetime"
            if idx.index.tz is not None:
                idx.index = idx.index.tz_localize(None)
            index_dfs[res] = idx

    instruments = CONFIG["instruments"]
    if args.instrument and args.resolution:
        instruments = [(i, r, t) for i, r, t in instruments
                       if i == args.instrument and r == args.resolution]

    if "start_date" not in state:
        state["start_date"] = date.today().isoformat()

    all_predictions = []
    for inst, res, ticker in instruments:
        try:
            pred = run_instrument(inst, res, ticker, state, device, index_dfs)
            if pred:
                all_predictions.append(pred)
        except Exception:
            log.error("Error on %s %s:\n%s", inst, res, traceback.format_exc())

    state["last_run"] = datetime.utcnow().isoformat()

    # Daily summary
    today_str   = date.today().isoformat()
    last_summary = state.get("last_summary", "")
    now_hour    = datetime.utcnow().hour

    if args.summary or (now_hour >= CONFIG["summary_hour"] and
                        not last_summary.startswith(today_str)):
        log.info("Sending daily summary...")
        start = state.get("start_date", today_str)
        days  = (date.today() - date.fromisoformat(start)).days + 1
        html  = daily_summary_html(state, all_predictions, days)
        send_email(
            f"📊 Intraday Summary — {date.today().strftime('%d %b %Y')}", html
        )
        state["last_summary"] = datetime.utcnow().isoformat()

    save_state(state)

    # Terminal output
    if all_predictions:
        print(f"\n{'─'*100}")
        print(f"{'Instrument':<28} {'Res':<4} {'Price':>9}  "
              f"{'H1':>6} {'H5':>6} {'H20':>6}  "
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


if __name__ == "__main__":
    main()