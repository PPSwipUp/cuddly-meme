"""
Backfill Simulation
===================
Replays the paper trading logic on historical closing prices to reconstruct
what the model would have done over the past N trading days.

Produces:
  logs/paper_trading/paper_state.json   — corrected open positions and capital
  logs/paper_trading/trade_log.csv      — full trade history with dates and P&L
  logs/paper_trading/backfill_report.txt — human-readable day-by-day summary

Usage:
  python backfill_simulation.py              # replay last 3 trading days
  python backfill_simulation.py --days 5     # replay last 5 trading days
  python backfill_simulation.py --dry-run    # print what would happen, don't save
"""

import os
import sys
import glob
import json
import logging
import argparse
import warnings
import math
from datetime import datetime, date, timedelta
from io import StringIO

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

# ─── Import CONFIG and helpers from paper_trading.py ─────────────────────────
# We import the key functions to ensure 100% consistency with live trading.
from paper_trading import (
    CONFIG,
    fetch_bars,
    build_feature_window,
    load_model,
    predict,
    interpret_prediction,
    classify_confidence,
    _asset_sigma,
)

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR = CONFIG["log_dir"]
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "backfill.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

HORIZONS  = [1, 5, 20]
MIN_PROB  = CONFIG["min_prob"]
KELLY     = CONFIG["kelly_fraction"]
MAX_RISK  = CONFIG["max_risk_per_trade"]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def trading_days_back(n: int) -> list[date]:
    """Return the last n weekdays (Mon-Fri) ending yesterday."""
    days = []
    d    = date.today() - timedelta(days=1)
    while len(days) < n:
        if d.weekday() < 5:   # Mon=0 … Fri=4
            days.append(d)
        d -= timedelta(days=1)
    return list(reversed(days))


def get_close_on_date(df: pd.DataFrame, target_date: date) -> float | None:
    """
    Return the closing price for target_date from a DataFrame with a DatetimeIndex.
    Falls back to the nearest prior date if exact date is missing (e.g. half-days).
    """
    target_str = target_date.isoformat()
    # Ensure index is datetime
    idx = pd.to_datetime(df.index)
    mask = idx.date <= target_date
    if not mask.any():
        return None
    price = float(df["Close"][mask].iloc[-1])
    return price if not math.isnan(price) and price > 0 else None


def compute_size(prob: float, capital: float,
                 instrument: str, resolution: str) -> tuple[float, float]:
    """Kelly position size and leverage."""
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
    fraction = KELLY * (2 * edge)
    size_gbp = min(capital * fraction * leverage,
                   capital * MAX_RISK * leverage)
    return round(max(size_gbp, 0.0), 2), round(leverage, 2)


# ─── Data download ─────────────────────────────────────────────────────────────

def download_all(instruments: list, index_ticker: str,
                 bars: int = 300) -> tuple[dict, dict]:
    """
    Download full OHLCV history for all instruments and the index reference.
    Returns (instrument_dfs, index_dfs) where keys are instrument names and
    "1D" for index_dfs.
    """
    log.info("Downloading historical data for %d instruments...", len(instruments))

    index_dfs = {}
    idx = yf.download(index_ticker, period="2y", interval="1d",
                      auto_adjust=False, progress=False)
    if idx is not None and not idx.empty:
        if isinstance(idx.columns, pd.MultiIndex):
            idx.columns = idx.columns.get_level_values(0)
        idx.index.name = "Datetime"
        if idx.index.tz is not None:
            idx.index = idx.index.tz_localize(None)
        index_dfs["1D"] = idx

    inst_dfs = {}
    for inst, res, ticker in instruments:
        key = f"{inst}_{res}"
        df  = fetch_bars(ticker, res, bars)
        if df is not None and len(df) > 100:
            inst_dfs[key] = (inst, res, df)
        else:
            log.warning("  No data for %s (%s)", inst, ticker)

    log.info("Downloaded %d / %d instruments", len(inst_dfs), len(instruments))
    return inst_dfs, index_dfs


# ─── Single-day simulation ─────────────────────────────────────────────────────

def simulate_day(sim_date: date, inst_dfs: dict, index_dfs: dict,
                 state: dict, device: str,
                 report_lines: list) -> list[dict]:
    """
    Run one day of paper trading simulation.
    Uses all OHLCV data UP TO AND INCLUDING sim_date to build the feature
    window, then makes open/close/hold decisions.
    Returns list of trade dicts for the trade log.
    """
    log.info("─── Simulating %s ───", sim_date.isoformat())
    report_lines.append(f"\n{'='*60}")
    report_lines.append(f"DATE: {sim_date.strftime('%A %d %B %Y')}")
    report_lines.append(f"{'='*60}")

    trades     = []
    positions  = state["positions"]
    capital    = state["capital"]

    for key, (instrument, resolution, df_full) in inst_dfs.items():
        # Slice data to only include rows up to sim_date
        if hasattr(df_full.index, 'date'):
            df = df_full[df_full.index.date <= sim_date].copy()
        else:
            df = df_full[pd.to_datetime(df_full.index).date <= sim_date].copy()

        if len(df) < 130:
            continue

        # Current price = close on sim_date (or most recent prior)
        current_price = get_close_on_date(df, sim_date)
        if current_price is None:
            continue

        # Build feature window on data up to sim_date
        idx_df = index_dfs.get(resolution)
        # Slice index too
        if idx_df is not None:
            if hasattr(idx_df.index, 'date'):
                idx_slice = idx_df[idx_df.index.date <= sim_date]
            else:
                idx_slice = idx_df[pd.to_datetime(idx_df.index).date <= sim_date]
        else:
            idx_slice = None

        window = build_feature_window(df, resolution, idx_slice)
        if window is None:
            continue

        model, source = load_model(instrument, resolution, device)
        if model is None:
            continue

        pred    = predict(model, window, device)
        signals = interpret_prediction(pred, current_price, instrument)

        h1_prob = pred["h1_prob"]
        h1_dir  = 1 if h1_prob >= 0.5 else -1
        action_prob = h1_prob if h1_dir == 1 else (1 - h1_prob)

        existing = positions.get(key)
        action   = None
        pnl_gbp  = None

        if existing:
            existing["current_price"] = current_price
            existing["bars_held"]     = existing.get("bars_held", 0) + 1

            # Close conditions
            if action_prob < MIN_PROB:
                action = "CLOSE"
                reason = f"prob={h1_prob:.3f} below threshold"
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
                capital                = state["capital"]

                t = {
                    "date":        sim_date.isoformat(),
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
                    "pnl_pct":     round(pnl_gbp / (size + 1e-9) * 100, 3),
                    "h1_prob":     h1_prob,
                    "confidence":  signals["h1"]["conf_label"],
                    "reason":      reason,
                }
                trades.append(t)
                del positions[key]

                report_lines.append(
                    f"  CLOSE  {instrument:<28} {resolution}  "
                    f"{'LONG' if existing['direction']==1 else 'SHORT':<5}  "
                    f"entry={ep:.4g}  exit={current_price:.4g}  "
                    f"P&L={pnl_gbp:+.2f}"
                )

                # Immediately consider opening in new direction
                if action_prob < MIN_PROB:
                    action = None  # don't reopen if no conviction

        # Open new position
        if existing is None or action == "OPEN":
            if action_prob >= MIN_PROB:
                size_gbp, leverage = compute_size(
                    action_prob, capital, instrument, resolution
                )
                if size_gbp > 0:
                    action = "OPEN"
                    positions[key] = {
                        "direction":     h1_dir,
                        "entry_price":   current_price,
                        "current_price": current_price,
                        "entry_date":    sim_date.isoformat(),
                        "size_gbp":      size_gbp,
                        "leverage":      leverage,
                        "bars_held":     0,
                    }
                    state["trade_count"] = state.get("trade_count", 0) + 1
                    capital = state["capital"]

                    t = {
                        "date":        sim_date.isoformat(),
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
                        "pnl_pct":     "",
                        "h1_prob":     h1_prob,
                        "confidence":  signals["h1"]["conf_label"],
                        "reason":      f"signal {'long' if h1_dir==1 else 'short'} "
                                       f"prob={action_prob:.3f}",
                    }
                    trades.append(t)

                    report_lines.append(
                        f"  OPEN   {instrument:<28} {resolution}  "
                        f"{'LONG' if h1_dir==1 else 'SHORT':<5}  "
                        f"price={current_price:.4g}  "
                        f"size=£{size_gbp:.2f}  lev={leverage:.1f}x  "
                        f"prob={action_prob:.3f} ({signals['h1']['conf_label']})"
                    )

        if action == "HOLD":
            report_lines.append(
                f"  HOLD   {instrument:<28} {resolution}  "
                f"{'LONG' if existing['direction']==1 else 'SHORT':<5}  "
                f"current={current_price:.4g}"
            )

    # End of day summary
    unrealised = sum(
        (pos["current_price"] / pos["entry_price"] - 1) * pos["direction"]
        * pos["size_gbp"] * pos.get("leverage", 1.0)
        for pos in positions.values()
        if pos.get("current_price") and not math.isnan(pos.get("current_price", float("nan")))
    )

    report_lines.append(f"\n  End of day:")
    report_lines.append(f"    Capital (realised):  £{state['capital']:,.2f}")
    report_lines.append(f"    Unrealised P&L:      £{unrealised:+.2f}")
    report_lines.append(f"    Open positions:      {len(positions)}")
    report_lines.append(f"    Trades today:        {len(trades)}")

    log.info("  End of %s: capital=£%.2f  open=%d  trades_today=%d",
             sim_date, state["capital"], len(positions), len(trades))

    return trades


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Backfill paper trading simulation"
    )
    parser.add_argument(
        "--days", type=int, default=3,
        help="Number of trading days to replay (default: 3)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print simulation without saving state or trade log"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Backfill Simulation — device=%s  days=%d", device, args.days)

    # ── Get trading days to simulate ─────────────────────────────────────────
    sim_days = trading_days_back(args.days)
    log.info("Simulating: %s", [d.isoformat() for d in sim_days])

    # ── Fresh starting state ──────────────────────────────────────────────────
    state = {
        "positions":    {},
        "capital":      CONFIG["starting_capital"],
        "realised_pnl": 0.0,
        "trade_count":  0,
        "start_date":   sim_days[0].isoformat(),
        "last_run":     datetime.now().isoformat(),
        "last_summary": None,
    }

    # ── Download all data once ─────────────────────────────────────────────────
    instruments  = CONFIG["instruments"]
    inst_dfs, index_dfs = download_all(
        instruments, CONFIG["index_ticker"], bars=300
    )

    # ── Run simulation day by day ──────────────────────────────────────────────
    all_trades   = []
    report_lines = [
        "PAPER TRADING BACKFILL SIMULATION",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Starting capital: £{CONFIG['starting_capital']:,.2f}",
        f"Days simulated: {args.days} ({sim_days[0]} → {sim_days[-1]})",
        f"Instruments: {len(instruments)}",
    ]

    for sim_date in sim_days:
        day_trades = simulate_day(
            sim_date, inst_dfs, index_dfs, state, device, report_lines
        )
        all_trades.extend(day_trades)

    # ── Final state update ─────────────────────────────────────────────────────
    # Update current prices to today for all open positions
    log.info("Updating current prices for open positions...")
    for key, pos in state["positions"].items():
        inst, res = key.rsplit("_", 1)
        if key in inst_dfs:
            _, _, df = inst_dfs[key]
            cp = get_close_on_date(df, date.today())
            if cp and not math.isnan(cp):
                pos["current_price"] = cp

    state["last_run"] = datetime.now().isoformat()

    # ── Compute final P&L ──────────────────────────────────────────────────────
    unrealised = sum(
        (pos.get("current_price", pos["entry_price"]) / pos["entry_price"] - 1)
        * pos["direction"] * pos["size_gbp"] * pos.get("leverage", 1.0)
        for pos in state["positions"].values()
    )
    total_pnl = state["realised_pnl"] + unrealised

    # ── Report ─────────────────────────────────────────────────────────────────
    report_lines.append(f"\n{'='*60}")
    report_lines.append("FINAL SUMMARY")
    report_lines.append(f"{'='*60}")
    report_lines.append(f"Starting capital:  £{CONFIG['starting_capital']:,.2f}")
    report_lines.append(f"Current capital:   £{state['capital']:,.2f}")
    report_lines.append(f"Realised P&L:      £{state['realised_pnl']:+,.2f}")
    report_lines.append(f"Unrealised P&L:    £{unrealised:+,.2f}")
    report_lines.append(f"Total P&L:         £{total_pnl:+,.2f} "
                        f"({total_pnl/CONFIG['starting_capital']*100:+.2f}%)")
    report_lines.append(f"Total trades:      {state['trade_count']}")
    report_lines.append(f"Open positions:    {len(state['positions'])}")

    report_lines.append(f"\nOpen positions:")
    for key, pos in state["positions"].items():
        ep  = pos["entry_price"]
        cp  = pos.get("current_price", ep)
        pnl = (cp / ep - 1) * pos["direction"] * pos["size_gbp"] * pos.get("leverage", 1.0)
        report_lines.append(
            f"  {key:<35}  {'LONG' if pos['direction']==1 else 'SHORT':<5}  "
            f"entry={ep:.4g}  current={cp:.4g}  "
            f"P&L=£{pnl:+.2f}  held={pos.get('bars_held',0)} bars"
        )

    closed = [t for t in all_trades if t["action"] == "CLOSE"]
    if closed:
        wins     = sum(1 for t in closed if isinstance(t["pnl_gbp"], float) and t["pnl_gbp"] > 0)
        win_rate = wins / len(closed) * 100
        report_lines.append(f"\nClosed trades:     {len(closed)}")
        report_lines.append(f"Win rate:          {win_rate:.1f}%  "
                            f"({wins}W / {len(closed)-wins}L)")

    report = "\n".join(report_lines)
    print("\n" + report)

    # ── Save ──────────────────────────────────────────────────────────────────
    if not args.dry_run:
        os.makedirs(CONFIG["log_dir"], exist_ok=True)

        # Save state
        with open(CONFIG["state_file"], "w") as f:
            json.dump(state, f, indent=2, default=str)
        log.info("Saved state: %s", CONFIG["state_file"])

        # Save trade log
        if all_trades:
            df_trades = pd.DataFrame(all_trades)
            header    = not os.path.exists(CONFIG["trade_log"])
            df_trades.to_csv(CONFIG["trade_log"], mode="w", header=True, index=False)
            log.info("Saved trade log: %s  (%d trades)", CONFIG["trade_log"], len(all_trades))
        else:
            log.info("No trades to log")

        # Save report
        report_path = os.path.join(CONFIG["log_dir"], "backfill_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        log.info("Saved report: %s", report_path)

        log.info("✓ Backfill complete. Run `python paper_trading.py --summary` to send email.")
    else:
        log.info("DRY RUN — nothing saved. Remove --dry-run to save state and trade log.")


if __name__ == "__main__":
    main()