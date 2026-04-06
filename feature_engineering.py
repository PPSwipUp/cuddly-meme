"""
Stage 2 — Feature Engineering & Normalisation Pipeline
Trading Algorithm Blueprint

Reads raw CSVs from data/raw/, computes all features specified in the blueprint,
and saves normalised feature arrays to data/processed/ as .parquet files.

Features computed:
  2.2  OHLC-derived: log return, body ratio, wick ratios, range z-score,
                     volume z-score, volume delta, gap
  2.3  Technical:    ATR(7/14/28), ATR ratio, RSI(14/28), MACD histogram,
                     ROC(5/10/20/60), Bollinger Band position,
                     SMA20/50 distance, rolling correlation to index,
                     high-low range percentile,
                     Williams %R, Stochastic %K, realised vol ratio,
                     OBV z-score, 52w high/low distance, ATR momentum,
                     CCI(20), Ichimoku distance, VWAP distance,
                     consecutive bar direction, RSI momentum, Donchian position
  2.4  Calendar:     day-of-week sin/cos, month sin/cos, hour sin/cos (intraday),
                     forex session flags, quarter-end flag
  1.5  Regime label: trend regime + volatility regime (6 classes)

Usage:
    python feature_engineering.py
"""

import os
import glob
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ─── Configuration ────────────────────────────────────────────────────────────

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
LOG_DIR       = "logs"
LOOKBACK      = 60      # bars per training sample
ROLLING_WIN   = 20      # for z-score normalisation of volume/range
Z_WIN         = 60      # rolling z-score window (blueprint 2.5)
Z_CLIP        = 3.0     # clip z-scores to [-3, +3]

# Reference tickers used for rolling correlation feature
INDEX_REFS = {
    "1D":  "SP500_GSPC_1D",
    "1W":  "SP500_GSPC_1W",
    "1H":  "SP500_GSPC_1H",
    "4H":  "SP500_GSPC_4H",
}

# ─── Logging ──────────────────────────────────────────────────────────────────

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "feature_engineering.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ─── Low-level indicators ─────────────────────────────────────────────────────

def atr(high, low, close, period):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def rsi(close, period):
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def rolling_zscore(series, window):
    # Use min_periods=window so z-scores are only computed when the full
    # window is available. Earlier bars become NaN and are dropped by
    # dropna() downstream, preventing unreliable early-window z-scores
    # from entering training data.
    m = series.rolling(window, min_periods=window).mean()
    s = series.rolling(window, min_periods=window).std()
    return (series - m) / (s + 1e-10)

# ─── Feature builders ─────────────────────────────────────────────────────────

def compute_ohlc_features(df):
    """Section 2.2 — OHLC-derived features."""
    eps = 1e-10
    hl  = df["High"] - df["Low"] + eps

    feats = pd.DataFrame(index=df.index)
    feats["log_return"]       = np.log(df["Close"] / (df["Close"].shift(1) + eps))
    feats["body_ratio"]       = (df["Close"] - df["Open"]) / hl
    feats["upper_wick_ratio"] = (df["High"] - df[["Open","Close"]].max(axis=1)) / hl
    feats["lower_wick_ratio"] = (df[["Open","Close"]].min(axis=1) - df["Low"]) / hl
    feats["range_zscore"]     = rolling_zscore(hl, ROLLING_WIN)
    vol_mean = df["Volume"].rolling(ROLLING_WIN).mean()
    vol_std  = df["Volume"].rolling(ROLLING_WIN).std()
    feats["volume_zscore"]    = (df["Volume"] - vol_mean) / (vol_std + eps)
    feats["volume_delta"]     = df["Volume"] / (df["Volume"].shift(1) + eps) - 1
    feats["gap"]              = (df["Open"] - df["Close"].shift(1)) / (df["Close"].shift(1) + eps)
    return feats


def compute_technical_features(df):
    """Section 2.3 — Technical indicator features."""
    eps   = 1e-10
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    feats = pd.DataFrame(index=df.index)

    # Volatility
    atr7  = atr(high, low, close, 7)
    atr14 = atr(high, low, close, 14)
    atr28 = atr(high, low, close, 28)
    feats["atr7_norm"]   = atr7  / (close + eps)
    feats["atr14_norm"]  = atr14 / (close + eps)
    feats["atr28_norm"]  = atr28 / (close + eps)
    feats["atr_ratio"]   = atr7  / (atr28 + eps)

    # Momentum
    feats["rsi14"] = rsi(close, 14) / 100
    feats["rsi28"] = rsi(close, 28) / 100

    ema12   = ema(close, 12)
    ema26   = ema(close, 26)
    macd    = ema12 - ema26
    signal  = ema(macd, 9)
    feats["macd_hist"] = (macd - signal) / (atr14 + eps)

    for period in [5, 10, 20, 60]:
        feats[f"roc_{period}"] = np.log(close / (close.shift(period) + eps))

    # Mean reversion
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_range = bb_upper - bb_lower + eps
    feats["bb_position"]   = (close - bb_lower) / bb_range
    feats["dist_sma20"]    = (close - sma20) / (atr14 + eps)
    feats["dist_sma50"]    = (close - close.rolling(50).mean()) / (atr14 + eps)

    # High-low range percentile
    bar_range = high - low
    feats["range_percentile"] = bar_range.rolling(60).rank(pct=True)

    # ── New features ──────────────────────────────────────────────────────────

    # Williams %R (14-bar): momentum oscillator, mapped to [0, 1]
    # Measures where close is within the 14-bar high-low range.
    # 0 = at 14-bar high (overbought zone), 1 = at 14-bar low (oversold zone).
    h14 = high.rolling(14).max()
    l14 = low.rolling(14).min()
    feats["williams_r_14"] = (h14 - close) / (h14 - l14 + eps)

    # Stochastic %K (14-bar): similar to Williams %R but low-anchored [0, 1]
    # 1 = close at 14-bar high, 0 = close at 14-bar low.
    feats["stoch_k_14"] = (close - l14) / (h14 - l14 + eps)

    # Realised volatility ratio: short-term vol / long-term vol.
    # > 1 = vol is expanding (regime shift), < 1 = vol contracting.
    # ATR-normalised to make it dimensionless across instruments.
    rv5  = close.diff().rolling(5).std()
    rv20 = close.diff().rolling(20).std()
    feats["realized_vol_ratio"] = rv5 / (rv20 + eps)

    # On-Balance Volume z-score: volume trend confirmation.
    # OBV rises when close rises (accumulation), falls when close falls (distribution).
    obv_raw = (np.sign(close.diff()) * df["Volume"]).cumsum()
    feats["obv_zscore"] = rolling_zscore(obv_raw, 20).clip(-Z_CLIP, Z_CLIP)

    # Distance to 52-week high and low, ATR-normalised.
    # Large negative dist_52w_high = far below 52w high (potential support break).
    # Close to 0 dist_52w_low = near 52w low (potential support level).
    h252 = high.rolling(252, min_periods=60).max()
    l252 = low.rolling(252, min_periods=60).min()
    feats["dist_52w_high"] = (close - h252) / (atr14 + eps)  # always <= 0
    feats["dist_52w_low"]  = (close - l252) / (atr14 + eps)  # always >= 0

    # ATR momentum: rate of change of ATR14 over 5 bars.
    # Positive = volatility expanding (potential breakout / trend acceleration).
    # Negative = volatility contracting (potential mean reversion).
    feats["atr_momentum"] = (atr14 - atr14.shift(5)) / (atr14.shift(5) + eps)

    # Commodity Channel Index (CCI, 20-bar): trend strength and extremes.
    # Normalised to roughly [-1, +1] range by dividing by 200.
    # +1 = strongly overbought trend, -1 = strongly oversold trend.
    typical_price = (high + low + close) / 3
    tp_sma20      = typical_price.rolling(20).mean()
    tp_mad20      = typical_price.rolling(20).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    feats["cci_20"] = (typical_price - tp_sma20) / (0.015 * tp_mad20 + eps) / 200


    # ── New features — batch 2 ────────────────────────────────────────────────

    # Ichimoku distance: gap between Tenkan-sen (9-bar) and Kijun-sen (26-bar),
    # ATR14-normalised. Positive = bullish momentum (fast line above slow line).
    # Widely used by institutional FX and equity participants.
    tenkan  = (high.rolling(9).max()  + low.rolling(9).min())  / 2
    kijun   = (high.rolling(26).max() + low.rolling(26).min()) / 2
    feats["ichimoku_distance"] = (tenkan - kijun) / (atr14 + eps)

    # VWAP distance: how far price is from its 20-bar volume-weighted average.
    # Institutional traders use VWAP as a reference price for order execution.
    # Positive = price above VWAP (bullish), negative = below (bearish).
    vwap20 = (close * df["Volume"]).rolling(20).sum() / (df["Volume"].rolling(20).sum() + eps)
    feats["vwap_distance"] = (close - vwap20) / (atr14 + eps)

    # Consecutive bar direction: number of consecutive bars moving in the same
    # direction, clipped to ±5 and normalised. Long up/down streaks often precede
    # reversals; short streaks indicate indecision.
    direction_sign = np.sign(close.diff()).fillna(0)
    consec = pd.Series(0.0, index=close.index)
    for i in range(1, len(close)):
        prev = consec.iloc[i - 1]
        cur  = direction_sign.iloc[i]
        if cur == 0:
            consec.iloc[i] = 0
        elif (prev >= 0 and cur > 0) or (prev <= 0 and cur < 0):
            consec.iloc[i] = prev + cur
        else:
            consec.iloc[i] = cur
    feats["consec_bars"] = (consec.clip(-5, 5) / 5)

    # RSI momentum: 3-bar rate of change of RSI14.
    # Captures whether momentum is accelerating (positive) or decelerating
    # (negative), independently of the RSI level itself.
    rsi14_series = rsi(close, 14)
    feats["rsi_momentum"] = (rsi14_series - rsi14_series.shift(3)) / 100

    # Donchian channel position: where close sits within the 20-bar high-low
    # range. 1.0 = at 20-bar high (potential resistance), 0.0 = at 20-bar low.
    # Range-based breakout signal, distinct from Bollinger (std-based).
    high20 = high.rolling(20).max()
    low20  = low.rolling(20).min()
    feats["donchian_position"] = (close - low20) / (high20 - low20 + eps)

    return feats


def compute_calendar_features(df, resolution):
    """Section 2.4 — Calendar and session features."""
    feats = pd.DataFrame(index=df.index)
    idx   = df.index

    # Day of week (0=Mon … 4=Fri)
    dow = idx.dayofweek
    feats["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    feats["dow_cos"] = np.cos(2 * np.pi * dow / 5)

    # Month of year
    month = idx.month
    feats["month_sin"] = np.sin(2 * np.pi * month / 12)
    feats["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Hour of day (intraday only)
    if resolution in ("1H", "4H"):
        hour = idx.hour
        feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)

        # Forex session flags (UTC hours)
        feats["session_asian"]   = ((hour >= 0)  & (hour < 8)).astype(float)
        feats["session_london"]  = ((hour >= 8)  & (hour < 16)).astype(float)
        feats["session_newyork"] = ((hour >= 13) & (hour < 21)).astype(float)
        feats["session_overlap"] = ((hour >= 13) & (hour < 16)).astype(float)

    # Quarter-end flag (within 5 trading days of quarter end).
    # Vectorised: compute days to next quarter-end for each date using
    # modular arithmetic on month-within-quarter and day-of-month.
    # Quarter ends: Mar 31, Jun 30, Sep 30, Dec 31.
    month = idx.month
    day   = idx.day
    # Days remaining in current quarter (approximate — ignores exact month lengths)
    # Month within quarter: 1=first, 2=mid, 3=last
    month_in_q = ((month - 1) % 3) + 1
    # Last month of quarter has <=31 days; flag if within last 7 calendar days
    is_last_month_of_q = (month_in_q == 3)
    # Approximate days left in month: use 31 as upper bound
    days_left_approx = 31 - day
    near_quarter_end = is_last_month_of_q & (days_left_approx <= 7)
    feats["quarter_end_flag"] = near_quarter_end.astype(float)

    return feats


def compute_regime_label(df):
    """Section 1.5 — Regime labelling (trend × volatility = 6 classes)."""
    atr14 = atr(df["High"], df["Low"], df["Close"], 14)
    close = df["Close"]

    # Trend regime over LOOKBACK bars
    # Trend regime: price moved > 1.5 × ATR directionally over the window
    # Blueprint spec — do NOT multiply by sqrt(LOOKBACK)
    price_move   = (close - close.shift(LOOKBACK)).abs()
    trend_thresh = 1.5 * atr14
    trend_regime = (price_move > trend_thresh).astype(int)   # 1=trending, 0=range

    # Volatility regime
    atr_pct = atr14 / (close + 1e-10)
    vol_regime = pd.cut(
        atr_pct,
        bins=[-np.inf, 0.01, 0.025, np.inf],
        labels=[0, 1, 2]          # 0=low, 1=normal, 2=high
    ).astype(float)

    # Combined: 0-5 (trend × 3 + vol)
    regime = trend_regime * 3 + vol_regime.fillna(1)
    return regime.rename("regime")


def add_index_correlation(df_instrument, df_index, resolution):
    """Rolling 20-bar correlation of instrument log-return to index log-return.

    When the index reference has different trading hours than the instrument
    (e.g. ASX 1H vs S&P 500 1H, which trade on opposite sides of the clock),
    reindex() finds no matching timestamps and produces all-NaN. This caused
    ASX/European 1H instruments to produce zero windows because the subsequent
    dropna() eliminated every row.

    Fix: after reindex+ffill, fill any remaining NaN with 0.0. A correlation
    of zero (no relationship to index) is an honest representation of the
    situation when the index has no overlapping bars. It does NOT drop the row.
    """
    if df_index is None:
        return pd.Series(0.0, index=df_instrument.index, name="corr_index")
    eps    = 1e-10
    r_inst = np.log(df_instrument["Close"] / (df_instrument["Close"].shift(1) + eps))
    r_idx  = np.log(df_index["Close"] / (df_index["Close"].shift(1) + eps))
    # Align index to instrument timestamps. For instruments that trade on
    # different hours/days, unmatched positions become NaN after reindex.
    r_idx  = r_idx.reindex(df_instrument.index).ffill().fillna(0.0)
    corr   = r_inst.rolling(20).corr(r_idx).rename("corr_index")
    # Fill any residual NaN from the rolling window warm-up with 0.0
    # so that dropna() downstream never eliminates rows on account of
    # this single feature being unavailable.
    return corr.fillna(0.0)

# ─── Normalisation pipeline (section 2.5) ────────────────────────────────────

def apply_rolling_zscore_pipeline(feature_df, non_zscore_cols):
    """
    Apply rolling z-score (window=Z_WIN) to all features except calendar features
    which are already normalised. Then clip to [-Z_CLIP, +Z_CLIP].
    """
    result = feature_df.copy()
    for col in feature_df.columns:
        if col in non_zscore_cols:
            continue
        result[col] = rolling_zscore(feature_df[col], Z_WIN).clip(-Z_CLIP, Z_CLIP)
    return result

# ─── Window builder ───────────────────────────────────────────────────────────

def build_windows(feature_df, regime_series):
    """
    Slide a LOOKBACK-bar window over the feature DataFrame.
    Returns:
        X      : np.ndarray [n_windows, LOOKBACK, n_features]
        regimes: np.ndarray [n_windows]        (regime label at window end)
        dates  : list of timestamps             (window end date)
    """
    arr     = feature_df.values.astype(np.float32)
    reg_arr = regime_series.values
    n       = len(arr)
    windows, regimes, dates = [], [], []

    for i in range(LOOKBACK, n):
        window = arr[i - LOOKBACK: i]
        if np.isnan(window).any():
            continue
        windows.append(window)
        regimes.append(reg_arr[i])
        dates.append(feature_df.index[i])

    if not windows:
        return None, None, None

    return np.array(windows), np.array(regimes), dates

# ─── Per-file processor ───────────────────────────────────────────────────────

def process_file(filepath, index_dfs):
    fname      = os.path.basename(filepath)
    parts      = fname.replace(".csv", "").split("_")
    resolution = parts[2] if len(parts) >= 3 else "1D"

    log.info("[PROC] %s", fname)

    # Load
    try:
        df = pd.read_csv(filepath, parse_dates=["Datetime"])
        df = df.set_index("Datetime").sort_index()
        # Strip timezone info if present — tz-aware vs tz-naive index
        # mismatch causes all correlation values to become NaN after reindex,
        # which then wipes every row via dropna() and produces zero windows.
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
    except Exception as e:
        log.error("  ✗  Failed to load %s: %s", fname, e)
        return None

    # Need at least LOOKBACK + Z_WIN bars to produce any samples
    raw_row_count = len(df)
    if raw_row_count < LOOKBACK + Z_WIN + 10:
        log.warning("  ✗  Too few rows (%d < %d) in %s — skipping.",
                    raw_row_count, LOOKBACK + Z_WIN + 10, fname)
        return None

    # Drop rows with zero/NaN close
    df = df[df["Close"] > 0].dropna(subset=["Open","High","Low","Close"])


    # Build features
    ohlc_feats = compute_ohlc_features(df)
    tech_feats = compute_technical_features(df)
    cal_feats  = compute_calendar_features(df, resolution)
    regime     = compute_regime_label(df)

    # Index correlation
    idx_df   = index_dfs.get(resolution)
    corr_col = add_index_correlation(df, idx_df, resolution).to_frame()

    # Combine all features
    all_feats = pd.concat([ohlc_feats, tech_feats, corr_col, cal_feats], axis=1)
    all_feats = all_feats.loc[df.index]   # align

    # Calendar cols are already normalised — skip z-scoring them
    cal_cols       = list(cal_feats.columns)
    non_zscore     = cal_cols + [
        "bb_position", "rsi14", "rsi28", "range_percentile", "regime",
        # Already bounded — z-scoring would distort their meaning
        "donchian_position",   # [0, 1] by construction
        "stoch_k_14",          # [0, 1] by construction
        "williams_r_14",       # [0, 1] by construction
        "consec_bars",         # [-1, +1] by construction
    ]

    # Apply rolling z-score pipeline
    feat_normalised = apply_rolling_zscore_pipeline(all_feats, non_zscore_cols=set(cal_cols))
    feat_normalised = feat_normalised.dropna()
    regime_aligned  = regime.reindex(feat_normalised.index)

    if len(feat_normalised) < LOOKBACK + 5:
        log.warning("  ✗  Only %d rows remain after dropna() in %s "
                   "(needed >%d) — likely NaN propagation in features.",
                    len(feat_normalised), fname, LOOKBACK + 5)
        return None

    # Build sample windows
    X, regimes, dates = build_windows(feat_normalised, regime_aligned)
    if X is None:
        log.warning("  ✗  No valid windows produced for %s", fname)
        return None

    log.info("  ✓  %d windows × %d features", X.shape[0], X.shape[2])

    # Regime class distribution check
    unique, counts = np.unique(regimes[~np.isnan(regimes)], return_counts=True)
    total = len(regimes)
    for cls, cnt in zip(unique, counts):
        pct = 100 * cnt / total
        if pct < 5:
            log.warning("  ⚠  Regime class %d is only %.1f%% of samples", int(cls), pct)

    return {
        "X":           X,
        "regimes":     regimes,
        "dates":       dates,
        "feature_cols": list(feat_normalised.columns),
        "source_file":  fname,
        "resolution":   resolution,
    }

# ─── Main ─────────────────────────────────────────────────────────────────────

def load_index_refs():
    """Load S&P 500 (or fallback) DataFrames for correlation feature."""
    index_dfs = {}
    for res, prefix in INDEX_REFS.items():
        pattern = os.path.join(RAW_DIR, f"*GSPC*{res}*.csv")
        matches = glob.glob(pattern)
        if matches:
            try:
                df = pd.read_csv(matches[0], parse_dates=["Datetime"])
                df = df.set_index("Datetime").sort_index()
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                index_dfs[res] = df
                log.info("[IDX]  Loaded index ref for %s: %s", res, os.path.basename(matches[0]))
            except Exception as e:
                log.warning("[IDX]  Could not load index ref for %s: %s", res, e)
    return index_dfs


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 2 Feature Engineering")
    parser.add_argument(
        "--refresh", action="store_true",
        help="Delete all existing .npy, .parquet, and _inspect.csv files from "
             "data/processed/ before processing. Use this after collect_data.py "
             "--refresh to rebuild everything from scratch."
    )
    args = parser.parse_args()

    log.info("Stage 2 Feature Engineering — Start")
    log.info("Raw data dir  : %s", os.path.abspath(RAW_DIR))
    log.info("Output dir    : %s", os.path.abspath(PROCESSED_DIR))

    # ── Refresh: delete all processed outputs ─────────────────────────────
    if args.refresh:
        deleted = 0
        for ext in ("*.npy", "*.parquet", "*_inspect.csv"):
            for f in glob.glob(os.path.join(PROCESSED_DIR, ext)):
                os.remove(f)
                deleted += 1
        if deleted:
            log.info("--refresh: deleted %d files from %s", deleted, PROCESSED_DIR)
        else:
            log.info("--refresh: no existing processed files found — clean run")

    csv_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not csv_files:
        log.error("No CSV files found in %s", RAW_DIR)
        return

    log.info("Found %d CSV files", len(csv_files))

    index_dfs = load_index_refs()

    results    = []
    skipped    = []
    already    = []
    nan_checks = []

    for filepath in csv_files:
        fname    = os.path.basename(filepath)
        out_stem = fname.replace(".csv", "")
        out_base = os.path.join(PROCESSED_DIR, out_stem)

        # ── Incremental skip: if .npy already exists, don't reprocess ────
        # This means only newly downloaded instruments (from collect_data.py
        # without --refresh) get processed. Instruments whose .npy files are
        # up to date are left alone, saving significant time.
        # Use --refresh to force a full rebuild.
        if not args.refresh and os.path.exists(out_base + ".npy"):
            log.debug("[SKIP] %s — .npy already exists", fname)
            already.append(fname)
            continue

        result = process_file(filepath, index_dfs)

        if result is None:
            skipped.append(fname)
            continue

        # NaN / Inf check
        nan_count = np.isnan(result["X"]).sum()
        inf_count = np.isinf(result["X"]).sum()
        if nan_count > 0 or inf_count > 0:
            nan_checks.append((result["source_file"], nan_count, inf_count))
            log.warning("  ⚠  NaN/Inf in %s: %d NaN, %d Inf",
                        result["source_file"], nan_count, inf_count)

        # out_stem and out_base already computed above (before the skip check)
        meta_df = pd.DataFrame({
            "date":    result["dates"],
            "regime":  result["regimes"],
        })

        # 1. Feature windows as numpy array  [n_windows, 60, n_features]
        np.save(out_base + ".npy", result["X"])

        # 2. Metadata (dates + regime labels) as parquet
        meta_df.to_parquet(out_base + ".parquet", index=False)

        # 3. Human-readable CSV — last bar of each window so you can open it in Excel
        last_bar   = result["X"][:, -1, :]
        inspect_df = pd.DataFrame(last_bar, columns=result["feature_cols"])
        inspect_df.insert(0, "date",   [str(d) for d in result["dates"]])
        inspect_df.insert(1, "regime", result["regimes"])
        inspect_df.to_csv(out_base + "_inspect.csv", index=False)
        log.info("  → Inspect CSV written: %s_inspect.csv", out_stem)

        results.append(result)

    # ── Summary ──────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Feature engineering complete.")
    log.info("  Processed  : %d files", len(results))
    log.info("  Up to date : %d files (skipped — .npy already exists)", len(already))
    log.info("  Failed     : %d files", len(skipped))

    if already:
        log.info("  Run with --refresh to reprocess all files from scratch.")
    if skipped:
        log.info("  Failed list: %s", skipped)

    if nan_checks:
        log.warning("  Files with NaN/Inf values:")
        for fname, n, i in nan_checks:
            log.warning("    %s — NaN: %d  Inf: %d", fname, n, i)
    else:
        log.info("  ✓  No NaN or Inf values detected in any output array.")

    total_windows = sum(r["X"].shape[0] for r in results)
    log.info("  Total training windows: %d", total_windows)

    # Regime class balance across all data
    all_regimes = np.concatenate([r["regimes"] for r in results])
    valid_reg   = all_regimes[~np.isnan(all_regimes)]
    log.info("\nGlobal regime class distribution:")
    for cls in range(6):
        cnt = (valid_reg == cls).sum()
        pct = 100 * cnt / len(valid_reg) if len(valid_reg) > 0 else 0
        flag = " ⚠  BELOW 5%" if pct < 5 else ""
        log.info("  Class %d: %5d samples  (%5.1f%%)%s", cls, cnt, pct, flag)

    if results:
        sample = results[0]
        log.info("\nFeature columns (%d total):", len(sample["feature_cols"]))
        for col in sample["feature_cols"]:
            log.info("  - %s", col)


if __name__ == "__main__":
    main()