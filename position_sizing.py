"""
Position Sizing Module — Probability-Scaled Sizing
Trading Algorithm Blueprint

Translates the model's direction probability into a position size.
Position size scales with conviction — higher probability = larger bet.

Sizing formula:
    raw_signal = (prob - 0.5) * 2          # maps [0.5, 1.0] → [0.0, 1.0]
    kelly_f    = 2p - 1                    # full Kelly fraction at this accuracy
    half_kelly = kelly_f * KELLY_FRACTION  # use fractional Kelly (default 0.5)
    position   = half_kelly * raw_signal   # scale by conviction

This ensures:
  - prob = 0.50 → position = 0 (no trade)
  - prob = 0.75 → position = half_kelly * 0.5
  - prob = 1.00 → position = half_kelly * 1.0
  - Below MIN_PROB threshold → position = 0 (skip low-conviction signals)

Usage:
    from position_sizing import PositionSizer

    sizer = PositionSizer(
        starting_capital=10000,
        max_risk_per_trade=0.02,   # 2% of capital per trade
        kelly_fraction=0.5,        # half-Kelly
        min_prob=0.55,             # ignore signals below 55% confidence
    )

    size, direction, notional = sizer.size(prob=0.68, current_capital=10500)
    # → size=0.018, direction='long', notional=189.0
"""

from __future__ import annotations
import numpy as np


# ─── Constants ────────────────────────────────────────────────────────────────

DEFAULT_STARTING_CAPITAL = 10_000.0   # £10,000 default
DEFAULT_MAX_RISK         = 0.1       # 2% of capital per trade
DEFAULT_KELLY_FRACTION   = 0.7        # half-Kelly
DEFAULT_MIN_PROB         = 0.57       # minimum confidence to trade
DEFAULT_MAX_POSITION     = 0.20       # never more than 20% of capital in one trade
TRANSACTION_COST         = 0.0005     # 0.05% round-trip


# ─── Core sizer ───────────────────────────────────────────────────────────────

class PositionSizer:
    """
    Probability-scaled position sizer.

    Parameters
    ----------
    starting_capital : float
        Initial portfolio value in base currency.
    max_risk_per_trade : float
        Maximum fraction of current capital to risk on any single trade.
        Acts as a ceiling regardless of Kelly output.
    kelly_fraction : float
        Fraction of full Kelly to use. 0.5 = half-Kelly (recommended).
        Full Kelly (1.0) maximises long-run growth but has high variance.
    min_prob : float
        Direction probability below which no trade is taken.
        Must be > 0.5. Signals below this threshold return size=0.
    max_position_fraction : float
        Hard cap on position size as fraction of capital.
    """

    def __init__(
        self,
        starting_capital:     float = DEFAULT_STARTING_CAPITAL,
        max_risk_per_trade:   float = DEFAULT_MAX_RISK,
        kelly_fraction:       float = DEFAULT_KELLY_FRACTION,
        min_prob:             float = DEFAULT_MIN_PROB,
        max_position_fraction: float = DEFAULT_MAX_POSITION,
    ):
        assert 0.5 < min_prob < 1.0,      "min_prob must be in (0.5, 1.0)"
        assert 0.0 < kelly_fraction <= 1.0, "kelly_fraction must be in (0, 1]"
        assert 0.0 < max_risk_per_trade <= 0.5

        self.starting_capital      = starting_capital
        self.max_risk_per_trade    = max_risk_per_trade
        self.kelly_fraction        = kelly_fraction
        self.min_prob              = min_prob
        self.max_position_fraction = max_position_fraction

    def size(
        self,
        prob: float,
        current_capital: float | None = None,
    ) -> tuple[float, str, float]:
        """
        Compute position size for a single signal.

        Parameters
        ----------
        prob : float
            Model's direction probability in [0, 1].
            > 0.5 = bullish, < 0.5 = bearish, = 0.5 = no signal.
        current_capital : float
            Current portfolio value. Defaults to starting_capital.

        Returns
        -------
        fraction : float
            Position size as fraction of current capital [0, max_position].
        direction : str
            'long', 'short', or 'no_trade'.
        notional : float
            Monetary value of the position.
        """
        capital = current_capital if current_capital is not None else self.starting_capital

        # Determine direction
        if prob > 0.5:
            direction = "long"
            p = prob
        elif prob < 0.5:
            direction = "short"
            p = 1 - prob   # mirror: short at prob=0.3 ↔ long at prob=0.7
        else:
            return 0.0, "no_trade", 0.0

        # Skip low-conviction signals
        if p < self.min_prob:
            return 0.0, "no_trade", 0.0

        # Probability-scaled sizing
        # conviction ∈ [0, 1]: how far above min_prob this signal is
        conviction   = (p - 0.5) * 2.0          # maps [0.5, 1.0] → [0.0, 1.0]

        # Kelly fraction at this accuracy level
        kelly_full   = 2 * p - 1                 # E[r]/Var[r] under ±1 payoff
        kelly_used   = kelly_full * self.kelly_fraction

        # Scale by conviction and cap
        raw_fraction = kelly_used * conviction
        fraction     = min(raw_fraction, self.max_risk_per_trade, self.max_position_fraction)
        fraction     = max(fraction, 0.0)

        notional = fraction * capital

        return fraction, direction, notional

    def size_batch(
        self,
        probs: np.ndarray,
        capital_series: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorised sizing for an array of probabilities.

        Parameters
        ----------
        probs : np.ndarray [n]
            Array of direction probabilities.
        capital_series : np.ndarray [n] or None
            Capital at each bar. If None, uses starting_capital for all.

        Returns
        -------
        fractions  : np.ndarray [n]  — position sizes as fraction of capital
        directions : np.ndarray [n]  — +1=long, -1=short, 0=no_trade
        notionals  : np.ndarray [n]  — monetary values
        """
        n       = len(probs)
        capital = capital_series if capital_series is not None else \
                  np.full(n, self.starting_capital)

        fractions  = np.zeros(n)
        directions = np.zeros(n)

        long_mask  = probs > 0.5
        short_mask = probs < 0.5
        p_long     = np.where(long_mask,  probs,     0.5)
        p_short    = np.where(short_mask, 1 - probs, 0.5)

        # Long signals
        lm = long_mask & (p_long >= self.min_prob)
        if lm.any():
            conv    = (p_long[lm] - 0.5) * 2
            kelly   = (2 * p_long[lm] - 1) * self.kelly_fraction
            raw     = kelly * conv
            fractions[lm]  = np.clip(raw, 0, min(self.max_risk_per_trade,
                                                   self.max_position_fraction))
            directions[lm] = 1.0

        # Short signals
        sm = short_mask & (p_short >= self.min_prob)
        if sm.any():
            conv    = (p_short[sm] - 0.5) * 2
            kelly   = (2 * p_short[sm] - 1) * self.kelly_fraction
            raw     = kelly * conv
            fractions[sm]  = np.clip(raw, 0, min(self.max_risk_per_trade,
                                                   self.max_position_fraction))
            directions[sm] = -1.0

        notionals = fractions * capital
        return fractions, directions, notionals


# ─── Full backtest simulation ─────────────────────────────────────────────────

def run_backtest(
    probs:           np.ndarray,
    true_directions: np.ndarray,
    sizer:           PositionSizer,
    resolution:      str = "1D",
    calendar_years:  float | None = None,
    n_instruments:   int = 1,
) -> dict:
    """
    Run a full backtest using probability-scaled position sizing.
    """
    BARS_PER_YEAR = {"1D": 252, "4H": 1575, "1H": 6300, "1W": 52}.get(resolution, 252)
    UNIT_RETURN = {"1D": 0.010, "4H": 0.004, "1H": 0.002, "1W": 0.020}.get(resolution, 0.010)

    n            = len(probs)
    capital      = sizer.starting_capital
    equity       = [capital]
    trade_log    = []
    n_long        = 0
    n_short       = 0
    n_no_trade    = 0
    gross_profit  = 0.0
    gross_loss    = 0.0
    prev_dir      = 0
    current_hold  = 0          # bars held in the current open position
    hold_durations = []        # completed hold lengths (bars)

    for i in range(n):
        frac, direction, notional = sizer.size(probs[i], capital)

        if direction == "no_trade":
            n_no_trade += 1
            equity.append(capital)
            continue

        dir_val    = 1 if direction == "long" else -1
        correct    = (dir_val > 0) == (true_directions[i] > 0.5)
        bar_return = UNIT_RETURN if correct else -UNIT_RETURN

        tx_cost  = TRANSACTION_COST * notional if dir_val != prev_dir and prev_dir != 0 else 0.0
        pnl      = notional * bar_return - tx_cost

        capital += pnl
        capital  = max(capital, 0.01)
        equity.append(capital)

        if pnl > 0: gross_profit += pnl
        else:       gross_loss   += abs(pnl)

        if direction == "long":  n_long  += 1
        else:                    n_short += 1

        trade_log.append({
            "bar": i, "prob": float(probs[i]), "direction": direction,
            "size_frac": frac, "notional": notional,
            "correct": correct, "pnl": pnl, "capital": capital,
        })

        # Track position hold duration
        if dir_val != prev_dir and prev_dir != 0:
            # Direction flipped — record the just-closed hold
            hold_durations.append(current_hold)
            current_hold = 1
        else:
            current_hold += 1

        prev_dir = dir_val

    # Close final open position
    if current_hold > 0 and prev_dir != 0:
        hold_durations.append(current_hold)

    equity_arr = np.array(equity)

    n_trades = n_long + n_short
    # Use calendar_years if provided — avoids bar-count inflation for multi-instrument data
    n_years  = calendar_years if calendar_years is not None else n / BARS_PER_YEAR

    # Hold duration statistics
    BAR_LABEL = {"1D": "days", "4H": "4H bars", "1H": "hours", "1W": "weeks"}
    bar_label = BAR_LABEL.get(resolution, "bars")
    # Convert bars to calendar time for readability
    BAR_TO_HOURS = {"1D": 24, "4H": 4, "1H": 1, "1W": 168}
    hours_per_bar = BAR_TO_HOURS.get(resolution, 24)

    if hold_durations:
        arr_h = np.array(hold_durations, dtype=float)
        avg_hold     = float(arr_h.mean())
        med_hold     = float(np.median(arr_h))
        min_hold     = int(arr_h.min())
        max_hold     = int(arr_h.max())
        pct_1bar     = float((arr_h == 1).mean() * 100)   # flipped next bar
        pct_le5      = float((arr_h <= 5).mean() * 100)
        avg_hold_hrs = avg_hold * hours_per_bar
    else:
        avg_hold = med_hold = avg_hold_hrs = 0.0
        min_hold = max_hold = 0
        pct_1bar = pct_le5 = 0.0
    total_return  = (capital - sizer.starting_capital) / sizer.starting_capital
    annual_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    peak   = np.maximum.accumulate(equity_arr)
    max_dd = float(np.abs(((equity_arr - peak) / (peak + 1e-10)).min()))

    pred_dir    = (probs >= 0.5).astype(float)
    true_dir    = (true_directions > 0.5).astype(float)
    p_acc       = (pred_dir == true_dir).mean()
    e_r         = 2 * p_acc - 1
    std_r       = 2 * np.sqrt(p_acc * (1 - p_acc) + 1e-10)
    sharpe      = float(e_r / std_r * np.sqrt(BARS_PER_YEAR))

    wins          = sum(1 for t in trade_log if t["correct"])
    win_rate      = wins / n_trades if n_trades > 0 else 0.0
    profit_factor = gross_profit / (gross_loss + 1e-10)
    avg_win       = gross_profit / (wins + 1e-10)
    avg_loss      = gross_loss   / (n_trades - wins + 1e-10)
    bars_per_trade  = n / n_trades if n_trades > 0 else float("inf")
    trades_per_year = n_trades / max(n_years, 0.01)

    return {
        "starting_capital":   sizer.starting_capital,
        "ending_capital":     round(capital, 2),
        "total_return_pct":   round(total_return * 100, 2),
        "annual_return_pct":  round(annual_return * 100, 2),
        "max_drawdown_pct":   round(max_dd * 100, 2),
        "sharpe_ratio":       round(sharpe, 3),
        "n_bars_total":       n,
        "n_trades":           n_trades,
        "n_long":             n_long,
        "n_short":            n_short,
        "n_no_trade":         n_no_trade,
        # Hold duration
        "avg_hold_bars":      round(avg_hold, 1),
        "avg_hold_hrs":       round(avg_hold_hrs, 1),
        "med_hold_bars":      round(med_hold, 1),
        "min_hold_bars":      min_hold,
        "max_hold_bars":      max_hold,
        "pct_held_1bar":      round(pct_1bar, 1),
        "pct_held_le5bars":   round(pct_le5, 1),
        "hold_bar_label":     bar_label,
        "trades_per_year":    round(trades_per_year, 1),
        "bars_per_trade":     round(bars_per_trade, 1),
        "pct_bars_traded":    round(100 * n_trades / n, 1),
        "n_years":            round(n_years, 2),
        "n_instruments":      n_instruments,
        "win_rate_pct":       round(win_rate * 100, 2),
        "profit_factor":      round(profit_factor, 3),
        "avg_win":            round(avg_win, 4),
        "avg_loss":           round(avg_loss, 4),
        "gross_profit":       round(gross_profit, 2),
        "gross_loss":         round(gross_loss, 2),
        "kelly_fraction":     sizer.kelly_fraction,
        "max_risk_per_trade": sizer.max_risk_per_trade,
        "min_prob_threshold": sizer.min_prob,
        "resolution":         resolution,
        "equity_curve":       equity_arr.tolist(),
        "trade_log":          trade_log,
    }



def print_backtest_report(results: dict, label: str = ""):
    """Print a formatted backtest report."""
    PASS, FAIL, WARN = "✓", "✗", "⚠"
    SEP = "─" * 58

    print(f"\n{SEP}")
    print(f"BACKTEST REPORT{' — ' + label if label else ''}")
    print(SEP)

    sc  = results["starting_capital"]
    ec  = results["ending_capital"]
    ret = results["total_return_pct"]
    ann = results["annual_return_pct"]
    dd  = results["max_drawdown_pct"]
    sh  = results["sharpe_ratio"]
    pf  = results["profit_factor"]
    wr  = results["win_rate_pct"]

    currency = "£"

    print(f"\n  Capital:")
    print(f"    Starting capital   : {currency}{sc:>12,.2f}")
    print(f"    Ending capital     : {currency}{ec:>12,.2f}  ({ret:+.2f}%)")
    print(f"    Annualised return  : {ann:+.2f}%")

    print(f"\n  Risk:")
    sh_flag = PASS if sh >= 0.8 else (WARN if sh >= 0.5 else FAIL)
    dd_flag = PASS if dd <= 25 else FAIL
    print(f"    Sharpe ratio       :  {sh:.3f}  {sh_flag}")
    print(f"    Max drawdown       :  {dd:.2f}%  {dd_flag}")
    print(f"    Profit factor      :  {pf:.3f}")

    print(f"\n  Trades ({results['resolution']} bars):")
    print(f"    Total bars         :  {results['n_bars_total']:,}")
    print(f"    Total trades taken :  {results['n_trades']:,}  "
          f"({results['pct_bars_traded']:.1f}% of bars)")
    print(f"    Trades per year    :  {results['trades_per_year']:,.1f}")
    print(f"    Long / Short       :  {results['n_long']:,} / {results['n_short']:,}")
    print(f"    No-trade (low conf):  {results['n_no_trade']:,}")
    print(f"    Bars per trade     :  {results['bars_per_trade']:.1f}")
    print(f"    Timespan           :  {results['n_years']:.1f} years")

    print(f"\n  Edge:")
    wr_flag = PASS if wr >= 53 else (WARN if wr >= 51 else FAIL)
    print(f"    Win rate           :  {wr:.2f}%  {wr_flag}")
    print(f"    Avg win            :  {currency}{results['avg_win']:.4f}")
    print(f"    Avg loss           :  {currency}{results['avg_loss']:.4f}")
    print(f"    Gross profit       :  {currency}{results['gross_profit']:,.2f}")
    print(f"    Gross loss         :  {currency}{results['gross_loss']:,.2f}")

    # Trade duration
    avg_h = results.get("avg_hold_bars", 0)
    med_h = results.get("med_hold_bars", 0)
    min_h = results.get("min_hold_bars", 0)
    max_h = results.get("max_hold_bars", 0)
    avg_d = results.get("avg_hold_hrs", 0) / 24 if results.get("resolution") in ("1H","4H")             else avg_h  # for daily, bars = days
    p1    = results.get("pct_held_1bar", 0)
    p5    = results.get("pct_held_le5bars", 0)
    unit  = results.get("hold_bar_label", "bars")

    print(f"\n  Trade Duration:")
    print(f"    Avg hold           :  {avg_h:.1f} {unit}  (~{avg_d:.1f} calendar days)")
    print(f"    Median hold        :  {med_h:.1f} {unit}")
    print(f"    Min / Max hold     :  {min_h} / {max_h} {unit}")
    print(f"    Held 1 bar only    :  {p1:.1f}%  (flipped next bar)")
    print(f"    Held ≤5 bars       :  {p5:.1f}%  of all trades")

    print(f"\n  Sizing config:")
    print(f"    Kelly fraction     :  {results['kelly_fraction']:.1f}x")
    print(f"    Max risk/trade     :  {results['max_risk_per_trade']*100:.1f}%")
    print(f"    Min prob threshold :  {results['min_prob_threshold']*100:.0f}%")

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    import sys

    # ── Quick demo ────────────────────────────────────────────────────────
    print("Position Sizer — Demo\n")

    sizer = PositionSizer(
        starting_capital=10_000,
        max_risk_per_trade=0.1,
        kelly_fraction=0.7,
        min_prob=0.57,
    )

    # Show how size scales with probability
    print("  Probability → Position size (% of capital):\n")
    print(f"  {'Prob':<8} {'Direction':<10} {'Size %':<10} {'Notional (£10k)'}")
    print(f"  {'-'*45}")
    for prob in [0.50, 0.52, 0.53, 0.55, 0.58, 0.62, 0.68, 0.75, 0.85]:
        frac, direction, notional = sizer.size(prob, 10_000)
        print(f"  {prob:<8.2f} {direction:<10} {frac*100:<10.3f} £{notional:.2f}")

    print()

    # Demo backtest — 54% accurate model, realistic persistent signals
    np.random.seed(42)
    n = 252 * 3   # 3 years daily

    # True direction persists in trends (~5 bars average)
    true = np.zeros(n)
    d = 1.0
    for i in range(n):
        if np.random.random() < 0.15: d = -d
        true[i] = 1.0 if d > 0 else 0.0

    # 54% accurate probs with persistence (realistic model output)
    raw = np.zeros(n)
    for i in range(n):
        correct = np.random.random() < 0.54
        if correct:
            raw[i] = np.random.uniform(0.54, 0.72) if true[i] > 0.5 else np.random.uniform(0.28, 0.46)
        else:
            raw[i] = np.random.uniform(0.28, 0.46) if true[i] > 0.5 else np.random.uniform(0.54, 0.72)
        if i > 0: raw[i] = np.clip(0.6 * raw[i] + 0.4 * raw[i-1], 0.01, 0.99)

    for res in ["1D", "4H", "1W"]:
        r = run_backtest(raw, true, sizer, resolution=res)
        print_backtest_report(r, f"Demo — 54% acc, 3yr, {res}")