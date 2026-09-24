"""Walk-forward backtest of Apex-Alpha's quantitative BUY / HOLD / SELL calls on real prices.

Every 30 trading days, for each stock, the model sees only trailing data (beta, realized volatility,
52-week high), simulates 30-day price paths with the app's own Monte Carlo engine, and makes a call:

    BUY   fractional Kelly allocation > 0 (same rule the risk engine uses to size positions)
    SELL  probability of profit below 45%
    HOLD  otherwise

The call is scored against the realized 30-trading-day forward return. Calls do not overlap.
Tuning uses 2011-2018 only; 2019 onward is held out and reported separately.

Run:  pip install -r backtest/requirements.txt && python -m backtest.run_backtest
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.quant_engine import QuantitativeForecaster as QF

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache" / "prices.csv"

# Large caps with long histories. Current constituents only, so results carry survivorship bias.
UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "JPM", "JNJ", "PG", "KO", "PEP", "WMT", "XOM",
    "CVX", "HD", "UNH", "V", "MA", "DIS", "INTC", "CSCO", "ORCL", "IBM", "MRK", "PFE", "VZ", "BA", "CAT",
    "MCD", "NKE", "COST",
]
MARKET = "SPY"
START, END = "2009-12-01", "2026-09-01"
SPLIT = pd.Timestamp("2019-01-01")
HORIZON = 30          # trading days, matches the app's 30-day horizon
LOOKBACK = 252
PATHS = 3000
SELL_BELOW = 0.45     # probability-of-profit threshold for SELL, fixed up front (not tuned)
# Run 1 used [0, 0.25, 0.5, 1.0] (momentum only); every positive tilt lowered in-sample IC, so run 2
# widened the grid to include reversal (negative) tilts. Out-of-sample data was scored once per run.
TILT_GRID = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]


def load_prices() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_csv(CACHE, index_col=0, parse_dates=True)
    import yfinance as yf

    prices = yf.download(UNIVERSE + [MARKET], start=START, end=END, auto_adjust=True, progress=False)["Close"]
    CACHE.parent.mkdir(exist_ok=True)
    prices.to_csv(CACHE)
    return prices


def legacy_simulate(price: float, drift: float, vol: float, horizon: int, paths: int, seed: int) -> np.ndarray:
    """The simulator as it shipped before this change: 0.05 jumps/day, mean -1%, no drift compensation."""
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    log_ret = np.zeros(paths)
    for _ in range(horizon):
        jumps = rng.poisson(0.05, paths) * rng.normal(-0.01, 0.04, paths)
        log_ret += (drift - 0.5 * vol**2) * dt + vol * math.sqrt(dt) * rng.normal(0, 1, paths) + jumps
    return price * np.exp(log_ret)


@dataclass
class Features:
    ticker: str
    date: pd.Timestamp
    price: float
    beta: float
    vol: float
    high_52w_gap: float   # price / 52-week high - 1, in [-1, 0]
    fwd_return: float


def build_features(prices: pd.DataFrame) -> list[Features]:
    rets = np.log(prices).diff()
    rows = []
    for ticker in UNIVERSE:
        px = prices[ticker].dropna()
        idx = px.index
        for i in range(LOOKBACK, len(idx) - HORIZON, HORIZON):
            window = slice(idx[i - LOOKBACK + 1], idx[i])
            r_i, r_m = rets[ticker][window], rets[MARKET][window]
            ok = r_i.notna() & r_m.notna()
            beta = float(np.cov(r_i[ok], r_m[ok])[0, 1] / np.var(r_m[ok], ddof=1))
            rows.append(Features(
                ticker=ticker,
                date=idx[i],
                price=float(px.iloc[i]),
                beta=beta,
                vol=float(r_i[ok].std() * math.sqrt(252)),
                high_52w_gap=float(px.iloc[i] / px[window].max() - 1),
                fwd_return=float(px.iloc[i + HORIZON] / px.iloc[i] - 1),
            ))
    return rows


def call_from_paths(price: float, terminal: np.ndarray) -> tuple[str, float, float]:
    p = float(np.mean(terminal > price))
    p10, p50, p90 = np.percentile(terminal, [10, 50, 90])
    b = max(0.01, p90 - price) / max(0.01, price - p10)
    kelly = (p * b - (1 - p)) / b
    call = "BUY" if kelly > 0 else "SELL" if p < SELL_BELOW else "HOLD"
    return call, p, float(p50 / price - 1)


def run_model(features: list[Features], drift_fn, simulate) -> pd.DataFrame:
    out = []
    for n, f in enumerate(features):
        drift = drift_fn(f)
        terminal = simulate(f.price, drift, f.vol, HORIZON, PATHS, seed=n)
        call, p, median_ret = call_from_paths(f.price, terminal)
        out.append((f.ticker, f.date, call, p, median_ret, f.fwd_return))
    return pd.DataFrame(out, columns=["ticker", "date", "call", "p_profit", "pred_median", "fwd_return"])


def summarize(df: pd.DataFrame) -> dict:
    buy, sell, hold = (df[df.call == c] for c in ("BUY", "SELL", "HOLD"))
    # Portfolio view: each period, equal-weight the BUY names (cash at 0 if none) vs equal-weight everything.
    by_date = df.groupby("date")
    strat = by_date.apply(lambda g: g.loc[g.call == "BUY", "fwd_return"].mean() if (g.call == "BUY").any() else 0.0)
    bench = by_date["fwd_return"].mean()
    periods_per_year = 252 / HORIZON

    def ann(r: pd.Series) -> float:
        return float((1 + r).prod() ** (periods_per_year / len(r)) - 1)

    def mean(s: pd.Series) -> float | None:
        return float(s.mean()) if len(s) else None

    return {
        "n_calls": int(len(df)),
        "share": {c: round(float((df.call == c).mean()), 3) for c in ("BUY", "HOLD", "SELL")},
        "base_rate_up": round(float((df.fwd_return > 0).mean()), 3),
        "buy_hit_rate": round(float((buy.fwd_return > 0).mean()), 3) if len(buy) else None,
        "sell_hit_rate": round(float((sell.fwd_return < 0).mean()), 3) if len(sell) else None,
        "avg_fwd_return": {"BUY": mean(buy.fwd_return), "HOLD": mean(hold.fwd_return), "SELL": mean(sell.fwd_return)},
        "rank_ic": round(float(df.pred_median.rank().corr(df.fwd_return.rank())), 4),
        "strategy_cagr": round(ann(strat), 4),
        "equal_weight_cagr": round(ann(bench), 4),
    }


def main() -> None:
    prices = load_prices()
    features = build_features(prices)
    train = [f for f in features if f.date < SPLIT]
    center = float(np.median([f.high_52w_gap for f in train]))
    print(f"{len(features)} calls, {len(train)} in-sample; median 52w-high gap in-sample {center:.3f}")

    rf, erp = QF.RISK_FREE_RATE, QF.EQUITY_RISK_PREMIUM
    legacy = lambda f: max(-0.15, min(0.35, 0.12 * (1.0 / max(0.5, f.beta))))  # noqa: E731
    capm = lambda f: rf + f.beta * erp  # noqa: E731
    fixed_sim = lambda price, drift, vol, h, paths, seed: QF.simulate_gbm_paths(price, drift, vol, h, paths, seed)  # noqa: E731

    # Pick the momentum tilt on 2011-2018 only, by rank IC.
    tilt_scores = {}
    for tilt in TILT_GRID:
        drift = lambda f, t=tilt: capm(f) + t * (f.high_52w_gap - center)  # noqa: E731
        tilt_scores[tilt] = summarize(run_model(train, drift, fixed_sim))["rank_ic"]
        print(f"  tilt {tilt}: in-sample rank IC {tilt_scores[tilt]}")
    best_tilt = max(tilt_scores, key=tilt_scores.get)

    models = {
        "shipped: 0.12/beta drift, uncompensated jumps": (legacy, legacy_simulate),
        "0.12/beta drift, fixed jumps": (legacy, fixed_sim),
        "CAPM drift, fixed jumps": (capm, fixed_sim),
        f"CAPM + 52w-high tilt {best_tilt}, fixed jumps": (
            lambda f: capm(f) + best_tilt * (f.high_52w_gap - center), fixed_sim),
    }
    results = {"config": {
        "universe": UNIVERSE, "start": START, "end": END, "split": str(SPLIT.date()), "horizon_trading_days": HORIZON,
        "paths": PATHS, "sell_below_p_profit": SELL_BELOW, "tilt_grid": TILT_GRID, "tilt_in_sample_ic": tilt_scores,
        "chosen_tilt": best_tilt, "momentum_center": round(center, 4), "risk_free": rf, "equity_risk_premium": erp,
    }, "models": {}}
    for name, (drift_fn, sim) in models.items():
        df = run_model(features, drift_fn, sim)
        results["models"][name] = {
            "in_sample_2011_2018": summarize(df[df.date < SPLIT]),
            "out_of_sample_2019_on": summarize(df[df.date >= SPLIT]),
        }
        oos = results["models"][name]["out_of_sample_2019_on"]
        print(f"\n{name}\n  OOS: {json.dumps(oos)}")

    (HERE / "results.json").write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
