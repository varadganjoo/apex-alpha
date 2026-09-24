"""Quantitative Forecasting Engine using Geometric Brownian Motion (GBM) with Jump Diffusion.
Simulates multi-horizon price distributions and calculates exact probabilistic quantiles (P10, P50, P90).
"""

import math
import numpy as np
from typing import Dict, List, Optional
from app.schemas import MarketQuote, MonteCarloResult, QuantileForecast


class QuantitativeForecaster:
    """Simulates asset price paths and computes multi-horizon quantile forecasts."""

    HORIZONS = [5, 30, 90]  # 5-day (tactical), 30-day (monthly), 90-day (quarterly)
    NUM_PATHS = 10000

    RISK_FREE_RATE = 0.045       # matches RiskGuardEngine's Sharpe calculation
    EQUITY_RISK_PREMIUM = 0.055  # long-run US equity premium used for CAPM expected returns

    # Merton jumps: rare earnings/macro shocks, about four a year.
    JUMPS_PER_YEAR = 4.0
    JUMP_MEAN = -0.01
    JUMP_STD = 0.05

    # 52-week-high tilt on top of CAPM, chosen on 2011-2018 by backtest/run_backtest.py. Negative = short-term
    # reversal: stocks far below their high get a higher expected return. Held-out rank IC +0.07 (backtest/RESULTS.md).
    # ponytail: -1.0 is the edge of the searched grid; widening it would need a fresh holdout period.
    HIGH_52W_TILT = -1.0
    HIGH_52W_CENTER = -0.046  # in-sample median of price / 52-week high - 1

    @classmethod
    def estimate_drift(cls, quote: MarketQuote) -> float:
        """Expected annual return: CAPM (risk-free + beta x equity premium), plus the backtested 52-week-high tilt."""
        gap = quote.price / quote.week_52_high - 1 if quote.week_52_high > 0 else cls.HIGH_52W_CENTER
        return cls.RISK_FREE_RATE + quote.beta * cls.EQUITY_RISK_PREMIUM + cls.HIGH_52W_TILT * (gap - cls.HIGH_52W_CENTER)

    @classmethod
    def simulate_gbm_paths(
        cls,
        current_price: float,
        annualized_drift: float,
        annualized_volatility: float,
        horizon_days: int,
        num_paths: int = NUM_PATHS,
        random_seed: Optional[int] = 42,
    ) -> np.ndarray:
        """Simulates terminal prices under Merton jump diffusion.

        `annualized_drift` is the expected (arithmetic) return and `annualized_volatility` the total
        volatility, jumps included. The drift is jump-compensated and the jump variance is carved out of
        the diffusion, so the simulated paths match both inputs instead of adding a hidden drag.
        """
        rng = np.random.default_rng(random_seed)
        dt = 1.0 / 252.0

        lam = cls.JUMPS_PER_YEAR
        kappa = math.exp(cls.JUMP_MEAN + 0.5 * cls.JUMP_STD**2) - 1  # expected relative jump size
        jump_var = lam * (cls.JUMP_MEAN**2 + cls.JUMP_STD**2)
        # ponytail: floor keeps at least half the volatility in the diffusion for very quiet stocks.
        diffusion_var = max(annualized_volatility**2 - jump_var, 0.25 * annualized_volatility**2)
        step_drift = (annualized_drift - lam * kappa - 0.5 * diffusion_var) * dt

        log_returns = np.zeros(num_paths)
        for _ in range(horizon_days):
            jumps = rng.poisson(lam * dt, num_paths)
            log_returns += (
                step_drift
                + math.sqrt(diffusion_var * dt) * rng.normal(0, 1, num_paths)
                + rng.normal(cls.JUMP_MEAN, cls.JUMP_STD, num_paths) * jumps
            )
        return current_price * np.exp(log_returns)

    @classmethod
    def compute_forecast(
        cls,
        quote: MarketQuote,
        annualized_drift: Optional[float] = None,
        num_paths: int = NUM_PATHS,
        random_seed: Optional[int] = 42,
    ) -> MonteCarloResult:
        """Generates multi-horizon probabilistic forecasts for a given asset."""
        drift = annualized_drift if annualized_drift is not None else cls.estimate_drift(quote)
        vol = max(0.15, quote.annualized_volatility)

        horizons_dict: Dict[int, QuantileForecast] = {}

        for h in cls.HORIZONS:
            terminal_prices = cls.simulate_gbm_paths(
                current_price=quote.price,
                annualized_drift=drift,
                annualized_volatility=vol,
                horizon_days=h,
                num_paths=num_paths,
                random_seed=random_seed + h if random_seed is not None else None,
            )

            p10 = float(np.percentile(terminal_prices, 10))
            p50 = float(np.percentile(terminal_prices, 50))
            p90 = float(np.percentile(terminal_prices, 90))

            expected_return = float(((p50 - quote.price) / quote.price) * 100.0)
            prob_profit = float((np.sum(terminal_prices > quote.price) / num_paths) * 100.0)

            horizons_dict[h] = QuantileForecast(
                horizon_days=h,
                p10_bear=round(p10, 2),
                p50_median=round(p50, 2),
                p90_bull=round(p90, 2),
                expected_return_pct=round(expected_return, 2),
                probability_of_profit_pct=round(prob_profit, 1),
            )

        return MonteCarloResult(
            symbol=quote.symbol,
            current_price=quote.price,
            simulated_paths=num_paths,
            annualized_drift=round(drift, 4),
            annualized_volatility=round(vol, 4),
            horizons=horizons_dict,
        )
