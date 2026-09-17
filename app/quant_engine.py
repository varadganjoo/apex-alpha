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
        """Simulates price trajectories using Geometric Brownian Motion with Merton Jump Diffusion."""
        if random_seed is not None:
            np.random.seed(random_seed)

        dt = 1.0 / 252.0  # Daily time step (252 trading days per year)
        num_steps = horizon_days

        # Jump diffusion parameters (rare earnings/macro jump shocks)
        lambda_jumps = 0.05  # Average 0.05 jumps per day
        jump_mean = -0.01  # Slight negative asymmetry
        jump_std = 0.04

        # Pre-allocate path array
        prices = np.zeros((num_paths, num_steps + 1))
        prices[:, 0] = current_price

        # Standard normal random variates
        drift = (annualized_drift - 0.5 * (annualized_volatility ** 2)) * dt
        vol_step = annualized_volatility * math.sqrt(dt)

        for step in range(1, num_steps + 1):
            z = np.random.normal(0, 1, num_paths)
            # Poisson jump arrivals
            jumps_occurred = np.random.poisson(lambda_jumps, num_paths)
            jump_magnitudes = np.random.normal(jump_mean, jump_std, num_paths) * jumps_occurred

            # Log return update
            log_returns = drift + vol_step * z + jump_magnitudes
            prices[:, step] = prices[:, step - 1] * np.exp(log_returns)

        return prices[:, -1]  # Return terminal prices

    @classmethod
    def compute_forecast(
        cls,
        quote: MarketQuote,
        annualized_drift: Optional[float] = None,
        num_paths: int = NUM_PATHS,
        random_seed: Optional[int] = 42,
    ) -> MonteCarloResult:
        """Generates multi-horizon probabilistic forecasts for a given asset."""
        # Baseline drift estimated from forward P/E, revenue growth, or historical return
        drift = annualized_drift if annualized_drift is not None else max(-0.15, min(0.35, 0.12 * (1.0 / max(0.5, quote.beta))))
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
