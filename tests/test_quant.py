"""Unit tests for Quantitative Forecasting Engine & Monte Carlo simulation."""

import pytest
import numpy as np
from app.quant_engine import QuantitativeForecaster
from app.schemas import MarketQuote


@pytest.fixture
def sample_quote():
    return MarketQuote(
        symbol="NVDA",
        company_name="NVIDIA Corporation",
        price=128.40,
        change=4.20,
        change_pct=3.38,
        volume=50000000,
        pe_ratio=45.0,
        forward_pe=30.0,
        beta=1.70,
        market_cap_b=3150.0,
        week_52_high=140.0,
        week_52_low=45.0,
        annualized_volatility=0.45,
    )


def test_monte_carlo_quantiles_ordering(sample_quote):
    """Mathematical Invariant: P10 (bear floor) < P50 (median) < P90 (bull ceiling)."""
    forecast = QuantitativeForecaster.compute_forecast(
        quote=sample_quote,
        num_paths=1000,
        random_seed=42,
    )

    for h, q in forecast.horizons.items():
        assert q.p10_bear < q.p50_median < q.p90_bull, (
            f"Quantiles violated for horizon {h}d: P10={q.p10_bear}, P50={q.p50_median}, P90={q.p90_bull}"
        )


def test_probability_of_profit_bounds(sample_quote):
    """Probability of profit must be bounded between 0% and 100%."""
    forecast = QuantitativeForecaster.compute_forecast(
        quote=sample_quote,
        num_paths=500,
        random_seed=42,
    )

    for h, q in forecast.horizons.items():
        assert 0.0 <= q.probability_of_profit_pct <= 100.0


def test_horizon_dispersion_increases_with_time(sample_quote):
    """Statistical Invariant: Price dispersion (P90 - P10) expands as horizon increases (sqrt(t) law)."""
    forecast = QuantitativeForecaster.compute_forecast(
        quote=sample_quote,
        num_paths=2000,
        random_seed=42,
    )

    h5_spread = forecast.horizons[5].p90_bull - forecast.horizons[5].p10_bear
    h90_spread = forecast.horizons[90].p90_bull - forecast.horizons[90].p10_bear

    assert h90_spread > h5_spread, "90-day price distribution must have wider dispersion than 5-day."


def test_tail_losses_and_mean_come_from_the_simulated_distribution(sample_quote, monkeypatch):
    # 20 equally weighted paths: the worst 5% is the single $50 outcome.
    prices = np.array([50.0, 80.0] + [100.0] * 17 + [200.0])
    monkeypatch.setattr(QuantitativeForecaster, "simulate_gbm_paths", lambda **kwargs: prices)
    quote = sample_quote.model_copy(update={"price": 100.0})
    h = QuantitativeForecaster.compute_forecast(quote, num_paths=20).horizons[30]
    assert h.var_95_pct == 21.5  # interpolated P5 price = $78.50
    assert h.cvar_95_pct == 50.0
    assert h.expected_return_pct == 1.5  # mean $101.50, median $100
    assert h.median_return_pct == 0.0


def test_profitable_paths_do_not_report_negative_loss(sample_quote, monkeypatch):
    monkeypatch.setattr(QuantitativeForecaster, "simulate_gbm_paths", lambda **kwargs: np.full(20, 200.0))
    h = QuantitativeForecaster.compute_forecast(sample_quote, num_paths=20).horizons[30]
    assert h.var_95_pct == h.cvar_95_pct == 0.0


@pytest.mark.parametrize("paths", [0, -1, 1.5])
def test_invalid_path_counts_are_rejected(sample_quote, paths):
    with pytest.raises(ValueError, match="positive integer"):
        QuantitativeForecaster.compute_forecast(sample_quote, num_paths=paths)
