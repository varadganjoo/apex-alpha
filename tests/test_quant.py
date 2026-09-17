"""Unit tests for Quantitative Forecasting Engine & Monte Carlo simulation."""

import pytest
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
