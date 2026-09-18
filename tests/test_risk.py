"""Unit tests for Deterministic Risk Guardrail Engine & Kelly Sizing."""

import pytest
from app.quant_engine import QuantitativeForecaster
from app.risk_guard import RiskGuardEngine
from app.schemas import MarketQuote


@pytest.fixture
def high_vol_quote():
    return MarketQuote(
        symbol="TSLA",
        company_name="Tesla, Inc.",
        price=245.0,
        change=5.0,
        change_pct=2.0,
        volume=60000000,
        pe_ratio=65.0,
        forward_pe=50.0,
        beta=2.15,  # High beta > 1.6
        market_cap_b=780.0,
        week_52_high=270.0,
        week_52_low=140.0,
        annualized_volatility=0.55,
    )


@pytest.fixture
def low_vol_quote():
    return MarketQuote(
        symbol="AAPL",
        company_name="Apple Inc.",
        price=225.0,
        change=0.5,
        change_pct=0.2,
        volume=40000000,
        pe_ratio=32.0,
        forward_pe=28.0,
        beta=1.05,
        market_cap_b=3400.0,
        week_52_high=235.0,
        week_52_low=165.0,
        annualized_volatility=0.22,
    )


def test_kelly_allocation_respects_institutional_cap(high_vol_quote):
    """Safety Invariant: Recommended allocation must never exceed 15.0% institutional ceiling."""
    forecast = QuantitativeForecaster.compute_forecast(quote=high_vol_quote, num_paths=1000)
    risk = RiskGuardEngine.evaluate_risk(quote=high_vol_quote, forecast=forecast)

    assert risk.recommended_allocation_pct <= 15.0


def test_high_beta_triggers_pm_review(high_vol_quote):
    """Safety Invariant: Beta >= 1.6 triggers mandatory Portfolio Manager authorization."""
    forecast = QuantitativeForecaster.compute_forecast(quote=high_vol_quote, num_paths=1000)
    risk = RiskGuardEngine.evaluate_risk(quote=high_vol_quote, forecast=forecast)

    assert risk.requires_pm_approval is True
    assert any("Beta:" in r for r in risk.risk_reasons)


def test_cvar_exceeds_var(low_vol_quote):
    """Mathematical Invariant: Conditional VaR (Expected Shortfall) must be strictly >= VaR."""
    forecast = QuantitativeForecaster.compute_forecast(quote=low_vol_quote, num_paths=1000)
    risk = RiskGuardEngine.evaluate_risk(quote=low_vol_quote, forecast=forecast)

    assert risk.cvar_95_pct >= risk.var_95_pct
