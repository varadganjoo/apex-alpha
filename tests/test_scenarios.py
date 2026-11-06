"""Research comparisons must isolate assumptions and never call an LLM."""

import pytest
from app.quant_engine import QuantitativeForecaster
from app.risk_guard import RiskGuardEngine
from app.sec_rag import SECFilingRAG


def test_forecast_includes_reproducible_assumption_comparisons():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    data = client.get("/api/tickers/NVDA/forecast").json()
    rows = {row["id"]: row for row in data["scenarios"]}
    assert set(rows) == {"base", "capm_only", "volatility_shock"}
    assert rows["base"]["var_95_pct"] == data["risk"]["var_95_pct"]
    assert rows["base"]["allocation_pct"] == data["risk"]["recommended_allocation_pct"]
    # Offline rate 4.5%, NVDA sample beta 1.72: CAPM = .045 + 1.72 * .055.
    assert rows["capm_only"]["annualized_drift"] == pytest.approx(.1396)
    assert rows["volatility_shock"]["annualized_volatility"] == pytest.approx(data["monte_carlo"]["annualized_volatility"] * 1.5)
    assert rows["volatility_shock"]["var_95_pct"] > rows["base"]["var_95_pct"]
    assert client.get("/api/tickers/NVDA/forecast").json()["scenarios"] == data["scenarios"]
    assert all(0 <= row["allocation_pct"] <= 15 for row in rows.values())


def test_risk_uses_forecast_tail_losses():
    quote = SECFilingRAG.get_quote("NVDA")
    forecast = QuantitativeForecaster.compute_forecast(quote)
    h30 = forecast.horizons[30].model_copy(update={"var_95_pct": 22.0, "cvar_95_pct": 35.0})
    forecast = forecast.model_copy(update={"horizons": {30: h30}})
    risk = RiskGuardEngine.evaluate_risk(quote, forecast)
    assert risk.var_95_pct == 22.0
    assert risk.cvar_95_pct == 35.0


def test_scenarios_reuse_the_base_risk_free_snapshot(monkeypatch):
    from app.scenarios import compare_scenarios
    quote = SECFilingRAG.get_quote("NVDA")
    base = QuantitativeForecaster.compute_forecast(quote, risk_free=(.043276, "test snapshot"))
    def fail():
        raise AssertionError("Scenarios must not refetch the risk-free rate")
    monkeypatch.setattr(QuantitativeForecaster, "risk_free_rate", fail)
    rows = compare_scenarios(quote, base)
    assert rows[1]["annualized_drift"] == pytest.approx(.137876)


def test_custom_seed_comparison_keeps_common_random_draws():
    from app.scenarios import compare_scenarios
    quote = SECFilingRAG.get_quote("NVDA")
    # Set the base itself to CAPM: removing its tilt should now change nothing.
    base = QuantitativeForecaster.compute_forecast(quote, annualized_drift=.1396, num_paths=137, random_seed=7)
    rows = compare_scenarios(quote, base)
    assert rows[0]["var_95_pct"] == rows[1]["var_95_pct"]
    assert rows[0]["allocation_pct"] == rows[1]["allocation_pct"]
