"""Fixed assumption checks, not new trading signals or historical stress replays."""

from app.quant_engine import QuantitativeForecaster
from app.risk_guard import RiskGuardEngine
from app.schemas import MarketQuote, MonteCarloResult


def compare_scenarios(quote: MarketQuote, base: MonteCarloResult) -> list[dict]:
    capm = base.risk_free_rate + quote.beta * QuantitativeForecaster.EQUITY_RISK_PREMIUM
    rows = []
    for scenario_id, label, drift, vol in (
        ("base", "Base model", base.annualized_drift, base.annualized_volatility),
        ("capm_only", "CAPM only (no reversal tilt)", capm, base.annualized_volatility),
        ("volatility_shock", "Volatility shock (1.5x)", base.annualized_drift, base.annualized_volatility * 1.5),
    ):
        forecast = base if scenario_id == "base" else QuantitativeForecaster.compute_forecast(
            quote.model_copy(update={"annualized_volatility": vol}), annualized_drift=drift,
            num_paths=base.simulated_paths,
            random_seed=base.random_seed,
            risk_free=(base.risk_free_rate, base.risk_free_source),
        )
        risk = RiskGuardEngine.evaluate_risk(quote, forecast)
        h30 = forecast.horizons[30]
        rows.append({
            "id": scenario_id, "label": label,
            "annualized_drift": forecast.annualized_drift,
            "annualized_volatility": forecast.annualized_volatility,
            "median_return_pct": h30.median_return_pct,
            "probability_of_profit_pct": h30.probability_of_profit_pct,
            "var_95_pct": risk.var_95_pct, "cvar_95_pct": risk.cvar_95_pct,
            "allocation_pct": risk.recommended_allocation_pct,
            "quant_call": RiskGuardEngine.quant_call(h30.probability_of_profit_pct, risk.recommended_allocation_pct).value,
        })
    return rows
