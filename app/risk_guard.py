"""Deterministic Portfolio Risk Engine & Fractional Kelly Criterion Position Sizer.
Enforces hard mathematical risk invariants: Value-at-Risk (VaR 95%), CVaR, and allocation caps.
"""

import math
from typing import List
from app.schemas import MarketQuote, MonteCarloResult, OrderAction, RiskMetrics


class RiskGuardEngine:
    """Calculates risk parameters and enforces portfolio allocation invariants."""

    MAX_ALLOCATION_CAP_PCT = 15.0  # Hard ceiling: No single position may exceed 15% of portfolio
    MAX_UNMEDIATED_VAR_PCT = 8.0   # If 30-day VaR > 8%, requires Portfolio Manager sign-off
    KELLY_SAFETY_FRACTION = 0.33   # 1/3rd fractional Kelly to protect against model uncertainty
    SELL_BELOW_P_PROFIT = 45.0     # 30-day probability of profit (%) under which the quant call is SELL

    @classmethod
    def quant_call(cls, probability_of_profit_pct: float, recommended_allocation_pct: float) -> OrderAction:
        """The rule scored by backtest/run_backtest.py: BUY when Kelly sizes a position, SELL when the
        30-day probability of profit is below the threshold, otherwise HOLD."""
        if recommended_allocation_pct > 0:
            return OrderAction.BUY
        if probability_of_profit_pct < cls.SELL_BELOW_P_PROFIT:
            return OrderAction.SELL
        return OrderAction.HOLD

    @classmethod
    def evaluate_risk(cls, quote: MarketQuote, forecast: MonteCarloResult) -> RiskMetrics:
        """Evaluates portfolio risk and calculates safe Kelly position sizing."""
        reasons: List[str] = []
        requires_pm = False

        # 1. 30-day horizon forecast
        h30 = forecast.horizons.get(30)
        if not h30:
            h30 = list(forecast.horizons.values())[0]

        # 2. Value-at-Risk (VaR 95%) and CVaR
        # VaR 95% = loss percentage at 5th percentile
        # Using P10 as conservative proxy scaled to 95%
        downside_loss = max(0.0, quote.price - h30.p10_bear)
        var_95_pct = round((downside_loss / quote.price) * 100.0 * 1.2, 2)
        cvar_95_pct = round(var_95_pct * 1.35, 2)  # Expected shortfall beyond VaR

        # 3. Sharpe and Sortino Ratios (Annualized)
        risk_free_rate = 0.045  # 4.5% US Treasury risk-free rate
        excess_return = forecast.annualized_drift - risk_free_rate
        sharpe_ratio = round(excess_return / max(0.05, forecast.annualized_volatility), 2)

        downside_vol = forecast.annualized_volatility * 0.7  # Approximation of semi-deviation
        sortino_ratio = round(excess_return / max(0.05, downside_vol), 2)

        # 4. Maximum Drawdown Estimate
        max_drawdown_pct = round(min(55.0, forecast.annualized_volatility * 1.65 * 100.0), 1)

        # 5. Fractional Kelly Criterion Sizing
        p = h30.probability_of_profit_pct / 100.0
        q = 1.0 - p
        upside = max(0.01, h30.p90_bull - quote.price)
        downside = max(0.01, quote.price - h30.p10_bear)
        b = upside / downside  # Win-to-loss payoff ratio

        # Full Kelly formula: f* = (p*b - q) / b
        raw_kelly = ((p * b) - q) / b if b > 0 else 0.0
        raw_kelly_pct = round(max(0.0, raw_kelly * 100.0), 2)

        # Fractional Kelly with safety factor
        fractional_kelly = raw_kelly_pct * cls.KELLY_SAFETY_FRACTION

        # Apply hard institutional cap
        recommended_allocation = round(min(fractional_kelly, cls.MAX_ALLOCATION_CAP_PCT), 2)

        # 6. Enforce Deterministic Risk Invariants
        if var_95_pct >= cls.MAX_UNMEDIATED_VAR_PCT:
            requires_pm = True
            reasons.append(
                f"Value-at-Risk ({var_95_pct}%) exceeds threshold ({cls.MAX_UNMEDIATED_VAR_PCT}%). Requires PM authorization."
            )

        if fractional_kelly > cls.MAX_ALLOCATION_CAP_PCT:
            reasons.append(
                f"Raw Kelly sizing ({fractional_kelly:.1f}%) exceeds institutional cap ({cls.MAX_ALLOCATION_CAP_PCT}%). Capped at {cls.MAX_ALLOCATION_CAP_PCT}%."
            )

        if quote.beta >= 1.6:
            requires_pm = True
            reasons.append(
                f"High systematic market sensitivity (Beta: {quote.beta} >= 1.6). Portfolio vulnerable to macro drawdowns."
            )

        if sharpe_ratio < 0.5:
            requires_pm = True
            reasons.append(
                f"Sub-optimal risk-adjusted return profile (Sharpe Ratio: {sharpe_ratio} < 0.50)."
            )

        return RiskMetrics(
            symbol=quote.symbol,
            var_95_pct=var_95_pct,
            cvar_95_pct=cvar_95_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown_pct=max_drawdown_pct,
            raw_kelly_pct=raw_kelly_pct,
            recommended_allocation_pct=recommended_allocation,
            requires_pm_approval=requires_pm,
            risk_reasons=reasons,
        )
