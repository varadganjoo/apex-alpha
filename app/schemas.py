"""Data models and Pydantic schemas for Apex-Alpha Quantitative Market Intelligence Platform.
Defines market quotes, Monte Carlo quantile forecasts, SEC filings, risk metrics, and debate structures.
"""

from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MarketStance(str, Enum):
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


class OrderAction(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    REBALANCE = "rebalance"


class MarketQuote(BaseModel):
    symbol: str
    company_name: str
    price: float
    change: float
    change_pct: float
    volume: int
    pe_ratio: float
    forward_pe: float
    beta: float
    market_cap_b: float
    week_52_high: float
    week_52_low: float
    annualized_volatility: float = 0.32
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class FinancialMetrics(BaseModel):
    symbol: str
    fiscal_quarter: str
    revenue_b: float
    revenue_growth_yoy_pct: float
    gross_margin_pct: float
    operating_margin_pct: float
    free_cash_flow_b: float
    debt_to_equity: float
    sec_filing_ref: str  # e.g. "SEC Form 10-Q Q3 2026, Item 1"


class QuantileForecast(BaseModel):
    horizon_days: int
    p10_bear: float
    p50_median: float
    p90_bull: float
    expected_return_pct: float
    probability_of_profit_pct: float


class MonteCarloResult(BaseModel):
    symbol: str
    current_price: float
    simulated_paths: int = 10000
    annualized_drift: float
    annualized_volatility: float
    horizons: Dict[int, QuantileForecast]
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RiskMetrics(BaseModel):
    symbol: str
    var_95_pct: float  # Value-at-Risk at 95% confidence over 30 days
    cvar_95_pct: float  # Conditional VaR (Expected Shortfall)
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    raw_kelly_pct: float
    recommended_allocation_pct: float
    requires_pm_approval: bool
    risk_reasons: List[str] = Field(default_factory=list)


class Citation(BaseModel):
    source: str
    quote: str

    def __getitem__(self, item: str):
        return getattr(self, item)


class DebateArgument(BaseModel):
    agent_type: str  # "bull" or "bear"
    claim: str
    thesis: str
    evidence_points: List[str]
    citations: List[Citation] = Field(default_factory=list)
    confidence_score: float = 0.85


class ExecutiveSynthesis(BaseModel):
    symbol: str
    stance: MarketStance
    confidence_score: float
    expected_annualized_return_pct: float
    key_catalysts: List[str]
    key_downside_risks: List[str]
    synthesis_narrative: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class OrderProposal(BaseModel):
    order_id: str
    symbol: str
    action: OrderAction
    target_allocation_pct: float
    estimated_capital_usd: float
    suggested_entry_price: float
    stop_loss_price: float
    take_profit_price: float
    risk_reward_ratio: float
    rationale: str


class MarketAnalysisState(BaseModel):
    symbol: str
    status: str = "ingesting"
    quote: Optional[MarketQuote] = None
    financials: Optional[FinancialMetrics] = None
    monte_carlo: Optional[MonteCarloResult] = None
    risk: Optional[RiskMetrics] = None
    bull_case: Optional[DebateArgument] = None
    bear_case: Optional[DebateArgument] = None
    synthesis: Optional[ExecutiveSynthesis] = None
    proposal: Optional[OrderProposal] = None
    pm_approved: Optional[bool] = None
    pm_adjusted_allocation: Optional[float] = None
    execution_result: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
