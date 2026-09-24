"""LangGraph StateGraph for Apex-Alpha Market Intelligence & Quantitative Forecasting.
Orchestrates data ingestion, Monte Carlo simulation, adversarial debate, risk guards,
and the Human-in-the-Loop Portfolio Manager interrupt() review gate.
"""

from typing import Any, Dict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from app.debate import AdversarialDebateEngine
from app.quant_engine import QuantitativeForecaster
from app.risk_guard import RiskGuardEngine
from app.schemas import (
    MarketAnalysisState,
    MarketQuote,
    OrderAction,
    OrderProposal,
)
from app.sec_rag import SECFilingRAG


def ingest_market_data_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Ingests market quotes, SEC filings, and fundamental metrics."""
    symbol = state.get("symbol", "NVDA").upper()
    quote = SECFilingRAG.get_quote(symbol)
    financials = SECFilingRAG.get_financials(symbol)

    if not quote or not financials:
        raise ValueError(f"Ticker symbol '{symbol}' not found in SEC EDGAR market database.")

    state["symbol"] = symbol
    state["quote"] = quote
    state["financials"] = financials
    state["status"] = "market_data_ingested"
    return state


def run_quantitative_forecast_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Runs Monte Carlo simulations and calculates multi-horizon quantiles."""
    quote: MarketQuote = state["quote"]
    forecast = QuantitativeForecaster.compute_forecast(quote=quote)
    state["monte_carlo"] = forecast
    state["status"] = "forecast_computed"
    return state


def evaluate_risk_guardrails_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Calculates Value-at-Risk (VaR 95%), CVaR, and fractional Kelly Criterion sizing."""
    quote: MarketQuote = state["quote"]
    forecast = state["monte_carlo"]
    risk = RiskGuardEngine.evaluate_risk(quote=quote, forecast=forecast)
    state["risk"] = risk
    state["status"] = "risk_evaluated"
    return state


def adversarial_debate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Conducts adversarial Bull vs. Bear debate grounded in SEC disclosures."""
    quote: MarketQuote = state["quote"]
    financials = state["financials"]
    bull = AdversarialDebateEngine.generate_bull_case(quote, financials)
    bear = AdversarialDebateEngine.generate_bear_case(quote, financials)
    state["bull_case"] = bull
    state["bear_case"] = bear
    state["status"] = "debate_concluded"
    return state


def executive_synthesis_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Synthesizes debate and quantitative forecast into executive recommendation."""
    quote: MarketQuote = state["quote"]
    bull = state["bull_case"]
    bear = state["bear_case"]
    forecast = state["monte_carlo"]
    synthesis = AdversarialDebateEngine.synthesize_committee_decision(quote, bull, bear, forecast)
    state["synthesis"] = synthesis
    state["status"] = "synthesis_ready"
    return state


def stage_order_proposal_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Prepares structured trade proposal with entry, stop-loss, and take-profit levels."""
    quote: MarketQuote = state["quote"]
    risk = state["risk"]
    synthesis = state["synthesis"]
    h30 = state["monte_carlo"].horizons[30]

    # Calculate stop loss (below P10) and take profit (at P90)
    stop_loss = round(h30.p10_bear * 0.98, 2)
    take_profit = round(h30.p90_bull, 2)
    risk_reward = round((take_profit - quote.price) / max(0.01, quote.price - stop_loss), 2)

    total_portfolio_usd = 1000000.0  # $1,000,000 reference institutional portfolio
    target_capital = round(total_portfolio_usd * (risk.recommended_allocation_pct / 100.0), 2)

    proposal = OrderProposal(
        order_id=f"ORD-{quote.symbol}-001",
        symbol=quote.symbol,
        # A bullish committee cannot override the sizer: zero Kelly allocation means there is nothing to buy.
        action=OrderAction.BUY if "bullish" in synthesis.stance.value and risk.recommended_allocation_pct > 0 else OrderAction.HOLD,
        target_allocation_pct=risk.recommended_allocation_pct,
        estimated_capital_usd=target_capital,
        suggested_entry_price=quote.price,
        stop_loss_price=stop_loss,
        take_profit_price=take_profit,
        risk_reward_ratio=risk_reward,
        rationale=f"Quantitative Kelly sizing ({risk.recommended_allocation_pct}%) with R:R ratio of {risk_reward}:1.",
    )
    state["proposal"] = proposal
    return state


def portfolio_manager_gate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph interrupt() gate: Pauses execution for Portfolio Manager sign-off."""
    risk = state["risk"]

    if risk.requires_pm_approval:
        state["status"] = "awaiting_pm_approval"
        resume_data = interrupt({
            "symbol": state["symbol"],
            "proposal": state["proposal"].model_dump() if state.get("proposal") else None,
            "risk_metrics": risk.model_dump(),
            "reasons": risk.risk_reasons,
        })

        # Resumed via Command(resume=...)
        state["pm_approved"] = resume_data.get("approved", True)
        state["pm_adjusted_allocation"] = resume_data.get("adjusted_allocation_pct")
        state["pm_notes"] = resume_data.get("notes")
    else:
        state["pm_approved"] = True

    return state


def commit_portfolio_rebalance_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Executes the rebalance or logs human portfolio manager override."""
    is_approved = state.get("pm_approved", True)
    adjusted_pct = state.get("pm_adjusted_allocation")
    proposal: OrderProposal = state["proposal"]

    if is_approved:
        alloc = adjusted_pct if adjusted_pct is not None else proposal.target_allocation_pct
        state["status"] = "executed"
        state["execution_result"] = (
            f"ORDER COMMITTED: {proposal.action.value.upper()} {proposal.symbol} @ ${proposal.suggested_entry_price} "
            f"| Target Allocation: {alloc}% | Stop-Loss: ${proposal.stop_loss_price} | Take-Profit: ${proposal.take_profit_price}."
        )
    else:
        state["status"] = "rejected_by_pm"
        state["execution_result"] = f"ORDER CANCELLED by Portfolio Manager: {state.get('pm_notes', 'Allocation withheld.')}"

    return state


# --- Build LangGraph StateGraph ---

def build_apex_alpha_graph() -> Any:
    workflow = StateGraph(dict)

    workflow.add_node("ingest_market_data", ingest_market_data_node)
    workflow.add_node("run_quantitative_forecast", run_quantitative_forecast_node)
    workflow.add_node("evaluate_risk_guardrails", evaluate_risk_guardrails_node)
    workflow.add_node("adversarial_debate", adversarial_debate_node)
    workflow.add_node("executive_synthesis", executive_synthesis_node)
    workflow.add_node("stage_order_proposal", stage_order_proposal_node)
    workflow.add_node("portfolio_manager_gate", portfolio_manager_gate_node)
    workflow.add_node("commit_portfolio_rebalance", commit_portfolio_rebalance_node)

    workflow.set_entry_point("ingest_market_data")

    workflow.add_edge("ingest_market_data", "run_quantitative_forecast")
    workflow.add_edge("run_quantitative_forecast", "evaluate_risk_guardrails")
    workflow.add_edge("evaluate_risk_guardrails", "adversarial_debate")
    workflow.add_edge("adversarial_debate", "executive_synthesis")
    workflow.add_edge("executive_synthesis", "stage_order_proposal")
    workflow.add_edge("stage_order_proposal", "portfolio_manager_gate")
    workflow.add_edge("portfolio_manager_gate", "commit_portfolio_rebalance")
    workflow.add_edge("commit_portfolio_rebalance", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


apex_alpha_graph = build_apex_alpha_graph()
