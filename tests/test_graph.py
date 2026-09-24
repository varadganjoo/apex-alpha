"""Unit tests for LangGraph StateGraph (Forecasting, Risk, interrupt, and resume)."""

import pytest
from langgraph.types import Command
from app.graph import build_apex_alpha_graph


@pytest.fixture
def graph():
    return build_apex_alpha_graph()


def test_graph_pauses_at_portfolio_manager_gate(graph):
    """Verifies that high-risk assets (e.g. TSLA Beta > 1.6) pause at the interrupt() gate."""
    config = {"configurable": {"thread_id": "test-thread-pm-gate"}}
    initial_state = {"symbol": "TSLA"}

    # Run graph; should pause at interrupt()
    graph.invoke(initial_state, config=config)

    state_snap = graph.get_state(config)
    assert state_snap.next == ("portfolio_manager_gate",)
    assert len(state_snap.tasks) > 0
    assert len(state_snap.tasks[0].interrupts) > 0

    interrupt_data = state_snap.tasks[0].interrupts[0].value
    assert interrupt_data["symbol"] == "TSLA"
    assert "risk_metrics" in interrupt_data


def test_graph_resumes_with_pm_approval(graph):
    """Verifies that resuming with PM approval commits the order execution."""
    config = {"configurable": {"thread_id": "test-thread-pm-resume"}}
    initial_state = {"symbol": "NVDA"}

    # First run: pause at gate
    graph.invoke(initial_state, config=config)

    # Resume with approval and adjusted allocation
    resume_payload = {
        "approved": True,
        "adjusted_allocation_pct": 12.5,
        "notes": "Portfolio Manager signed off on 12.5% target allocation.",
    }

    result = graph.invoke(Command(resume=resume_payload), config=config)
    assert result.get("status") == "executed"
    assert "ORDER COMMITTED" in result.get("execution_result", "")
    assert "12.5%" in result.get("execution_result", "")


@pytest.mark.parametrize(
    "p_profit, allocation, expected",
    [(55.0, 6.5, "buy"), (48.0, 0.0, "hold"), (40.0, 0.0, "sell"), (40.0, 2.0, "buy")],
)
def test_quant_call_rule(p_profit, allocation, expected):
    from app.risk_guard import RiskGuardEngine

    assert RiskGuardEngine.quant_call(p_profit, allocation).value == expected


@pytest.mark.parametrize(
    "allocation, stance, expected_action",
    [(6.5, "bullish", "buy"), (6.5, "neutral", "buy"), (6.5, "bearish", "hold"), (0.0, "strong_bullish", "hold")],
)
def test_committee_can_veto_a_buy_but_not_create_one(allocation, stance, expected_action):
    from app.graph import stage_order_proposal_node
    from app.quant_engine import QuantitativeForecaster
    from app.risk_guard import RiskGuardEngine
    from app.schemas import ExecutiveSynthesis, MarketStance
    from app.sec_rag import SECFilingRAG

    quote = SECFilingRAG.get_quote("NVDA")
    forecast = QuantitativeForecaster.compute_forecast(quote=quote)
    h30 = forecast.horizons[30].model_copy(update={"probability_of_profit_pct": 50.0})
    forecast = forecast.model_copy(update={"horizons": {**forecast.horizons, 30: h30}})
    risk = RiskGuardEngine.evaluate_risk(quote=quote, forecast=forecast).model_copy(
        update={"recommended_allocation_pct": allocation}
    )
    synthesis = ExecutiveSynthesis(
        symbol="NVDA", stance=MarketStance(stance), confidence_score=0.8, expected_annualized_return_pct=7.0,
        key_catalysts=[], key_downside_risks=[], synthesis_narrative="",
    )
    state = stage_order_proposal_node({"quote": quote, "risk": risk, "synthesis": synthesis, "monte_carlo": forecast})
    assert state["proposal"].action.value == expected_action
