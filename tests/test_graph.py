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
    paused = graph.invoke(initial_state, config=config)
    allocation = round(paused["proposal"].target_allocation_pct / 2, 2)

    # Resume with approval and adjusted allocation
    resume_payload = {
        "approved": True,
        "adjusted_allocation_pct": allocation,
        "notes": "Portfolio Manager signed off on half size.",
    }

    result = graph.invoke(Command(resume=resume_payload), config=config)
    assert result.get("status") == "executed"
    assert "ORDER COMMITTED" in result.get("execution_result", "")
    assert f"{allocation}%" in result.get("execution_result", "")


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
    if expected_action == "hold":
        assert state["proposal"].target_allocation_pct == 0.0
        assert state["proposal"].estimated_capital_usd == 0.0


def test_low_risk_proposal_still_requires_a_human_decision(graph, monkeypatch):
    from app.risk_guard import RiskGuardEngine
    original = RiskGuardEngine.evaluate_risk
    monkeypatch.setattr(RiskGuardEngine, "evaluate_risk", lambda **kw: original(**kw).model_copy(
        update={"requires_pm_approval": False, "risk_reasons": []}
    ))
    result = graph.invoke({"symbol": "AAPL"}, config={"configurable": {"thread_id": "low-risk-gate"}})
    assert result.get("__interrupt__")
    assert result.get("execution_result") is None


@pytest.mark.parametrize("payload", [{"approved": "yes"}, {"approved": True, "adjusted_allocation_pct": 15.0},
                                     {"approved": True, "adjusted_allocation_pct": float("nan")}])
def test_direct_graph_resume_cannot_bypass_decision_validation(graph, payload):
    config = {"configurable": {"thread_id": "invalid-decision"}}
    graph.invoke({"symbol": "NVDA"}, config=config)
    with pytest.raises(ValueError):
        graph.invoke(Command(resume=payload), config=config)


def test_commit_without_explicit_approval_cancels():
    from app.graph import commit_portfolio_rebalance_node
    assert commit_portfolio_rebalance_node({})["status"] == "rejected_by_pm"


def test_empty_graph_resume_keeps_order_uncommitted(graph):
    config = {"configurable": {"thread_id": "empty-resume"}}
    graph.invoke({"symbol": "NVDA"}, config=config)
    result = graph.invoke(Command(resume={}), config=config)
    assert result.get("execution_result") is None
    assert result.get("__interrupt__")
