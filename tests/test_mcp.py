"""Unit tests for Model Context Protocol (MCP) Server in Apex-Alpha."""

import json
import pytest
from mcp_server.server import (
    evaluate_risk_guardrails,
    get_market_quote,
    get_sec_filing_resource,
    get_ticker_resource,
    run_adversarial_debate,
    run_monte_carlo_simulation,
)


def test_mcp_ticker_resource():
    """Verifies that market://tickers/{symbol} returns valid quote and financials."""
    res_json = get_ticker_resource("NVDA")
    data = json.loads(res_json)
    assert "quote" in data
    assert "financials" in data
    assert data["quote"]["symbol"] == "NVDA"


def test_mcp_sec_filing_resource():
    """Verifies that market://sec/{symbol} returns SEC filing excerpts."""
    res_json = get_sec_filing_resource("NVDA")
    data = json.loads(res_json)
    assert data["symbol"] == "NVDA"
    assert "excerpts" in data
    assert "data_center_growth" in data["excerpts"]


def test_mcp_tools_monte_carlo_and_risk():
    """Verifies MCP tools for Monte Carlo forecasting and risk evaluation."""
    mc_res = run_monte_carlo_simulation("AAPL")
    assert mc_res["symbol"] == "AAPL"
    assert "horizons" in mc_res

    risk_res = evaluate_risk_guardrails("AAPL")
    assert risk_res["symbol"] == "AAPL"
    assert "var_95_pct" in risk_res
    assert "recommended_allocation_pct" in risk_res


def test_mcp_tools_adversarial_debate():
    """Verifies MCP tool for adversarial Bull vs. Bear debate."""
    debate_res = run_adversarial_debate("TSLA")
    assert "bull_case" in debate_res
    assert "bear_case" in debate_res
    assert "synthesis" in debate_res
