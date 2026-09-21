"""Model Context Protocol (MCP) Server for Apex-Alpha Market Intelligence.
Exposes market quotes, SEC filings, Monte Carlo quantiles, and risk evaluation tools over protocol.
"""

import json
from typing import Any, Dict
from mcp.server.mcpserver import MCPServer

from app.debate import AdversarialDebateEngine
from app.quant_engine import QuantitativeForecaster
from app.risk_guard import RiskGuardEngine
from app.sec_rag import SECFilingRAG

mcp = MCPServer("apex-alpha-mcp")


# --- Resources ---

@mcp.resource("market://tickers/{symbol}")
def get_ticker_resource(symbol: str) -> str:
    """Returns quote and fundamental metrics for a ticker as JSON."""
    quote = SECFilingRAG.get_quote(symbol)
    financials = SECFilingRAG.get_financials(symbol)
    if not quote or not financials:
        return json.dumps({"error": f"Ticker '{symbol}' not found."})

    return json.dumps({
        "quote": quote.model_dump(),
        "financials": financials.model_dump(),
    }, indent=2)


@mcp.resource("market://sec/{symbol}")
def get_sec_filing_resource(symbol: str) -> str:
    """Returns official SEC 10-K/10-Q disclosures and excerpts for a ticker."""
    excerpts = SECFilingRAG.get_excerpts(symbol)
    financials = SECFilingRAG.get_financials(symbol)
    return json.dumps({
        "symbol": symbol.upper(),
        "filing_ref": financials.sec_filing_ref if financials else None,
        "excerpts": excerpts,
    }, indent=2)


# --- Tools ---

@mcp.tool()
def get_market_quote(symbol: str) -> Dict[str, Any]:
    """Retrieves real-time quote, valuation multiples, and volatility for a ticker."""
    quote = SECFilingRAG.get_quote(symbol)
    if not quote:
        return {"error": f"Symbol '{symbol}' not found."}
    return quote.model_dump()


@mcp.tool()
def run_monte_carlo_simulation(symbol: str) -> Dict[str, Any]:
    """Runs 10,000-path Monte Carlo simulation and returns P10, P50, and P90 quantiles."""
    quote = SECFilingRAG.get_quote(symbol)
    if not quote:
        return {"error": f"Symbol '{symbol}' not found."}
    forecast = QuantitativeForecaster.compute_forecast(quote=quote)
    return forecast.model_dump()


@mcp.tool()
def evaluate_risk_guardrails(symbol: str) -> Dict[str, Any]:
    """Calculates Value-at-Risk (VaR 95%), CVaR, Sharpe ratio, and Fractional Kelly sizing."""
    quote = SECFilingRAG.get_quote(symbol)
    if not quote:
        return {"error": f"Symbol '{symbol}' not found."}
    forecast = QuantitativeForecaster.compute_forecast(quote=quote)
    risk = RiskGuardEngine.evaluate_risk(quote=quote, forecast=forecast)
    return risk.model_dump()


@mcp.tool()
def run_adversarial_debate(symbol: str) -> Dict[str, Any]:
    """Executes Bull vs. Bear debate grounded in SEC disclosures and returns synthesis."""
    quote = SECFilingRAG.get_quote(symbol)
    financials = SECFilingRAG.get_financials(symbol)
    if not quote or not financials:
        return {"error": f"Symbol '{symbol}' not found."}
    bull = AdversarialDebateEngine.generate_bull_case(quote, financials)
    bear = AdversarialDebateEngine.generate_bear_case(quote, financials)
    forecast = QuantitativeForecaster.compute_forecast(quote=quote)
    synthesis = AdversarialDebateEngine.synthesize_committee_decision(quote, bull, bear, forecast)
    return {
        "bull_case": bull.model_dump(),
        "bear_case": bear.model_dump(),
        "synthesis": synthesis.model_dump(),
    }


if __name__ == "__main__":
    mcp.run()
