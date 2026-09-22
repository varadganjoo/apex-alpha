# Apex-Alpha: Quantitative Market Intelligence & Forecasting Toolkit

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-indigo.svg)](https://github.com/langchain-ai/langgraph)
[![Methodology](https://img.shields.io/badge/Math-Methodology%20Spec-blue.svg)](docs/METHODOLOGY.md)
[![Tests](https://img.shields.io/badge/Tests-15%20Passed%20(100%25)-emerald.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-gray.svg)](LICENSE)

> **Apex-Alpha** is a quantitative research toolkit and market intelligence terminal. It integrates **Geometric Brownian Motion jump-diffusion modeling ($P_{10}, P_{50}, P_{90}$)**, **adversarial Bull vs. Bear multi-agent analysis grounded in SEC EDGAR disclosures**, and **mathematical risk guardrails (VaR 95%, CVaR, Fractional Kelly Criterion)** with LangGraph Human-in-the-Loop checkpoints for order staging.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph DataIngestion["1. Data & Filings Ingestion"]
        Quotes["Real-Time Market Quotes<br/>(OHLCV, Volatility, Beta)"]
        SEC["SEC EDGAR 10-K / 10-Q<br/>(Financials, Cash Flows, Disclosures)"]
    end

    subgraph QuantEngine["2. Quantitative Forecasting & Risk"]
        MC["Monte Carlo Simulator<br/>(10,000 Paths: P10, P50, P90 Quantiles)"]
        Risk["Deterministic Risk Guard<br/>(VaR 95%, CVaR, Kelly Criterion)"]
    end

    subgraph DebateEngine["3. Adversarial Multi-Agent Debate"]
        Bull["Bull Researcher<br/>(Growth Catalysts, Margin Expansion)"]
        Bear["Bear Risk Officer<br/>(Downside Scenarios, Multiple Compression)"]
        Synth["Executive Synthesizer<br/>(Probability-Weighted Expected Value)"]
    end

    subgraph HITL["4. LangGraph HITL Control Plane"]
        PMGate["portfolio_manager_gate<br/>(interrupt() Checkpoint for Sign-Off)"]
        Execution["Order Staging & Rebalance Node<br/>(Command(resume) Commit)"]
    end

    Quotes --> MC
    SEC --> Bull & Bear
    MC --> Risk
    Bull & Bear --> Synth
    Synth --> Risk
    Risk --> PMGate
    PMGate -.->|"interrupt() pauses"| PM["Human Portfolio Manager"]
    PM -->|"Approve / Adjust Allocation"| PMGate
    PMGate --> Execution
```

---

## 📸 Quantitative Terminal & Workstation

A Bloomberg-style quantitative trading workstation featuring real-time ticker tape, multi-horizon probability cones, grounded SEC filing citations, and portfolio risk sizing:

### 1. Market Screener & Benchmark Coverage
Displays institutional universe coverage (NVDA, AAPL, MSFT, TSLA) with live OHLCV price ticks, daily returns, market capitalizations, and real-time financial ratios.

![Market Screener](docs/images/01_market_screener_overview.png)

---

### 2. Monte Carlo Lab & Quantile Fan Chart
Simulates 10,000 price paths using Geometric Brownian Motion with jump diffusion, plotting multi-horizon probability cones ($P_{10}$ bear floor, $P_{50}$ median, $P_{90}$ bull ceiling) across 5d, 30d, and 90d horizons.

![Monte Carlo Lab](docs/images/02_monte_carlo_fanchart_lab.png)

---

### 3. SEC 10-Q RAG & Citation Engine
Extracts verbatim disclosures from SEC EDGAR filings (Item 1A Risk Factors, MD&A, segment revenues) with character-level grounding to ensure zero hallucination in financial analysis.

![SEC 10-Q RAG](docs/images/03_sec_10q_rag_evidence.png)

---

### 4. Adversarial Multi-Agent Debate (Bull vs. Bear)
Orchestrates dialectical debate between a Bull Researcher (catalysts, gross margin expansion) and a Bear Risk Officer (customer concentration, multiple compression), adjudicated by an Executive Synthesizer.

![Adversarial Debate](docs/images/04_adversarial_debate_bull_bear.png)

---

### 5. Portfolio Desk & Deterministic Risk Guardrails
Enforces strict mathematical risk limits: Value-at-Risk (VaR 95%), Expected Shortfall (CVaR), and Fractional Kelly Criterion position sizing capped at institutional limits (15.0%).

![Portfolio Desk & Risk Guard](docs/images/05_portfolio_risk_kelly_sizing.png)

---

## 🧪 Automated Test Suite (15 / 15 Passed)

Run the full automated verification suite:

```bash
python -m pytest tests/ -v
```

```
============================= test session starts =============================
tests/test_debate.py::test_sec_citation_verification PASSED              [  6%]
tests/test_debate.py::test_bear_case_identifies_downside_risks PASSED    [ 13%]
tests/test_debate.py::test_executive_synthesis_reaches_consensus PASSED  [ 20%]
tests/test_graph.py::test_graph_pauses_at_portfolio_manager_gate PASSED  [ 26%]
tests/test_graph.py::test_graph_resumes_with_pm_approval PASSED          [ 33%]
tests/test_mcp.py::test_mcp_ticker_resource PASSED                       [ 40%]
tests/test_mcp.py::test_mcp_sec_filing_resource PASSED                   [ 46%]
tests/test_mcp.py::test_mcp_tools_monte_carlo_and_risk PASSED            [ 53%]
tests/test_mcp.py::test_mcp_tools_adversarial_debate PASSED              [ 60%]
tests/test_quant.py::test_monte_carlo_quantiles_ordering PASSED          [ 66%]
tests/test_quant.py::test_probability_of_profit_bounds PASSED            [ 73%]
tests/test_quant.py::test_horizon_dispersion_increases_with_time PASSED  [ 80%]
tests/test_risk.py::test_kelly_allocation_respects_institutional_cap PASSED [ 86%]
tests/test_risk.py::test_high_beta_triggers_pm_review PASSED             [ 93%]
tests/test_risk.py::test_cvar_exceeds_var PASSED                         [100%]
============================= 15 passed in 1.79s ==============================
```

---

## 🛠️ Quickstart

### 1. Launch the Bloomberg Terminal
```bash
uvicorn app.main:app --port 8030 --reload
```
Open **[http://127.0.0.1:8030](http://127.0.0.1:8030)** in your browser.

### 2. Launch the MCP Server
```bash
python -m mcp_server.server
```

---

## Engineering Attribution & AI Pair-Programming

This repository was developed with Gemini and Claude as AI pair-programming assistants. I designed the architecture, the quantitative jump-diffusion model, the Kelly criterion risk bounds, and the verification test suites, and reviewed all code.


