# Apex-Alpha: Quantitative Market Intelligence & Forecasting Toolkit

[![Apex-Alpha CI](https://github.com/varadganjoo/apex-alpha/actions/workflows/ci.yml/badge.svg)](https://github.com/varadganjoo/apex-alpha/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-indigo.svg)](https://github.com/langchain-ai/langgraph)
[![Methodology](https://img.shields.io/badge/Math-Methodology%20Spec-blue.svg)](docs/METHODOLOGY.md)
[![License](https://img.shields.io/badge/License-MIT-gray.svg)](LICENSE)

> **Apex-Alpha** is a quantitative research terminal. A **Merton jump-diffusion Monte Carlo** engine produces 5, 30 and 90-day price quantiles; a **deterministic risk engine** sizes positions with fractional Kelly and VaR limits; an **adversarial Bull vs. Bear LLM committee** argues the case from filing excerpts; and a **LangGraph `interrupt()`** holds every staged order for a portfolio manager's sign-off.

**Live demo:** https://apex-alpha-tan.vercel.app (NVDA, AAPL, TSLA, MSFT with live prices and SEC filings; the committee runs on free-tier models, so runs per day are limited)

## Data sources

| Data | Source | Cached | If unavailable |
| :--- | :--- | :--- | :--- |
| Prices, beta, volatility, 52-week range | Yahoo Finance via `yfinance` (about 15 months of daily bars; beta and volatility computed exactly as in the backtest) | 15 min | Tiingo if `TIINGO_API_KEY` is set, then the illustrative sample |
| Market cap, trailing and forward P/E | Yahoo Finance; market cap and P/E fall back to SEC shares outstanding and trailing net income | 24 h | shown as n/a |
| Revenue, margins, free cash flow, debt/equity | SEC EDGAR XBRL company facts (latest quarter, derived from year-to-date filings where needed) | 6 h | illustrative sample |
| Filing excerpts the committee cites | SEC EDGAR, latest 10-Q or 10-K (MD&A and risk factors) | 6 h | illustrative sample |
| Risk-free rate (expected returns, Sharpe) | U.S. Treasury daily par yield curve, 3-month | 12 h | 4.5% assumption |

Read endpoints also carry CDN cache headers, so serverless instances rarely refetch. Every response and the UI say which source each number came from.

---

## Does it predict? The backtest

The quant BUY / HOLD / SELL rule is scored walk-forward on real prices: 32 large-cap US stocks, 2011 to 2026, one non-overlapping call every 30 trading days, using only data available at the time. Parameters were chosen on 2011-2018; 2019 onward was held out. Full tables and caveats are in [backtest/RESULTS.md](backtest/RESULTS.md).

| Held out, 2019-2026 | Rank IC | BUY hit rate (base rate 60.5%) | BUY-only CAGR (equal-weight 26.1%) |
| :--- | ---: | ---: | ---: |
| Original model (`0.12 / beta` drift, uncompensated jumps) | -0.117 | 59.0% | 3.0% |
| Current model (CAPM + 52-week-high reversal tilt) | **+0.071** | 60.5% | 22.0% |

What this says, plainly:

* The original simulator added about -12.6% a year of hidden drift. Its calls were **anti-predictive**: the stocks it avoided outperformed the ones it bought.
* The current model **ranks** stocks slightly better than chance on data it never saw (rank IC +0.07, in line with its in-sample +0.078).
* Its **BUY / SELL calls do not beat the base rate**. With realistic expected returns almost every stock clears the Kelly threshold, so the call is BUY about 99.6% of the time. Treat the call as position-sizing guidance, not a prediction.

The LLM committee is not part of the backtest: it cannot be replayed without look-ahead, because the model has already seen what happened next.

---

## System Architecture

```mermaid
flowchart TD
    subgraph DataIngestion["1. Data & Filings"]
        Quotes["Live Quotes<br/>(Yahoo Finance: price, beta, volatility, 52-week range)"]
        SEC["SEC EDGAR<br/>(XBRL fundamentals, 10-Q / 10-K excerpts)"]
    end

    subgraph QuantEngine["2. Quantitative Forecasting & Risk"]
        Drift["Expected Return<br/>(CAPM + 52-week-high tilt)"]
        MC["Monte Carlo Jump Diffusion<br/>(10,000 paths: P10 / P50 / P90)"]
        Risk["Deterministic Risk Guard<br/>(VaR, CVaR, fractional Kelly, 15% cap)"]
    end

    subgraph DebateEngine["3. Adversarial LLM Committee (Gemini 3.6 / 3.7 / 3.8, then Groq)"]
        Bull["Bull Analyst"]
        Bear["Bear Risk Officer"]
        Synth["CIO Synthesis"]
    end

    subgraph HITL["4. LangGraph HITL"]
        Stage["Stage Order<br/>(quant BUY / HOLD / SELL, committee can veto a BUY)"]
        PMGate["portfolio_manager_gate<br/>(interrupt())"]
        Execution["Commit or Cancel<br/>(Command(resume))"]
    end

    Quotes --> Drift --> MC --> Risk
    SEC --> Bull & Bear
    Bull & Bear --> Synth
    Risk & Synth --> Stage --> PMGate
    PMGate -.->|"pauses for"| PM["Portfolio Manager"]
    PM -->|"Approve / resize / reject"| PMGate
    PMGate --> Execution
```

The UI streams each graph node as it finishes (`POST /api/tickers/{symbol}/analyze/stream`), shows whether every citation was found verbatim in the filing excerpts, and resumes the paused graph through `POST /api/tickers/{symbol}/resume`. A portfolio manager can shrink a position but never exceed the 15% cap.

---

## Screenshots

Captured from the live demo.

![Screener with live prices](docs/images/01_screener.png)

![Monte Carlo forecast](docs/images/02_monte_carlo.png)

![Excerpts from the latest SEC filing](docs/images/03_filings.png)

![Committee run: every citation checked against the filing](docs/images/04_committee.png)

![Order staged for portfolio manager sign-off](docs/images/05_risk_desk.png)

---

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env          # add GEMINI_API_KEY, GROQ_API_KEY, SEC_USER_AGENT
uvicorn app.main:app --port 8030 --reload
```

Open http://127.0.0.1:8030.

| Variable | Purpose |
| :--- | :--- |
| `GEMINI_API_KEY` | Committee models. `GEMINI_MODELS` sets the order (default `gemini-3.6-flash,gemini-3.7-flash,gemini-3.8-flash`); each has its own free-tier quota. |
| `GROQ_API_KEY` | Backup when every Gemini model is out of quota or overloaded (`GROQ_MODEL`, default `openai/gpt-oss-120b`). |
| `SEC_USER_AGENT` | Required by SEC EDGAR: a name and contact email, e.g. `Apex-Alpha demo you@example.com`. No API key needed. |
| `TIINGO_API_KEY` | Optional second price source (free key at tiingo.com). |
| `APEX_DATA=sample` | Use the offline sample dataset only (the test suite does this). |

MCP server:

```bash
python -m mcp_server.server
```

Reproduce the backtest (downloads prices from Yahoo Finance, about 7 minutes):

```bash
pip install -r backtest/requirements.txt
python -m backtest.run_backtest && python -m backtest.report
```

---

## Tests

```bash
python -m pytest tests/ -v
```

Covers Monte Carlo quantiles, the risk engine and order rule, the LangGraph interrupt/resume lifecycle, the HTTP API (streaming, sign-off validation, error handling, cache headers), live-data parsing (SEC quarter derivation, filing excerpts, fallbacks), the MCP server, and the Gemini and Groq fallback chain. Tests run offline against the sample dataset; CI runs them on every push.

---

## Engineering Attribution & AI Pair-Programming

This repository was developed with Gemini and Claude as AI pair-programming assistants. I designed the architecture, the quantitative jump-diffusion model, the Kelly criterion risk bounds, and the verification test suites, and reviewed all code.

## License

MIT, see [LICENSE](LICENSE).
