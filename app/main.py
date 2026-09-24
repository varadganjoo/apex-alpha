"""FastAPI Application for Apex-Alpha Quantitative Market Intelligence Platform.
Serves REST APIs for quantitative forecasting, adversarial debate, and the dark-mode research terminal UI.
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, Optional
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from langgraph.types import Command

from app.debate import AdversarialDebateEngine
from app.graph import apex_alpha_graph
from app.llm import describe_llm_error
from app.quant_engine import QuantitativeForecaster
from app.risk_guard import RiskGuardEngine
from app.sec_rag import SEC_DATABASE, SECFilingRAG

logger = logging.getLogger("apex_alpha.api")

app = FastAPI(
    title="Apex-Alpha Quantitative Market Intelligence Platform",
    description="Multi-Horizon Stock Price Forecasting, SEC 10-K RAG, and Adversarial Bull vs. Bear Debate.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

SSE_HEADERS = {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}

# State keys each graph node produces; streamed to the UI as the run progresses.
NODE_OUTPUT_KEYS = {
    "ingest_market_data": ["quote", "financials"],
    "run_quantitative_forecast": ["monte_carlo"],
    "evaluate_risk_guardrails": ["risk"],
    "adversarial_debate": ["bull_case", "bear_case"],
    "executive_synthesis": ["synthesis"],
    "stage_order_proposal": ["proposal"],
    "commit_portfolio_rebalance": ["status", "execution_result"],
}


class ResumeAllocationRequest(BaseModel):
    thread_id: str = Field(..., max_length=64, description="thread_id returned by the analyze stream's interrupt event")
    approved: bool = True
    # A PM override may shrink the position but never exceed the hard institutional cap.
    adjusted_allocation_pct: Optional[float] = Field(None, ge=0.0, le=RiskGuardEngine.MAX_ALLOCATION_CAP_PCT)
    notes: Optional[str] = Field(None, max_length=500)


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(jsonable_encoder(data))}\n\n"


def _sse_guard(events: Iterator[str]) -> Iterator[str]:
    """Turns a mid-stream failure into an `error` event instead of a silently truncated response."""
    try:
        yield from events
    except Exception as exc:
        logger.exception("Streaming run failed")
        status, message, retryable = describe_llm_error(exc)
        yield _sse("error", {"message": message, "retryable": retryable})


def _require_ticker(symbol: str) -> str:
    sym = symbol.upper()
    if not SECFilingRAG.get_quote(sym):
        raise HTTPException(status_code=404, detail=f"Ticker '{sym}' not found.")
    return sym


def _with_verified_citations(symbol: str, argument: Dict[str, Any]) -> Dict[str, Any]:
    for citation in argument.get("citations", []):
        citation["verified"] = SECFilingRAG.verify_citation(symbol, citation.get("quote", ""))
    return argument


def _pending_interrupt(config: Dict[str, Any]) -> Optional[Any]:
    snapshot = apex_alpha_graph.get_state(config)
    for task in snapshot.tasks if snapshot else ():
        if task.interrupts:
            return task.interrupts[0].value
    return None


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "platform": "Apex-Alpha Quantitative Market Intelligence",
        "benchmark_tickers": list(SEC_DATABASE.keys()),
    }


@app.get("/api/tickers")
def list_tickers():
    """Lists benchmark assets with current quotes."""
    return [
        {
            "symbol": sym,
            "company_name": data["quote"].company_name,
            "price": data["quote"].price,
            "change": data["quote"].change,
            "change_pct": data["quote"].change_pct,
            "pe_ratio": data["quote"].pe_ratio,
            "beta": data["quote"].beta,
            "market_cap_b": data["quote"].market_cap_b,
            "is_sample_data": data.get("is_sample_data", False),
        }
        for sym, data in SEC_DATABASE.items()
    ]


@app.get("/api/tickers/{symbol}/quote")
def get_quote(symbol: str):
    """Retrieves quote and SEC financial statement metrics."""
    sym = _require_ticker(symbol)
    return {
        "quote": SECFilingRAG.get_quote(sym).model_dump(),
        "financials": SECFilingRAG.get_financials(sym).model_dump(),
        "excerpts": SECFilingRAG.get_excerpts(sym),
    }


@app.get("/api/tickers/{symbol}/forecast")
def get_forecast(symbol: str):
    """Deterministic Monte Carlo forecast and risk scorecard (seeded; no LLM calls)."""
    sym = _require_ticker(symbol)
    quote = SECFilingRAG.get_quote(sym)
    forecast = QuantitativeForecaster.compute_forecast(quote=quote)
    risk = RiskGuardEngine.evaluate_risk(quote=quote, forecast=forecast)
    return {
        "monte_carlo": forecast.model_dump(),
        "risk": risk.model_dump(),
        "allocation_cap_pct": RiskGuardEngine.MAX_ALLOCATION_CAP_PCT,
        "kelly_fraction": RiskGuardEngine.KELLY_SAFETY_FRACTION,
    }


@app.post("/api/tickers/{symbol}/analyze/stream")
def stream_analysis(symbol: str):
    """Runs the full LangGraph pipeline, streaming each node's output as Server-Sent Events.

    Ends with `interrupt` (PM sign-off required; resume via /resume with the thread_id),
    `complete` (no sign-off required), or `error`.
    """
    sym = _require_ticker(symbol)
    thread_id = f"{sym}-{uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": thread_id}}

    def events() -> Iterator[str]:
        yield _sse("start", {"symbol": sym, "thread_id": thread_id})
        for update in apex_alpha_graph.stream({"symbol": sym}, config=config, stream_mode="updates"):
            for node, state in update.items():
                if node == "__interrupt__":
                    yield _sse("interrupt", {"thread_id": thread_id, "payload": state[0].value})
                    return
                data = {key: state.get(key) for key in NODE_OUTPUT_KEYS.get(node, [])}
                data = jsonable_encoder(data)
                for side in ("bull_case", "bear_case"):
                    if data.get(side):
                        _with_verified_citations(sym, data[side])
                yield _sse("node", {"node": node, "data": data})
        yield _sse("complete", {"thread_id": thread_id})

    return StreamingResponse(_sse_guard(events()), media_type="text/event-stream", headers=SSE_HEADERS)


@app.post("/api/tickers/{symbol}/analyze")
def run_analysis(symbol: str):
    """Runs the full pipeline in one request; returns the pending PM interrupt, if any, with its thread_id."""
    sym = _require_ticker(symbol)
    thread_id = f"{sym}-{uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = apex_alpha_graph.invoke({"symbol": sym}, config=config)
    except Exception as exc:
        logger.exception("Analysis run failed")
        status, message, _ = describe_llm_error(exc)
        raise HTTPException(status_code=status, detail=message) from exc

    interrupts = result.get("__interrupt__") or ()
    keys = ["quote", "financials", "monte_carlo", "risk", "bull_case", "bear_case", "synthesis", "proposal", "execution_result"]
    return jsonable_encoder({
        "symbol": sym,
        "thread_id": thread_id,
        "status": "awaiting_pm_approval" if interrupts else result.get("status"),
        **{key: result.get(key) for key in keys},
        "interrupt_details": interrupts[0].value if interrupts else None,
    })


@app.get("/api/tickers/{symbol}/debate/stream")
def stream_debate_endpoint(symbol: str):
    """Streams live Bull, Bear, and CIO synthesis tokens via Server-Sent Events (SSE)."""
    sym = _require_ticker(symbol)
    quote = SECFilingRAG.get_quote(sym)
    financials = SECFilingRAG.get_financials(sym)
    forecast = QuantitativeForecaster.compute_forecast(quote=quote)
    return StreamingResponse(
        _sse_guard(AdversarialDebateEngine.stream_debate(quote=quote, financials=financials, forecast=forecast)),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@app.post("/api/tickers/{symbol}/resume")
def resume_analysis(symbol: str, body: ResumeAllocationRequest):
    """Resumes an interrupted trade proposal via LangGraph Command(resume=...)."""
    sym = _require_ticker(symbol)
    if not body.thread_id.startswith(f"{sym}-"):
        raise HTTPException(status_code=400, detail=f"thread_id does not belong to {sym}.")

    config = {"configurable": {"thread_id": body.thread_id}}
    if _pending_interrupt(config) is None:
        # ponytail: MemorySaver is per-instance; a recycled serverless instance loses paused runs.
        # Swap in a Postgres checkpointer if sign-offs must survive restarts.
        raise HTTPException(
            status_code=409,
            detail="No pending sign-off for this run (it was already decided, or the server restarted). Run the committee again.",
        )

    resume_payload = {
        "approved": body.approved,
        "adjusted_allocation_pct": body.adjusted_allocation_pct,
        "notes": body.notes,
    }
    result = apex_alpha_graph.invoke(Command(resume=resume_payload), config=config)
    return {
        "symbol": sym,
        "thread_id": body.thread_id,
        "status": result.get("status"),
        "execution_result": result.get("execution_result"),
        "pm_approved": result.get("pm_approved"),
        "pm_adjusted_allocation": result.get("pm_adjusted_allocation"),
    }


# Serve UI
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")
