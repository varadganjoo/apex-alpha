"""FastAPI Application for Apex-Alpha Quantitative Market Intelligence Platform.
Serves REST APIs for quantitative forecasting, adversarial debate, and the dark-mode Bloomberg terminal UI.
"""

from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langgraph.types import Command

from app.debate import AdversarialDebateEngine
from app.graph import apex_alpha_graph
from app.quant_engine import QuantitativeForecaster
from app.sec_rag import SEC_DATABASE, SECFilingRAG

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


class ResumeAllocationRequest(BaseModel):
    approved: bool = True
    adjusted_allocation_pct: Optional[float] = None
    notes: Optional[str] = None


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
        }
        for sym, data in SEC_DATABASE.items()
    ]


@app.get("/api/tickers/{symbol}/quote")
def get_quote(symbol: str):
    """Retrieves quote and SEC financial statement metrics."""
    sym = symbol.upper()
    quote = SECFilingRAG.get_quote(sym)
    financials = SECFilingRAG.get_financials(sym)
    if not quote or not financials:
        raise HTTPException(status_code=404, detail=f"Ticker '{sym}' not found.")
    return {
        "quote": quote.model_dump(),
        "financials": financials.model_dump(),
        "excerpts": SECFilingRAG.get_excerpts(sym),
    }


@app.post("/api/tickers/{symbol}/analyze")
def run_analysis(symbol: str):
    """Executes the full LangGraph quantitative forecasting and debate pipeline."""
    sym = symbol.upper()
    quote = SECFilingRAG.get_quote(sym)
    if not quote:
        raise HTTPException(status_code=404, detail=f"Ticker '{sym}' not found.")

    thread_id = f"thread-{sym}"
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {"symbol": sym}

    try:
        result = apex_alpha_graph.invoke(initial_state, config=config)
        return {
            "symbol": sym,
            "status": result.get("status"),
            "quote": result["quote"].model_dump() if result.get("quote") else None,
            "financials": result["financials"].model_dump() if result.get("financials") else None,
            "monte_carlo": result["monte_carlo"].model_dump() if result.get("monte_carlo") else None,
            "risk": result["risk"].model_dump() if result.get("risk") else None,
            "bull_case": result["bull_case"].model_dump() if result.get("bull_case") else None,
            "bear_case": result["bear_case"].model_dump() if result.get("bear_case") else None,
            "synthesis": result["synthesis"].model_dump() if result.get("synthesis") else None,
            "proposal": result["proposal"].model_dump() if result.get("proposal") else None,
            "execution_result": result.get("execution_result"),
        }
    except Exception as e:
        # Check if paused at interrupt()
        state_snap = apex_alpha_graph.get_state(config)
        if state_snap and state_snap.tasks:
            for task in state_snap.tasks:
                if task.interrupts:
                    int_val = task.interrupts[0].value
                    return {
                        "symbol": sym,
                        "status": "awaiting_pm_approval",
                        "quote": state_snap.values["quote"].model_dump(),
                        "financials": state_snap.values["financials"].model_dump(),
                        "monte_carlo": state_snap.values["monte_carlo"].model_dump(),
                        "risk": state_snap.values["risk"].model_dump(),
                        "bull_case": state_snap.values["bull_case"].model_dump(),
                        "bear_case": state_snap.values["bear_case"].model_dump(),
                        "synthesis": state_snap.values["synthesis"].model_dump(),
                        "proposal": state_snap.values["proposal"].model_dump(),
                        "interrupt_details": int_val,
                    }
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tickers/{symbol}/debate/stream")
def stream_debate_endpoint(symbol: str):
    """Streams live Bull, Bear, and CIO synthesis tokens via Server-Sent Events (SSE)."""
    sym = symbol.upper()
    quote = SECFilingRAG.get_quote(sym)
    financials = SECFilingRAG.get_financials(sym)
    if not quote or not financials:
        raise HTTPException(status_code=404, detail=f"Ticker '{sym}' not found.")

    forecast = QuantitativeForecaster.compute_forecast(quote=quote)
    return StreamingResponse(
        AdversarialDebateEngine.stream_debate(quote=quote, financials=financials, forecast=forecast),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/tickers/{symbol}/resume")
def resume_analysis(symbol: str, body: ResumeAllocationRequest):
    """Resumes an interrupted trade proposal via LangGraph Command(resume=...)."""
    sym = symbol.upper()
    thread_id = f"thread-{sym}"
    config = {"configurable": {"thread_id": thread_id}}

    resume_payload = {
        "approved": body.approved,
        "adjusted_allocation_pct": body.adjusted_allocation_pct,
        "notes": body.notes,
    }

    try:
        result = apex_alpha_graph.invoke(Command(resume=resume_payload), config=config)
        return {
            "symbol": sym,
            "status": result.get("status"),
            "execution_result": result.get("execution_result"),
            "pm_approved": result.get("pm_approved"),
            "pm_adjusted_allocation": result.get("pm_adjusted_allocation"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to resume analysis: {e}")


# Serve UI
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")
