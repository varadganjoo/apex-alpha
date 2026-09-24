"""API tests for the UI-facing endpoints: deterministic data, streamed LangGraph runs, and PM sign-off."""

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.quant_engine import QuantitativeForecaster
from app.sec_rag import SECFilingRAG

client = TestClient(app)


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = block.splitlines()
        event = next(line[7:] for line in lines if line.startswith("event: "))
        data = json.loads(next(line[6:] for line in lines if line.startswith("data: ")))
        events.append((event, data))
    return events


def run_committee(symbol: str) -> list[tuple[str, dict]]:
    response = client.post(f"/api/tickers/{symbol}/analyze/stream")
    assert response.status_code == 200
    return parse_sse(response.text)


def test_tickers_come_from_backend_dataset():
    rows = {row["symbol"]: row for row in client.get("/api/tickers").json()}
    assert rows["AAPL"]["price"] == SECFilingRAG.get_quote("AAPL").price
    assert rows["NVDA"]["market_cap_b"] == SECFilingRAG.get_quote("NVDA").market_cap_b


def test_forecast_endpoint_matches_quant_engine():
    body = client.get("/api/tickers/NVDA/forecast").json()
    expected = QuantitativeForecaster.compute_forecast(quote=SECFilingRAG.get_quote("NVDA"))
    assert body["monte_carlo"]["horizons"]["30"]["p50_median"] == expected.horizons[30].p50_median
    assert body["allocation_cap_pct"] == 15.0
    assert body["risk"]["recommended_allocation_pct"] <= 15.0
    assert body["quant_call"] in {"buy", "hold", "sell"}


def test_unknown_ticker_is_404():
    assert client.get("/api/tickers/ZZZZ/forecast").status_code == 404
    assert client.post("/api/tickers/ZZZZ/analyze/stream").status_code == 404


def test_analyze_stream_emits_nodes_then_pm_interrupt():
    events = run_committee("NVDA")
    names = [name for name, _ in events]
    assert names[0] == "start"
    assert names[-1] == "interrupt"
    nodes = [data["node"] for name, data in events if name == "node"]
    assert nodes == [
        "ingest_market_data",
        "run_quantitative_forecast",
        "evaluate_risk_guardrails",
        "adversarial_debate",
        "executive_synthesis",
        "stage_order_proposal",
    ]
    interrupt = events[-1][1]
    assert interrupt["thread_id"].startswith("NVDA-")
    assert interrupt["payload"]["proposal"]["symbol"] == "NVDA"
    assert interrupt["payload"]["reasons"]


def test_debate_citations_are_verified_against_excerpts():
    debate = next(data["data"] for name, data in run_committee("NVDA") if name == "node" and data["node"] == "adversarial_debate")
    for side in ("bull_case", "bear_case"):
        assert debate[side]["citations"], side
        assert all(c["verified"] for c in debate[side]["citations"])


def test_each_run_gets_its_own_thread():
    first = run_committee("NVDA")[-1][1]["thread_id"]
    second = run_committee("NVDA")[-1][1]["thread_id"]
    assert first != second


def test_resume_approval_commits_order_with_pm_allocation():
    thread_id = run_committee("NVDA")[-1][1]["thread_id"]
    response = client.post(
        "/api/tickers/NVDA/resume",
        json={"thread_id": thread_id, "approved": True, "adjusted_allocation_pct": 5.0, "notes": "Half size."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "executed"
    assert "Target Allocation: 5.0%" in body["execution_result"]


def test_resume_rejection_cancels_order():
    thread_id = run_committee("TSLA")[-1][1]["thread_id"]
    body = client.post("/api/tickers/TSLA/resume", json={"thread_id": thread_id, "approved": False, "notes": "Too volatile."}).json()
    assert body["status"] == "rejected_by_pm"
    assert "Too volatile." in body["execution_result"]


def test_resume_twice_is_rejected():
    thread_id = run_committee("NVDA")[-1][1]["thread_id"]
    assert client.post("/api/tickers/NVDA/resume", json={"thread_id": thread_id}).status_code == 200
    assert client.post("/api/tickers/NVDA/resume", json={"thread_id": thread_id}).status_code == 409


def test_resume_unknown_thread_is_409():
    assert client.post("/api/tickers/NVDA/resume", json={"thread_id": "NVDA-doesnotexist"}).status_code == 409


def test_resume_rejects_thread_from_other_ticker():
    thread_id = run_committee("NVDA")[-1][1]["thread_id"]
    assert client.post("/api/tickers/TSLA/resume", json={"thread_id": thread_id}).status_code == 400


def test_pm_override_cannot_exceed_allocation_cap():
    thread_id = run_committee("NVDA")[-1][1]["thread_id"]
    response = client.post("/api/tickers/NVDA/resume", json={"thread_id": thread_id, "adjusted_allocation_pct": 40.0})
    assert response.status_code == 422


@pytest.mark.parametrize(
    "failure, retryable, phrase",
    [
        ("Gemini API structured generation failed: 429 RESOURCE_EXHAUSTED", True, "rate limit"),
        ("Gemini API structured generation failed: 503 UNAVAILABLE", True, "overloaded"),
        ("GEMINI_API_KEY environment variable is not configured.", False, "not configured"),
        ("Gemini API structured generation failed: GenerateRequestsPerDayPerProjectPerModel-FreeTier 429 RESOURCE_EXHAUSTED", False, "daily"),
    ],
)
def test_llm_failure_mid_stream_becomes_error_event(monkeypatch, failure, retryable, phrase):
    def boom(*args, **kwargs):
        raise RuntimeError(failure)

    monkeypatch.setattr("app.debate.generate_structured", boom)
    events = run_committee("NVDA")
    name, data = events[-1]
    assert name == "error"
    assert data["retryable"] is retryable
    assert phrase in data["message"]
    # Deterministic nodes still streamed before the model failed.
    assert "evaluate_risk_guardrails" in [d["node"] for n, d in events if n == "node"]


def test_debate_stream_failure_becomes_error_event(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("Gemini streaming generation failed: 503 UNAVAILABLE")
        yield  # pragma: no cover

    monkeypatch.setattr("app.debate.stream_text", boom)
    events = parse_sse(client.get("/api/tickers/NVDA/debate/stream").text)
    assert events[-1][0] == "error"


def test_analyze_json_endpoint_returns_thread_and_interrupt():
    body = client.post("/api/tickers/NVDA/analyze").json()
    assert body["status"] == "awaiting_pm_approval"
    assert body["thread_id"].startswith("NVDA-")
    assert body["interrupt_details"]["symbol"] == "NVDA"
