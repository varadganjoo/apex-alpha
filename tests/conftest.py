"""Pytest fixtures for Apex-Alpha test suite.
Provides deterministic mock responses for Gemini structured outputs and streaming
so tests run reliably, fast, and offline without requiring active API keys,
mirroring the setup in CareFlow.
"""

import pytest
from app.schemas import Citation, DebateArgument, ExecutiveSynthesis, MarketStance
from app.sec_rag import SECFilingRAG


@pytest.fixture(autouse=True)
def offline_sample_data(monkeypatch):
    """Tests never hit Yahoo or SEC; test_market_data.py opts back in with mocked fetchers."""
    monkeypatch.setenv("APEX_DATA", "sample")


@pytest.fixture(autouse=True)
def mock_gemini_apex(monkeypatch, request):
    """Mocks Gemini LLM structured outputs and text streaming for deterministic testing.
    Skips mocking for tests in test_llm.py so error handling and missing-key checks can be verified.
    """
    if "test_llm" in request.node.nodeid:
        return

    def fake_generate_structured(prompt: str, system_instruction: str, schema, **kwargs):
        symbol = "NVDA"
        for s in ["NVDA", "AAPL", "TSLA", "MSFT"]:
            if s in prompt:
                symbol = s
                break

        excerpts = SECFilingRAG.get_excerpts(symbol)
        quotes = list(excerpts.values())
        quote1 = quotes[0] if quotes else "Sample SEC filing disclosure."
        quote2 = quotes[1] if len(quotes) > 1 else quote1

        if schema == DebateArgument:
            is_bull = (
                "BULL" in system_instruction
                or "Bull" in prompt
                or "agent_type must be 'bull'" in prompt
            )
            if is_bull:
                return DebateArgument(
                    agent_type="bull",
                    claim=f"High conviction long thesis on {symbol} supported by accelerating demand.",
                    thesis=f"{symbol} exhibits strong operational execution and expanding gross margins.",
                    evidence_points=[
                        quote1,
                        "Gross margin expansion reflecting pricing power",
                        "Free cash flow conversion remains robust",
                    ],
                    citations=[
                        Citation(
                            source=f"SEC Form 10-Q ({symbol})",
                            quote=quote1,
                        )
                    ],
                    confidence_score=0.91,
                )
            else:
                return DebateArgument(
                    agent_type="bear",
                    claim=f"Downside risk warning for {symbol} from customer concentration and multiple compression.",
                    thesis=f"Elevated valuation multiple and customer concentration present downside risk.",
                    evidence_points=[
                        quote2,
                        "Customer concentration risk in hyperscale accounts",
                        "Elevated valuation multiple relative to historical median",
                    ],
                    citations=[
                        Citation(
                            source=f"SEC Form 10-Q ({symbol})",
                            quote=quote2,
                        )
                    ],
                    confidence_score=0.88,
                )
        elif schema == ExecutiveSynthesis:
            return ExecutiveSynthesis(
                symbol=symbol,
                stance=MarketStance.BULLISH,
                confidence_score=0.85,
                expected_annualized_return_pct=18.4,
                key_catalysts=["Accelerating demand", "Gross margin expansion"],
                key_downside_risks=["Customer concentration", "Valuation multiple compression"],
                synthesis_narrative=f"Institutional committee consensus for {symbol} (NVIDIA Corporation) remains bullish based on empirical Monte Carlo distribution.",
            )
        return schema()

    def fake_stream_text(prompt: str, system_instruction: str = "", **kwargs):
        yield "Institutional "
        yield "committee "
        yield "analysis "
        yield "streamed."

    monkeypatch.setattr("app.debate.generate_structured", fake_generate_structured)
    monkeypatch.setattr("app.debate.stream_text", fake_stream_text)
