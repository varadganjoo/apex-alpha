"""Unit tests for Adversarial Bull vs. Bear Debate Engine & SEC Citations."""

import pytest
from app.debate import AdversarialDebateEngine
from app.quant_engine import QuantitativeForecaster
from app.sec_rag import SECFilingRAG


def test_sec_citation_verification():
    """Verifies that generated claims cite verbatim substrings from SEC disclosures."""
    quote = SECFilingRAG.get_quote("NVDA")
    financials = SECFilingRAG.get_financials("NVDA")
    bull = AdversarialDebateEngine.generate_bull_case(quote, financials)

    assert len(bull.citations) > 0
    # Every citation must be verified in the SEC database
    for cit in bull.citations:
        assert SECFilingRAG.verify_citation("NVDA", cit["quote"]) is True


def test_bear_case_identifies_downside_risks():
    """Verifies that Bear Risk Officer identifies concrete valuation or customer risks."""
    quote = SECFilingRAG.get_quote("NVDA")
    financials = SECFilingRAG.get_financials("NVDA")
    bear = AdversarialDebateEngine.generate_bear_case(quote, financials)

    assert bear.agent_type == "bear"
    assert len(bear.evidence_points) >= 3
    assert any("Customer" in p or "multiple" in p.lower() or "Beta" in p for p in bear.evidence_points)


def test_executive_synthesis_reaches_consensus():
    """Verifies that committee synthesis produces a valid stance and expected return."""
    quote = SECFilingRAG.get_quote("NVDA")
    financials = SECFilingRAG.get_financials("NVDA")
    bull = AdversarialDebateEngine.generate_bull_case(quote, financials)
    bear = AdversarialDebateEngine.generate_bear_case(quote, financials)
    forecast = QuantitativeForecaster.compute_forecast(quote, num_paths=1000)

    synthesis = AdversarialDebateEngine.synthesize_committee_decision(quote, bull, bear, forecast)
    assert synthesis.symbol == "NVDA"
    assert synthesis.stance is not None
    assert ("NVDA" in synthesis.synthesis_narrative or "NVIDIA" in synthesis.synthesis_narrative)
