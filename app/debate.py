"""Adversarial Bull vs. Bear Multi-Agent Debate Engine & Executive Synthesizer.
Orchestrates an institutional investment committee debate grounded in SEC disclosures
using Gemini LLM structured outputs and streaming.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any, Dict, Iterator, Tuple
from app.llm import generate_structured, stream_text
from app.schemas import (
    DebateArgument,
    ExecutiveSynthesis,
    FinancialMetrics,
    MarketQuote,
    MarketStance,
    MonteCarloResult,
)
from app.sec_rag import SECFilingRAG


def _multiple(value: float | None) -> str:
    return f"{value}x" if value is not None else "n/a"


def _billions(value: float | None) -> str:
    return f"${value}B" if value is not None else "n/a"


class AdversarialDebateEngine:
    """Orchestrates structured Bull vs. Bear debate and executive committee synthesis using Gemini."""

    BULL_SYSTEM_INSTRUCTION = (
        "You are a Senior Wall Street Equity Research Analyst representing the BULL case on an institutional "
        "investment committee. Your role is to construct a rigorous, data-driven long thesis grounded strictly "
        "in the provided SEC 10-K/10-Q filing disclosures and financial statement metrics. Cite exact verbatim "
        "phrases from the SEC excerpts where relevant. Maintain high professional standards and do not hallucinate."
    )

    BEAR_SYSTEM_INSTRUCTION = (
        "You are a Principal Short-Seller and Chief Risk Officer representing the BEAR case on an institutional "
        "investment committee. Your role is to construct a rigorous, data-driven short/downside thesis grounded strictly "
        "in the provided SEC 10-K/10-Q filing disclosures and valuation metrics. Highlight valuation multiple "
        "vulnerabilities, customer concentration, competitive threats, and margin compression risks using verbatim "
        "citations from the SEC excerpts. Maintain high professional standards."
    )

    CIO_SYSTEM_INSTRUCTION = (
        "You are the Chief Investment Officer (CIO) and Chair of an Institutional Investment Committee. "
        "Your mandate is to synthesize adversarial Bull and Bear arguments alongside quantitative Monte Carlo "
        "price path simulations into an authoritative, probability-weighted allocation recommendation. "
        "Weigh the empirical quantitative forecast against the fundamental SEC risks."
    )

    @classmethod
    def generate_bull_case(cls, quote: MarketQuote, financials: FinancialMetrics) -> DebateArgument:
        """Constructs the long thesis with verbatim SEC citations using Gemini."""
        symbol = quote.symbol
        excerpts = SECFilingRAG.get_excerpts(symbol)

        prompt = (
            f"Construct the institutional Bull thesis for {symbol} ({quote.company_name}).\n\n"
            f"Market Quote Data:\n"
            f"- Price: ${quote.price}\n"
            f"- P/E Ratio: {_multiple(quote.pe_ratio)}\n"
            f"- Forward P/E: {_multiple(quote.forward_pe)}\n"
            f"- Beta: {quote.beta}\n"
            f"- Market Cap: {_billions(quote.market_cap_b)}\n"
            f"- 52-Week Range: ${quote.week_52_low} - ${quote.week_52_high}\n\n"
            f"Financial Statement Metrics ({financials.sec_filing_ref}):\n"
            f"- Revenue: ${financials.revenue_b}B (YoY Growth: {financials.revenue_growth_yoy_pct}%)\n"
            f"- Gross Margin: {financials.gross_margin_pct}%\n"
            f"- Operating Margin: {financials.operating_margin_pct}%\n"
            f"- Free Cash Flow: ${financials.free_cash_flow_b}B\n"
            f"- Debt-to-Equity: {financials.debt_to_equity}\n\n"
            f"SEC Filing Excerpts:\n"
            f"{json.dumps(excerpts, indent=2)}\n\n"
            f"Requirements:\n"
            f"1. agent_type must be 'bull'\n"
            f"2. claim: 1 sentence high-conviction claim\n"
            f"3. thesis: 2-3 sentence strategic long thesis\n"
            f"4. evidence_points: 3 bullet points with specific quantitative metrics from above\n"
            f"5. citations: list of objects with 'source' ({financials.sec_filing_ref}) and 'quote' (verbatim substring from SEC excerpts)\n"
            f"6. confidence_score: float between 0.70 and 0.95"
        )

        return generate_structured(
            prompt=prompt,
            system_instruction=cls.BULL_SYSTEM_INSTRUCTION,
            schema=DebateArgument,
        )

    @classmethod
    def generate_bear_case(cls, quote: MarketQuote, financials: FinancialMetrics) -> DebateArgument:
        """Constructs the short/risk thesis with verbatim SEC citations using Gemini."""
        symbol = quote.symbol
        excerpts = SECFilingRAG.get_excerpts(symbol)

        prompt = (
            f"Construct the institutional Bear/Risk thesis for {symbol} ({quote.company_name}).\n\n"
            f"Market Quote Data:\n"
            f"- Price: ${quote.price}\n"
            f"- P/E Ratio: {_multiple(quote.pe_ratio)}\n"
            f"- Forward P/E: {_multiple(quote.forward_pe)}\n"
            f"- Beta: {quote.beta}\n"
            f"- Market Cap: {_billions(quote.market_cap_b)}\n"
            f"- 52-Week Range: ${quote.week_52_low} - ${quote.week_52_high}\n\n"
            f"Financial Statement Metrics ({financials.sec_filing_ref}):\n"
            f"- Revenue: ${financials.revenue_b}B (YoY Growth: {financials.revenue_growth_yoy_pct}%)\n"
            f"- Gross Margin: {financials.gross_margin_pct}%\n"
            f"- Operating Margin: {financials.operating_margin_pct}%\n"
            f"- Free Cash Flow: ${financials.free_cash_flow_b}B\n"
            f"- Debt-to-Equity: {financials.debt_to_equity}\n\n"
            f"SEC Filing Excerpts:\n"
            f"{json.dumps(excerpts, indent=2)}\n\n"
            f"Requirements:\n"
            f"1. agent_type must be 'bear'\n"
            f"2. claim: 1 sentence downside risk warning\n"
            f"3. thesis: 2-3 sentence thesis focusing on multiple compression, concentration, and headwinds\n"
            f"4. evidence_points: 3 bullet points with specific quantitative metrics from above\n"
            f"5. citations: list of objects with 'source' ({financials.sec_filing_ref}) and 'quote' (verbatim substring from SEC excerpts)\n"
            f"6. confidence_score: float between 0.70 and 0.95"
        )

        return generate_structured(
            prompt=prompt,
            system_instruction=cls.BEAR_SYSTEM_INSTRUCTION,
            schema=DebateArgument,
        )

    @classmethod
    def synthesize_committee_decision(
        cls,
        quote: MarketQuote,
        bull: DebateArgument,
        bear: DebateArgument,
        forecast: MonteCarloResult,
    ) -> ExecutiveSynthesis:
        """Synthesizes Bull vs. Bear debate and quantitative forecast into CIO executive recommendation using Gemini."""
        symbol = quote.symbol
        h30 = forecast.horizons.get(30, list(forecast.horizons.values())[0])

        prompt = (
            f"Synthesize the Institutional Investment Committee consensus for {symbol} ({quote.company_name}).\n"
            f"Today's date: {date.today().isoformat()}\n\n"
            f"Current Market Price: ${quote.price}\n\n"
            f"Quantitative Monte Carlo 30-Day Simulation (10,000 paths):\n"
            f"- Median Expected Target (P50): ${h30.p50_median} ({h30.expected_return_pct:+.1f}%)\n"
            f"- 90% Confidence Interval: [${h30.p10_bear}, ${h30.p90_bull}]\n"
            f"- Probability of Profit: {h30.probability_of_profit_pct:.1f}%\n"
            f"- Annualized Drift: {forecast.annualized_drift * 100:.1f}%\n"
            f"- Annualized Volatility: {forecast.annualized_volatility * 100:.1f}%\n\n"
            f"Adversarial Bull Agent Argument:\n"
            f"- Claim: {bull.claim}\n"
            f"- Thesis: {bull.thesis}\n"
            f"- Evidence: {bull.evidence_points}\n"
            f"- Confidence: {bull.confidence_score}\n\n"
            f"Adversarial Bear Agent Argument:\n"
            f"- Claim: {bear.claim}\n"
            f"- Thesis: {bear.thesis}\n"
            f"- Evidence: {bear.evidence_points}\n"
            f"- Confidence: {bear.confidence_score}\n\n"
            f"Requirements:\n"
            f"1. symbol: '{symbol}'\n"
            f"2. stance: exactly one of 'strong_bullish', 'bullish', 'neutral', 'bearish', 'strong_bearish'\n"
            f"3. confidence_score: float between 0.60 and 0.95 reflecting consensus confidence\n"
            f"4. expected_annualized_return_pct: float matching annualized drift percentage\n"
            f"5. key_catalysts: list of 2-3 primary bullish growth drivers\n"
            f"6. key_downside_risks: list of 2-3 primary bearish vulnerabilities\n"
            f"7. synthesis_narrative: 2-3 paragraphs synthesizing the quantitative distribution with qualitative SEC findings, addressing position sizing and risk boundaries."
        )

        return generate_structured(
            prompt=prompt,
            system_instruction=cls.CIO_SYSTEM_INSTRUCTION,
            schema=ExecutiveSynthesis,
        )

    @classmethod
    def stream_debate(
        cls,
        quote: MarketQuote,
        financials: FinancialMetrics,
        forecast: MonteCarloResult,
    ) -> Iterator[str]:
        """Streams Bull, Bear, and Executive synthesis tokens via Server-Sent Events (SSE)."""
        symbol = quote.symbol
        excerpts = SECFilingRAG.get_excerpts(symbol)
        h30 = forecast.horizons.get(30, list(forecast.horizons.values())[0])

        # Step 1: Stream Bull Case
        yield f"event: stage\ndata: {json.dumps({'stage': 'bull', 'title': f'Bull Equity Analyst: {symbol}'})}\n\n"
        bull_prompt = (
            f"Provide the Bull thesis for {symbol} ({quote.company_name}) trading at ${quote.price}. "
            f"Revenue is ${financials.revenue_b}B (+{financials.revenue_growth_yoy_pct}% YoY) with FCF of ${financials.free_cash_flow_b}B. "
            f"SEC Excerpts: {json.dumps(excerpts)}. "
            f"Write a 2-paragraph investment thesis with bullet points and SEC citations in Markdown."
        )
        bull_chunks = []
        for chunk in stream_text(prompt=bull_prompt, system_instruction=cls.BULL_SYSTEM_INSTRUCTION):
            bull_chunks.append(chunk)
            yield f"event: chunk\ndata: {json.dumps({'stage': 'bull', 'text': chunk})}\n\n"

        # Step 2: Stream Bear Case
        yield f"event: stage\ndata: {json.dumps({'stage': 'bear', 'title': f'Bear Risk Officer: {symbol}'})}\n\n"
        bear_prompt = (
            f"Provide the Bear/Risk thesis for {symbol} ({quote.company_name}) at P/E {_multiple(quote.pe_ratio)}, Beta {quote.beta}. "
            f"SEC Excerpts: {json.dumps(excerpts)}. "
            f"Highlight multiple risk, customer concentration, and macro headwinds. "
            f"Write a 2-paragraph risk thesis with bullet points and SEC citations in Markdown."
        )
        bear_chunks = []
        for chunk in stream_text(prompt=bear_prompt, system_instruction=cls.BEAR_SYSTEM_INSTRUCTION):
            bear_chunks.append(chunk)
            yield f"event: chunk\ndata: {json.dumps({'stage': 'bear', 'text': chunk})}\n\n"

        # Step 3: Stream Executive Synthesis
        yield f"event: stage\ndata: {json.dumps({'stage': 'synthesis', 'title': f'CIO Committee Synthesis: {symbol}'})}\n\n"
        synth_prompt = (
            f"Synthesize the committee decision for {symbol}. Today's date: {date.today().isoformat()}. "
            f"Monte Carlo 30-day forecast: Median ${h30.p50_median} ({h30.expected_return_pct:+.1f}%), "
            f"Win Rate: {h30.probability_of_profit_pct:.1f}%. "
            f"Bull arguments: {''.join(bull_chunks)[:400]}... "
            f"Bear arguments: {''.join(bear_chunks)[:400]}... "
            f"Write a comprehensive CIO synthesis with Stance, Key Catalysts, Key Risks, and Kelly Sizing recommendation in Markdown."
        )
        for chunk in stream_text(prompt=synth_prompt, system_instruction=cls.CIO_SYSTEM_INSTRUCTION):
            yield f"event: chunk\ndata: {json.dumps({'stage': 'synthesis', 'text': chunk})}\n\n"

        yield f"event: complete\ndata: {json.dumps({'status': 'complete', 'symbol': symbol})}\n\n"
