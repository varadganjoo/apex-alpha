"""Fundamental Analysis & Citation Verification Engine.

DATA SOURCE & INTEGRATION ARCHITECTURE:
- The dataset below represents sample data modeled after SEC 10-K and 10-Q filing structures
  for offline development, multi-agent evaluation, and deterministic unit testing.
- The system can be extended to pull from SEC EDGAR for live filing ingestion.
- All excerpts and financial statement metrics are clearly marked as illustrative benchmark sample data.
"""

from typing import Any, Dict, List, Optional
from app.schemas import FinancialMetrics, MarketQuote

SEC_DATABASE: Dict[str, Dict[str, Any]] = {
    "NVDA": {
        "is_sample_data": True,
        "quote": MarketQuote(
            symbol="NVDA",
            company_name="NVIDIA Corporation",
            price=128.40,
            change=4.20,
            change_pct=3.38,
            volume=52410000,
            pe_ratio=46.8,
            forward_pe=31.2,
            beta=1.72,
            market_cap_b=3150.0,
            week_52_high=140.76,
            week_52_low=45.60,
            annualized_volatility=0.48,
        ),
        "financials": FinancialMetrics(
            symbol="NVDA",
            fiscal_quarter="Q2 (Illustrative Sample Benchmark)",
            revenue_b=30.04,
            revenue_growth_yoy_pct=122.4,
            gross_margin_pct=75.1,
            operating_margin_pct=62.1,
            free_cash_flow_b=13.48,
            debt_to_equity=0.18,
            sec_filing_ref="SEC Form 10-Q (Illustrative Sample Benchmark - NVIDIA Corporation)",
        ),
        "excerpts": {
            "data_center_growth": "Data Center revenue was $26.3 billion, up 154% from a year ago, driven by robust demand for our Hopper and Blackwell GPU computing platforms.",
            "customer_concentration": "Customer A accounted for approximately 14% of total revenue for the second quarter of fiscal 2027, and Customer B accounted for 11%.",
            "supply_constraints": "Blackwell production ramp is scheduled for the second half of fiscal 2027; demand continues to outpace supply across all hyperscale cloud customers.",
        },
    },
    "AAPL": {
        "is_sample_data": True,
        "quote": MarketQuote(
            symbol="AAPL",
            company_name="Apple Inc.",
            price=228.60,
            change=-1.10,
            change_pct=-0.48,
            volume=38900000,
            pe_ratio=33.4,
            forward_pe=28.1,
            beta=1.08,
            market_cap_b=3480.0,
            week_52_high=237.23,
            week_52_low=164.08,
            annualized_volatility=0.22,
        ),
        "financials": FinancialMetrics(
            symbol="AAPL",
            fiscal_quarter="Q3 (Illustrative Sample Benchmark)",
            revenue_b=85.78,
            revenue_growth_yoy_pct=4.9,
            gross_margin_pct=46.3,
            operating_margin_pct=30.1,
            free_cash_flow_b=24.12,
            debt_to_equity=1.45,
            sec_filing_ref="SEC Form 10-Q (Illustrative Sample Benchmark - Apple Inc.)",
        ),
        "excerpts": {
            "services_margin": "Services gross margin reached an all-time high of 74.0%, driven by recurring subscriptions in App Store, Cloud, and Apple Pay.",
            "china_revenue": "Greater China revenue declined 6.5% year-over-year to $14.7 billion amid intensified domestic smartphone competition.",
            "apple_intelligence": "Apple Intelligence features began rolling out to iPhone 16 and 17 platforms, establishing an on-device privacy-first AI paradigm.",
        },
    },
    "TSLA": {
        "is_sample_data": True,
        "quote": MarketQuote(
            symbol="TSLA",
            company_name="Tesla, Inc.",
            price=245.20,
            change=8.90,
            change_pct=3.77,
            volume=68200000,
            pe_ratio=64.2,
            forward_pe=52.0,
            beta=2.15,
            market_cap_b=785.0,
            week_52_high=271.00,
            week_52_low=138.80,
            annualized_volatility=0.55,
        ),
        "financials": FinancialMetrics(
            symbol="TSLA",
            fiscal_quarter="Q2 (Illustrative Sample Benchmark)",
            revenue_b=25.50,
            revenue_growth_yoy_pct=2.3,
            gross_margin_pct=18.0,
            operating_margin_pct=6.3,
            free_cash_flow_b=1.34,
            debt_to_equity=0.08,
            sec_filing_ref="SEC Form 10-Q (Illustrative Sample Benchmark - Tesla, Inc.)",
        ),
        "excerpts": {
            "energy_storage": "Energy storage deployments reached a record 9.4 GWh in the quarter, representing year-over-year revenue growth of 100%.",
            "automotive_margin": "Automotive gross margin excluding regulatory credits compressed to 14.6% as price reductions offset manufacturing cost reductions.",
            "autonomous_fleet": "Unsupervised Full Self-Driving (FSD) fleet testing began in select metropolitan jurisdictions ahead of planned Cybercab commercial launch.",
        },
    },
    "MSFT": {
        "is_sample_data": True,
        "quote": MarketQuote(
            symbol="MSFT",
            company_name="Microsoft Corporation",
            price=438.10,
            change=2.40,
            change_pct=0.55,
            volume=21400000,
            pe_ratio=35.1,
            forward_pe=29.4,
            beta=1.18,
            market_cap_b=3260.0,
            week_52_high=468.35,
            week_52_low=366.50,
            annualized_volatility=0.24,
        ),
        "financials": FinancialMetrics(
            symbol="MSFT",
            fiscal_quarter="Q4 (Illustrative Sample Benchmark)",
            revenue_b=64.73,
            revenue_growth_yoy_pct=15.2,
            gross_margin_pct=69.8,
            operating_margin_pct=43.1,
            free_cash_flow_b=23.32,
            debt_to_equity=0.38,
            sec_filing_ref="SEC Form 10-K (Illustrative Sample Benchmark - Microsoft Corporation)",
        ),
        "excerpts": {
            "azure_growth": "Intelligent Cloud revenue was $28.5 billion, with Azure and other cloud services growing 29%, including 8 points of growth from AI services.",
            "capital_expenditures": "Capital expenditures including finance leases were $19.0 billion to support cloud and AI infrastructure demand, with continued expansion in fiscal 2027.",
            "copilot_adoption": "M365 Copilot active seats expanded 60% quarter-over-quarter across Fortune 500 enterprises.",
        },
    },
}


class SECFilingRAG:
    """Retrieval and citation verification for SEC 10-K and 10-Q disclosures."""

    @classmethod
    def is_sample_data(cls, symbol: str) -> bool:
        """Returns True if the data for symbol is illustrative sample benchmark data."""
        data = SEC_DATABASE.get(symbol.upper())
        return data.get("is_sample_data", False) if data else False

    @classmethod
    def get_quote(cls, symbol: str) -> Optional[MarketQuote]:
        data = SEC_DATABASE.get(symbol.upper())
        return data["quote"] if data else None

    @classmethod
    def get_financials(cls, symbol: str) -> Optional[FinancialMetrics]:
        data = SEC_DATABASE.get(symbol.upper())
        return data["financials"] if data else None

    @classmethod
    def get_excerpts(cls, symbol: str) -> Dict[str, str]:
        data = SEC_DATABASE.get(symbol.upper())
        return data["excerpts"] if data else {}

    @classmethod
    def verify_citation(cls, symbol: str, quote_text: str) -> bool:
        """Verifies whether a claim's citation is a verbatim substring of sample SEC filing disclosures."""
        excerpts = cls.get_excerpts(symbol)
        clean_target = " ".join(quote_text.lower().split())
        for raw_text in excerpts.values():
            clean_raw = " ".join(raw_text.lower().split())
            if clean_target in clean_raw:
                return True
        return False

