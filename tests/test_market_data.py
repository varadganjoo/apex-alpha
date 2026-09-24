"""Offline tests for live market data: parsing, derivations, fallbacks, and caching. No network access."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import market_data as md
from app.sec_rag import SEC_DATABASE, SECFilingRAG


CACHED = (
    md.price_table, md.yahoo_info, md.company_facts, md.submissions, md.sec_fundamentals, md.sec_excerpts,
    md.treasury_3m_yield,
)


@pytest.fixture
def live(monkeypatch):
    monkeypatch.setenv("APEX_DATA", "live")
    monkeypatch.setenv("SEC_USER_AGENT", "Apex-Alpha tests test@example.com")
    for fn in CACHED:  # module-level references to the real cached functions, even if a test monkeypatches them
        fn.cache_clear()
    yield
    for fn in CACHED:
        fn.cache_clear()


def _prices(n=400, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2025-01-01", periods=n)
    market = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n))), index=index)
    stock = pd.Series(50 * np.exp(np.cumsum(1.3 * np.log(market).diff().fillna(0) + rng.normal(0, 0.012, n))), index=index)
    return stock, market


# ---------------------------------------------------------------- risk inputs


def test_risk_inputs_match_the_backtest_features(monkeypatch):
    from backtest import run_backtest as bt

    stock, market = _prices()
    monkeypatch.setattr(bt, "UNIVERSE", ["AAA"])
    features = bt.build_features(pd.DataFrame({"AAA": stock, "SPY": market}))
    row = features[3]
    i = stock.index.get_loc(row.date)
    live = md.risk_inputs(stock.iloc[: i + 1], market.iloc[: i + 1])

    assert live["beta"] == pytest.approx(row.beta)
    assert live["annualized_volatility"] == pytest.approx(row.vol)
    assert row.price / live["week_52_high"] - 1 == pytest.approx(row.high_52w_gap)


def test_risk_inputs_recover_a_known_beta():
    stock, market = _prices(n=600, seed=3)
    assert md.risk_inputs(stock, market)["beta"] == pytest.approx(1.3, abs=0.15)


# ---------------------------------------------------------------- SEC facts


def _fact(start, end, val, filed="2026-01-01", form="10-Q"):
    row = {"end": end, "val": val, "filed": filed, "form": form}
    if start:
        row["start"] = start
    return row


def test_quarterly_derives_quarters_from_year_to_date_figures():
    group = {"Revenues": {"units": {"USD": [
        _fact("2025-01-01", "2025-03-31", 10),
        _fact("2025-01-01", "2025-06-30", 25),
        _fact("2025-01-01", "2025-09-30", 45),
        _fact("2025-01-01", "2025-12-31", 70, form="10-K"),
    ]}}}
    assert md.quarterly(group, ["Revenues"]) == {
        "2025-03-31": 10, "2025-06-30": 15, "2025-09-30": 20, "2025-12-31": 25,
    }


def test_quarterly_prefers_reported_quarters_and_latest_restatement():
    group = {"Revenues": {"units": {"USD": [
        _fact("2025-04-01", "2025-06-30", 14, filed="2025-08-01"),
        _fact("2025-04-01", "2025-06-30", 16, filed="2026-08-01"),  # restated in a later filing
        _fact("2025-01-01", "2025-06-30", 99),
    ]}}}
    assert md.quarterly(group, ["Revenues"])["2025-06-30"] == 16


def test_tag_with_most_recent_data_wins():
    group = {
        "Revenues": {"units": {"USD": [_fact("2018-01-01", "2018-03-31", 1)]}},
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [_fact("2026-01-01", "2026-03-31", 7)]}},
    }
    assert md.quarterly(group, md.REVENUE_TAGS) == {"2026-03-31": 7}


def _company_facts():
    usd = lambda rows: {"units": {"USD": rows}}  # noqa: E731
    quarters = [("2025-04-01", "2025-06-30"), ("2025-07-01", "2025-09-30"), ("2025-10-01", "2025-12-31"),
                ("2026-01-01", "2026-03-31"), ("2026-04-01", "2026-06-30")]
    return {"facts": {
        "us-gaap": {
            "Revenues": usd([_fact(s, e, v) for (s, e), v in zip(quarters, [80e9, 85e9, 90e9, 95e9, 100e9])]),
            "GrossProfit": usd([_fact("2026-04-01", "2026-06-30", 60e9)]),
            "OperatingIncomeLoss": usd([_fact("2026-04-01", "2026-06-30", 40e9)]),
            "NetCashProvidedByUsedInOperatingActivities": usd([_fact("2026-04-01", "2026-06-30", 30e9)]),
            "PaymentsToAcquireProductiveAssets": usd([_fact("2026-04-01", "2026-06-30", 5e9)]),
            "NetIncomeLoss": usd([_fact(s, e, 20e9) for s, e in quarters]),
            "LongTermDebt": usd([_fact(None, "2026-06-30", 10e9)]),
            "StockholdersEquity": usd([_fact(None, "2026-06-30", 100e9)]),
        },
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [_fact(None, "2026-07-15", 2e9)]}}},
    }}


def _submissions():
    return {"filings": {"recent": {
        "form": ["8-K", "10-Q"], "accessionNumber": ["x", "0001045810-26-000075"],
        "filingDate": ["2026-08-30", "2026-08-26"], "reportDate": ["", "2026-06-30"],
        "primaryDocument": ["a.htm", "nvda-20260630.htm"],
    }}}


def test_sec_fundamentals_from_company_facts(live, monkeypatch):
    monkeypatch.setattr(md, "company_facts", lambda cik: _company_facts())
    monkeypatch.setattr(md, "submissions", lambda cik: _submissions())
    f = md.sec_fundamentals("NVDA")
    assert f["fiscal_quarter"] == "Quarter ended 2026-06-30"
    assert f["revenue_b"] == 100.0
    assert f["revenue_growth_yoy_pct"] == 25.0          # vs 80B a year earlier
    assert (f["gross_margin_pct"], f["operating_margin_pct"]) == (60.0, 40.0)
    assert f["free_cash_flow_b"] == 25.0
    assert f["debt_to_equity"] == 0.1
    assert f["filing_url"].endswith("/1045810/000104581026000075/nvda-20260630.htm")

    valuation = md.sec_valuation_inputs("NVDA")
    assert valuation == {"shares": 2e9, "net_income_ttm": 80e9}


def test_sec_requires_contact_email(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "no email here")
    with pytest.raises(RuntimeError, match="contact email"):
        md._sec_get("https://data.sec.gov/anything")


# ---------------------------------------------------------------- filing excerpts

FILING = """<html><head><title>10-Q</title></head><body>
<div style="display:none"><ix:header><p>Hidden XBRL header revenue 99%</p></ix:header></div>
<p>Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations</p>
<p>Item 1A. Risk Factors</p>
<p>Item 2. Management&#8217;s Discussion and Analysis of Financial Condition and Results of Operations</p>
<p>Revenue was $30.0 billion, up 12% from a year ago, driven by strong demand for our data center products. Growth
was broad based across regions and customers.</p>
<p>costs, and higher logistics costs. Gross margin increased to 71.2% from 68.0% a year ago, reflecting a favorable
mix of higher-margin products and lower inventory provisions across all segments.</p>
<p>As of quarter end we had $40.1 billion in cash, cash equivalents and marketable securities, which we believe is
sufficient to fund operations and capital returns for at least the next twelve months.</p>
<p>Item 3. Quantitative and Qualitative Disclosures About Market Risk</p>
<p>Item 1A. Risk Factors</p>
<p>There have been no material changes to our risk factors from those disclosed in our annual report on Form 10-K
for the fiscal year, other than the updates described in this section of the report.</p>
<p>We depend on a limited number of customers for a significant portion of our revenue, and the loss of any of these
customers could adversely affect our results of operations and financial condition going forward.</p>
<p>Item 2. Unregistered Sales of Equity Securities</p>
</body></html>"""


def test_extract_excerpts_uses_the_real_sections():
    lines = md.filing_lines(FILING)
    excerpts = md.extract_excerpts(lines)
    text = " ".join(lines)

    assert set(excerpts) >= {"revenue_trend", "margins", "liquidity", "key_risk"}
    assert excerpts["revenue_trend"].startswith("Revenue was $30.0 billion, up 12%")
    assert excerpts["margins"].startswith("Gross margin increased to 71.2%")   # page-break fragment dropped
    assert excerpts["key_risk"].startswith("We depend on a limited number of customers")
    assert "Hidden XBRL header" not in text
    for excerpt in excerpts.values():
        assert excerpt in text  # verbatim, so citation verification can match it


def test_excerpts_are_used_for_citation_verification(live, monkeypatch):
    monkeypatch.setattr(md, "submissions", lambda cik: _submissions())
    monkeypatch.setattr(md, "_sec_get", lambda url: FILING.encode("utf-8"))
    excerpts, meta = SECFilingRAG.get_excerpts_with_source("NVDA")
    assert meta["source"] == "SEC EDGAR" and meta["form"] == "10-Q"
    assert SECFilingRAG.verify_citation("NVDA", "Gross margin increased to 71.2% from 68.0% a year ago")
    assert not SECFilingRAG.verify_citation("NVDA", "Revenue tripled overnight")


def test_windows_1252_filings_decode(live, monkeypatch):
    monkeypatch.setattr(md, "submissions", lambda cik: _submissions())
    monkeypatch.setattr(md, "_sec_get", lambda url: FILING.replace("&#8217;", "’").encode("cp1252"))
    excerpts, _ = md.sec_excerpts("NVDA")
    assert "revenue_trend" in excerpts


# ---------------------------------------------------------------- fallbacks and caching


def test_each_piece_falls_back_to_sample_independently(live, monkeypatch):
    def down(*args):
        raise RuntimeError("source down")

    monkeypatch.setattr(md, "price_table", down)
    monkeypatch.setattr(md, "company_facts", lambda cik: _company_facts())
    monkeypatch.setattr(md, "submissions", lambda cik: _submissions())
    monkeypatch.setattr(md, "_sec_get", down)

    quote = SECFilingRAG.get_quote("NVDA")
    financials = SECFilingRAG.get_financials("NVDA")
    excerpts, meta = SECFilingRAG.get_excerpts_with_source("NVDA")

    assert quote == SEC_DATABASE["NVDA"]["quote"] and quote.source == "sample"
    assert financials.source == "SEC EDGAR"                     # SEC facts still worked
    assert meta == {"source": "sample"} and excerpts == SEC_DATABASE["NVDA"]["excerpts"]


def test_live_quote_never_mixes_in_stale_sample_valuations(live, monkeypatch):
    stock, market = _prices()
    closes = pd.DataFrame({s: stock for s in md.CIKS} | {"SPY": market})
    monkeypatch.setattr(md, "price_table", lambda: (closes, closes * 0 + 1_000_000, "Yahoo Finance"))

    def no_info(symbol):
        raise RuntimeError("info blocked")

    monkeypatch.setattr(md, "yahoo_info", no_info)
    monkeypatch.setattr(md, "sec_valuation_inputs", no_info)

    quote = SECFilingRAG.get_quote("NVDA")
    assert quote.source == "Yahoo Finance"
    assert quote.price == round(stock.iloc[-1], 2)
    assert quote.pe_ratio is None and quote.market_cap_b is None   # unknown, not the sample's stale numbers
    assert quote.company_name == "NVIDIA Corporation"


def test_ttl_cache_caches_values_and_briefly_caches_failures(monkeypatch):
    calls = []
    clock = [100.0]
    monkeypatch.setattr(md.time, "monotonic", lambda: clock[0])

    @md.ttl_cache(10, failure_seconds=2)
    def fetch(x):
        calls.append(x)
        if x == "bad":
            raise RuntimeError("down")
        return x * 2

    assert fetch("a") == "aa" and fetch("a") == "aa" and calls == ["a"]
    clock[0] += 11
    fetch("a")
    assert calls == ["a", "a"]

    for _ in range(3):
        with pytest.raises(RuntimeError):
            fetch("bad")
    assert calls.count("bad") == 1
    clock[0] += 3
    with pytest.raises(RuntimeError):
        fetch("bad")
    assert calls.count("bad") == 2


def test_read_endpoints_send_cdn_cache_headers():
    from app.main import app

    client = TestClient(app)
    response = client.get("/api/tickers")
    assert "s-maxage" in response.headers["cache-control"]
    assert all(row["source"] == "sample" for row in response.json())
    assert "s-maxage" in client.get("/api/tickers/NVDA/quote").headers["cache-control"]
    stream = client.get("/api/tickers/NVDA/debate/stream")
    assert "s-maxage" not in stream.headers.get("cache-control", "")


# ---------------------------------------------------------------- risk-free rate

TREASURY = '''\ufeffDate,"1 Mo","2 Mo","3 Mo","6 Mo"
09/23/2026,3.99,4.10,4.19,4.31
09/24/2026,4.01,4.18,4.24,4.34
09/22/2026,3.97,4.09,4.16,4.26
'''


def test_parse_treasury_csv_takes_the_latest_date():
    assert md.parse_treasury_csv(TREASURY.lstrip("\ufeff")) == (pytest.approx(0.0424), "2026-09-24")
    assert md.parse_treasury_csv('Date,"3 Mo"\n') is None


def test_treasury_falls_back_to_last_year_file_in_january(live, monkeypatch):
    files = {md.dt.date.today().year: 'Date,"3 Mo"\n', md.dt.date.today().year - 1: TREASURY}
    monkeypatch.setattr(md, "_get", lambda url, headers: files[int(url.split("/")[-2])].encode("utf-8"))
    assert md.treasury_3m_yield() == (pytest.approx(0.0424), "2026-09-24")


def test_forecast_and_sharpe_use_the_live_risk_free_rate(live, monkeypatch):
    from app.quant_engine import QuantitativeForecaster as QF
    from app.risk_guard import RiskGuardEngine

    monkeypatch.setattr(md, "treasury_3m_yield", lambda: (0.03, "2026-09-24"))
    quote = SEC_DATABASE["NVDA"]["quote"]
    forecast = QF.compute_forecast(quote=quote)
    assert (forecast.risk_free_rate, forecast.risk_free_source) == (0.03, "US Treasury 3M, 2026-09-24")
    assert forecast.annualized_drift == pytest.approx(QF.estimate_drift(quote, 0.03), abs=1e-4)
    assert QF.estimate_drift(quote, 0.05) - QF.estimate_drift(quote, 0.03) == pytest.approx(0.02)

    risk = RiskGuardEngine.evaluate_risk(quote=quote, forecast=forecast)
    expected = round((forecast.annualized_drift - 0.03) / max(0.05, forecast.annualized_volatility), 2)
    assert risk.sharpe_ratio == expected


def test_risk_free_rate_falls_back_to_the_assumption(live, monkeypatch):
    from app.quant_engine import QuantitativeForecaster as QF

    def down():
        raise RuntimeError("treasury down")

    monkeypatch.setattr(md, "treasury_3m_yield", down)
    assert QF.risk_free_rate() == (QF.RISK_FREE_RATE, "assumed")
