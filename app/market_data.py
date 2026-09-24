"""Live market data for Apex-Alpha's covered tickers.

    prices, market cap, P/E   Yahoo Finance via yfinance; Tiingo if TIINGO_API_KEY is set
    fundamentals              SEC EDGAR XBRL company facts
    filing excerpts           SEC EDGAR, latest 10-Q or 10-K primary document
    risk-free rate            U.S. Treasury daily par yield curve, 3-month

SEC requires a User-Agent with a contact email (SEC_USER_AGENT). Results are cached in-process; the API also sets
CDN cache headers so serverless instances rarely refetch. Callers fall back to the sample dataset on any failure.
Set APEX_DATA=sample to stay offline (tests do).
"""

from __future__ import annotations

import csv
import datetime as dt
import gzip
import io
import json
import logging
import math
import os
import re
import tempfile
import threading
import time
import urllib.request
from collections import defaultdict
from html.parser import HTMLParser
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger("apex_alpha.market_data")

CIKS = {"NVDA": 1045810, "AAPL": 320193, "TSLA": 1318605, "MSFT": 789019}
MARKET = "SPY"
LOOKBACK = 252  # trading days, same window as the backtest

REVENUE_TAGS = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"]
CAPEX_TAGS = ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"]
DEBT_TAGS = ["LongTermDebt", "LongTermDebtNoncurrent"]


def live_enabled() -> bool:
    return os.getenv("APEX_DATA", "live").lower() != "sample"


def ttl_cache(seconds: float, failure_seconds: float = 60.0) -> Callable:
    """Per-process TTL cache. Failures are cached briefly so a dead source is not re-hit on every request."""

    def decorator(fn: Callable) -> Callable:
        store: dict[tuple, tuple[float, bool, Any]] = {}
        lock = threading.Lock()

        def wrapper(*args):
            now = time.monotonic()
            with lock:
                hit = store.get(args)
            if hit and hit[0] > now:
                ok, value = hit[1], hit[2]
                if ok:
                    return value
                raise value
            try:
                value = fn(*args)
            except Exception as exc:
                with lock:
                    store[args] = (now + failure_seconds, False, exc)
                raise
            with lock:
                store[args] = (now + seconds, True, value)
            return value

        wrapper.cache_clear = store.clear
        return wrapper

    return decorator


def _get(url: str, headers: dict[str, str], timeout: float = 20.0) -> bytes:
    request = urllib.request.Request(url, headers={**headers, "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
        return gzip.decompress(body) if response.headers.get("Content-Encoding") == "gzip" else body


# ---------------------------------------------------------------- prices


@ttl_cache(900)
def price_table() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Adjusted closes and volumes for the covered tickers plus SPY, about 15 months of daily bars."""
    symbols = list(CIKS) + [MARKET]
    errors = []
    try:
        import yfinance as yf

        # Serverless filesystems are read-only outside /tmp.
        yf.set_tz_cache_location(os.path.join(tempfile.gettempdir(), "yfinance"))
        data = yf.download(symbols, period="15mo", auto_adjust=True, progress=False, threads=True)
        closes, volumes = data["Close"], data["Volume"]
        if closes.dropna(how="all").shape[0] > LOOKBACK and closes.iloc[-1].notna().all():
            closes.index = pd.DatetimeIndex(closes.index).tz_localize(None).normalize()
            volumes.index = closes.index
            return closes, volumes, "Yahoo Finance"
        errors.append(f"yahoo: incomplete data {closes.shape}")
    except Exception as exc:
        errors.append(f"yahoo: {exc}")

    key = os.getenv("TIINGO_API_KEY", "").strip()
    if key:
        try:
            start = (dt.date.today() - dt.timedelta(days=460)).isoformat()
            frames = {}
            for symbol in symbols:
                rows = json.loads(_get(
                    f"https://api.tiingo.com/tiingo/daily/{symbol}/prices?startDate={start}",
                    {"Authorization": f"Token {key}", "Content-Type": "application/json"},
                ))
                frame = pd.DataFrame(rows)
                frame.index = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
                frames[symbol] = frame
            closes = pd.DataFrame({s: f["adjClose"] for s, f in frames.items()})
            volumes = pd.DataFrame({s: f["adjVolume"] for s, f in frames.items()})
            return closes, volumes, "Tiingo"
        except Exception as exc:
            errors.append(f"tiingo: {exc}")
    raise RuntimeError("; ".join(errors))


def risk_inputs(closes: pd.Series, market: pd.Series) -> dict[str, float]:
    """Beta vs SPY, annualized volatility, and 52-week range over the last 252 trading days.

    Same math as backtest/run_backtest.py, so the live model sees the inputs it was tested on.
    """
    closes = closes.dropna()
    window = closes.iloc[-LOOKBACK:]
    r_i = np.log(closes).diff().loc[window.index]
    r_m = np.log(market.dropna()).diff().reindex(window.index)
    ok = r_i.notna() & r_m.notna()
    beta = float(np.cov(r_i[ok], r_m[ok])[0, 1] / np.var(r_m[ok], ddof=1))
    return {
        "beta": beta,
        "annualized_volatility": float(r_i[ok].std() * math.sqrt(252)),
        "week_52_high": float(window.max()),
        "week_52_low": float(window.min()),
    }


@ttl_cache(24 * 3600, failure_seconds=600)
def yahoo_info(symbol: str) -> dict:
    import yfinance as yf

    return yf.Ticker(symbol).info


def live_quote_fields(symbol: str) -> dict[str, Any]:
    """Everything MarketQuote needs from live prices; market cap and P/E fall back to SEC-derived values."""
    closes, volumes, source = price_table()
    series = closes[symbol].dropna()
    price, previous = float(series.iloc[-1]), float(series.iloc[-2])
    fields: dict[str, Any] = {
        "price": round(price, 2),
        "change": round(price - previous, 2),
        "change_pct": round((price / previous - 1) * 100, 2),
        "volume": int(volumes[symbol].dropna().iloc[-1]),
        **{k: round(v, 4) for k, v in risk_inputs(series, closes[MARKET]).items()},
        "as_of": series.index[-1].date().isoformat(),
        "source": source,
    }
    try:
        info = yahoo_info(symbol)
        fields.update(
            company_name=info.get("longName"),
            market_cap_b=round(info["marketCap"] / 1e9, 1) if info.get("marketCap") else None,
            pe_ratio=round(info["trailingPE"], 1) if info.get("trailingPE") else None,
            forward_pe=round(info["forwardPE"], 1) if info.get("forwardPE") else None,
        )
    except Exception as exc:
        logger.warning(f"Yahoo info unavailable for {symbol}: {exc}")
    if not fields.get("market_cap_b") or not fields.get("pe_ratio"):
        try:
            sec = sec_valuation_inputs(symbol)
            market_cap = price * sec["shares"]
            fields.setdefault("market_cap_b", None)
            fields["market_cap_b"] = fields["market_cap_b"] or round(market_cap / 1e9, 1)
            if not fields.get("pe_ratio") and sec["net_income_ttm"] > 0:
                fields["pe_ratio"] = round(market_cap / sec["net_income_ttm"], 1)
        except Exception as exc:
            logger.warning(f"SEC valuation inputs unavailable for {symbol}: {exc}")
    return fields


# ---------------------------------------------------------------- risk-free rate

TREASURY_CSV = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/"
    "{year}/all?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv"
)


def parse_treasury_csv(text: str) -> tuple[float, str] | None:
    """Latest 3-month yield (as a decimal) and its ISO date from a Treasury yield-curve CSV, or None if empty."""
    rows = [r for r in csv.DictReader(io.StringIO(text)) if r.get("3 Mo")]
    if not rows:
        return None
    latest = max(rows, key=lambda r: dt.datetime.strptime(r["Date"], "%m/%d/%Y"))
    return float(latest["3 Mo"]) / 100, dt.datetime.strptime(latest["Date"], "%m/%d/%Y").date().isoformat()


@ttl_cache(12 * 3600, failure_seconds=600)
def treasury_3m_yield() -> tuple[float, str]:
    """(yield, date) of the latest 3-month Treasury. Early January the current-year file can be empty."""
    today = dt.date.today()
    for year in (today.year, today.year - 1):
        text = _get(TREASURY_CSV.format(year=year), {"User-Agent": "apex-alpha (github.com/varadganjoo/apex-alpha)"})
        parsed = parse_treasury_csv(text.decode("utf-8-sig"))
        if parsed:
            return parsed
    raise RuntimeError("no Treasury yield rows for this year or last")


# ---------------------------------------------------------------- SEC EDGAR


def _sec_get(url: str) -> bytes:
    agent = os.getenv("SEC_USER_AGENT", "").strip()
    if "@" not in agent:
        raise RuntimeError("SEC_USER_AGENT must be set to a name and contact email (SEC fair-access policy).")
    return _get(url, {"User-Agent": agent})


@ttl_cache(6 * 3600, failure_seconds=300)
def company_facts(cik: int) -> dict:
    return json.loads(_sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"))


@ttl_cache(6 * 3600, failure_seconds=300)
def submissions(cik: int) -> dict:
    return json.loads(_sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json"))


def _days(start: str, end: str) -> int:
    return (dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days


def _facts(group: dict, tags: list[str], unit: str) -> list[dict]:
    """Facts for whichever tag has the most recent data (companies switch tags over the years)."""
    candidates = [group[t]["units"][unit] for t in tags if t in group and unit in group[t]["units"]]
    if not candidates:
        return []
    rows = max(candidates, key=lambda rs: max(r["end"] for r in rs))
    # The same period appears in several filings (comparatives, restatements); keep the latest filed.
    latest: dict[tuple, dict] = {}
    for r in rows:
        if r.get("form") not in ("10-Q", "10-K", "10-Q/A", "10-K/A"):
            continue
        key = (r.get("start"), r["end"])
        if key not in latest or r["filed"] > latest[key]["filed"]:
            latest[key] = r
    return list(latest.values())


def quarterly(group: dict, tags: list[str], unit: str = "USD") -> dict[str, float]:
    """Quarter-end date -> value for a duration concept.

    Uses reported three-month facts where they exist and otherwise differences year-to-date figures
    (10-Qs report cash flow year-to-date, and 10-Ks report no fourth quarter at all).
    """
    rows = [r for r in _facts(group, tags, unit) if r.get("start")]
    quarters: dict[str, float] = {}
    for r in rows:
        if 80 <= _days(r["start"], r["end"]) <= 100:
            quarters[r["end"]] = r["val"]
    by_start: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_start[r["start"]].append(r)
    for start, cumulative in by_start.items():
        previous: tuple[int, float] | None = None
        for r in sorted(cumulative, key=lambda x: x["end"]):
            days = _days(start, r["end"])
            n = round(days / 91.3)
            if not 1 <= n <= 4 or abs(days - 91.3 * n) > 25:
                continue
            if n == 1:
                quarters.setdefault(r["end"], r["val"])
            elif previous and previous[0] == n - 1:
                quarters.setdefault(r["end"], r["val"] - previous[1])
            previous = (n, r["val"])
    return dict(sorted(quarters.items()))


def latest_instant(group: dict, tags: list[str], unit: str = "USD") -> float | None:
    rows = [r for r in _facts(group, tags, unit) if not r.get("start")]
    return max(rows, key=lambda r: r["end"])["val"] if rows else None


def _year_ago(series: dict[str, float], end: str) -> float | None:
    for other, value in series.items():
        if 350 <= _days(other, end) <= 380:
            return value
    return None


def _ttm(series: dict[str, float], end: str) -> float | None:
    window = [v for e, v in series.items() if e <= end and _days(e, end) < 340]
    return sum(window) if len(window) == 4 else None


def latest_filing(cik: int) -> dict[str, str]:
    recent = submissions(cik)["filings"]["recent"]
    i = next(i for i, form in enumerate(recent["form"]) if form in ("10-Q", "10-K"))
    accession = recent["accessionNumber"][i]
    return {
        "form": recent["form"][i],
        "filed": recent["filingDate"][i],
        "period": recent["reportDate"][i],
        "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{recent['primaryDocument'][i]}",
    }


@ttl_cache(6 * 3600, failure_seconds=300)
def sec_fundamentals(symbol: str) -> dict[str, Any]:
    """FinancialMetrics fields for the latest reported quarter, plus the filing it came from."""
    cik = CIKS[symbol]
    g = company_facts(cik)["facts"]["us-gaap"]
    revenue = quarterly(g, REVENUE_TAGS)
    end = max(revenue)
    rev = revenue[end]
    gross = quarterly(g, ["GrossProfit"]).get(end)
    if gross is None and (cost := quarterly(g, ["CostOfRevenue", "CostOfGoodsAndServicesSold"]).get(end)) is not None:
        gross = rev - cost
    operating = quarterly(g, ["OperatingIncomeLoss"]).get(end)
    ocf = quarterly(g, ["NetCashProvidedByUsedInOperatingActivities"]).get(end)
    capex = quarterly(g, CAPEX_TAGS).get(end)
    prior = _year_ago(revenue, end)
    debt, equity = latest_instant(g, DEBT_TAGS), latest_instant(g, ["StockholdersEquity"])
    filing = latest_filing(cik)
    return {
        "fiscal_quarter": f"Quarter ended {end}",
        "revenue_b": round(rev / 1e9, 2),
        "revenue_growth_yoy_pct": round((rev / prior - 1) * 100, 1) if prior else 0.0,
        "gross_margin_pct": round(gross / rev * 100, 1) if gross is not None else 0.0,
        "operating_margin_pct": round(operating / rev * 100, 1) if operating is not None else 0.0,
        "free_cash_flow_b": round((ocf - capex) / 1e9, 2) if ocf is not None and capex is not None else 0.0,
        "debt_to_equity": round(debt / equity, 2) if debt and equity else 0.0,
        "sec_filing_ref": f"SEC Form {filing['form']} filed {filing['filed']} (period ended {filing['period']})",
        "filing_url": filing["url"],
        "source": "SEC EDGAR",
    }


def sec_valuation_inputs(symbol: str) -> dict[str, float]:
    facts = company_facts(CIKS[symbol])["facts"]
    net_income = quarterly(facts["us-gaap"], ["NetIncomeLoss"])
    shares = latest_instant(facts.get("dei", {}), ["EntityCommonStockSharesOutstanding"], unit="shares")
    ttm = _ttm(net_income, max(net_income))
    if not shares or ttm is None:
        raise RuntimeError("shares outstanding or trailing net income missing")
    return {"shares": shares, "net_income_ttm": ttm}


# ---------------------------------------------------------------- filing excerpts


class _TextExtractor(HTMLParser):
    """HTML to text with a line break at block elements. Skips scripts, styles, and the hidden inline-XBRL header."""

    BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "td"}
    SKIP = {"script", "style", "ix:header", "head"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            # Source newlines are just HTML whitespace; only block elements start a new line.
            self.parts.append(data.replace("\r", " ").replace("\n", " "))


def filing_lines(html: str) -> list[str]:
    parser = _TextExtractor()
    parser.feed(html)
    text = "".join(parser.parts).replace("\xa0", " ")
    return [line for line in (" ".join(raw.split()) for raw in text.split("\n")) if line]


def _section(lines: list[str], start: str, stop: str) -> list[str]:
    """Lines of the last heading matching `start` (the table of contents comes first) up to the next `stop` heading."""
    starts = [i for i, line in enumerate(lines) if len(line) < 150 and re.match(start, line, re.I)]
    if not starts:
        return []
    body = lines[starts[-1] + 1:]
    end = next((i for i, line in enumerate(body) if len(line) < 150 and re.match(stop, line, re.I)), len(body))
    return body[:end]


def _first_sentences(paragraph: str, limit: int = 420) -> str:
    """Leading sentences up to `limit` characters, cut at a sentence boundary so it stays a verbatim substring."""
    sentences = re.split(r"(?<=[.;])\s+(?=[A-Z(])", paragraph)
    if len(sentences) > 1 and paragraph[:1].islower():
        sentences = sentences[1:]  # paragraph split mid-sentence by a page break; drop the fragment
    out = sentences[0]
    for sentence in sentences[1:]:
        if len(out) + 1 + len(sentence) > limit:
            break
        out += " " + sentence
    return out


# (label, section, keywords that must appear in the excerpt itself, whether the excerpt must contain a figure)
EXCERPT_SLOTS = [
    ("revenue_trend", "mdna", ("revenue", "net sales"), True),
    ("margins", "mdna", ("gross margin",), True),
    ("liquidity", "mdna", ("cash",), True),
    ("demand_and_growth", "mdna", ("demand", "growth", "increase", "deliver"), True),
    ("key_risk", "risk", ("customer", "supply", "competition", "export", "regulat"), False),
]


def extract_excerpts(lines: list[str]) -> dict[str, str]:
    mdna = _section(lines, r"^item\s*[27]\.?\s*management", r"^item\s*(3|7a|8)\b")
    risk = _section(lines, r"^item\s*1a\.?\s*risk factors", r"^item\s*(1b|1c|2)\b")
    pools = {
        "mdna": [p for p in mdna if len(p) >= 120],
        "risk": [p for p in risk if len(p) >= 120 and "no material changes" not in p.lower()],
    }
    excerpts: dict[str, str] = {}
    used: set[str] = set()
    figure = re.compile(r"\d+(\.\d+)?\s?(%|percent|billion|million)")
    for label, pool, keywords, wants_figure in EXCERPT_SLOTS:
        # Prefer excerpts that quote a figure; some filers keep figures in tables and write qualitative prose.
        for require_figure in ([True, False] if wants_figure else [False]):
            match = next(
                (
                    (paragraph, excerpt)
                    for paragraph in pools[pool]
                    if paragraph not in used
                    and any(k in (excerpt := _first_sentences(paragraph)).lower() for k in keywords)
                    and (not require_figure or figure.search(excerpt))
                ),
                None,
            )
            if match:
                used.add(match[0])
                excerpts[label] = match[1]
                break
    return excerpts


@ttl_cache(6 * 3600, failure_seconds=300)
def sec_excerpts(symbol: str) -> tuple[dict[str, str], dict[str, str]]:
    filing = latest_filing(CIKS[symbol])
    raw = _sec_get(filing["url"])
    try:
        html = raw.decode("utf-8")
    except UnicodeDecodeError:
        html = raw.decode("cp1252", "replace")  # some filers still publish Windows-1252 HTML
    excerpts = extract_excerpts(filing_lines(html))
    if len(excerpts) < 2:
        raise RuntimeError(f"only {len(excerpts)} excerpts found in {filing['url']}")
    return excerpts, {**filing, "source": "SEC EDGAR"}
