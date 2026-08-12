"""IBJA daily bullion report loader.

IBJA publishes a dated PDF at a predictable URL. Parsing it is far more stable than
scraping the homepage, whose rate block is rendered client-side, and it yields three
things Aurex needs from one fetch:

* ``gold_999_am`` / ``gold_999_pm`` — the India 24K reference rate, **Rs per 10g and
  exclusive of GST**. The exclusivity matters: comparing it against a GST-inclusive
  parity would manufacture a spurious ~-300bps premium.
* ``spdr_gold_tonnes`` — SPDR holdings, the ETF-flow proxy. SPDR's own published
  ``.csv`` endpoint now serves a PDF, so this report is the working route to it.
* ``london_pm_usd`` — the LBMA PM fix, an independent cross-check on the gold series.

**History accumulates.** Each report carries only about four days of spot history, so
a fresh clone starts with a short observed window that the nightly job extends. Aurex
never backfills observed IBJA rates with computed parity — doing so would make
``local_premium_bps`` identically zero and fabricate the exact signal it is meant to
measure. Where there is no observed rate there is no premium.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

import pandas as pd

from aurex.data.base import LoadedSeries, SourceCitation, build_meta
from aurex.data.sources import http

REPORT_URL = "https://www.ibja.co/Upload/IBJA_Bullion Daily Report - {stamp}.pdf"

#: "Gold 999 142469 142224" -> AM, PM in Rs/10g.
_PURITY_ROW = re.compile(r"Gold\s+999\s+([\d,]+)\s+([\d,]+)")
#: "29th July 2026 142224 217335" from the Daily India Spot Market Rates block.
_SPOT_ROW = re.compile(r"(\d{1,2})(?:st|nd|rd|th)\s+([A-Z][a-z]+)\s+(\d{4})\s+([\d,]+)\s+([\d,]+)")
#: "SPDR Gold 1,008.73 -0.57"
_SPDR_ROW = re.compile(r"SPDR\s+Gold\s+([\d,]+\.\d+)\s+(-?[\d,]+\.\d+)")
#: "Gold London PM Fix($/oz) 4000.85"
_LONDON_PM = re.compile(r"Gold\s+London\s+PM\s+Fix\(\$/oz\)\s+([\d,]+\.?\d*)")
#: "Date: 30th July 2026"
_REPORT_DATE = re.compile(r"Date:\s*(\d{1,2})(?:st|nd|rd|th)\s+([A-Z][a-z]+)\s+(\d{4})")


def _to_float(text: str) -> float:
    return float(text.replace(",", ""))


def _to_date(day: str, month: str, year: str) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(f"{year}-{month}-{day}")
    except ValueError:
        return None


def parse_report(text: str) -> dict[str, object]:
    """Extract the fields Aurex uses from one report's extracted text.

    Returns a dict with ``report_date``, ``gold_999_am``, ``gold_999_pm``,
    ``spdr_gold_tonnes``, ``london_pm_usd`` and ``spot_history`` (a list of
    ``(date, gold_per_10g)``). Missing fields come back as ``None`` rather than
    raising — IBJA's layout drifts, and a partial report is still useful.
    """
    out: dict[str, object] = {
        "report_date": None,
        "gold_999_am": None,
        "gold_999_pm": None,
        "spdr_gold_tonnes": None,
        "london_pm_usd": None,
        "spot_history": [],
    }

    if match := _REPORT_DATE.search(text):
        out["report_date"] = _to_date(*match.groups())
    if match := _PURITY_ROW.search(text):
        out["gold_999_am"] = _to_float(match.group(1))
        out["gold_999_pm"] = _to_float(match.group(2))
    if match := _SPDR_ROW.search(text):
        out["spdr_gold_tonnes"] = _to_float(match.group(1))
    if match := _LONDON_PM.search(text):
        out["london_pm_usd"] = _to_float(match.group(1))

    history: list[tuple[pd.Timestamp, float]] = []
    for day, month, year, gold, _silver in _SPOT_ROW.findall(text):
        stamp = _to_date(day, month, year)
        if stamp is not None:
            history.append((stamp, _to_float(gold)))
    out["spot_history"] = history
    return out


class IbjaReportLoader:
    """Fetch and parse recent IBJA daily reports.

    Args:
        series_id: Aurex's internal series name.
        max_reports: How many recent publication dates to try. Each PDF is ~1MB and
            the session enforces a courtesy delay, so this is deliberately small;
            history accumulates in the cache across nightly runs instead.
    """

    source_name = "IBJA:daily-bullion-report"

    #: Primary: the association publishes its own reference rate. The URL cited is the
    #: report index rather than any one dated PDF, because ``source_url`` on the meta
    #: already records the exact report a given fetch read.
    citation = SourceCitation(
        source_url="https://www.ibja.co/",
        source_confidence="primary",
    )

    def __init__(self, series_id: str = "ibja_gold", max_reports: int = 5) -> None:
        self.series_id = series_id
        self.max_reports = max_reports

    @staticmethod
    def report_url(day: date) -> str:
        return REPORT_URL.format(stamp=day.strftime("%d-%m-%Y"))

    def _fetch_one(self, day: date) -> dict[str, object] | None:
        from pypdf import PdfReader

        try:
            response = http.get(self.report_url(day))
        except Exception:
            # No report published that day (holiday, or not posted yet). Not an
            # error — the caller walks back to the previous publication date.
            return None

        import io

        try:
            reader = PdfReader(io.BytesIO(response.content))
            text = reader.pages[0].extract_text() or ""
        except Exception:
            # A malformed or non-PDF response is treated as no report rather than
            # taking down the run; the chain records the gap.
            return None
        return parse_report(text)

    def fetch(self, start: date, end: date) -> LoadedSeries:
        rows: dict[pd.Timestamp, dict[str, float]] = {}
        attempted = 0
        cursor = end

        while attempted < self.max_reports and cursor >= start:
            # IBJA does not publish on weekends; skip without spending a request.
            if cursor.weekday() >= 5:
                cursor -= timedelta(days=1)
                continue

            attempted += 1
            parsed = self._fetch_one(cursor)
            cursor -= timedelta(days=1)
            if parsed is None:
                continue

            report_date = parsed["report_date"]
            if isinstance(report_date, pd.Timestamp):
                entry = rows.setdefault(report_date, {})
                for field in ("gold_999_am", "gold_999_pm", "spdr_gold_tonnes", "london_pm_usd"):
                    value = parsed[field]
                    if isinstance(value, float):
                        entry[field] = value

            # The embedded spot table extends history a few days per report.
            history = parsed["spot_history"]
            if isinstance(history, list):
                for stamp, gold in history:
                    rows.setdefault(stamp, {}).setdefault("gold_999_pm", gold)

        if not rows:
            raise ValueError(
                f"{self.source_name}: no reports parsed in {start}..{end} "
                f"({attempted} publication dates tried)"
            )

        frame = pd.DataFrame.from_dict(rows, orient="index").sort_index()
        frame.index.name = "date"
        frame = frame.loc[str(start) : str(end)]
        if frame.empty:
            raise ValueError(f"{self.source_name}: no rows within {start}..{end}")

        return LoadedSeries(
            frame=frame,
            meta=build_meta(
                series_id=self.series_id,
                source_name=self.source_name,
                source_url=self.report_url(end),
                citation=self.citation,
                frame=frame,
            ),
        )
