"""Source parsers, exercised against recorded payload shapes. No network."""

from __future__ import annotations

import json
from datetime import date
from typing import ClassVar

import pandas as pd
import pytest
import responses

from aurex.data.sources.fred import CSV_ENDPOINT, FredLoader
from aurex.data.sources.ibja import IbjaReportLoader, parse_report
from aurex.data.sources.lbma import GOLD_PM_URL, LbmaGoldLoader
from aurex.data.sources.yahoo import YahooLoader

START, END = date(2020, 1, 1), date(2026, 12, 31)


@pytest.fixture(autouse=True)
def _no_courtesy_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 1s per-host delay is right in production and pointless in tests."""
    monkeypatch.setattr("aurex.data.sources.http.HTTP_COURTESY_DELAY", 0.0)


class TestFred:
    FRED_CSV = "observation_date,DFII10\n2026-07-27,1.85\n2026-07-28,.\n2026-07-29,1.88\n"

    @responses.activate
    def test_parses_csv_and_drops_missing_markers(self) -> None:
        """FRED writes '.' on non-trading days; those are absences, not zeros."""
        responses.add(
            responses.GET, CSV_ENDPOINT, body=self.FRED_CSV, status=200, content_type="text/csv"
        )
        loaded = FredLoader("real_yield_10y", "DFII10", "real_yield").fetch(START, END)

        assert list(loaded.frame.columns) == ["real_yield"]
        assert len(loaded.frame) == 2
        assert loaded.frame["real_yield"].tolist() == [1.85, 1.88]
        assert loaded.meta.source_name == "FRED:DFII10"
        assert loaded.meta.has_ohlc is False

    @responses.activate
    def test_records_the_url_it_actually_called(self) -> None:
        responses.add(responses.GET, CSV_ENDPOINT, body=self.FRED_CSV, status=200)
        loaded = FredLoader("real_yield_10y", "DFII10").fetch(START, END)
        assert "DFII10" in loaded.meta.source_url

    @responses.activate
    def test_empty_window_raises_so_the_chain_falls_through(self) -> None:
        responses.add(responses.GET, CSV_ENDPOINT, body=self.FRED_CSV, status=200)
        with pytest.raises(ValueError, match="no observations"):
            FredLoader("real_yield_10y", "DFII10").fetch(date(1990, 1, 1), date(1990, 12, 31))

    @responses.activate
    def test_malformed_csv_raises(self) -> None:
        responses.add(responses.GET, CSV_ENDPOINT, body="only_one_column\n1\n", status=200)
        with pytest.raises(ValueError, match="unexpected CSV shape"):
            FredLoader("x", "DFII10").fetch(START, END)


class TestLbma:
    PAYLOAD: ClassVar[list[dict[str, object]]] = [
        {"d": "1968-04-01", "v": [37.7, 15.68, None]},
        {"d": "2026-07-28", "v": [4022.2, 3023.89, 3537.12]},
        {"d": "2026-07-29", "v": [4000.85, 3009.44, 3511.95]},
    ]

    @responses.activate
    def test_parses_usd_fix(self) -> None:
        responses.add(responses.GET, GOLD_PM_URL, json=self.PAYLOAD, status=200)
        loaded = LbmaGoldLoader("xauusd").fetch(START, END)

        assert loaded.frame["close"].tolist() == [4022.2, 4000.85]
        assert loaded.meta.has_ohlc is False, "LBMA is close-only; vol layer must know"

    @responses.activate
    def test_skips_nulls_for_currencies_that_did_not_exist_yet(self) -> None:
        """The euro postdates 1968; those entries are null, not zero."""
        responses.add(responses.GET, GOLD_PM_URL, json=self.PAYLOAD, status=200)
        loaded = LbmaGoldLoader("xaueur", currency="EUR").fetch(date(1960, 1, 1), END)
        assert pd.Timestamp("1968-04-01") not in loaded.frame.index
        assert len(loaded.frame) == 2

    def test_unknown_currency_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="unsupported currency"):
            LbmaGoldLoader("x", currency="INR")

    @responses.activate
    def test_empty_payload_raises(self) -> None:
        responses.add(responses.GET, GOLD_PM_URL, json=[], status=200)
        with pytest.raises(ValueError, match="no usable observations"):
            LbmaGoldLoader("xauusd").fetch(START, END)


class TestIbjaParser:
    def test_extracts_the_fields_aurex_uses(self, ibja_text: str) -> None:
        parsed = parse_report(ibja_text)

        assert parsed["report_date"] == pd.Timestamp("2026-07-30")
        assert parsed["gold_999_am"] == 142469.0
        assert parsed["gold_999_pm"] == 142224.0
        assert parsed["spdr_gold_tonnes"] == 1008.73
        assert parsed["london_pm_usd"] == 4000.85

    def test_extracts_the_embedded_spot_history(self, ibja_text: str) -> None:
        """Each report carries a few days of history, which is how the observed
        series accumulates without hammering their server."""
        history = parse_report(ibja_text)["spot_history"]
        assert isinstance(history, list)
        as_dict = dict(history)
        assert as_dict[pd.Timestamp("2026-07-29")] == 142224.0
        assert as_dict[pd.Timestamp("2026-07-27")] == 144466.0

    def test_thousands_separators_are_handled(self, ibja_text: str) -> None:
        assert parse_report(ibja_text)["spdr_gold_tonnes"] == 1008.73

    def test_missing_fields_come_back_none_rather_than_raising(self) -> None:
        """IBJA's layout drifts; a partial report is still worth having."""
        parsed = parse_report("Daily Bullion Physical Market Report  Date: 1st June 2026")
        assert parsed["report_date"] == pd.Timestamp("2026-06-01")
        assert parsed["gold_999_pm"] is None
        assert parsed["spdr_gold_tonnes"] is None
        assert parsed["spot_history"] == []

    def test_unparseable_text_yields_all_none(self) -> None:
        parsed = parse_report("this is not a bullion report")
        assert parsed["report_date"] is None
        assert parsed["gold_999_am"] is None

    def test_report_url_uses_the_published_date_format(self) -> None:
        url = IbjaReportLoader.report_url(date(2026, 7, 30))
        assert url.endswith("30-07-2026.pdf")


class TestYahoo:
    def _frame(self, multiindex: bool) -> pd.DataFrame:
        index = pd.bdate_range("2026-07-01", periods=3, tz="UTC")
        frame = pd.DataFrame(
            {
                "Open": [4000.0, 4010.0, 4020.0],
                "High": [4050.0, 4060.0, 4070.0],
                "Low": [3990.0, 4000.0, 4010.0],
                "Close": [4040.0, 4050.0, 4060.0],
                "Adj Close": [4040.0, 4050.0, 4060.0],
                "Volume": [1, 2, 3],
            },
            index=index,
        )
        if multiindex:
            frame.columns = pd.MultiIndex.from_product([frame.columns, ["GC=F"]])
        return frame

    @pytest.mark.parametrize("multiindex", [False, True])
    def test_normalises_both_column_shapes(
        self, monkeypatch: pytest.MonkeyPatch, multiindex: bool
    ) -> None:
        """Recent yfinance returns a (field, ticker) MultiIndex even for one symbol."""
        import yfinance

        monkeypatch.setattr(yfinance, "download", lambda *a, **k: self._frame(multiindex))
        loaded = YahooLoader("xauusd", "GC=F").fetch(date(2026, 7, 1), date(2026, 7, 3))

        assert list(loaded.frame.columns) == ["open", "high", "low", "close"]
        assert loaded.meta.has_ohlc is True
        assert loaded.frame.index.tz is None, "index must be tz-naive to align with FRED"

    def test_empty_response_raises_so_the_chain_falls_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """This is the HTTP 429 path that made fallbacks necessary."""
        import yfinance

        monkeypatch.setattr(yfinance, "download", lambda *a, **k: pd.DataFrame())
        with pytest.raises(ValueError, match="no rows"):
            YahooLoader("xauusd", "GC=F").fetch(date(2026, 7, 1), date(2026, 7, 3))

    def test_missing_columns_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import yfinance

        partial = pd.DataFrame(
            {"Close": [1.0]}, index=pd.bdate_range("2026-07-01", periods=1, tz="UTC")
        )
        monkeypatch.setattr(yfinance, "download", lambda *a, **k: partial)
        with pytest.raises(ValueError, match="missing columns"):
            YahooLoader("xauusd", "GC=F").fetch(date(2026, 7, 1), date(2026, 7, 3))


class TestHttpPoliteness:
    def test_robots_disallow_is_respected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from aurex.data.sources import http

        monkeypatch.setattr(http, "robots_allows", lambda url: False)
        with pytest.raises(PermissionError, match=r"robots\.txt"):
            http.get("https://example.invalid/blocked")

    @responses.activate
    def test_non_2xx_raises(self) -> None:
        from aurex.data.sources import http

        responses.add(responses.GET, "https://example.invalid/x", status=503)
        with pytest.raises(Exception):  # noqa: B017 - requests raises HTTPError
            http.get("https://example.invalid/x", check_robots=False)

    @responses.activate
    def test_identifies_itself_by_user_agent(self) -> None:
        from aurex.config import USER_AGENT
        from aurex.data.sources import http

        responses.add(responses.GET, "https://example.invalid/x", body="ok", status=200)
        http.get("https://example.invalid/x", check_robots=False)
        assert responses.calls[0].request.headers["User-Agent"] == USER_AGENT
        assert "aurex" in USER_AGENT


def test_fixture_matches_recorded_lbma_shape() -> None:
    """Guards the assumption that v = [USD, GBP, EUR]."""
    record = json.loads('{"d": "2026-07-29", "v": [4000.85, 3009.44, 3511.95]}')
    assert record["v"][0] > record["v"][2] > record["v"][1]
