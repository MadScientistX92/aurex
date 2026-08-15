"""Source parsers, exercised against recorded payload shapes. No network."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest
import responses

from aurex.data.base import LoadedSeries, Loader, SourceCitation
from aurex.data.schedules.provenance import VALID_CONFIDENCE
from aurex.data.sources.fred import CSV_ENDPOINT, FredLoader
from aurex.data.sources.gpr import DAILY_XLS_URL, GprDailyLoader, parse_daily
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


class TestGeopoliticalRisk:
    """The index the driver set declared and went without until step 4.

    The parse is tested against a recorded slice of the authors' own workbook rather
    than a hand-written frame, because the two things that can break here are both
    properties of the real file: the columns it names, and the data dictionary it keeps
    in trailing columns beside the observations.
    """

    @pytest.fixture
    def sheet(self, fixture_dir: Path) -> pd.DataFrame:
        return pd.read_csv(fixture_dir / "gpr_daily_2026-08-10.csv")

    def test_parses_the_three_series_it_reads(self, sheet: pd.DataFrame) -> None:
        frame = parse_daily(sheet)

        assert list(frame.columns) == ["gpr", "gpr_threats", "gpr_acts"]
        assert isinstance(frame.index, pd.DatetimeIndex)
        assert frame.index[0] == pd.Timestamp("1985-01-01")
        assert frame.index[-1] == pd.Timestamp("2026-08-10")
        assert frame["gpr"].iloc[-1] == pytest.approx(154.9739227294922)

    def test_the_data_dictionary_never_reaches_a_regressor(self, sheet: pd.DataFrame) -> None:
        """``var_name``/``var_label`` sit beside the observations, not under them."""
        assert {"var_name", "var_label"} <= set(sheet.columns)
        frame = parse_daily(sheet)

        assert not {"var_name", "var_label"} & set(frame.columns)
        assert frame.notna().all().all(), "a dictionary cell leaked into a numeric column"

    def test_the_index_covers_weekends(self, sheet: pd.DataFrame) -> None:
        """Calendar-daily, which is why weekly aggregation is a mean and not a last."""
        frame = parse_daily(sheet)
        assert pd.Timestamp("1985-01-05").dayofweek == 5
        assert pd.Timestamp("1985-01-05") in frame.index

    @pytest.mark.parametrize("column", ["date", "GPRD", "GPRD_THREAT", "GPRD_ACT"])
    def test_a_renamed_column_fails_loudly(self, sheet: pd.DataFrame, column: str) -> None:
        """Schema drift must raise, so the chain falls through to the cache."""
        with pytest.raises(ValueError, match=column):
            parse_daily(sheet.drop(columns=[column]))

    @responses.activate
    def test_fetch_windows_and_carries_the_citation(
        self, sheet: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The workbook is binary, so only its decoding is stubbed — not the parse."""
        responses.add(responses.GET, DAILY_XLS_URL, body=b"<xls bytes>", status=200)
        monkeypatch.setattr(pd, "read_excel", lambda *a, **k: sheet)

        loaded = GprDailyLoader("gpr").fetch(date(2026, 1, 1), date(2026, 12, 31))

        assert loaded.frame.index[0] == pd.Timestamp("2026-08-05"), "window not applied"
        assert loaded.meta.source_url == DAILY_XLS_URL
        assert loaded.meta.citation.source_confidence == "primary"
        assert "Caldara" in (loaded.meta.citation.cite_as or "")
        assert loaded.meta.has_ohlc is False

    @responses.activate
    def test_an_empty_window_raises_rather_than_serving_nothing(
        self, sheet: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        responses.add(responses.GET, DAILY_XLS_URL, body=b"<xls bytes>", status=200)
        monkeypatch.setattr(pd, "read_excel", lambda *a, **k: sheet)

        with pytest.raises(ValueError, match="no observations"):
            GprDailyLoader("gpr").fetch(date(2000, 1, 1), date(2000, 12, 31))

    @pytest.mark.network
    def test_the_published_workbook_still_has_the_columns_we_read(self) -> None:
        """The one thing a recorded fixture cannot tell us: whether the file still exists."""
        loaded = GprDailyLoader("gpr").fetch(date(2026, 1, 1), date(2026, 12, 31))
        assert list(loaded.frame.columns) == ["gpr", "gpr_threats", "gpr_acts"]
        assert loaded.frame["gpr"].gt(0).all()


class TestEverySourceCitesItself:
    """The provenance rule, applied to series the way it is applied to schedules.

    Every dated schedule entry has carried its own ``source_url`` and
    ``source_confidence`` since the duty table existed. Series were the last external
    facts here not held to that rule; these assert the rule now reaches them, and that
    it is enforced at the boundary rather than by a convention each loader remembers.
    """

    def _loaders(self) -> list[object]:
        from aurex.assets import GOLD
        from aurex.assets.synthetic import SYNTHETIC
        from aurex.data.macro import macro_chains

        chains = dict(macro_chains())
        chains |= GOLD.price_sources()
        chains |= SYNTHETIC.price_sources()
        return [loader for chain in chains.values() for loader in chain.loaders]

    def test_the_guard_is_looking_at_something(self) -> None:
        assert len(self._loaders()) > 5

    def test_every_registered_loader_declares_a_citation(self) -> None:
        for loader in self._loaders():
            citation = loader.citation  # type: ignore[attr-defined]
            assert citation.source_url.startswith("http"), loader
            assert citation.source_confidence in VALID_CONFIDENCE, loader

    def test_a_loader_without_a_citation_is_not_a_loader(self) -> None:
        """Why the citation is a protocol member and not something read with getattr.

        Under duck typing this class serves numbers with no recorded confidence and
        nothing anywhere fails. As a protocol member it is rejected at the boundary.
        """

        class UncitedLoader:
            series_id = "x"
            source_name = "test:uncited"

            def fetch(self, start: date, end: date) -> LoadedSeries:
                raise NotImplementedError

        assert not isinstance(UncitedLoader(), Loader)
        assert isinstance(FredLoader("x", "DFII10"), Loader)

    def test_a_confidence_outside_the_vocabulary_is_refused(self) -> None:
        with pytest.raises(ValueError, match="source_confidence"):
            SourceCitation(source_url="https://example.invalid", source_confidence="probably")  # type: ignore[arg-type]

    def test_a_citation_without_a_url_cites_nothing(self) -> None:
        with pytest.raises(ValueError, match="cites nothing"):
            SourceCitation(source_url="", source_confidence="primary")

    def test_redistributors_and_publishers_are_told_apart(self) -> None:
        """The label records how many hands the number passed through, nothing more."""
        from aurex.data.sources import LbmaGoldLoader

        assert FredLoader("wti", "DCOILWTICO").citation.source_confidence == "secondary"
        assert LbmaGoldLoader("xauusd").citation.source_confidence == "primary"


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


class TestJsonDecodeDiagnostics:
    """What came back, when what came back was not JSON.

    A 2xx carrying an HTML challenge page and a 2xx carrying an empty body raise the
    identical ``JSONDecodeError`` — "Expecting value: line 1 column 1 (char 0)" — and
    need opposite repairs. The LBMA fetch failed this way five times in eight days and
    not one of those records can say which it was, because the body was discarded.
    """

    @responses.activate
    def test_a_challenge_page_is_distinguishable_from_an_empty_body(self) -> None:
        from aurex.data.sources import http

        responses.add(
            responses.GET,
            "https://example.invalid/challenge",
            body="<!DOCTYPE html><html><head><title>Just a moment...</title>",
            status=200,
            content_type="text/html",
            headers={"cf-ray": "a2aa1371ee7d3c06-BLR", "cf-mitigated": "challenge"},
        )
        responses.add(responses.GET, "https://example.invalid/empty", body="", status=200)

        with pytest.raises(http.NonJsonResponseError) as challenge:
            http.get_json("https://example.invalid/challenge", check_robots=False)
        with pytest.raises(http.NonJsonResponseError) as empty:
            http.get_json("https://example.invalid/empty", check_robots=False)

        assert "DOCTYPE html" in str(challenge.value)
        assert "cf-mitigated=challenge" in str(challenge.value)
        assert "cf-ray=a2aa1371ee7d3c06-BLR" in str(challenge.value)
        assert "0 bytes" in str(empty.value)
        assert str(challenge.value) != str(empty.value), (
            "the two failures that need opposite fixes must not read identically"
        )

    @responses.activate
    def test_the_excerpt_is_bounded_and_escaped(self) -> None:
        """It is unknown-encoding text heading for a JSON skip record.

        A raw newline or quote landing mid-record is a second defect on top of the one
        being recorded, and an unbounded body would put 913KB of HTML in a git commit.
        """
        from aurex.data.sources import http

        responses.add(
            responses.GET,
            "https://example.invalid/noisy",
            body='<html>\n"quoted"\n' + "x" * 5_000,
            status=200,
        )

        with pytest.raises(http.NonJsonResponseError) as caught:
            http.get_json("https://example.invalid/noisy", check_robots=False)

        message = str(caught.value)
        assert len(message) < 600, "the excerpt must fit in a skip record"
        assert "\n" not in message
        assert "5016 bytes" in message, "the full size is reported even when excerpted"

    @responses.activate
    def test_valid_json_is_returned_unchanged(self) -> None:
        """The guard must not be the reason a healthy fetch fails."""
        from aurex.data.sources import http

        responses.add(
            responses.GET,
            "https://example.invalid/ok",
            json=[{"d": "2026-07-29", "v": [4000.85, 3009.44, 3511.95]}],
            status=200,
        )

        payload = http.get_json("https://example.invalid/ok", check_robots=False)
        assert payload[0]["v"][0] == 4000.85

    @responses.activate
    def test_the_lbma_loader_reports_the_body_through_the_chain(self, tmp_path: Path) -> None:
        """End to end: the description must survive into what a skip record files.

        The chain formats a failure as ``{source}: {type}: {exc}``, so a diagnostic
        that is not on the exception's own message never reaches the record. This is
        the exact path that produced the four uninformative skip records.
        """
        from aurex.data.base import DataUnavailableError
        from aurex.data.cache import CacheStore
        from aurex.data.chain import SourceChain

        responses.add(
            responses.GET,
            GOLD_PM_URL,
            body="<!DOCTYPE html><html><title>Just a moment...</title>",
            status=200,
            content_type="text/html",
        )

        chain = SourceChain("xauusd", (LbmaGoldLoader("xauusd"),), cache=CacheStore(tmp_path))
        with pytest.raises(DataUnavailableError) as caught:
            chain.load(START, END)

        message = str(caught.value)
        assert "NonJsonResponseError" in message
        assert "DOCTYPE html" in message
        assert "content-type=text/html" in message


def test_fixture_matches_recorded_lbma_shape() -> None:
    """Guards the assumption that v = [USD, GBP, EUR]."""
    record = json.loads('{"d": "2026-07-29", "v": [4000.85, 3009.44, 3511.95]}')
    assert record["v"][0] > record["v"][2] > record["v"][1]
