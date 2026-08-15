"""The guard against a forecast dated today built on last week's prices.

This is the failure mode the nightly job exists to prevent, so these tests are written
against the *fabrication*, not against the plumbing: each one describes a way a real
run could quietly publish a stale price, and asserts that it refuses instead.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from aurex.assets import REGISTRY
from aurex.assets.gold import GOLD
from aurex.cli import app
from aurex.data.base import LoadedSeries, SeriesMeta
from aurex.data.cache import CacheStore
from aurex.data.freshness import (
    _REFUSAL_PHRASE,
    EMPTY,
    FRESH,
    STALE,
    UNAVAILABLE,
    UNDECLARED,
    SeriesFreshness,
    StaleDataError,
    assess,
)
from aurex.data.registry import blocking_series, freshness_for
from aurex.pipeline import run
from tests.conftest import TEST_CITATION, make_series

RUN_DATE = date(2026, 8, 5)

TOLERANCE = SeriesFreshness(
    max_lag_days=4,
    calendar="test calendar",
    rationale="a tolerance for tests, declared like any other",
)

runner = CliRunner()


def series_ending(
    end: str,
    *,
    series_id: str = "xauusd",
    periods: int = 30,
    column: str = "close",
    trailing_nans: int = 0,
    seed: int = 4,
) -> LoadedSeries:
    """A series whose index ends on ``end``, optionally with a NaN tail.

    Prices wander rather than sitting flat. A constant series has zero variance, and
    everything downstream of a variance — the GARCH fit, the copula, the standardised
    residuals — divides by it, so a flat fixture fails the pipeline with a linear
    algebra error instead of exercising the freshness question under test.
    """
    index = pd.bdate_range(end=end, periods=periods, name="date")
    rng = np.random.default_rng(seed)
    values = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=len(index))))
    if trailing_nans:
        values[-trailing_nans:] = np.nan
    frame = pd.DataFrame({column: values}, index=index)
    return LoadedSeries(
        frame=frame,
        meta=SeriesMeta(
            series_id=series_id,
            source_name="test:source",
            source_url="https://example.invalid/series",
            citation=TEST_CITATION,
            fetched_at=datetime(2026, 8, 5, tzinfo=UTC),
            rows=len(frame),
            start=index[0].date(),
            end=index[-1].date(),
        ),
    )


def verdict_for(
    loaded: dict[str, LoadedSeries],
    *,
    policies: dict[str, SeriesFreshness | None] | None = None,
    blocking: frozenset[str] = frozenset({"xauusd"}),
    unavailable: dict[str, str] | None = None,
):
    return assess(
        series=loaded,
        unavailable=unavailable or {},
        policies=policies if policies is not None else dict.fromkeys(loaded, TOLERANCE),
        blocking=blocking,
        run_date=RUN_DATE,
    )


class TestTheRefusal:
    def test_a_price_series_reaching_the_run_date_publishes(self) -> None:
        verdict = verdict_for({"xauusd": series_ending("2026-08-04")})

        assert verdict.publishable
        assert verdict.ages[0].verdict == FRESH
        assert verdict.ages[0].lag_days == 1

    def test_a_stale_price_series_refuses(self) -> None:
        """The headline. Seven days behind against a four-day tolerance."""
        verdict = verdict_for({"xauusd": series_ending("2026-07-29")})

        assert not verdict.publishable
        assert verdict.ages[0].verdict == STALE
        assert verdict.ages[0].lag_days == 7

    def test_the_refusal_raises_rather_than_returning_a_flag(self) -> None:
        """A flag can be ignored by the caller; the fabrication cannot be un-published."""
        verdict = verdict_for({"xauusd": series_ending("2026-07-29")})

        with pytest.raises(StaleDataError) as excinfo:
            verdict.raise_if_stale()

        message = str(excinfo.value)
        assert "xauusd" in message
        assert "2026-08-05" in message, "the refusal must name the date it refused to publish"

    def test_the_tuesday_after_a_monday_holiday_still_publishes(self) -> None:
        """The case that sets the tolerance, asserted on the real calendar.

        A 02:00 UTC run on Tuesday 11 August, after a Monday holiday, reads Friday the
        7th's fix — four days behind and entirely healthy. A guard that fired here
        would be switched off within a week, so the boundary is tested at the exact
        date that motivated the number rather than at a synthetic one.
        """
        verdict = assess(
            series={"xauusd": series_ending("2026-08-07")},
            unavailable={},
            policies={"xauusd": TOLERANCE},
            blocking=frozenset({"xauusd"}),
            run_date=date(2026, 8, 11),
        )

        assert verdict.ages[0].last_observation == date(2026, 8, 7)
        assert verdict.ages[0].lag_days == 4
        assert verdict.publishable, "four days behind is at the tolerance, not past it"


class TestWhatBlocksAndWhatDoesNot:
    def test_a_stale_factor_does_not_block(self) -> None:
        """A stale factor moves a loading. A stale price fabricates a published number."""
        verdict = verdict_for(
            {"xauusd": series_ending("2026-08-04"), "dxy": series_ending("2026-07-20")},
            blocking=frozenset({"xauusd"}),
        )

        stale = next(age for age in verdict.ages if age.series_id == "dxy")
        assert stale.verdict == STALE
        assert not stale.blocks_publication
        assert verdict.publishable, "a non-blocking series must never refuse the run"

    def test_the_exchange_rate_of_a_published_lens_blocks(self) -> None:
        """A stale rate over a fresh fix publishes a rupee price that was never quoted."""
        verdict = verdict_for(
            {
                "xauusd": series_ending("2026-08-04"),
                "usdinr": series_ending("2026-07-25", series_id="usdinr"),
            },
            blocking=frozenset({"xauusd", "usdinr"}),
        )

        assert not verdict.publishable
        assert [age.series_id for age in verdict.failures] == ["usdinr"]

    def test_gold_declares_both_its_price_and_its_fx_as_blocking(self) -> None:
        assert blocking_series([GOLD]) == frozenset({"xauusd", "usdinr"})


class TestFailingClosed:
    def test_an_undeclared_blocking_series_refuses(self) -> None:
        """An undeclared tolerance is not a permissive one.

        Otherwise a series added without a policy inherits "anything goes" and the
        guard silently stops covering it — the exact rot this is meant to prevent.
        """
        verdict = verdict_for(
            {"xauusd": series_ending("2026-08-04")},
            policies={"xauusd": None},
        )

        assert not verdict.publishable
        assert verdict.ages[0].verdict == UNDECLARED

    def test_an_unavailable_blocking_series_refuses(self) -> None:
        verdict = verdict_for({}, unavailable={"xauusd": "every source declined"})

        assert not verdict.publishable
        assert verdict.ages[0].verdict == UNAVAILABLE
        assert "every source declined" in verdict.ages[0].detail

    def test_an_index_reaching_today_over_a_nan_close_is_not_fresh(self) -> None:
        """The partial-fetch vector: the frame's dates arrive, the prices do not.

        Judging freshness on the index alone would pass this, and the run would then
        price from whatever the last real observation was — which is the fabrication
        wearing a fresh timestamp.
        """
        verdict = verdict_for({"xauusd": series_ending("2026-08-04", trailing_nans=5)})

        assert not verdict.publishable
        assert verdict.ages[0].last_observation == date(2026, 7, 28)


class TestTheReasonARefusalIsFiledUnder:
    """The line a reader acts on, and it used to name the wrong cause.

    Four public skip records — 2026-08-06, 08-08, 08-10 and 08-11 — say the price
    series "did not reach the run date within its declared tolerance" when in fact it
    never resolved at all. The distinction matters because of what each one licenses:
    a stale series invites a look at the tolerance, and on those four nights widening
    the tolerance would have published nothing truer while retiring the signal that
    the only source for the anchor series had failed five times in eight days.
    """

    def test_an_unavailable_series_is_not_reported_as_stale(self) -> None:
        verdict = verdict_for({}, unavailable={"xauusd": "every source declined"})

        reason = verdict.refusal_reason()
        assert "xauusd" in reason
        assert "did not resolve" in reason
        assert "tolerance" not in reason, (
            "an unavailable series has no lag to compare against a tolerance, and "
            "naming one points the reader at the repair that must not be made"
        )

    def test_a_stale_series_is_still_reported_as_stale(self) -> None:
        """The other half. A reason that never says 'tolerance' is as wrong as one
        that always does."""
        verdict = verdict_for({"xauusd": series_ending("2026-07-20")})

        assert verdict.ages[0].verdict == STALE
        assert "did not reach the run date" in verdict.refusal_reason()

    def test_each_verdict_reads_differently(self) -> None:
        """Mechanically: four distinct verdicts, four distinct sentences.

        Asserted on the phrases rather than on one worked example, because the bug was
        that two causes collapsed into one string and nothing noticed.
        """
        phrases = {
            verdict: _REFUSAL_PHRASE[verdict] for verdict in (STALE, UNAVAILABLE, EMPTY, UNDECLARED)
        }
        assert len(set(phrases.values())) == len(phrases), phrases

    def test_one_outage_across_several_series_reads_as_one_outage(self) -> None:
        verdict = verdict_for(
            {},
            unavailable={"xauusd": "every source declined", "usdinr": "every source declined"},
            policies={"xauusd": TOLERANCE, "usdinr": TOLERANCE},
            blocking=frozenset({"xauusd", "usdinr"}),
        )

        reason = verdict.refusal_reason()
        assert reason.count("did not resolve") == 1, reason
        assert "usdinr, xauusd" in reason

    def test_mixed_causes_are_both_named(self) -> None:
        verdict = verdict_for(
            {"usdinr": series_ending("2026-07-20", series_id="usdinr")},
            unavailable={"xauusd": "every source declined"},
            policies={"xauusd": TOLERANCE, "usdinr": TOLERANCE},
            blocking=frozenset({"xauusd", "usdinr"}),
        )

        reason = verdict.refusal_reason()
        assert "usdinr: did not reach the run date" in reason
        assert "xauusd: did not resolve" in reason

    def test_a_publishable_verdict_has_no_refusal_to_explain(self) -> None:
        verdict = verdict_for({"xauusd": series_ending("2026-08-04")})

        assert verdict.publishable
        with pytest.raises(ValueError, match="no refusal"):
            verdict.refusal_reason()


class TestTheDeclarationsThemselves:
    def test_every_production_series_declares_a_tolerance(self) -> None:
        """The anti-rot test: a new series cannot ship without a freshness policy.

        Without this, the guard degrades one series at a time and nothing fails until
        a night that should have refused does not.
        """
        undeclared = [
            series_id
            for series_id, policy in freshness_for(REGISTRY.values()).items()
            if policy is None
        ]

        assert not undeclared, f"series with no declared staleness tolerance: {undeclared}"

    def test_every_tolerance_carries_a_calendar_and_a_rationale(self) -> None:
        """A bare number cannot be argued with, so it never gets revisited."""
        for series_id, policy in freshness_for(REGISTRY.values()).items():
            assert policy is not None
            assert policy.calendar.strip(), f"{series_id} declares no publication calendar"
            assert len(policy.rationale) > 40, f"{series_id}'s rationale explains nothing"

    def test_a_tolerance_without_a_rationale_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="calendar and rationale"):
            SeriesFreshness(max_lag_days=4, calendar="", rationale="")

    def test_a_negative_tolerance_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            SeriesFreshness(max_lag_days=-1, calendar="c", rationale="r")


class TestTheArtifactCitesIt:
    def test_the_verdict_block_carries_the_tolerance_that_licensed_publication(self) -> None:
        """A reader deciding whether to trust a forecast needs the rule it passed."""
        block = verdict_for({"xauusd": series_ending("2026-08-04")}).describe()

        assert block["run_date"] == "2026-08-05"
        assert block["publishable"] is True
        entry = block["series"][0]
        assert entry["tolerance"]["max_lag_days"] == 4
        assert entry["tolerance"]["calendar"] == "test calendar"
        assert entry["lag_days"] == 1

    def test_the_pipeline_artifact_carries_the_freshness_block(self, cache: CacheStore) -> None:
        """A published forecast must cite the freshness rule it satisfied."""
        end = date(2026, 3, 31)
        cache.write(make_series("xauusd", start="2026-01-01", periods=64, value=4000.0))
        cache.write(make_series("usdinr", start="2026-01-01", periods=64, value=96.5))

        artifact = run(offline=True, start=date(2026, 1, 1), end=end, cache=cache).artifact

        block = artifact["freshness"]
        assert block["run_date"] == end.isoformat()
        judged = {entry["series_id"]: entry for entry in block["series"]}
        assert judged["xauusd"]["tolerance"]["max_lag_days"] == 4
        assert "London business days" in judged["xauusd"]["tolerance"]["calendar"]
        assert "policy" in block


class TestTheNightlyExit:
    def test_a_stale_run_exits_non_zero_and_writes_no_forecast(
        self, stale_cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Not a warning, not a flag in the artifact — a non-zero exit and no artifact."""
        target = tmp_path / "public-data"
        monkeypatch.setattr("aurex.config.PUBLIC_DATA_DIR", target)

        result = runner.invoke(app, ["pipeline", "--offline"])

        assert result.exit_code == 1, result.output
        assert not (target / "latest.json").exists(), "a refused run must publish nothing"
        assert not list((target / "forecasts").glob("20*.json"))

    def test_a_refused_night_still_leaves_a_trace(
        self, stale_cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The refusal is itself data. A gap nothing explains is the weaker position."""
        target = tmp_path / "public-data"
        monkeypatch.setattr("aurex.config.PUBLIC_DATA_DIR", target)

        runner.invoke(app, ["pipeline", "--offline"])

        skipped = list((target / "forecasts" / "skipped").glob("*.json"))
        assert len(skipped) == 1
        record = json.loads(skipped[0].read_text())
        assert "tolerance" in record["reason"]
        assert record["detail"]["series"], "the skip record must carry what it measured"

        # Named series and a verdict that matches the detail block. A hardcoded reason
        # passes the assertion above and fails these two, which is how the four records
        # of 2026-08-06 through 08-11 came to describe a cause that had not occurred.
        blocked = [
            age for age in record["detail"]["series"] if age["blocking"] and age["verdict"] != FRESH
        ]
        assert blocked, "this fixture is supposed to block"
        for age in blocked:
            assert age["series_id"] in record["reason"], record["reason"]
            assert _REFUSAL_PHRASE[age["verdict"]] in record["reason"], record["reason"]

        # And only those. This fixture also leaves dxy and wti unresolved, but neither
        # blocks publication, so neither is a reason the night was skipped. A reason
        # listing every series that was merely unwell buries the one that refused.
        for age in record["detail"]["series"]:
            if not age["blocking"]:
                assert age["series_id"] not in record["reason"], record["reason"]

    def test_a_fresh_run_publishes(
        self, fresh_cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other half of the guard, and the half that keeps it honest.

        A guard asserted only against stale data would still pass if it refused
        everything, which is the failure mode that gets it disabled rather than fixed.
        """
        monkeypatch.setattr("aurex.config.PUBLIC_DATA_DIR", tmp_path / "public-data")

        result = runner.invoke(app, ["pipeline", "--offline"])

        assert result.exit_code == 0, result.output
        assert (tmp_path / "public-data" / "latest.json").exists()

    def test_allow_stale_is_opt_in_and_publishes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Local exploration from the committed seed cache must stay possible."""
        target = tmp_path / "public-data"
        monkeypatch.setattr("aurex.config.PUBLIC_DATA_DIR", target)

        result = runner.invoke(app, ["pipeline", "--offline", "--allow-stale"])

        assert result.exit_code == 0, result.output
        assert (target / "latest.json").exists()
