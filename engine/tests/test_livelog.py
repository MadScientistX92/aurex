"""The live log, and the separation it exists to maintain.

The claim the live log makes is stronger than the backtest's, and it will be tiny for
months. Both facts are load-bearing: the first is why it is published at all, and the
second is why it must never be quietly topped up with walk-forward observations to
reach a testable count sooner.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aurex.livelog import MIN_INDEPENDENT_FOR_TEST, collect, summarise, write_live_log
from aurex.record import write_forecast_log


def artifact(
    *,
    generated_at: str,
    as_of: str,
    anchor: float = 4000.0,
    horizons: tuple[int, ...] = (5, 21),
) -> dict[str, Any]:
    return {
        "generated_at": generated_at,
        "code": {"commit": "deadbeef"},
        "assets": {
            "gold": {
                "lenses": {
                    "USD": {
                        "latest": {"as_of": as_of},
                        "distribution": {
                            "available": True,
                            "anchor": anchor,
                            "horizons": {
                                str(h): {
                                    "quantiles": {
                                        "q05": anchor * 0.90,
                                        "q25": anchor * 0.96,
                                        "q50": anchor,
                                        "q75": anchor * 1.04,
                                        "q95": anchor * 1.10,
                                    }
                                }
                                for h in horizons
                            },
                        },
                    }
                }
            }
        },
    }


def realised_series(*, end: str, periods: int = 400, value: float = 4000.0) -> pd.Series:
    index = pd.bdate_range(end=end, periods=periods, name="date")
    return pd.Series(np.full(len(index), value), index=index, name="price")


class TestScoringWhatElapsed:
    def test_an_elapsed_horizon_is_scored(self, tmp_path: Path) -> None:
        write_forecast_log(
            artifact(generated_at="2026-08-05T02:00:00+00:00", as_of="2026-08-04"),
            tmp_path,
            today=date(2026, 8, 5),
        )
        realised = {"gold": {"USD": realised_series(end="2026-10-01", value=4050.0)}}

        observations = collect(realised=realised, directory=tmp_path)

        assert {obs.horizon for obs in observations} == {5, 21}
        five = next(obs for obs in observations if obs.horizon == 5)
        assert five.as_of == date(2026, 8, 4)
        assert five.realised == 4050.0
        assert five.pit is not None
        assert five.commit == "deadbeef", "the live log must carry the code that produced it"

    def test_a_horizon_that_has_not_elapsed_is_not_scored(self, tmp_path: Path) -> None:
        """Scoring an unfinished window against the last price available is lookahead
        in the one place this repository claims to have none."""
        write_forecast_log(
            artifact(generated_at="2026-08-05T02:00:00+00:00", as_of="2026-08-04"),
            tmp_path,
            today=date(2026, 8, 5),
        )
        realised = {"gold": {"USD": realised_series(end="2026-08-07")}}

        observations = collect(realised=realised, directory=tmp_path)

        assert {obs.horizon for obs in observations} == set(), "5 sessions have not passed"

    def test_the_horizon_is_counted_in_sessions_on_the_real_calendar(self, tmp_path: Path) -> None:
        """A five-session horizon from Tuesday 4 August settles on Tuesday 11 August."""
        write_forecast_log(
            artifact(generated_at="2026-08-05T02:00:00+00:00", as_of="2026-08-04"),
            tmp_path,
            today=date(2026, 8, 5),
        )
        realised = {"gold": {"USD": realised_series(end="2026-10-01")}}

        five = next(
            obs for obs in collect(realised=realised, directory=tmp_path) if obs.horizon == 5
        )

        assert five.realised_on == date(2026, 8, 11)

    def test_a_lens_with_no_realised_series_is_skipped_not_guessed(self, tmp_path: Path) -> None:
        """Substituting a different series would score the engine against a price it
        never published."""
        write_forecast_log(
            artifact(generated_at="2026-08-05T02:00:00+00:00", as_of="2026-08-04"),
            tmp_path,
            today=date(2026, 8, 5),
        )

        assert collect(realised={"gold": {}}, directory=tmp_path) == []


class TestCensoring:
    def test_a_value_outside_the_published_grid_is_censored_not_clamped(
        self, tmp_path: Path
    ) -> None:
        """A PIT clamped to 1.0 looks measured and is not.

        The published grid stops at q95, so an outcome above it is only known to be
        somewhere in the top 5%. Recording 1.0 would put a fabricated precision into
        the one table whose whole value is that it is not fabricated.
        """
        write_forecast_log(
            artifact(generated_at="2026-08-05T02:00:00+00:00", as_of="2026-08-04"),
            tmp_path,
            today=date(2026, 8, 5),
        )
        realised = {"gold": {"USD": realised_series(end="2026-10-01", value=9999.0)}}

        five = next(
            obs for obs in collect(realised=realised, directory=tmp_path) if obs.horizon == 5
        )

        assert five.pit is None
        assert five.censored is not None
        assert "q95" in five.censored

    def test_the_summary_counts_censored_observations(self, tmp_path: Path) -> None:
        write_forecast_log(
            artifact(generated_at="2026-08-05T02:00:00+00:00", as_of="2026-08-04"),
            tmp_path,
            today=date(2026, 8, 5),
        )
        realised = {"gold": {"USD": realised_series(end="2026-10-01", value=9999.0)}}

        summary = summarise(collect(realised=realised, directory=tmp_path))

        assert summary["horizons"][0]["censored"] == 1
        assert summary["horizons"][0]["mean_pit"] is None


class TestRefusingToTestTooEarly:
    def test_an_empty_log_says_so_rather_than_reporting_nothing(self) -> None:
        summary = summarise([])

        assert summary["total_observations"] == 0
        assert summary["horizons"] == []
        assert summary["scope"] == "live"

    def test_a_small_count_is_published_with_no_test(self, tmp_path: Path) -> None:
        """Publishing n=4 with 'no test possible' is the honest version.

        The alternative — waiting until the number looks like something — is a
        publication decision made on the basis of the result, which is the thing this
        repository exists to not do.
        """
        write_forecast_log(
            artifact(generated_at="2026-08-05T02:00:00+00:00", as_of="2026-08-04"),
            tmp_path,
            today=date(2026, 8, 5),
        )
        realised = {"gold": {"USD": realised_series(end="2026-10-01", value=4050.0)}}

        summary = summarise(collect(realised=realised, directory=tmp_path))

        five = next(h for h in summary["horizons"] if h["horizon_sessions"] == 5)
        assert five["observations"] == 1
        assert five["test_possible"] is False
        assert "no test is possible" in five["test_note"]
        assert str(MIN_INDEPENDENT_FOR_TEST) in five["test_note"]

    def test_the_threshold_counts_independent_windows_not_forecasts(self, tmp_path: Path) -> None:
        """Consecutive nightly forecasts at 21 sessions share twenty of their days."""
        for day in pd.bdate_range("2026-08-04", periods=10):
            write_forecast_log(
                artifact(
                    generated_at=f"{day.date().isoformat()}T02:00:00+00:00",
                    as_of=day.date().isoformat(),
                ),
                tmp_path,
                today=day.date(),
            )
        realised = {"gold": {"USD": realised_series(end="2026-12-31", value=4050.0)}}

        summary = summarise(collect(realised=realised, directory=tmp_path))

        twenty_one = next(h for h in summary["horizons"] if h["horizon_sessions"] == 21)
        assert twenty_one["observations"] == 10
        assert twenty_one["independent_observations"] == 1


class TestTheSeparation:
    def test_the_live_log_is_labelled_live_and_reports_no_crps(self, tmp_path: Path) -> None:
        """CRPS skill needs the null's distribution for the same date, which a
        published artifact does not carry. Reporting one would mean it came from the
        backtest — which is precisely the merge that must never happen."""
        summary = summarise([])

        assert summary["scope"] == "live"
        assert "not computed" in summary["conventions"]["crps"].lower()
        assert "never pooled with it" in summary["conventions"]["what_this_is"]

    def test_the_written_file_is_separate_from_the_calibration_report(self, tmp_path: Path) -> None:
        path = write_live_log([], tmp_path)

        assert path.name == "live-log.json"
        payload = json.loads(path.read_text())
        assert payload["scope"] == "live"
        assert "pit" in payload["conventions"]
