"""The line between correcting a live forecast and editing the past.

Both of these rules are the kind that survive on discipline right up until the night
nobody is watching, which is why they are code. The tests are written from the
attacker's side: each describes a way the published record could be quietly improved
after the fact, and asserts that it cannot be.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from aurex.record import (
    ElapsedForecastError,
    build_index,
    earliest_elapse,
    status_of,
    write_forecast_log,
    write_index,
    write_skip_record,
)


def artifact(
    *,
    generated_at: str = "2026-08-05T02:00:00+00:00",
    as_of: str = "2026-08-04",
    horizons: tuple[int, ...] = (5, 21, 63),
    available: bool = True,
    quantile: float = 4000.0,
) -> dict[str, Any]:
    """A published artifact, cut down to the fields the record rules read."""
    return {
        "generated_at": generated_at,
        "schema_version": 4,
        "code": {"commit": "abc123", "dirty": False},
        "assets": {
            "gold": {
                "lenses": {
                    "USD": {
                        "latest": {"as_of": as_of, "price": quantile},
                        "distribution": {
                            "available": available,
                            "anchor": quantile,
                            "horizons": {
                                str(h): {
                                    "quantiles": {
                                        "q05": quantile * 0.95,
                                        "q25": quantile * 0.98,
                                        "q50": quantile,
                                        "q75": quantile * 1.02,
                                        "q95": quantile * 1.05,
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


class TestTheFreezeBoundary:
    def test_a_live_forecast_may_be_rewritten(self, tmp_path: Path) -> None:
        """A rerun before the horizon elapses is a correction, and git carries both."""
        first = write_forecast_log(artifact(), tmp_path, today=date(2026, 8, 5))
        second = write_forecast_log(artifact(quantile=4100.0), tmp_path, today=date(2026, 8, 6))

        assert first == second
        assert (
            json.loads(second.read_text())["assets"]["gold"]["lenses"]["USD"]["latest"]["price"]
            == 4100.0
        )

    def test_an_elapsed_forecast_may_never_be_rewritten(self, tmp_path: Path) -> None:
        """The whole point. Five sessions from 2026-08-04 can have run by 2026-08-11."""
        write_forecast_log(artifact(), tmp_path, today=date(2026, 8, 5))

        with pytest.raises(ElapsedForecastError, match="refusing to rewrite"):
            write_forecast_log(artifact(quantile=9999.0), tmp_path, today=date(2026, 8, 20))

    def test_the_original_survives_a_refused_rewrite(self, tmp_path: Path) -> None:
        """A refusal that had already truncated the file would be worse than no rule."""
        path = write_forecast_log(artifact(), tmp_path, today=date(2026, 8, 5))

        with pytest.raises(ElapsedForecastError):
            write_forecast_log(artifact(quantile=9999.0), tmp_path, today=date(2026, 8, 20))

        assert (
            json.loads(path.read_text())["assets"]["gold"]["lenses"]["USD"]["latest"]["price"]
            == 4000.0
        )

    def test_the_shortest_horizon_decides_the_freeze(self, tmp_path: Path) -> None:
        """Not the longest.

        Once any horizon has an outcome, rewriting the file revises a claim whose
        result is known — even though the quarterly horizon beside it is still live.
        The ability to correct the live part is not worth the ability to quietly
        revise the settled one.
        """
        status = status_of(write_forecast_log(artifact(), tmp_path, today=date(2026, 8, 5)))

        assert status.horizons == (5, 21, 63)
        assert status.frozen_from == earliest_elapse(date(2026, 8, 4), 5)
        assert status.frozen_on(date(2026, 8, 11))
        assert not status.frozen_on(date(2026, 8, 10))

    def test_the_clock_starts_at_the_anchor_not_the_run_time(self, tmp_path: Path) -> None:
        """A Monday run prices from Friday's fix, so the horizon started on Friday.

        Dating the freeze from `generated_at` would start the clock late and leave the
        forecast rewritable after its horizon had actually run.
        """
        path = write_forecast_log(
            artifact(generated_at="2026-08-10T02:00:00+00:00", as_of="2026-08-07"),
            tmp_path,
            today=date(2026, 8, 10),
        )

        assert status_of(path).as_of == date(2026, 8, 7)
        assert status_of(path).frozen_from == earliest_elapse(date(2026, 8, 7), 5)


class TestFailingClosed:
    def test_an_artifact_with_no_distribution_is_not_frozen(self, tmp_path: Path) -> None:
        """A run that resolved data but produced no forecast makes no claim to protect.

        Freezing it would block legitimate reruns on the days the engine had nothing
        to say, which is a cost with no corresponding benefit.
        """
        path = write_forecast_log(artifact(available=False), tmp_path, today=date(2026, 8, 5))
        status = status_of(path)

        assert not status.carries_forecast
        assert status.frozen_from is None
        assert not status.frozen_on(date(2030, 1, 1))

    def test_an_unreadable_forecast_is_frozen(self, tmp_path: Path) -> None:
        """It might be a real forecast. Destroying evidence beats blocking a rerun."""
        target = tmp_path / "forecasts"
        target.mkdir(parents=True)
        (target / "2026-08-05.json").write_text("{ this is not json")

        assert status_of(target / "2026-08-05.json").frozen_on(date(2026, 8, 5))

    def test_a_distribution_with_no_readable_horizon_is_frozen(self, tmp_path: Path) -> None:
        """Schema drift must not silently switch the rule off."""
        payload = artifact()
        payload["assets"]["gold"]["lenses"]["USD"]["distribution"]["horizons"] = {}
        target = tmp_path / "forecasts"
        target.mkdir(parents=True)
        (target / "2026-08-05.json").write_text(json.dumps(payload))

        status = status_of(target / "2026-08-05.json")
        assert status.carries_forecast
        assert status.frozen_on(date(2026, 8, 5))


class TestTheIndex:
    def test_a_gap_with_a_skip_record_is_explained(self, tmp_path: Path) -> None:
        write_forecast_log(artifact(), tmp_path, today=date(2026, 8, 5))
        write_skip_record(
            when=date(2026, 8, 6),
            reason="price series did not reach the run date",
            directory=tmp_path,
        )

        index = build_index(tmp_path, today=date(2026, 8, 6))

        gaps = {gap["date"]: gap for gap in index["gaps"]}
        assert gaps["2026-08-06"]["explained"] is True
        assert "price series" in gaps["2026-08-06"]["reason"]

    def test_a_gap_with_no_trace_is_reported_as_unexplained(self, tmp_path: Path) -> None:
        """The case that matters: three weeks of silence nothing accounts for.

        An outage the engine refused is a decision. An outage nothing survives from is
        indistinguishable from three weeks of forecasts nobody scored, and the index
        has to be able to tell a reader which one it is looking at.
        """
        write_forecast_log(artifact(), tmp_path, today=date(2026, 8, 5))

        index = build_index(tmp_path, today=date(2026, 8, 12))

        assert index["counts"]["gaps_unexplained"] == 5
        assert all(gap["reason"] is None for gap in index["gaps"])

    def test_weekends_are_not_gaps(self, tmp_path: Path) -> None:
        """The freshness guard refuses weekends by design, so counting them as holes
        would bury the real ones under two false alarms a week."""
        write_forecast_log(artifact(), tmp_path, today=date(2026, 8, 5))

        index = build_index(tmp_path, today=date(2026, 8, 10))

        assert "2026-08-08" not in {gap["date"] for gap in index["gaps"]}
        assert "2026-08-09" not in {gap["date"] for gap in index["gaps"]}

    def test_the_index_records_the_commit_behind_each_forecast(self, tmp_path: Path) -> None:
        write_forecast_log(artifact(), tmp_path, today=date(2026, 8, 5))

        entry = build_index(tmp_path, today=date(2026, 8, 5))["forecasts"][0]

        assert entry["commit"] == "abc123"
        assert entry["horizons"] == [5, 21, 63]

    def test_the_index_is_not_mistaken_for_a_forecast(self, tmp_path: Path) -> None:
        """index.json lives beside the dated files and must never be counted as one."""
        write_forecast_log(artifact(), tmp_path, today=date(2026, 8, 5))
        write_index(tmp_path, today=date(2026, 8, 5))

        index = build_index(tmp_path, today=date(2026, 8, 5))

        assert index["counts"]["published"] == 1
        assert [entry["date"] for entry in index["forecasts"]] == ["2026-08-05"]


class TestEarliestElapse:
    def test_five_sessions_is_seven_calendar_days(self) -> None:
        """Monday to the following Monday, the shortest that window can be."""
        assert earliest_elapse(date(2026, 8, 3), 5) == date(2026, 8, 10)

    @pytest.mark.parametrize("horizon", [1, 5, 10, 21, 42, 63])
    def test_it_never_lands_after_the_real_business_day_count(self, horizon: int) -> None:
        """Erring early costs a rerun nobody needed; erring late leaves the past editable.

        A weekday-only calendar is already the most generous one — real holidays only
        push the true elapse date further out — so the bound must sit on or before it
        at every horizon this engine publishes.
        """
        anchor = date(2026, 8, 3)
        real = pd.bdate_range(start=anchor, periods=horizon + 1)[-1].date()

        assert earliest_elapse(anchor, horizon) <= real
