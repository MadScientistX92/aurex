"""Tests for the walk-forward's progress output.

The interesting assertion here is about the **denominator**. A progress line reading
``[12/290]`` is only useful if 290 is the number of windows the loop will actually run,
and that number is derived separately from the loop that runs them: the loop stops at the
first as-of date whose shortest horizon would run past the end of the sample, while the
count is computed from a ``range`` up front. Two derivations of one quantity is exactly
the arrangement where a projected finish can be quietly, confidently wrong — a run
reporting 94% when it is really at 100% would have looked fine right up to the ceiling.
So the count is not asserted against a literal; it is asserted against the loop, over a
grid of steps and horizons where the two derivations can disagree, and against the record
counts the artifact publishes.

The rest is boundaries: the harness must stay silent without a reporter, the reporter must
write to stderr so ``--dry-run``'s JSON on stdout stays parseable, and the projection has
to be arithmetic rather than an impression, which is why the clock is injectable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from aurex.assets.transforms import LogReturn
from aurex.score import RandomWalkForecaster, WalkForwardRequest, walk_forward
from aurex.score.progress import ElapsedProgress


def price_path(periods: int, *, seed: int = 5) -> pd.Series:
    """A driftless path. Nothing here grades a model, so a random walk is enough."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.01, size=periods)
    index = pd.bdate_range("2016-01-04", periods=periods)
    return pd.Series(100.0 * np.exp(np.cumsum(returns)), index=index, name="price")


@dataclass(slots=True)
class Recorder:
    """A reporter that keeps what it was told instead of printing it."""

    planned_windows: int | None = None
    planned_observations: int | None = None
    indices: list[int] = field(default_factory=list)
    skips: list[int] = field(default_factory=list)
    summary: tuple[int, int] | None = None

    def planned(
        self,
        *,
        windows: int,
        first_as_of: str,
        last_as_of: str,
        step: int,
        horizons: object,
        observations: int,
    ) -> None:
        self.planned_windows = windows
        self.planned_observations = observations

    def window(self, *, index: int, as_of: str, seed: int, skipped: bool) -> None:
        self.indices.append(index)
        if skipped:
            self.skips.append(index)

    def finished(self, *, scored: int, skipped: int) -> None:
        self.summary = (scored, skipped)


def run(prices: pd.Series, request: WalkForwardRequest, recorder: Recorder) -> object:
    walk = RandomWalkForecaster(transform=LogReturn(), n_paths=40, min_observations=40)
    return walk_forward(
        prices,
        subject=walk,
        baseline=RandomWalkForecaster(transform=LogReturn(), n_paths=40, min_observations=40),
        request=request,
        progress=recorder,
    )


class TestTheDenominator:
    @pytest.mark.parametrize(
        ("periods", "step", "horizons"),
        [
            (200, 1, (5,)),
            (200, 5, (5,)),
            (200, 5, (5, 21)),
            (200, 7, (5, 21, 63)),
            (201, 5, (5, 21, 63)),
            (202, 4, (3, 63)),
            (203, 10, (1,)),
            # The horizon eats most of the series, so nearly every planned window falls
            # off the end. If the count came from ``range(first, total, step)`` rather
            # than from where the loop stops, this row is where it would be wrong.
            (160, 5, (100,)),
        ],
    )
    def test_the_planned_count_is_the_number_of_windows_the_loop_runs(
        self, periods: int, step: int, horizons: tuple[int, ...]
    ) -> None:
        recorder = Recorder()
        run(
            price_path(periods),
            WalkForwardRequest(horizons=horizons, step=step, min_observations=60),
            recorder,
        )

        assert recorder.planned_windows == len(recorder.indices)
        assert recorder.indices == list(range(1, len(recorder.indices) + 1))

    def test_the_planned_count_matches_the_records_the_artifact_publishes(self) -> None:
        """Tied to a published quantity, not only to the loop.

        Every window that ran was scored at the shortest horizon — that is what the
        loop's exit condition guarantees — unless it was skipped outright. So the
        denominator is checkable against the observation count that reaches the artifact,
        which is the number a reader can see.
        """
        recorder = Recorder()
        request = WalkForwardRequest(horizons=(5, 21, 63), step=5, min_observations=60)
        result = run(price_path(400), request, recorder)

        shortest = len(result.for_horizon(5))  # type: ignore[attr-defined]
        assert recorder.planned_windows == shortest + len(recorder.skips)
        assert recorder.summary is not None
        assert recorder.summary[1] == len(recorder.skips)

    def test_a_sample_too_short_to_forecast_plans_no_windows(self) -> None:
        recorder = Recorder()
        run(
            price_path(70),
            WalkForwardRequest(horizons=(5,), step=5, min_observations=68),
            recorder,
        )

        assert recorder.planned_windows == 0
        assert recorder.indices == []


class TestTheHarnessStaysSilent:
    def test_nothing_is_printed_without_a_reporter(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A scoring function that writes to a stream on its own initiative is a scoring
        function that cannot be called from anything but a terminal."""
        walk = RandomWalkForecaster(transform=LogReturn(), n_paths=40, min_observations=40)
        walk_forward(
            price_path(200),
            subject=walk,
            baseline=RandomWalkForecaster(transform=LogReturn(), n_paths=40, min_observations=40),
            request=WalkForwardRequest(horizons=(5,), step=5, min_observations=60),
        )

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""


class TestElapsedProgress:
    def test_it_writes_to_stderr_and_never_to_stdout(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``--dry-run`` puts the whole artifact on stdout. A progress line in the middle
        of it would make the JSON unparseable."""
        reporter = ElapsedProgress(label="direction")
        reporter.planned(
            windows=2,
            first_as_of="2015-01-02",
            last_as_of="2026-07-24",
            step=10,
            horizons=(5, 10),
            observations=5_408,
        )
        reporter.window(index=1, as_of="2015-01-02", seed=22_526_000, skipped=False)
        reporter.finished(scored=10, skipped=0)

        captured = capsys.readouterr()
        assert captured.out == ""
        assert "2 windows planned" in captured.err
        assert "[1/2]" in captured.err
        assert "seed 22526000" in captured.err

    def test_the_projection_is_arithmetic(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Ten windows, one done, forty seconds gone: nine left at forty seconds each."""
        start = datetime(2026, 8, 17, 16, 0, tzinfo=UTC)
        ticks = iter([start, start + timedelta(seconds=40)])
        reporter = ElapsedProgress(label="direction", now=lambda: next(ticks))
        reporter.planned(
            windows=10,
            first_as_of="2015-01-02",
            last_as_of="2026-07-24",
            step=10,
            horizons=(5,),
            observations=100,
        )
        reporter.window(index=1, as_of="2015-01-02", seed=1_000, skipped=False)

        line = capsys.readouterr().err.splitlines()[-1]
        assert "elapsed 0:00:40" in line
        assert "40.0s/window" in line
        assert "remaining 0:06:00" in line
        # 16:00:00 + 40s elapsed + 360s remaining = 16:06:40, truncated to the minute.
        assert "finish 2026-08-17T16:06Z" in line

    def test_a_skipped_window_says_so(self, capsys: pytest.CaptureFixture[str]) -> None:
        reporter = ElapsedProgress(label="direction")
        reporter.planned(
            windows=1,
            first_as_of="2015-01-02",
            last_as_of="2015-01-02",
            step=5,
            horizons=(5,),
            observations=100,
        )
        reporter.window(index=1, as_of="2015-01-02", seed=1_000, skipped=True)

        assert "SKIPPED" in capsys.readouterr().err

    def test_throttling_still_reports_the_final_window(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A throttled reporter that dropped the last line would leave a finished run
        looking stalled at 9/10, which is the reading error this whole module exists to
        remove."""
        reporter = ElapsedProgress(label="direction", every=5)
        reporter.planned(
            windows=10,
            first_as_of="2015-01-02",
            last_as_of="2026-07-24",
            step=10,
            horizons=(5,),
            observations=100,
        )
        for index in range(1, 11):
            reporter.window(index=index, as_of="2015-01-02", seed=index * 1_000, skipped=False)

        reported = [
            line.split("]")[0].lstrip("[").strip()
            for line in capsys.readouterr().err.splitlines()
            if line.startswith("[")
        ]
        assert reported == ["5/10", "10/10"]

    def test_a_reporter_never_told_the_plan_says_nothing(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """No total means no denominator and no projection, so there is nothing honest to
        print. Silence rather than a line with a zero in it."""
        reporter = ElapsedProgress(label="direction")
        reporter.window(index=1, as_of="2015-01-02", seed=1_000, skipped=False)
        reporter.finished(scored=0, skipped=0)

        assert capsys.readouterr().err == ""
