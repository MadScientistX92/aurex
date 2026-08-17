"""Progress for a run measured in hours rather than in seconds.

**Why this exists as code rather than as a note to run it in a terminal.** The direction
run's first execution on a hosted runner produced, between starting and being killed six
hours later, no stdout of its own at all — the only moving line in the log was
``lightning_fabric``'s "Seed set to N", printed by a dependency for its own reasons. That
number happens to advance by ``step`` times 1,000 per window, so it *could* be read as a
counter, and it was. Reading a third-party log line as a progress meter is a guess that
happens to be right: it carries no total, so it cannot say how far along the run is, and
it stops the moment the model that emits it is dropped from the set. There was no way to
tell working from hung, and no estimate of when the run would end — which is how a job
walked into a ceiling it needed twenty-three more minutes to clear.

So the counter is deliberate here: the window index, the total, the as-of date, the seed
the accident exposed, the elapsed time, the per-window rate and a projected finish, one
line per window, flushed. The projection is the part that matters. A run that will end
after the platform's cap is a run to kill and re-scope in its first ten minutes, not its
last, and that judgement needs a number rather than a feeling about how long it has been
quiet.

**Two properties this must not violate.**

*It writes to stderr.* ``aurex direction --dry-run`` writes the whole artifact to stdout,
and a progress line interleaved into that makes it unparseable. Whatever consumes the
JSON gets the JSON; whatever reads the log gets both, in order, because a CI log
interleaves the two streams anyway.

*The harness stays silent unless asked.* :func:`~aurex.score.walkforward.walk_forward`
takes a reporter and defaults to ``None``, so nothing in the library prints. A test suite
that ran a thousand short walk-forwards would otherwise emit a thousand progress lines,
and a scoring function that writes to a stream on its own initiative is a scoring
function that cannot be called from anything but a terminal.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Protocol, TextIO


def _clock(seconds: float) -> str:
    """Seconds as ``H:MM:SS``, because a run this long is not readable in seconds."""
    total = max(0, round(seconds))
    hours, rest = divmod(total, 3_600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


class ProgressReporter(Protocol):
    """What a walk-forward may say about itself while it is still running.

    A protocol rather than a callback so the three moments stay distinct: the plan is
    known before any work happens, each window is one unit of it, and the summary can
    report what the plan turned into. A single ``on_event`` callable would collapse the
    three into one signature whose meaning depends on which fields happen to be set.
    """

    def planned(
        self,
        *,
        windows: int,
        first_as_of: str,
        last_as_of: str,
        step: int,
        horizons: Sequence[int],
        observations: int,
    ) -> None:
        """Called once, before the first fit, with the whole plan."""
        ...

    def window(self, *, index: int, as_of: str, seed: int, skipped: bool) -> None:
        """Called once per window, after its work is done. ``index`` is 1-based."""
        ...

    def finished(self, *, scored: int, skipped: int) -> None:
        """Called once, after the last window."""
        ...


@dataclass(slots=True)
class ElapsedProgress:
    """Windows done over windows planned, with elapsed, rate and projected finish.

    The projection is a straight-line extrapolation of the mean window so far, and that
    is the honest model of this workload: every window refits every model on an expanding
    history, so the per-window cost drifts up slowly and the projection runs slightly
    early rather than late. Stated here because a projected finish that quietly reads as a
    guarantee is the failure mode of a progress meter.
    """

    #: Names the run in the log, so two jobs' lines are not confusable.
    label: str
    #: Progress goes to stderr; see the module docstring for why that is not incidental.
    stream: TextIO = field(default_factory=lambda: sys.stderr)
    #: Print every ``every``-th window. 1 by default: at ~40 seconds a window, one line
    #: per window is not noise, and a run whose lines stop arriving is the signal.
    every: int = 1
    #: Injected so the projection can be tested without waiting for a real clock.
    now: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))
    _total: int = field(default=0, init=False)
    _started: datetime | None = field(default=None, init=False)

    def _say(self, line: str) -> None:
        print(line, file=self.stream, flush=True)

    def planned(
        self,
        *,
        windows: int,
        first_as_of: str,
        last_as_of: str,
        step: int,
        horizons: Sequence[int],
        observations: int,
    ) -> None:
        self._total = windows
        self._started = self.now()
        listed = ",".join(str(h) for h in horizons)
        self._say(
            f"{self.label}: {windows} windows planned, {first_as_of} to {last_as_of}, "
            f"step {step} sessions, horizons {listed}, {observations} observations"
        )

    def window(self, *, index: int, as_of: str, seed: int, skipped: bool) -> None:
        if self._started is None or (index % self.every and index != self._total):
            return
        # One reading of the clock, used for both the elapsed time and the projection.
        # Two readings would put a line's own arithmetic a few milliseconds out of step
        # with itself, and make the projection untestable against a fake clock.
        moment = self.now()
        elapsed = (moment - self._started).total_seconds()
        per = elapsed / index if index else 0.0
        remaining = per * max(0, self._total - index)
        finish = moment + timedelta(seconds=remaining)
        width = len(str(self._total))
        self._say(
            f"[{index:>{width}}/{self._total}] {as_of}  seed {seed}  "
            f"elapsed {_clock(elapsed)}  {per:.1f}s/window  "
            f"remaining {_clock(remaining)}  finish {finish.strftime('%Y-%m-%dT%H:%MZ')}"
            + ("  SKIPPED" if skipped else "")
        )

    def finished(self, *, scored: int, skipped: int) -> None:
        if self._started is None:
            return
        elapsed = (self.now() - self._started).total_seconds()
        per = elapsed / self._total if self._total else 0.0
        self._say(
            f"{self.label}: {self._total} windows in {_clock(elapsed)} ({per:.1f}s each), "
            f"{scored} forecasts scored, {skipped} windows skipped"
        )
