"""The track record: what was published, what was not, and what may still change.

``latest.json`` is state. This module owns the *record* — the dated forecasts under
``public-data/forecasts/``, the refusals beside them, and the index that makes the
difference between them legible. Three rules, each enforced here rather than promised
in prose, because all three are the kind of discipline that survives right up until
the night nobody is watching.

**A forecast whose horizon has elapsed may never be rewritten.** Before the horizon
elapses a rerun is a correction to a live forecast and git carries both versions.
After it elapses, the outcome exists, and rewriting the forecast that preceded it is
editing the past. The line between those two is not a matter of judgement and should
not be left to one: :func:`write_forecast_log` refuses.

**A gap must be detectable in the data, not merely absent.** A three-week outage that
leaves no trace is indistinguishable from three weeks of forecasts nobody scored. So a
refusal writes a *skip record* — the refusal is itself data — and :func:`build_index`
lists every date that should carry a forecast and does not, separating the ones that
explained themselves from the ones that vanished.

**The live log is not the backtest.** See :mod:`aurex.livelog`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from aurex import config

#: Trading sessions per calendar week. Used only to convert a horizon in sessions into
#: the *earliest* calendar date it could possibly have elapsed on — five sessions from
#: a Monday land on the next Monday, seven calendar days later. Holidays only push the
#: real elapse date later, so freezing on this bound freezes early and never late,
#: which is the direction a rule protecting the past should err in.
SESSIONS_PER_WEEK = 5
DAYS_PER_WEEK = 7


class ElapsedForecastError(RuntimeError):
    """An attempt to rewrite a forecast whose horizon has already elapsed."""


def earliest_elapse(as_of: date, horizon_sessions: int) -> date:
    """First calendar date on which ``horizon_sessions`` could have elapsed from ``as_of``.

    Whole weeks cost seven days each and the remaining sessions cost one day each, which
    is the shortest a session count can possibly take: the remainder only stretches when
    it straddles a weekend, and every start weekday that would stretch it is a start
    weekday this bound is deliberately not assuming. Scaling by ``7/5`` throughout
    instead — the obvious formula — spreads the weekend across sessions that never meet
    one and lands *after* the true elapse date at 21 sessions and beyond, which would
    leave a settled forecast rewritable for two days. That is the one direction this
    function must never err in.
    """
    weeks, remainder = divmod(horizon_sessions, SESSIONS_PER_WEEK)
    return as_of + timedelta(days=weeks * DAYS_PER_WEEK + remainder)


def _anchor_date(artifact: Mapping[str, Any]) -> date | None:
    """The date the forecast is anchored to — the last observed price, not the run time.

    A Monday run prices from Friday's fix; the horizon starts at Friday, so that is the
    date every elapse question is asked from. Falling back to ``generated_at`` would
    start the clock late and keep a forecast rewritable after its horizon had run.
    """
    seen: list[date] = []
    assets = artifact.get("assets")
    if isinstance(assets, dict):
        for asset in assets.values():
            lenses = asset.get("lenses") if isinstance(asset, dict) else None
            if not isinstance(lenses, dict):
                continue
            for block in lenses.values():
                latest = block.get("latest") if isinstance(block, dict) else None
                if isinstance(latest, dict) and isinstance(latest.get("as_of"), str):
                    try:
                        seen.append(date.fromisoformat(latest["as_of"]))
                    except ValueError:
                        continue
    if seen:
        return min(seen)
    generated = artifact.get("generated_at")
    if isinstance(generated, str):
        try:
            return date.fromisoformat(generated[:10])
        except ValueError:
            return None
    return None


def _distributions(artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every per-lens distribution block in the artifact."""
    out: list[dict[str, Any]] = []
    assets = artifact.get("assets")
    if not isinstance(assets, dict):
        return out
    for asset in assets.values():
        lenses = asset.get("lenses") if isinstance(asset, dict) else None
        if not isinstance(lenses, dict):
            continue
        for block in lenses.values():
            dist = block.get("distribution") if isinstance(block, dict) else None
            if isinstance(dist, dict):
                out.append(dist)
    return out


def _horizons(artifact: Mapping[str, Any]) -> tuple[int, ...]:
    """Every simulated horizon in the artifact, in sessions."""
    found: set[int] = set()
    for dist in _distributions(artifact):
        horizons = dist.get("horizons")
        if isinstance(horizons, dict):
            found.update(int(key) for key in horizons if str(key).isdigit())
    return tuple(sorted(found))


def _carries_forecast(artifact: Mapping[str, Any]) -> bool:
    """True where at least one lens actually published a distribution.

    An artifact that resolved data but produced no distribution — too little history to
    fit, every source down — is a record of a run, not a forecast. There is no claim in
    it whose outcome could later become known, so freezing it protects nothing and only
    blocks legitimate reruns.
    """
    return any(dist.get("available") is True for dist in _distributions(artifact))


@dataclass(frozen=True, slots=True)
class ForecastStatus:
    """Whether a published forecast is still live, and until when."""

    path: Path
    as_of: date | None
    horizons: tuple[int, ...]
    #: Earliest date on which the *shortest* horizon could have elapsed. ``None`` where
    #: the file carries no forecast to elapse; :data:`date.min` where the file makes a
    #: claim we cannot date, which freezes it immediately.
    frozen_from: date | None
    carries_forecast: bool
    note: str

    def frozen_on(self, when: date) -> bool:
        """A forecast freezes when its shortest horizon elapses, not its longest.

        The shortest is the conservative choice and the right one: once any horizon has
        an outcome, rewriting the file edits a claim whose result is already known —
        even if the quarterly horizon in the same file is still live. The ability to
        correct the live part is not worth the ability to quietly revise the settled one.
        """
        return self.frozen_from is not None and when >= self.frozen_from


def status_of(path: Path) -> ForecastStatus:
    """Read a published forecast and work out whether it is still rewritable.

    Fails closed in both directions that matter. A file we cannot read might be a real
    forecast, so it freezes: destroying evidence is worse than blocking a rerun. A file
    that claims a distribution but carries no horizon we recognise freezes too, because
    that is schema drift and the alternative is a rule that silently stops applying the
    day the artifact shape changes.
    """
    try:
        artifact = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        artifact = None
    if not isinstance(artifact, dict):
        return ForecastStatus(
            path=path,
            as_of=None,
            horizons=(),
            frozen_from=date.min,
            carries_forecast=False,
            note="unreadable; frozen because it cannot be shown to be safe to rewrite",
        )

    as_of = _anchor_date(artifact)
    horizons = _horizons(artifact)

    if not _carries_forecast(artifact):
        return ForecastStatus(
            path=path,
            as_of=as_of,
            horizons=horizons,
            frozen_from=None,
            carries_forecast=False,
            note="no distribution was published, so there is no claim to freeze",
        )

    if as_of is None or not horizons:
        return ForecastStatus(
            path=path,
            as_of=as_of,
            horizons=horizons,
            frozen_from=date.min,
            carries_forecast=True,
            note=(
                "publishes a distribution but carries no readable anchor or horizon; "
                "frozen because an undatable claim cannot be shown to be still live"
            ),
        )

    return ForecastStatus(
        path=path,
        as_of=as_of,
        horizons=horizons,
        frozen_from=earliest_elapse(as_of, min(horizons)),
        carries_forecast=True,
        note="frozen once the shortest horizon could have elapsed",
    )


def forecasts_dir(directory: Path | None = None) -> Path:
    """Where the dated record lives.

    Resolved through the module rather than bound at import so there is exactly one
    place the output root can be redirected. Two independent bindings of it would let a
    caller redirect half the writes and leave the other half landing in the real
    ``public-data/`` — which is how a test suite ends up committing a skip record.
    """
    return (directory or config.PUBLIC_DATA_DIR) / "forecasts"


def write_forecast_log(
    artifact: Mapping[str, Any],
    directory: Path | None = None,
    *,
    today: date | None = None,
) -> Path:
    """Write the dated copy that will be scored once its horizon elapses.

    Re-running on the same day replaces that day's file while the horizon is still
    running: a rerun is a correction to a live forecast, and git carries both versions.
    Once the shortest horizon has elapsed the file is frozen and this raises.

    Raises:
        ElapsedForecastError: The target exists and its horizon has already elapsed.
    """
    when = today or datetime.now(UTC).date()
    target_dir = forecasts_dir(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = str(artifact["generated_at"])[:10]
    path = target_dir / f"{stamp}.json"

    if path.exists():
        existing = status_of(path)
        if existing.frozen_on(when):
            raise ElapsedForecastError(
                f"refusing to rewrite {path.name}: its shortest horizon "
                f"({min(existing.horizons) if existing.horizons else 'unknown'} sessions "
                f"from {existing.as_of}) elapsed on or before "
                f"{existing.frozen_from}, and today is {when}. A forecast whose horizon "
                f"has run is a claim with a known outcome; correcting a live forecast is "
                f"a different act from editing a settled one, and only the first is "
                f"allowed. Publish today's forecast under today's date instead."
            )

    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return path


def write_skip_record(
    *,
    when: date,
    reason: str,
    detail: Mapping[str, Any] | None = None,
    directory: Path | None = None,
) -> Path:
    """Record that a night produced no forecast, and why.

    This is what makes an outage a *hole* rather than an absence. It is written before
    the job exits non-zero, because a refusal that leaves no trace is exactly the
    silent gap the index exists to rule out.
    """
    target_dir = forecasts_dir(directory) / "skipped"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{when.isoformat()}.json"
    path.write_text(
        json.dumps(
            {
                "date": when.isoformat(),
                "recorded_at": datetime.now(UTC).isoformat(),
                "reason": reason,
                "detail": dict(detail) if detail else None,
                "code": None,
                "note": (
                    "No forecast was published for this date. This file is the record "
                    "of that decision: a night the engine refused is a visible hole, "
                    "and a night it fabricated would not be."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return path


def _expected_dates(first: date, last: date) -> list[date]:
    """Dates a nightly job should have produced a forecast for.

    Weekdays only. The job runs every night, but weekends carry no new fix and the
    freshness guard refuses them by design, so counting them as missing would bury the
    real gaps under two false ones a week.
    """
    out: list[date] = []
    day = first
    while day <= last:
        if day.weekday() < SESSIONS_PER_WEEK:
            out.append(day)
        day += timedelta(days=1)
    return out


def build_index(
    directory: Path | None = None,
    *,
    today: date | None = None,
) -> dict[str, Any]:
    """Enumerate the published record, including the dates missing from it.

    The point of the ``gaps`` list is the ``reason`` field. A gap carrying a recorded
    refusal is an outage the engine noticed and declined to paper over. A gap carrying
    ``null`` is one nothing survives from — the job died, never started, or was never
    scheduled — and that is a materially weaker position to be in. Publishing both
    under one heading, distinguished by whether an explanation exists, is what stops a
    three-week silence from reading like three weeks of unscored forecasts.
    """
    when = today or datetime.now(UTC).date()
    root = forecasts_dir(directory)
    published: list[dict[str, Any]] = []
    for path in published_forecasts(directory):
        try:
            artifact = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            published.append({"date": path.stem, "readable": False})
            continue
        status = status_of(path)
        code = artifact.get("code") if isinstance(artifact, dict) else None
        published.append(
            {
                "date": path.stem,
                "readable": True,
                "as_of": status.as_of.isoformat() if status.as_of else None,
                "horizons": list(status.horizons),
                "frozen": status.frozen_on(when),
                "frozen_from": status.frozen_from.isoformat() if status.frozen_from else None,
                "commit": code.get("commit") if isinstance(code, dict) else None,
            }
        )

    skipped: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "skipped").glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(record, dict) and isinstance(record.get("date"), str):
            skipped[record["date"]] = record

    have = {entry["date"] for entry in published}
    gaps: list[dict[str, Any]] = []
    if published or skipped:
        stamps = sorted(have | set(skipped))
        try:
            first = date.fromisoformat(stamps[0])
        except ValueError:
            first = when
        for day in _expected_dates(first, when):
            stamp = day.isoformat()
            if stamp in have:
                continue
            record = skipped.get(stamp)
            gaps.append(
                {
                    "date": stamp,
                    "reason": record.get("reason") if record else None,
                    "explained": record is not None,
                }
            )

    unexplained = [gap for gap in gaps if not gap["explained"]]
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "as_of_date": when.isoformat(),
        "counts": {
            "published": len(published),
            "gaps": len(gaps),
            "gaps_explained": len(gaps) - len(unexplained),
            "gaps_unexplained": len(unexplained),
        },
        "forecasts": published,
        "gaps": gaps,
        "conventions": {
            "expected": (
                "Weekdays between the first record and today. Weekends carry no fix and "
                "are refused by the freshness guard by design, so they are not gaps."
            ),
            "explained": (
                "A gap is explained when a skip record says why no forecast was "
                "published. An unexplained gap is one nothing survives from — the job "
                "died, never ran, or was never scheduled — and is the weaker case."
            ),
            "frozen": (
                "A forecast freezes once its shortest horizon could have elapsed. "
                "Frozen forecasts may not be rewritten; live ones may, and git carries "
                "every version of both."
            ),
        },
    }


def write_index(
    directory: Path | None = None,
    *,
    today: date | None = None,
) -> Path:
    """Write ``forecasts/index.json``."""
    target_dir = forecasts_dir(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "index.json"
    path.write_text(
        json.dumps(build_index(directory, today=today), indent=2, sort_keys=True) + "\n"
    )
    return path


def published_forecasts(directory: Path | None = None) -> Iterable[Path]:
    """Every dated forecast file, oldest first. ``index.json`` is not one of them."""
    return (p for p in sorted(forecasts_dir(directory).glob("*.json")) if p.stem != "index")
