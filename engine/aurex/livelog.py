"""The live track record, which is not the backtest and must never be merged with it.

The walk-forward in :mod:`aurex.backtest` is a *simulation of what the engine would
have said*. It is honest about lookahead, it refits weekly over eleven years, and it
produces thousands of scored forecasts — but every one of them was scored against an
outcome that already existed when the code ran. That is a legitimate and necessary
measurement, and it is a weaker claim than the one people assume it makes.

The nightly log is what the engine *did* say, published before the outcome existed.
There is no version of that which can be re-run, tuned, or accidentally given
lookahead, because the file was committed to a public repository on a date git
records. It is the stronger claim by some distance, and it will have `n` in single
digits for months.

Both get published. They get published *separately*, each with its own count, and this
module exists partly to make combining them awkward: a live observation and a
walk-forward observation have different types here and no function accepts both. The
temptation to pool them is real — pooling would make the live sample look testable
years earlier — and pooling is exactly the error that would turn the stronger claim
into the weaker one while appearing to strengthen it.

**What can and cannot be scored from a published artifact.** The artifact carries
quantiles, not paths, so the PIT here is interpolated on a five-point grid rather than
computed against the ensemble. That is coarser than :func:`aurex.score.pit_value` and
is labelled as such wherever it is reported. A realised value outside the published
grid is recorded as *censored* rather than clamped to zero or one, because a clamped
PIT is a number that looks measured and is not. CRPS skill is not computed at all: it
needs the null's distribution for the same date, and the artifact does not carry one.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from aurex import config
from aurex.record import earliest_elapse, published_forecasts

#: Independent observations required before any p-value is reported on the live log.
#:
#: Fixed here, in advance, rather than chosen once the numbers exist — the whole point
#: of the live log is that it cannot be revised after the fact, and a threshold picked
#: when `n` happens to look encouraging would defeat that. It follows the same
#: discipline every other p-value in this repository obeys: overlapping windows are not
#: independent observations, so this counts *thinned* windows. At a five-session
#: horizon a nightly job accrues roughly one per week, so 30 is on the order of half a
#: year. Reporting "no test is possible, n = 4" for that long is the honest output, not
#: a placeholder to be replaced by something more impressive.
MIN_INDEPENDENT_FOR_TEST = 30


@dataclass(frozen=True, slots=True)
class LiveObservation:
    """One published forecast, at one horizon, whose outcome is now known.

    Deliberately not interchangeable with a walk-forward record. The extra fields are
    the ones that make this the stronger claim — the date it was published and the
    commit that produced it — and a type that dropped them to match the backtest's
    shape would be discarding the evidence.
    """

    published_on: date
    commit: str | None
    asset_id: str
    lens: str
    horizon: int
    as_of: date
    anchor: float
    realised_on: date
    realised: float
    quantiles: dict[str, float]
    pit: float | None
    #: Set where the realised value fell outside the published quantile grid, which
    #: makes the PIT a bound rather than a measurement.
    censored: str | None

    def describe(self) -> dict[str, Any]:
        return {
            "published_on": self.published_on.isoformat(),
            "commit": self.commit,
            "asset_id": self.asset_id,
            "lens": self.lens,
            "horizon_sessions": self.horizon,
            "as_of": self.as_of.isoformat(),
            "anchor": round(self.anchor, 4),
            "realised_on": self.realised_on.isoformat(),
            "realised": round(self.realised, 4),
            "log_return": (
                round(math.log(self.realised / self.anchor), 6)
                if self.anchor > 0.0 and self.realised > 0.0
                else None
            ),
            "quantiles": self.quantiles,
            "pit": self.pit,
            "censored": self.censored,
        }


def _interpolated_pit(quantiles: Mapping[str, float], observed: float) -> tuple[float | None, str]:
    """PIT of ``observed`` against a published quantile grid.

    Linear interpolation between the published points. Coarse by construction: five
    points cannot resolve the tails, which is where the interesting failures live. It
    is reported anyway because a coarse measurement labelled coarse is worth more than
    no measurement, and censored where it would be a fiction.
    """
    points = sorted(
        (int(key[1:]) / 100.0, value) for key, value in quantiles.items() if key.startswith("q")
    )
    if len(points) < 2:
        return None, "fewer than two published quantiles"

    probabilities = [p for p, _ in points]
    levels = [v for _, v in points]

    if observed < levels[0]:
        return None, f"below the published q{int(probabilities[0] * 100):02d}"
    if observed > levels[-1]:
        return None, f"above the published q{int(probabilities[-1] * 100):02d}"

    for i in range(len(points) - 1):
        low, high = levels[i], levels[i + 1]
        if low <= observed <= high:
            if high == low:
                return probabilities[i], ""
            weight = (observed - low) / (high - low)
            return probabilities[i] + weight * (probabilities[i + 1] - probabilities[i]), ""
    return None, "could not be located on the published grid"


def _sessions_ahead(index: pd.DatetimeIndex, anchor: date, horizon: int) -> pd.Timestamp | None:
    """The date ``horizon`` sessions after ``anchor`` on this series' own calendar.

    Uses the realised trading calendar rather than a calendar-day approximation: the
    horizon is denominated in sessions, so counting anything else would score a
    forecast against the wrong day and do it silently.
    """
    stamp = pd.Timestamp(anchor)
    position = int(index.searchsorted(stamp, side="right")) - 1
    if position < 0:
        return None
    target = position + horizon
    if target >= len(index):
        return None
    return pd.Timestamp(index[target])


def collect(
    *,
    realised: Mapping[str, Mapping[str, pd.Series]],
    directory: Path | None = None,
) -> list[LiveObservation]:
    """Score every published forecast whose horizon has elapsed.

    Args:
        realised: ``{asset_id: {lens_code: price series}}`` — the outcomes, on the same
            basis the forecast was published on. A lens absent here is simply not
            scored; guessing a substitute series would score the engine against a
            price it never published.
        directory: Public-data root. Defaults to the configured one.
    """
    out: list[LiveObservation] = []

    for path in published_forecasts(directory):
        try:
            artifact = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(artifact, dict):
            continue

        try:
            published_on = date.fromisoformat(str(artifact.get("generated_at", ""))[:10])
        except ValueError:
            continue
        code = artifact.get("code")
        commit = code.get("commit") if isinstance(code, dict) else None

        assets = artifact.get("assets")
        if not isinstance(assets, dict):
            continue

        for asset_id, asset_block in assets.items():
            lenses = asset_block.get("lenses") if isinstance(asset_block, dict) else None
            if not isinstance(lenses, dict):
                continue
            for lens_code, lens_block in lenses.items():
                series = realised.get(asset_id, {}).get(lens_code)
                if series is None or series.empty:
                    continue
                out.extend(
                    _score_lens(
                        lens_block=lens_block,
                        series=series,
                        published_on=published_on,
                        commit=commit,
                        asset_id=asset_id,
                        lens_code=lens_code,
                    )
                )
    return out


def _score_lens(
    *,
    lens_block: Any,
    series: pd.Series,
    published_on: date,
    commit: str | None,
    asset_id: str,
    lens_code: str,
) -> list[LiveObservation]:
    if not isinstance(lens_block, dict):
        return []
    latest = lens_block.get("latest")
    dist = lens_block.get("distribution")
    if (
        not isinstance(latest, dict)
        or not isinstance(dist, dict)
        or dist.get("available") is not True
    ):
        return []

    try:
        as_of = date.fromisoformat(str(latest["as_of"]))
    except (KeyError, ValueError, TypeError):
        return []

    anchor = dist.get("anchor")
    horizons = dist.get("horizons")
    if not isinstance(anchor, int | float) or not isinstance(horizons, dict):
        return []

    index = series.dropna().index
    if not isinstance(index, pd.DatetimeIndex):
        return []

    out: list[LiveObservation] = []
    for key, block in horizons.items():
        if not str(key).isdigit() or not isinstance(block, dict):
            continue
        horizon = int(key)
        realised_on = _sessions_ahead(index, as_of, horizon)
        if realised_on is None:
            continue  # not yet elapsed on this calendar
        quantiles = block.get("quantiles")
        if not isinstance(quantiles, dict):
            continue
        grid = {k: float(v) for k, v in quantiles.items() if isinstance(v, int | float)}
        observed = float(series.dropna().loc[realised_on])
        pit, censored = _interpolated_pit(grid, observed)
        out.append(
            LiveObservation(
                published_on=published_on,
                commit=commit,
                asset_id=asset_id,
                lens=lens_code,
                horizon=horizon,
                as_of=as_of,
                anchor=float(anchor),
                realised_on=realised_on.date(),
                realised=observed,
                quantiles=grid,
                pit=pit,
                censored=censored or None,
            )
        )
    return out


def _thin(observations: Sequence[LiveObservation], horizon: int) -> int:
    """Count non-overlapping windows among same-horizon observations.

    Greedy from the earliest anchor: take one, skip every forecast whose window starts
    before it ends. The same convention :mod:`aurex.score.sampling` applies to the
    backtest, applied here for the same reason — consecutive nightly forecasts at a
    21-session horizon share twenty of their twenty-one days.
    """
    anchors = sorted(obs.as_of for obs in observations)
    if not anchors:
        return 0
    kept = 0
    frontier: date | None = None
    for anchor in anchors:
        if frontier is None or anchor >= frontier:
            kept += 1
            # Earliest calendar date the window can close on; see aurex.record. Using
            # the earliest bound keeps more windows than the true calendar would, so
            # this over-counts independence rather than under-counting it — the wrong
            # direction for a p-value, which is why none is reported below the
            # threshold regardless.
            frontier = earliest_elapse(anchor, horizon)
    return kept


def summarise(observations: Sequence[LiveObservation]) -> dict[str, Any]:
    """The live log's own report, with its counts and without a test it has not earned."""
    by_horizon: dict[int, list[LiveObservation]] = {}
    for obs in observations:
        by_horizon.setdefault(obs.horizon, []).append(obs)

    horizons: list[dict[str, Any]] = []
    for horizon in sorted(by_horizon):
        group = by_horizon[horizon]
        measured = [obs.pit for obs in group if obs.pit is not None]
        independent = _thin(group, horizon)
        horizons.append(
            {
                "horizon_sessions": horizon,
                "observations": len(group),
                "independent_observations": independent,
                "censored": sum(1 for obs in group if obs.censored),
                "mean_pit": round(sum(measured) / len(measured), 5) if measured else None,
                "test_possible": independent >= MIN_INDEPENDENT_FOR_TEST,
                "test_note": (
                    f"{independent} independent windows against a threshold of "
                    f"{MIN_INDEPENDENT_FOR_TEST} fixed in advance: no test is possible "
                    f"yet, and none is reported."
                    if independent < MIN_INDEPENDENT_FOR_TEST
                    else f"{independent} independent windows; a test is now admissible."
                ),
            }
        )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": "live",
        "total_observations": len(observations),
        "horizons": horizons,
        "conventions": {
            "what_this_is": (
                "Forecasts this engine published to a public git repository before the "
                "outcome existed, scored after it did. This is not the walk-forward "
                "backtest and is never pooled with it: the backtest simulates what the "
                "engine would have said, and this records what it did say."
            ),
            "why_separate": (
                "Pooling would make this sample look testable years earlier than it is, "
                "by diluting it with observations that carry a weaker guarantee. The "
                "counts are reported apart so a reader can see exactly how much of the "
                "evidence is of which kind."
            ),
            "pit": (
                "Interpolated on the published five-point quantile grid, not computed "
                "against the simulated ensemble, because the artifact carries quantiles "
                "rather than paths. Coarser than the backtest's PIT and not comparable "
                "with it point for point. A realised value outside the grid is recorded "
                "as censored rather than clamped."
            ),
            "crps": (
                "Not computed. CRPS skill needs the null's distribution for the same "
                "date and the published artifact does not carry one."
            ),
            "independence": (
                "Consecutive nightly forecasts overlap almost entirely at every horizon "
                "beyond a day. Independent counts are thinned to non-overlapping "
                "windows, the same rule the backtest follows."
            ),
        },
    }


def write_live_log(
    observations: Sequence[LiveObservation],
    directory: Path | None = None,
) -> Path:
    """Write ``public-data/live-log.json``."""
    target_dir = directory or config.PUBLIC_DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "live-log.json"
    payload = summarise(observations) | {"observations": [obs.describe() for obs in observations]}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
