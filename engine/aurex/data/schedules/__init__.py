"""Dated duty / GST / policy-break schedules.

Parity over twenty years cannot use scalar rates: GST did not exist before July 2017
and the import duty has moved ten times. These loaders resolve the rate in force on a
given date and enforce the provenance rule — every entry carries its own
``source_url`` and ``source_confidence``, and none inherits a table-level default.

The rule itself now lives in :mod:`aurex.data.schedules.provenance`, because the routes
table in :mod:`aurex.routes` is held to exactly the same standard and two
implementations of one rule is one too many.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache

from aurex.data.schedules.provenance import (
    VALID_CONFIDENCE,
    Confidence,
    ScheduleError,
    as_date,
    read_yaml,
    require_provenance,
)

_read_yaml = read_yaml
_require_provenance = require_provenance
_as_date = as_date


@dataclass(frozen=True, slots=True)
class DutyEntry:
    effective_from: date
    total: float
    components: dict[str, float]
    source_url: str
    source_confidence: Confidence
    note: str | None = None


@dataclass(frozen=True, slots=True)
class GstEntry:
    effective_from: date
    metal: float
    making_charges: float
    source_url: str
    source_confidence: Confidence
    note: str | None = None


@dataclass(frozen=True, slots=True)
class FuelExciseEntry:
    """Total central excise per litre on the two transport fuels, from one date."""

    effective_from: date
    petrol: float
    diesel: float
    source_url: str
    source_confidence: Confidence
    note: str | None = None

    @property
    def combined(self) -> float:
        """The two summed, which is the form the chain uses as a control.

        Summed rather than averaged so the units stay rupees per litre of tax levied
        across the pair, and because the two move together in every entry in the file:
        splitting them would put two near-collinear controls in a regression that has
        two hundred observations.
        """
        return self.petrol + self.diesel


@dataclass(frozen=True, slots=True)
class ScheduleGap:
    """A window whose level is unknown to a schedule, and why."""

    start: date
    end: date
    reason: str
    source_url: str
    source_confidence: Confidence

    def contains(self, when: date) -> bool:
        return self.start <= when <= self.end


@dataclass(frozen=True, slots=True)
class PolicyBreak:
    date: date
    kind: str
    description: str
    expected_effect: str
    source_url: str


@lru_cache(maxsize=1)
def load_duty_schedule() -> tuple[DutyEntry, ...]:
    """Duty entries, ascending by effective date."""
    raw = _read_yaml("duty.yaml")
    entries: list[DutyEntry] = []
    for i, item in enumerate(raw.get("schedule", [])):
        confidence = _require_provenance("duty.yaml", i, item)
        entries.append(
            DutyEntry(
                effective_from=_as_date(item["effective_from"]),
                total=float(item["total"]),
                components={k: float(v) for k, v in (item.get("components") or {}).items()},
                source_url=str(item["source_url"]),
                source_confidence=confidence,
                note=item.get("note"),
            )
        )
    if not entries:
        raise ScheduleError("duty.yaml: empty schedule")
    entries.sort(key=lambda e: e.effective_from)
    return tuple(entries)


@lru_cache(maxsize=1)
def load_gst_schedule() -> tuple[GstEntry, ...]:
    """GST entries, ascending by effective date."""
    raw = _read_yaml("gst.yaml")
    entries: list[GstEntry] = []
    for i, item in enumerate(raw.get("schedule", [])):
        confidence = _require_provenance("gst.yaml", i, item)
        entries.append(
            GstEntry(
                effective_from=_as_date(item["effective_from"]),
                metal=float(item["metal"]),
                making_charges=float(item["making_charges"]),
                source_url=str(item["source_url"]),
                source_confidence=confidence,
                note=item.get("note"),
            )
        )
    if not entries:
        raise ScheduleError("gst.yaml: empty schedule")
    entries.sort(key=lambda e: e.effective_from)
    return tuple(entries)


@lru_cache(maxsize=1)
def load_fuel_excise() -> tuple[tuple[FuelExciseEntry, ...], tuple[ScheduleGap, ...]]:
    """Fuel excise levels and the windows where the level is unknown.

    The gaps are loaded with the entries and held to the same provenance rule, because a
    declared hole is a claim about the world too — it says changes happened here that
    nobody has cited — and a claim with no source behind it is the thing this rule
    exists to prevent, whether it is a number or the absence of one.
    """
    raw = _read_yaml("fuel_excise.yaml")

    entries: list[FuelExciseEntry] = []
    for i, item in enumerate(raw.get("schedule", [])):
        confidence = _require_provenance("fuel_excise.yaml", i, item)
        entries.append(
            FuelExciseEntry(
                effective_from=_as_date(item["effective_from"]),
                petrol=float(item["petrol"]),
                diesel=float(item["diesel"]),
                source_url=str(item["source_url"]),
                source_confidence=confidence,
                note=item.get("note"),
            )
        )
    if not entries:
        raise ScheduleError("fuel_excise.yaml: empty schedule")

    gaps: list[ScheduleGap] = []
    for i, item in enumerate(raw.get("gaps", [])):
        confidence = _require_provenance("fuel_excise.yaml/gaps", i, item)
        start, end = _as_date(item["from"]), _as_date(item["until"])
        if end < start:
            raise ScheduleError(f"fuel_excise.yaml/gaps[{i}]: ends {end} before it starts {start}")
        gaps.append(
            ScheduleGap(
                start=start,
                end=end,
                reason=str(item["reason"]).strip(),
                source_url=str(item["source_url"]),
                source_confidence=confidence,
            )
        )

    entries.sort(key=lambda e: e.effective_from)
    return tuple(entries), tuple(gaps)


def fuel_excise_on(when: date) -> FuelExciseEntry | None:
    """Excise in force on ``when``; ``None`` before the schedule starts or inside a gap.

    Deliberately does not carry the last known level across a gap. Inside one the honest
    answer is that this repository does not know the rate, and a control silently held
    constant through the largest tax change in the sample would be worse than no control.
    """
    entries, gaps = load_fuel_excise()
    if any(gap.contains(when) for gap in gaps):
        return None
    applicable = [entry for entry in entries if entry.effective_from <= when]
    return applicable[-1] if applicable else None


@lru_cache(maxsize=1)
def load_policy_breaks() -> tuple[PolicyBreak, ...]:
    """Known structural breaks, ascending by date."""
    raw = _read_yaml("policy_breaks.yaml")
    breaks = [
        PolicyBreak(
            date=_as_date(item["date"]),
            kind=str(item["kind"]),
            description=str(item["description"]).strip(),
            expected_effect=str(item["expected_effect"]),
            source_url=str(item["source_url"]),
        )
        for item in raw.get("breaks", [])
    ]
    breaks.sort(key=lambda b: b.date)
    return tuple(breaks)


def duty_on(when: date) -> DutyEntry | None:
    """Duty in force on ``when``; ``None`` before the ad valorem regime began."""
    applicable = [e for e in load_duty_schedule() if e.effective_from <= when]
    return applicable[-1] if applicable else None


def gst_on(when: date) -> GstEntry | None:
    """GST in force on ``when``; ``None`` before the 2017 rollout."""
    applicable = [e for e in load_gst_schedule() if e.effective_from <= when]
    return applicable[-1] if applicable else None


__all__ = [
    "VALID_CONFIDENCE",
    "Confidence",
    "DutyEntry",
    "FuelExciseEntry",
    "GstEntry",
    "PolicyBreak",
    "ScheduleError",
    "ScheduleGap",
    "as_date",
    "duty_on",
    "fuel_excise_on",
    "gst_on",
    "load_duty_schedule",
    "load_fuel_excise",
    "load_gst_schedule",
    "load_policy_breaks",
    "read_yaml",
    "require_provenance",
]
