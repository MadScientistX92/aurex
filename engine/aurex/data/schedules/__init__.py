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
    "GstEntry",
    "PolicyBreak",
    "ScheduleError",
    "as_date",
    "duty_on",
    "gst_on",
    "load_duty_schedule",
    "load_gst_schedule",
    "load_policy_breaks",
    "read_yaml",
    "require_provenance",
]
