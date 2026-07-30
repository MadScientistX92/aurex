"""Dated duty / GST / policy-break schedules.

Parity over twenty years cannot use scalar rates: GST did not exist before July 2017
and the import duty has moved ten times. These loaders resolve the rate in force on a
given date and enforce the provenance rule — every entry carries its own
``source_url`` and ``source_confidence``, and none inherits a table-level default.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from aurex.config import SCHEDULE_DIR

Confidence = Literal["primary", "secondary"]
VALID_CONFIDENCE: frozenset[str] = frozenset({"primary", "secondary"})


class ScheduleError(ValueError):
    """A schedule file is malformed or violates the provenance rule."""


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


def _read_yaml(name: str) -> dict[str, Any]:
    path: Path = SCHEDULE_DIR / name
    if not path.exists():
        raise ScheduleError(f"missing schedule file {path}")
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise ScheduleError(f"{name}: expected a mapping at top level")
    return loaded


def _require_provenance(name: str, index: int, raw: dict[str, Any]) -> Confidence:
    """Enforce the per-entry provenance rule."""
    url = raw.get("source_url")
    confidence = raw.get("source_confidence")
    if not url:
        raise ScheduleError(f"{name}[{index}]: missing source_url")
    if not confidence:
        raise ScheduleError(f"{name}[{index}]: missing source_confidence")
    if confidence not in VALID_CONFIDENCE:
        raise ScheduleError(
            f"{name}[{index}]: source_confidence {confidence!r} not in {sorted(VALID_CONFIDENCE)}"
        )
    return cast("Confidence", confidence)


def _as_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


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
    "Confidence",
    "DutyEntry",
    "GstEntry",
    "PolicyBreak",
    "ScheduleError",
    "duty_on",
    "gst_on",
    "load_duty_schedule",
    "load_gst_schedule",
    "load_policy_breaks",
]
