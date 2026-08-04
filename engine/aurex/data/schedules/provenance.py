"""The provenance rule, shared by everything that carries a sourced constant.

Every entry in a schedule or a routes table carries its own ``source_url`` and its own
``source_confidence``. No entry inherits a table-level default, because a default is how
one verified citation ends up standing behind nine unverified rows.

Extracted from :mod:`aurex.data.schedules` when routes arrived so that the duty
schedule, the GST schedule and the routes table are guarded by the *same function*
rather than by three similar ones. A second implementation of this rule is a second
place for it to be relaxed.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from aurex.config import SCHEDULE_DIR

Confidence = Literal["primary", "secondary"]
VALID_CONFIDENCE: frozenset[str] = frozenset({"primary", "secondary"})


class ScheduleError(ValueError):
    """A schedule or routes file is malformed or violates the provenance rule."""


def read_yaml(name: str) -> dict[str, Any]:
    path: Path = SCHEDULE_DIR / name
    if not path.exists():
        raise ScheduleError(f"missing schedule file {path}")
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise ScheduleError(f"{name}: expected a mapping at top level")
    return loaded


def require_provenance(name: str, index: int | str, raw: dict[str, Any]) -> Confidence:
    """Enforce the per-entry provenance rule.

    ``index`` is an int for a positional schedule entry and a string for a keyed one,
    so the error message points at the row a human can find.
    """
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


def as_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))
