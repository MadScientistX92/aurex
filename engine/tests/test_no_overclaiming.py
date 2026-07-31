"""The §0 guard.

Aurex's credibility is the product. This test fails the build if overclaiming
vocabulary or point-forecast language reaches user-facing text. It is deliberately
crude — a curated phrase list rather than anything clever — because a guard nobody
can read is a guard nobody maintains.

Phrases are chosen so they cannot occur innocently. "prediction" alone is fine:
Aurex reads *prediction markets* for scenario priors, and the philosophy section
says the engine never predicts a price. Both must keep working.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from aurex.config import ENGINE_ROOT, REPO_ROOT

#: Marketing claims banned by §0.
BANNED_MARKETING = [
    "institutional-grade",
    "institutional grade",
    "beats the market",
    "beat the market",
    "ai-powered prediction",
    "ai powered prediction",
    "guaranteed return",
    "risk-free profit",
    "proven strategy",
    "never loses",
]

#: Point-forecast language banned by §0 — a number without a distribution behind it.
BANNED_POINT_FORECAST = [
    "price target",
    "we predict that",
    "our prediction is",
    "expected price of",
    "will reach rs",
    "will hit rs",
    "target price",
]

BANNED = BANNED_MARKETING + BANNED_POINT_FORECAST

#: This file necessarily contains the banned phrases.
EXEMPT_FILES = {"test_no_overclaiming.py"}

SCANNED_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".json", ".ts", ".tsx"}
SKIPPED_DIRS = {".venv", ".git", "node_modules", "__pycache__", ".cache", ".ruff_cache"}


def _scannable(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in SCANNED_SUFFIXES
        and path.name not in EXEMPT_FILES
        and not any(part in SKIPPED_DIRS for part in path.parts)
    ]


def scannable_files() -> list[Path]:
    files = _scannable(ENGINE_ROOT)
    files += _scannable(REPO_ROOT / "public-data")
    files += _scannable(REPO_ROOT / "web")
    readme = REPO_ROOT / "README.md"
    if readme.exists():
        files.append(readme)
    return files


@pytest.mark.parametrize("phrase", BANNED)
def test_banned_phrase_appears_nowhere(phrase: str) -> None:
    pattern = re.compile(re.escape(phrase), re.IGNORECASE)
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{i}"
        for path in scannable_files()
        for i, line in enumerate(path.read_text(errors="replace").splitlines(), start=1)
        if pattern.search(line)
    ]
    assert not offenders, f"banned phrase {phrase!r} found at: {offenders}"


def test_the_guard_actually_scans_something() -> None:
    """A guard over an empty file list passes vacuously and protects nothing."""
    files = scannable_files()
    assert len(files) > 10, f"only {len(files)} files scanned; the guard is not wired up"
    assert any(path.suffix == ".py" for path in files)
    assert any(path.suffix == ".yaml" for path in files)


def test_legitimate_uses_of_predict_are_not_caught() -> None:
    """Prediction markets and the philosophy statement must survive the guard."""
    allowed = [
        "priors come from prediction markets, not from us",
        "Aurex never predicts a price",
        "Short-horizon price direction is not reliably forecastable",
    ]
    for text in allowed:
        for phrase in BANNED:
            assert phrase not in text.lower(), f"{phrase!r} wrongly matches {text!r}"
