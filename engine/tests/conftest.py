"""Shared fixtures. No test in this suite touches the network."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aurex.data.base import LoadedSeries, SeriesMeta
from aurex.data.cache import CacheStore

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_dir() -> Path:
    return FIXTURE_DIR


@pytest.fixture
def ibja_text() -> str:
    return (FIXTURE_DIR / "ibja_report_2026-07-30.txt").read_text()


@pytest.fixture
def cache(tmp_path: Path) -> CacheStore:
    return CacheStore(tmp_path / "cache")


def make_series(
    series_id: str = "demo",
    *,
    start: str = "2026-01-01",
    periods: int = 30,
    value: float = 100.0,
    column: str = "close",
    source_name: str = "test:source",
    has_ohlc: bool = False,
) -> LoadedSeries:
    """Build a deterministic :class:`LoadedSeries` for tests."""
    index = pd.bdate_range(start, periods=periods, name="date")
    frame = pd.DataFrame({column: np.full(len(index), value, dtype=float)}, index=index)
    return LoadedSeries(
        frame=frame,
        meta=SeriesMeta(
            series_id=series_id,
            source_name=source_name,
            source_url="https://example.invalid/series",
            fetched_at=datetime(2026, 7, 31, tzinfo=UTC),
            rows=len(frame),
            start=index[0].date(),
            end=index[-1].date(),
            has_ohlc=has_ohlc,
        ),
    )


@pytest.fixture
def sample_series() -> LoadedSeries:
    return make_series()


class StubLoader:
    """A loader that returns a canned series or raises on demand."""

    def __init__(
        self,
        series_id: str,
        source_name: str,
        *,
        result: LoadedSeries | None = None,
        error: Exception | None = None,
    ) -> None:
        self.series_id = series_id
        self.source_name = source_name
        self._result = result
        self._error = error
        self.calls = 0

    def fetch(self, start: date, end: date) -> LoadedSeries:
        self.calls += 1
        if self._error is not None:
            raise self._error
        assert self._result is not None
        # Real loaders stamp their own identity onto the meta they return; the stub
        # must do the same or the chain's provenance assertions are meaningless.
        return LoadedSeries(
            frame=self._result.frame,
            meta=dataclasses.replace(self._result.meta, source_name=self.source_name),
        )


@pytest.fixture
def stub_loader_factory() -> Iterator[type[StubLoader]]:
    yield StubLoader
