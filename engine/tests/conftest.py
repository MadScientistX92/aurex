"""Shared fixtures. No test in this suite touches the network."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aurex.data.base import LoadedSeries, SeriesMeta, SourceCitation
from aurex.data.cache import CacheStore

FIXTURE_DIR = Path(__file__).parent / "fixtures"

#: Provenance for made-up data. Every fixture carries one because the loader protocol
#: requires it of every real source, and a test helper that could skip it would be the
#: one place the rule does not hold.
TEST_CITATION = SourceCitation(
    source_url="https://example.invalid/series",
    source_confidence="secondary",
)


@pytest.fixture
def fixture_dir() -> Path:
    return FIXTURE_DIR


@pytest.fixture
def ibja_text() -> str:
    return (FIXTURE_DIR / "ibja_report_2026-07-30.txt").read_text()


@pytest.fixture
def cache(tmp_path: Path) -> CacheStore:
    return CacheStore(tmp_path / "cache")


def wandering_series(
    end: str,
    *,
    series_id: str = "xauusd",
    periods: int = 600,
    column: str = "close",
    trailing_nans: int = 0,
    seed: int = 4,
) -> LoadedSeries:
    """A price series ending on ``end`` that actually moves.

    Prices wander rather than sitting flat, because everything downstream of a variance
    divides by it: a constant fixture fails a GARCH fit with a linear algebra error
    instead of exercising whatever was under test.
    """
    index = pd.bdate_range(end=end, periods=periods, name="date")
    rng = np.random.default_rng(seed)
    values = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=len(index))))
    if trailing_nans:
        values[-trailing_nans:] = np.nan
    frame = pd.DataFrame({column: values}, index=index)
    return LoadedSeries(
        frame=frame,
        meta=SeriesMeta(
            series_id=series_id,
            source_name="test:source",
            source_url="https://example.invalid/series",
            citation=TEST_CITATION,
            fetched_at=datetime.now(UTC),
            rows=len(frame),
            start=index[0].date(),
            end=index[-1].date(),
        ),
    )


def _pinned_cache(root: Path, monkeypatch: pytest.MonkeyPatch, *, last_observation: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    store = CacheStore(root)
    store.write(wandering_series(last_observation, series_id="xauusd"))
    store.write(wandering_series(last_observation, series_id="usdinr", seed=9))
    monkeypatch.setattr("aurex.config.CACHE_DIR", root)
    return root


@pytest.fixture
def stale_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A cache whose prices stopped a fortnight ago, wherever today happens to land.

    Pinned rather than borrowed from the developer's own ``.cache/``. Tests that
    asserted a refusal used to read whatever was lying there, so they passed only while
    that data happened to be old — one live pipeline run turned three assertions about
    refusing into assertions about publishing, and the guard's tests stopped testing
    the guard without failing.
    """
    stale = (date.today() - timedelta(days=14)).isoformat()
    return _pinned_cache(tmp_path / "stale-cache", monkeypatch, last_observation=stale)


@pytest.fixture
def fresh_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A cache reaching today, so the publishing path is exercised on its own terms."""
    return _pinned_cache(
        tmp_path / "fresh-cache", monkeypatch, last_observation=date.today().isoformat()
    )


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
            citation=TEST_CITATION,
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


def simulate_gjr_path(
    *,
    periods: int = 2_000,
    omega: float = 2.0e-6,
    alpha: float = 0.04,
    gamma: float = 0.08,
    beta: float = 0.88,
    seed: int = 7,
    start: str = "2016-01-01",
) -> pd.DataFrame:
    """Returns *and* the latent sigma that generated them, from a known GJR process.

    The sigma column is what makes a recovery test meaningful: a fitted conditional
    sigma correlates only about 0.2 with the absolute return, because ``|r|`` is a
    single noisy draw from the variance rather than the variance itself. Checking the
    fit against ``|r|`` would therefore fail on correct code.

    Normal innovations, deliberately: the recursion is what is under test, and a
    fat-tailed innovation makes parameter recovery noisy for unrelated reasons.
    """
    rng = np.random.default_rng(seed)
    variance = omega / (1.0 - (alpha + gamma / 2.0 + beta))
    returns = np.empty(periods)
    sigma = np.empty(periods)

    for t in range(periods):
        sigma[t] = np.sqrt(variance)
        shock = sigma[t] * rng.standard_normal()
        returns[t] = shock
        variance = omega + (alpha + gamma * (shock < 0.0)) * shock**2 + beta * variance

    index = pd.bdate_range(start, periods=periods, name="date")
    return pd.DataFrame({"returns": returns, "sigma": sigma}, index=index)


def simulate_gjr_returns(**kwargs: object) -> pd.Series:
    """Just the return series from :func:`simulate_gjr_path`."""
    return simulate_gjr_path(**kwargs)["returns"].rename("returns")  # type: ignore[arg-type]


@pytest.fixture
def gjr_returns() -> pd.Series:
    return simulate_gjr_returns()


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
        self.citation = TEST_CITATION
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
