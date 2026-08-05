"""Source-priority chains.

The spec names a preferred source per series. In practice those sources rate-limit,
change shape, or go down — Yahoo Finance was returning HTTP 429 while this was being
built. A chain tries each source in order and falls back to the cache, recording
which one actually answered so the artifact never implies a provenance it doesn't have.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from datetime import date

from aurex.data.base import DataUnavailableError, LoadedSeries, Loader
from aurex.data.cache import CacheStore
from aurex.data.freshness import SeriesFreshness

log = logging.getLogger(__name__)


class SourceChain:
    """Resolve one series from the first source that answers.

    The fallback behaviour this class exists for is also the reason
    :mod:`aurex.data.freshness` exists: serving a cached copy is right for a human at a
    terminal and dangerous for an unattended nightly run, which would publish it as
    today's price. The chain therefore carries the series' declared staleness tolerance
    so the guard can find it beside the loaders that motivated it. It is optional here
    and fails closed there — an undeclared tolerance blocks publication of a blocking
    series rather than defaulting to a permissive one.

    Args:
        series_id: Stable identifier; also the cache key.
        loaders: Sources in preference order.
        cache: Store to merge into. A fresh :class:`CacheStore` if omitted.
        freshness: How far behind the run date this series may fall.
    """

    def __init__(
        self,
        series_id: str,
        loaders: Sequence[Loader],
        cache: CacheStore | None = None,
        *,
        freshness: SeriesFreshness | None = None,
    ) -> None:
        if not loaders:
            raise ValueError(f"{series_id}: chain needs at least one loader")
        self.series_id = series_id
        self.loaders = tuple(loaders)
        self.cache = cache or CacheStore()
        self.freshness = freshness

    def load(self, start: date, end: date, *, offline: bool = False) -> LoadedSeries:
        """Return the series, preferring live sources unless ``offline``.

        Raises:
            DataUnavailableError: Every source failed and nothing is cached.
        """
        if offline:
            cached = self.cache.read(self.series_id)
            if cached is None:
                raise DataUnavailableError(
                    f"{self.series_id}: offline mode and no cached copy at {self.cache.root}"
                )
            return cached

        attempts: list[str] = []
        for loader in self.loaders:
            try:
                fetched = loader.fetch(start, end)
            except Exception as exc:
                # Deliberately broad: any failure mode — rate limit, schema drift,
                # DNS — means the same thing here, which is "try the next source".
                reason = f"{loader.source_name}: {type(exc).__name__}: {exc}"
                log.warning("%s: source declined — %s", self.series_id, reason)
                attempts.append(reason)
                continue

            merged = self.cache.merge_write(fetched)
            meta = dataclasses.replace(merged.meta, fallbacks=tuple(attempts))
            return LoadedSeries(frame=merged.frame, meta=meta)

        cached = self.cache.read(self.series_id)
        if cached is not None:
            log.warning(
                "%s: all %d sources failed; serving cached copy fetched %s",
                self.series_id,
                len(self.loaders),
                cached.meta.fetched_at.isoformat(),
            )
            meta = dataclasses.replace(
                cached.meta,
                fallbacks=(*attempts, "served from cache: all live sources failed"),
            )
            return LoadedSeries(frame=cached.frame, meta=meta)

        raise DataUnavailableError(
            f"{self.series_id}: all sources failed and no cache. Tried:\n  " + "\n  ".join(attempts)
        )
