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

import pandas as pd

from aurex.data.base import DataUnavailableError, LoadedSeries, Loader
from aurex.data.cache import CacheStore
from aurex.data.freshness import SeriesFreshness

log = logging.getLogger(__name__)


def clip_to_window(series: LoadedSeries, start: date, end: date) -> LoadedSeries:
    """Return only the observations inside ``[start, end]``, provenance included.

    The cache is a union of every fetch ever made on the machine it lives on, and
    :meth:`CacheStore.merge_write` returns that union. Serving it unclipped made the
    resolved sample a property of the *machine* rather than of the command: three runs
    of one published command returned three different sample starts — 2006-08-04 and
    2000-01-04 on the same Mac either side of a cache extension, and 2005-01-04 on a
    cacheless runner, which is the only one that got what the code asked for. Every
    expanding-window fit then sees a different history, so every number built on one
    moves. A published headline that changes with the local cache is not reproducible
    from its own recorded command, which is the failure this exists to close.

    ``meta`` is rebuilt from the clipped frame rather than carried over. The cache
    still holds — and still accumulates — everything it did before; what changes is
    that ``rows``, ``start`` and ``end`` now describe what the caller was handed,
    because provenance describing a wider frame than the one returned is the same
    class of error as an artifact naming a source it did not use.
    """
    # End of ``end``, not its midnight: a source that stamps observations intraday
    # would otherwise lose the last day it was asked for.
    last_moment = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    frame = series.frame.loc[pd.Timestamp(start) : last_moment]
    if len(frame) == len(series.frame):
        return series
    index = frame.index
    meta = dataclasses.replace(
        series.meta,
        rows=len(frame),
        start=index[0].date() if len(frame) else None,
        end=index[-1].date() if len(frame) else None,
    )
    return LoadedSeries(frame=frame, meta=meta)


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
        """Return the series over ``[start, end]``, preferring live sources unless ``offline``.

        The window is a bound on what comes back, not merely on what is fetched. Every
        path out of this method goes through :func:`clip_to_window`, including the
        offline and cache-fallback ones, so the sample a caller grades is decided by
        the arguments rather than by the history the local cache happens to hold.

        Raises:
            DataUnavailableError: No source could supply observations in the window.
        """
        if offline:
            cached = self.cache.read(self.series_id)
            if cached is None:
                raise DataUnavailableError(
                    f"{self.series_id}: offline mode and no cached copy at {self.cache.root}"
                )
            clipped = clip_to_window(cached, start, end)
            if clipped.frame.empty:
                raise DataUnavailableError(
                    f"{self.series_id}: offline mode and the cached copy "
                    f"({cached.meta.start}..{cached.meta.end}) holds nothing in {start}..{end}"
                )
            return clipped

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

            # Merged first, so the cache keeps accumulating history a later run with a
            # wider window can use, and clipped second, so this run is handed only what
            # it asked for. The two are not in tension: the store is a superset by
            # design and the return value is not.
            merged = clip_to_window(self.cache.merge_write(fetched), start, end)
            if merged.frame.empty:
                # The source answered and had nothing in the window. That is a decline
                # like any other — recording it and trying the next source is what the
                # chain is for, and returning the unclipped union instead is exactly
                # the substitution this method exists to refuse.
                reason = f"{loader.source_name}: no observations in {start}..{end}"
                log.warning("%s: source declined — %s", self.series_id, reason)
                attempts.append(reason)
                continue
            meta = dataclasses.replace(merged.meta, fallbacks=tuple(attempts))
            return LoadedSeries(frame=merged.frame, meta=meta)

        cached = self.cache.read(self.series_id)
        if cached is not None:
            clipped = clip_to_window(cached, start, end)
            if clipped.frame.empty:
                attempts.append(
                    f"cache ({cached.meta.start}..{cached.meta.end}): "
                    f"no observations in {start}..{end}"
                )
            else:
                log.warning(
                    "%s: all %d sources failed; serving cached copy fetched %s",
                    self.series_id,
                    len(self.loaders),
                    cached.meta.fetched_at.isoformat(),
                )
                meta = dataclasses.replace(
                    clipped.meta,
                    fallbacks=(*attempts, "served from cache: all live sources failed"),
                )
                return LoadedSeries(frame=clipped.frame, meta=meta)

        raise DataUnavailableError(
            f"{self.series_id}: all sources failed and no cache. Tried:\n  " + "\n  ".join(attempts)
        )
