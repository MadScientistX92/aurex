"""Parquet cache with a JSON provenance sidecar.

The cache is what makes ``aurex pipeline --dry-run`` work offline and what keeps the
test suite off the network. Each series is one parquet file plus one ``.meta.json``
holding its :class:`~aurex.data.base.SeriesMeta`.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from aurex import config
from aurex.data.base import LoadedSeries, SeriesMeta, SourceCitation


def _parse_meta(raw: dict[str, Any]) -> SeriesMeta:
    """Rebuild the sidecar's provenance, or raise so the entry reads as a miss.

    ``citation`` is read strictly. An entry written before citations existed has no
    such key, raises ``KeyError`` here, and is treated by :meth:`CacheStore.read` as a
    cache miss — which refetches it from a loader that does declare one. Defaulting the
    field instead would let a cached series publish with a confidence nobody recorded,
    which is the failure this rule exists to prevent.
    """
    cited = raw["citation"]
    return SeriesMeta(
        series_id=raw["series_id"],
        source_name=raw["source_name"],
        source_url=raw["source_url"],
        citation=SourceCitation(
            source_url=cited["source_url"],
            source_confidence=cited["source_confidence"],
            cite_as=cited.get("cite_as"),
            licence=cited.get("licence"),
        ),
        fetched_at=datetime.fromisoformat(raw["fetched_at"]),
        rows=int(raw["rows"]),
        start=date.fromisoformat(raw["start"]) if raw.get("start") else None,
        end=date.fromisoformat(raw["end"]) if raw.get("end") else None,
        has_ohlc=bool(raw.get("has_ohlc", False)),
        fallbacks=tuple(raw.get("fallbacks", ())),
    )


def _carries_ohlc(frame: pd.DataFrame) -> bool:
    return {"open", "high", "low", "close"} <= set(frame.columns)


class CacheStore:
    """Read/write cached series under ``root``."""

    def __init__(self, root: Path | None = None) -> None:
        # Resolved through the module rather than bound at import, for the same reason
        # PUBLIC_DATA_DIR is: one place to redirect. A test that redirects the output
        # but not the input silently exercises whatever the developer's own cache
        # happens to hold, which makes it pass or fail on the age of ambient data.
        self.root = root or config.CACHE_DIR

    def _paths(self, series_id: str) -> tuple[Path, Path]:
        return (
            self.root / f"{series_id}.parquet",
            self.root / f"{series_id}.meta.json",
        )

    def has(self, series_id: str) -> bool:
        data_path, meta_path = self._paths(series_id)
        return data_path.exists() and meta_path.exists()

    def read(self, series_id: str) -> LoadedSeries | None:
        """Return the cached series, or ``None`` if absent or unreadable."""
        data_path, meta_path = self._paths(series_id)
        if not (data_path.exists() and meta_path.exists()):
            return None
        try:
            frame = pd.read_parquet(data_path)
            meta = _parse_meta(json.loads(meta_path.read_text()))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            # A corrupt cache entry must not take down the pipeline; treat it as a
            # miss so the chain falls through to a live source.
            return None
        return LoadedSeries(frame=frame, meta=meta)

    def rows_only(self, series_id: str) -> pd.DataFrame | None:
        """The cached observations, ignoring whether their sidecar can still be read.

        Split out from :meth:`read` because the two are answering different questions.
        ``read`` asks "may this be *served*", and a series whose provenance cannot be
        parsed may not be — that is what makes the citation rule strict rather than
        advisory. This asks "is there history here worth keeping", and the answer does
        not depend on the sidecar at all, because :meth:`merge_write` replaces the
        provenance with the fresh fetch's regardless.

        The distinction has teeth for series whose history only accumulates. A daily
        report carries about four days of it, so treating a sidecar written before
        citations existed as a total miss would have silently discarded every earlier
        observation the cache had gathered — losing data to a provenance rule that was
        never about the data.
        """
        data_path, _ = self._paths(series_id)
        if not data_path.exists():
            return None
        try:
            frame: pd.DataFrame = pd.read_parquet(data_path)
        except (OSError, ValueError):
            return None
        return frame

    def write(self, series: LoadedSeries) -> None:
        """Persist ``series``, replacing any existing entry atomically."""
        self.root.mkdir(parents=True, exist_ok=True)
        data_path, meta_path = self._paths(series.meta.series_id)

        tmp_data = data_path.with_suffix(".parquet.tmp")
        tmp_meta = meta_path.with_suffix(".json.tmp")
        series.frame.to_parquet(tmp_data)
        tmp_meta.write_text(json.dumps(series.meta.to_dict(), indent=2, sort_keys=True))
        tmp_data.replace(data_path)
        tmp_meta.replace(meta_path)

    def merge_write(self, series: LoadedSeries) -> LoadedSeries:
        """Union fresh rows over any cached history, then persist.

        Sources differ in how much history they serve — LBMA goes back to 1968 while a
        Yahoo request may cover a year. Merging means a short fetch never truncates
        history we already hold. Fresh rows win on overlap.

        History is taken from :meth:`rows_only` rather than :meth:`read`, so an entry
        whose sidecar is no longer parseable still contributes its observations. Nothing
        about the served provenance is softened by that: every field below comes from
        the fetch that just answered, and ``has_ohlc`` is the one place the cached side
        still gets a vote — a merge of OHLC rows with close-only rows is close-only.
        """
        existing = self.rows_only(series.meta.series_id)
        if existing is None or existing.empty:
            self.write(series)
            return series

        cached_meta = self.read(series.meta.series_id)
        # Where the sidecar can be read it is the authority, because a source can serve
        # the four columns without them being a true session range. Where it cannot, the
        # rows themselves are the only evidence left, and reading them beats defaulting:
        # defaulting one way claims a range the cached half may not have, and the other
        # way silently degrades the volatility layer to close-only.
        cached_has_ohlc = (
            cached_meta.meta.has_ohlc if cached_meta is not None else _carries_ohlc(existing)
        )
        combined = pd.concat([existing, series.frame])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()

        meta = SeriesMeta(
            series_id=series.meta.series_id,
            source_name=series.meta.source_name,
            source_url=series.meta.source_url,
            # The fresh fetch's citation, not the cached one: the merged frame is served
            # under whichever route just answered, and that is the route to name.
            citation=series.meta.citation,
            fetched_at=series.meta.fetched_at,
            rows=len(combined),
            start=combined.index[0].date(),
            end=combined.index[-1].date(),
            has_ohlc=series.meta.has_ohlc and cached_has_ohlc,
            fallbacks=series.meta.fallbacks,
        )
        merged = LoadedSeries(frame=combined, meta=meta)
        self.write(merged)
        return merged
