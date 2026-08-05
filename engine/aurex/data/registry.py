"""Series resolution across assets and shared macro data.

This module knows nothing about any particular asset. It merges what the requested
assets declare in ``price_sources`` with the shared macro table and hands back one
chain per series id. An asset's own module is the only place its series are named.
"""

from __future__ import annotations

from collections.abc import Iterable

from aurex.assets.base import Asset
from aurex.data.cache import CacheStore
from aurex.data.chain import SourceChain
from aurex.data.freshness import SeriesFreshness
from aurex.data.macro import macro_chains


def chains_for(
    assets: Iterable[Asset],
    cache: CacheStore | None = None,
    *,
    include_macro: bool = True,
) -> dict[str, SourceChain]:
    """Every chain needed to run ``assets``.

    Only macro series some asset actually references in its ``factor_set`` are
    loaded — there is no reason to fetch VIX for an asset that never looks at it,
    and it keeps a self-contained asset genuinely self-contained.

    Asset-declared sources win on collision: an asset that has a better route to a
    series than the shared macro table is entitled to use it.
    """
    store = cache or CacheStore()
    assets = tuple(assets)

    chains: dict[str, SourceChain] = {}
    if include_macro:
        wanted = {factor.series_id for asset in assets for factor in asset.factor_set}
        chains = {sid: chain for sid, chain in macro_chains(store).items() if sid in wanted}

    for asset in assets:
        chains.update(asset.price_sources(store))
    return chains


def blocking_series(assets: Iterable[Asset]) -> frozenset[str]:
    """Series ids whose staleness would fabricate a *published price*.

    Two kinds and no more: the price series every lens is composed from, and the
    exchange rate of any lens that converts one. Both feed the number a reader would
    act on, so both refuse rather than degrade. A stale factor does not appear here —
    it moves a fitted loading, which is a different and smaller problem, and it is
    reported without blocking.

    Derived from what the assets declare rather than maintained as a separate list, so
    an asset that gains a lens gains the guard over that lens's exchange rate in the
    same commit.
    """
    out: set[str] = set()
    for asset in assets:
        out.add(asset.price_series_id)
        for lens in asset.currency_lenses:
            if lens.fx_series_id is not None:
                out.add(lens.fx_series_id)
    return frozenset(out)


def freshness_for(
    assets: Iterable[Asset],
    cache: CacheStore | None = None,
) -> dict[str, SeriesFreshness | None]:
    """Declared staleness tolerance per series, for every series ``assets`` need.

    Read off the chains rather than from a parallel table: a tolerance that can drift
    away from the loaders it describes is a tolerance that will.
    """
    return {sid: chain.freshness for sid, chain in chains_for(assets, cache).items()}


def chain_for(
    series_id: str,
    assets: Iterable[Asset],
    cache: CacheStore | None = None,
) -> SourceChain:
    """Resolve a single series id across ``assets`` plus macro."""
    chains = chains_for(assets, cache)
    try:
        return chains[series_id]
    except KeyError:
        raise KeyError(f"unknown series {series_id!r}; known: {sorted(chains)}") from None
