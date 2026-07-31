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
