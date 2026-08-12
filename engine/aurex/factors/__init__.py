"""Driver attribution and the transmission chain.

Elastic-net loadings for attribution and scenario propagation only — never for
direction forecasting. Loadings ship with bootstrap confidence intervals and an
honest out-of-sample R-squared, including when it is near zero.

Two estimates live here and they are deliberately not combined. The weekly loadings
decompose a return into contemporaneous drivers. The monthly chain traces one of those
drivers through an economy to the same asset's local price. They share a starting point,
which means they measure overlapping things, which means adding them would count that
overlap twice. :func:`describe` therefore emits them as sibling blocks with the overlap
named, and there is no field anywhere in the output that totals them.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from aurex.assets.base import Asset
from aurex.factors import chain as chain_module
from aurex.factors import design as design_module
from aurex.factors import loadings as loadings_module
from aurex.factors.chain import ChainError, ChainEstimate
from aurex.factors.design import DesignError, WeeklyDesign
from aurex.factors.loadings import Attribution

__all__ = [
    "Attribution",
    "ChainError",
    "ChainEstimate",
    "DesignError",
    "WeeklyDesign",
    "describe",
    "estimate_chain",
    "estimate_loadings",
]


def estimate_loadings(
    asset: Asset,
    frames: dict[str, pd.DataFrame],
    *,
    unavailable: dict[str, str] | None = None,
    window: int = loadings_module.DEFAULT_WINDOW,
    draws: int = loadings_module.DEFAULT_DRAWS,
    seed: int = 4,
) -> Attribution:
    """Build the weekly design for ``asset`` and estimate everything on it."""
    built = design_module.build(asset, frames, unavailable=unavailable)
    return loadings_module.estimate(built, window=window, draws=draws, seed=seed)


def estimate_chain(
    asset: Asset,
    frames: dict[str, pd.DataFrame],
    *,
    horizons: tuple[int, ...] = chain_module.DEFAULT_HORIZONS,
    draws: int = chain_module.DEFAULT_DRAWS,
    seed: int = 11,
) -> ChainEstimate | None:
    """Estimate the asset's declared chain, or ``None`` where it declares none."""
    declared = asset.transmission_chain
    if declared is None:
        return None
    return chain_module.estimate(declared, frames, horizons=horizons, draws=draws, seed=seed)


def describe(
    asset: Asset,
    attribution: Attribution,
    chain: ChainEstimate | None,
) -> dict[str, Any]:
    """The artifact block for one asset's attribution.

    ``chain`` is ``None`` where the asset declares none or where its inputs did not
    resolve, and the block says which. An absent chain is reported as absent; it is never
    replaced by the loadings, which answer a different question.
    """
    return {
        "asset": asset.describe(),
        "attribution": attribution.describe(),
        "chain": None if chain is None else chain.describe(),
        "what_this_is_not": (
            "Attribution, not a forecast. The loadings decompose a return that has "
            "already happened into drivers observed in the same week; the predictive "
            "out-of-sample R-squared beside them is what the same factor set achieves "
            "when asked to forecast instead, and it is reported so the distinction has a "
            "number rather than an assurance behind it."
        ),
        "never_summed": (
            "The chain starts from a driver that is also in the weekly factor set, so "
            "the direct loading and the chain are alternative decompositions of one "
            "overlapping thing. They are published side by side and are never added. "
            "Nothing in this artifact totals a chain response with a factor loading."
        ),
    }
