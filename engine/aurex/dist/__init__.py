"""Return distributions: filtered historical simulation, with the paths kept.

The pieces, in the order the pipeline uses them:

* :mod:`~aurex.dist.fhs` — block-bootstrap standardised residuals through a fitted
  variance recursion, walk the result into prices session by session, and apply any
  daily price limit on the way.
* :mod:`~aurex.dist.paths` — the ensemble that comes back. Terminal quantiles are one
  method on it rather than the whole of it, which is the §18 correction in one line.
* :mod:`~aurex.dist.passage` — barrier statistics: what share of paths touch a level,
  how long they take, and what the survivors are holding.
* :mod:`~aurex.dist.copula` — a bivariate t-copula, for a price and an exchange rate
  whose joint tail is not the product of two marginals.

Which series these run on is the asset's business, not this package's. Nothing here
names one, and ``tests/test_asset_abstraction.py`` fails the build if that changes.
"""

from __future__ import annotations

from aurex.dist.copula import (
    DependenceMode,
    TCopula,
    fit_t_copula,
    joint_shocks,
    pseudo_observations,
)
from aurex.dist.fhs import (
    SimulationSpec,
    block_indices,
    bootstrap_shocks,
    paths_from_returns,
    simulate,
)
from aurex.dist.passage import (
    Direction,
    FirstPassage,
    Headline,
    first_passage,
    headline_statistic,
    margin_call_barrier,
)
from aurex.dist.paths import DEFAULT_QUANTILES, PathEnsemble

__all__ = [
    "DEFAULT_QUANTILES",
    "DependenceMode",
    "Direction",
    "FirstPassage",
    "Headline",
    "PathEnsemble",
    "SimulationSpec",
    "TCopula",
    "block_indices",
    "bootstrap_shocks",
    "first_passage",
    "fit_t_copula",
    "headline_statistic",
    "joint_shocks",
    "margin_call_barrier",
    "paths_from_returns",
    "pseudo_observations",
    "simulate",
]
