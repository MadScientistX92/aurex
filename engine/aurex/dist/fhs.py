"""Filtered historical simulation, with the paths kept.

The filter is the fitted volatility model; the history is its standardised residuals.
Resampling those residuals rather than drawing from a parametric law is the whole
point: the simulated tail is then the *observed* tail, reshaped by today's conditional
variance rather than by an assumption about kurtosis.

**Blocks rather than single draws.** Residuals are resampled in contiguous blocks, so
whatever serial structure survived the filter — a cluster of one-sided days, a run of
small moves — survives into the simulation. The bootstrap is circular: block starts
wrap around the end of the sample, which keeps every observation equally likely to be
drawn. A non-circular moving block quietly underweights the first and last few
residuals, which are the most recent ones.

**Sessions are constructed one at a time.** A daily price limit truncates the session
it lands in and carries the remainder into the next one (§18), so prices must exist
between steps. That is what :meth:`~aurex.assets.transforms.ReturnTransform.advance`
is for, and it is also why the loop below is over sessions rather than over paths.

**The pool is demeaned by default, and that is a §0 decision rather than a numerical
one.** Resampling the empirical residuals of a sample that rose draws from a pool with a
positive mean, and a positive mean per session compounds into a median that sits above
spot — several per cent at a quarter, on any series that trended for a decade. Nothing
fitted a drift, but the simulation has one, and a distribution whose median sits above
spot *is* a directional forecast however it got there. §0 says the null is the random
walk and that direction is not forecastable, so the default is the pool with its mean
removed. The drift-carrying pool remains available as :func:`residual_pool` with
``demean=False``, because "what the sample actually did" is a legitimate thing to
simulate; it is a declared option that lands in the artifact, not a silent default.

That choice costs something measurable and the cost is published rather than absorbed:
the long-horizon direction forecast was scoring partly on the drift leaking in, and it
scores worse without it. A benchmark that improves because a directional bias happened
to point the right way for eleven years is not a result the project keeps.

**The seed is recorded.** A distribution nobody can reproduce cannot be scored, and
§3's forecast log needs to be able to regenerate exactly what was published.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from aurex.assets.transforms import ReturnTransform
from aurex.dist.paths import PathEnsemble
from aurex.vol.base import FittedVol
from aurex.vol.limits import SessionLimit

#: Remove the pool's mean before resampling. See the module docstring: this is §0's
#: position on drift expressed as a default, and flipping it is a declared choice.
DEMEAN_BY_DEFAULT = True


@dataclass(frozen=True, slots=True)
class SimulationSpec:
    """How much simulation to run, and how to make it reproducible."""

    horizon_days: int
    n_paths: int = 20_000
    #: Contiguous residuals per draw. Roughly two trading weeks by default.
    block_length: int = 10
    seed: int = 20260731
    #: Whether the resampled pool had its mean removed. Part of the spec rather than a
    #: constructor argument somewhere upstream, because it changes where the simulated
    #: median sits and therefore has to reach the artifact with everything else that
    #: does.
    demean_residuals: bool = DEMEAN_BY_DEFAULT

    def __post_init__(self) -> None:
        for name in ("horizon_days", "n_paths", "block_length"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")

    def describe(self) -> dict[str, Any]:
        return {
            "horizon_sessions": self.horizon_days,
            "n_paths": self.n_paths,
            "block_length": self.block_length,
            "seed": self.seed,
            "demean_residuals": self.demean_residuals,
        }


@dataclass(frozen=True, slots=True)
class ResidualPool:
    """What a bootstrap is allowed to draw from, and the drift it carries.

    A type rather than an array, because "these residuals" and "these residuals with
    their mean removed" are numerically similar and produce distributions that differ by
    a directional forecast. Making the pool the thing that gets passed around means the
    drift policy travels with the numbers instead of being a keyword argument two call
    sites away, and it means a simulation can check that what it resampled matches what
    its spec is about to claim in the artifact.

    For a conditional model these are the standardised residuals of the fit. For the
    random walk they are the raw increments — the same object, because the question
    "does this pool carry a drift" is the same question for both, and asking it in one
    place is what stops a driftless null being scored against a drifting model.
    """

    residuals: pd.Series
    demeaned: bool
    #: What was subtracted. Zero when ``demeaned`` is false, and reported either way so
    #: the size of the drift left in is visible rather than merely implied.
    removed_mean: float

    @property
    def values(self) -> np.ndarray:
        drawn: np.ndarray = self.residuals.to_numpy(dtype=float)
        return drawn

    def describe(self) -> dict[str, Any]:
        return {
            "size": int(self.residuals.size),
            "demeaned": self.demeaned,
            "removed_mean": round(self.removed_mean, 8),
            "drift": (
                "removed: the pool mean was subtracted, so the simulation has no drift "
                "the model did not fit"
                if self.demeaned
                else "left in: the pool carries the sample's own mean, so the simulated "
                "median is displaced from spot by it"
            ),
        }


def residual_pool(
    residuals: pd.Series, *, demean: bool = DEMEAN_BY_DEFAULT
) -> ResidualPool:
    """Build the pool a simulation will resample, declaring its drift policy.

    The mean is taken over the finite entries only, so an excluded break — which enters
    the pool as ``NaN`` rather than as a zero — does not drag it toward zero.
    """
    finite = residuals[np.isfinite(residuals.to_numpy(dtype=float))]
    if finite.empty:
        raise ValueError("a residual pool needs at least one finite observation")

    mean = float(finite.mean()) if demean else 0.0
    return ResidualPool(
        residuals=finite - mean if demean else finite,
        demeaned=demean,
        removed_mean=mean,
    )


def block_indices(
    n_residuals: int, *, n_paths: int, horizon: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    """``(n_paths, horizon)`` indices into the residual sample, drawn in blocks.

    Circular: a block starting near the end wraps to the beginning, so every residual
    is equally likely to appear at every position.
    """
    if n_residuals < 2:
        raise ValueError(f"need at least 2 residuals to bootstrap, got {n_residuals}")

    effective_block = min(block_length, max(n_residuals, 1))
    n_blocks = int(np.ceil(horizon / effective_block))
    starts = rng.integers(0, n_residuals, size=(n_paths, n_blocks))
    offsets = np.arange(effective_block)

    drawn = (starts[:, :, np.newaxis] + offsets[np.newaxis, np.newaxis, :]) % n_residuals
    return drawn.reshape(n_paths, n_blocks * effective_block)[:, :horizon]


def bootstrap_shocks(
    pool: ResidualPool,
    *,
    n_paths: int,
    horizon: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Block-bootstrapped standardised shocks, ``(n_paths, horizon)``.

    Takes a :class:`ResidualPool` rather than an array on purpose. A bare array of
    residuals has no drift policy attached, and the caller who forgets to demean one
    gets a distribution that is quietly a directional forecast; refusing the array is
    what makes that impossible to do by accident.
    """
    if not isinstance(pool, ResidualPool):
        raise TypeError(
            "bootstrap_shocks needs a ResidualPool, not a bare array: the pool declares "
            "whether its mean was removed, and resampling a pool that still carries the "
            "sample's drift produces a directional forecast. Build one with "
            "residual_pool()."
        )
    clean = pool.values
    picks = block_indices(
        len(clean), n_paths=n_paths, horizon=horizon, block_length=block_length, rng=rng
    )
    drawn: np.ndarray = clean[picks]
    return drawn


def paths_from_returns(
    transform: ReturnTransform,
    anchor: float,
    returns: np.ndarray,
    *,
    session_limit: SessionLimit | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Walk returns into prices session by session, applying any daily limit.

    Returns the price array and a diagnostics block recording how often the limit
    bound — a simulation where it binds constantly is describing a market that is
    closed, and the reader should be told rather than left to assume it never bound.
    """
    n_paths, horizon = returns.shape
    prices = np.empty((n_paths, horizon), dtype=float)
    current = np.full(n_paths, float(anchor))
    carry = np.zeros(n_paths)
    locked_previous = np.zeros(n_paths, dtype=bool)
    locked_sessions = 0

    for step in range(horizon):
        desired = returns[:, step] + carry
        target = transform.advance(current, desired)

        if session_limit is None:
            current, carry = target, np.zeros(n_paths)
            prices[:, step] = current
            continue

        cap_ordinary = session_limit.cap_for(previous_session_locked=False)
        cap_relaxed = session_limit.cap_for(previous_session_locked=True)
        cap = np.where(locked_previous, cap_relaxed, cap_ordinary)

        ceiling = current * (1.0 + cap)
        floor = current * (1.0 - cap)
        capped = np.clip(target, floor, ceiling)

        locked = capped != target
        locked_sessions += int(np.count_nonzero(locked))
        executed = transform.step_return(current, capped)
        carry = (desired - executed) if session_limit.carry_residual else np.zeros(n_paths)

        current, locked_previous = capped, locked
        prices[:, step] = current

    diagnostics: dict[str, Any] = {"session_limit": None}
    if session_limit is not None:
        diagnostics["session_limit"] = session_limit.describe() | {
            "sessions_truncated": locked_sessions,
            "share_of_sessions_truncated": locked_sessions / float(n_paths * horizon),
        }
    return prices, diagnostics


def simulate(
    fit: FittedVol,
    *,
    transform: ReturnTransform,
    anchor: float,
    spec: SimulationSpec,
    session_limit: SessionLimit | None = None,
    shocks: np.ndarray | None = None,
    pool: ResidualPool | None = None,
) -> PathEnsemble:
    """Simulate price paths from a fitted model.

    ``shocks`` lets a caller supply standardised shocks drawn elsewhere — the joint
    simulation in :mod:`aurex.dist.copula` needs the two series to share a dependence
    structure, which it cannot do if each series draws its own. A caller who does that
    must also hand over the :class:`ResidualPool` those shocks came from, because the
    ensemble is about to publish ``spec.demean_residuals`` as a fact about itself and
    this is the only place that can check it is one.
    """
    rng = np.random.default_rng(spec.seed)
    if shocks is None:
        pool = pool or residual_pool(
            fit.standardized_residuals, demean=spec.demean_residuals
        )
        shocks = bootstrap_shocks(
            pool,
            n_paths=spec.n_paths,
            horizon=spec.horizon_days,
            block_length=spec.block_length,
            rng=rng,
        )
    elif pool is None:
        raise ValueError(
            "shocks drawn elsewhere must arrive with the ResidualPool they came from, "
            "or the ensemble cannot honestly declare whether it carries a drift"
        )

    if pool.demeaned != spec.demean_residuals:
        raise ValueError(
            f"the spec declares demean_residuals={spec.demean_residuals} but the pool "
            f"resampled was demeaned={pool.demeaned}; the artifact would misreport the "
            f"drift of every path in this ensemble"
        )
    if shocks.shape != (spec.n_paths, spec.horizon_days):
        raise ValueError(
            f"shocks must be {(spec.n_paths, spec.horizon_days)}, got {shocks.shape}"
        )

    returns = fit.propagate(shocks)
    prices, diagnostics = paths_from_returns(
        transform, anchor, returns, session_limit=session_limit
    )
    return PathEnsemble(
        prices=prices,
        anchor=float(anchor),
        diagnostics=diagnostics
        | {
            "simulation": spec.describe(),
            "vol_model": fit.describe(),
            "residual_sample_size": int(pool.residuals.size),
            "residual_pool": pool.describe(),
            "transform": transform.describe(),
        },
    )
