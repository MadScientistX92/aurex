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

**The seed is recorded.** A distribution nobody can reproduce cannot be scored, and
§3's forecast log needs to be able to regenerate exactly what was published.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from aurex.assets.transforms import ReturnTransform
from aurex.dist.paths import PathEnsemble
from aurex.vol.base import FittedVol
from aurex.vol.limits import SessionLimit


@dataclass(frozen=True, slots=True)
class SimulationSpec:
    """How much simulation to run, and how to make it reproducible."""

    horizon_days: int
    n_paths: int = 20_000
    #: Contiguous residuals per draw. Roughly two trading weeks by default.
    block_length: int = 10
    seed: int = 20260731

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
        }


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
    residuals: np.ndarray,
    *,
    n_paths: int,
    horizon: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Block-bootstrapped standardised shocks, ``(n_paths, horizon)``."""
    clean = np.asarray(residuals, dtype=float)
    clean = clean[np.isfinite(clean)]
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
) -> PathEnsemble:
    """Simulate price paths from a fitted model.

    ``shocks`` lets a caller supply standardised shocks drawn elsewhere — the joint
    simulation in :mod:`aurex.dist.copula` needs the two series to share a dependence
    structure, which it cannot do if each series draws its own.
    """
    rng = np.random.default_rng(spec.seed)
    if shocks is None:
        shocks = bootstrap_shocks(
            fit.standardized_residuals.to_numpy(),
            n_paths=spec.n_paths,
            horizon=spec.horizon_days,
            block_length=spec.block_length,
            rng=rng,
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
            "residual_sample_size": int(fit.standardized_residuals.size),
            "transform": transform.describe(),
        },
    )
