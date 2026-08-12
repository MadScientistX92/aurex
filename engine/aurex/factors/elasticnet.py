"""Elastic net by coordinate descent, and why it is written out rather than imported.

The estimator is thirty lines of arithmetic with a closed form for every step, and the
alternative was a new default dependency an order of magnitude larger than the rest of
this package for one regression. That trade is only defensible if the arithmetic is
actually pinned down, so the tests check it against three things a correct
implementation cannot avoid agreeing with: the OLS solution at zero penalty, the
soft-thresholding closed form on an orthonormal design, and the penalty above which
every coefficient must be exactly zero.

**The objective**, on a centred target and standardised regressors::

    (1 / 2n) * ||y - Xb||^2  +  lam * ( l1_ratio * ||b||_1
                                        + (1 - l1_ratio) / 2 * ||b||^2 )

**Standardisation is not cosmetic here.** A penalty applied to raw coefficients would
shrink a regressor measured in basis points and one measured in index points by wildly
different amounts, and which drivers survive selection would then be a fact about their
units. Coefficients are returned in both forms: standardised, which is what makes two
drivers comparable, and per natural unit, which is what makes one interpretable.

**Nothing here selects a model.** The penalty is chosen by blocked cross-validation in
:mod:`aurex.factors.loadings`, on contiguous folds, because a shuffled fold in a serially
dependent series trains on the neighbours of what it is scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

#: Coordinate descent stops when no coefficient moves by more than this, measured on
#: the standardised scale. Tight enough that the fit is not the noise floor of a
#: bootstrap replicate, loose enough that a path of 60 penalties is not the slow part.
TOLERANCE = 1e-7

#: Ceiling on sweeps per penalty. Reached only on a degenerate design; the fit is still
#: returned, with the iteration count recorded so a caller can see it did not converge.
MAX_ITERATIONS = 5_000

#: Smallest penalty on the path, as a fraction of the one that zeroes every coefficient.
PATH_EPSILON = 1e-4


@dataclass(frozen=True, slots=True)
class Standardisation:
    """Column means and scales, kept so a fit can be read on either scale."""

    mean: np.ndarray
    scale: np.ndarray
    target_mean: float

    def raw(self, standardised: np.ndarray) -> np.ndarray:
        """Coefficients per natural unit of each regressor."""
        out: np.ndarray = standardised / self.scale
        return out

    def intercept(self, standardised: np.ndarray) -> float:
        return float(self.target_mean - np.dot(self.raw(standardised), self.mean))


@dataclass(frozen=True, slots=True)
class ElasticNetFit:
    """One fit at one penalty.

    ``coefficients`` is on the standardised scale — a change in the target per one
    standard deviation of the regressor — because that is the scale the penalty acted
    on and the only one on which two drivers are comparable. ``raw_coefficients`` is per
    natural unit and is what a reader interprets.
    """

    lam: float
    l1_ratio: float
    coefficients: np.ndarray
    raw_coefficients: np.ndarray
    intercept: float
    iterations: int
    converged: bool

    @property
    def support(self) -> np.ndarray:
        """Which regressors survived selection."""
        out: np.ndarray = self.coefficients != 0.0
        return out

    def predict(self, design: np.ndarray) -> np.ndarray:
        out: np.ndarray = self.intercept + design @ self.raw_coefficients
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "penalty": self.lam,
            "l1_ratio": self.l1_ratio,
            "intercept": self.intercept,
            "selected": int(np.count_nonzero(self.support)),
            "converged": self.converged,
            "iterations": self.iterations,
        }


def standardise(
    design: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, Standardisation]:
    """Centre both, scale the design to unit standard deviation.

    A constant column has no scale to divide by. It is left at zero rather than
    rescued: a regressor that does not vary over the window carries no information
    about the target, and dividing by an epsilon would turn rounding noise into a
    coefficient large enough to survive selection.
    """
    mean = design.mean(axis=0)
    scale = design.std(axis=0, ddof=0)
    safe = np.where(scale > 0.0, scale, 1.0)
    centred = (design - mean) / safe
    centred[:, scale <= 0.0] = 0.0

    target_mean = float(target.mean())
    return (
        centred,
        target - target_mean,
        Standardisation(mean=mean, scale=safe, target_mean=target_mean),
    )


def _soft_threshold(value: float, amount: float) -> float:
    if value > amount:
        return value - amount
    if value < -amount:
        return value + amount
    return 0.0


def max_penalty(standardised: np.ndarray, centred_target: np.ndarray, *, l1_ratio: float) -> float:
    """The smallest penalty at which every coefficient is exactly zero.

    At the first sweep from an all-zero start each coefficient's update is
    ``S(x_j'y / n, lam * l1_ratio)``, so no coefficient can leave zero once ``lam``
    exceeds the largest of those correlations divided by ``l1_ratio``. It is the top of
    the path, and the tests assert the boundary rather than trusting the derivation.
    """
    if l1_ratio <= 0.0:
        raise ValueError("a ridge-only fit has no penalty at which coefficients vanish")
    n = centred_target.size
    return float(np.max(np.abs(standardised.T @ centred_target)) / (n * l1_ratio))


def fit(
    design: np.ndarray,
    target: np.ndarray,
    *,
    lam: float,
    l1_ratio: float = 0.5,
    warm_start: np.ndarray | None = None,
) -> ElasticNetFit:
    """Fit one penalty by cyclic coordinate descent."""
    if design.ndim != 2:
        raise ValueError(f"design must be 2-d, got shape {design.shape}")
    if design.shape[0] != target.shape[0]:
        raise ValueError(f"design has {design.shape[0]} rows, target has {target.shape[0]}")
    if not 0.0 <= l1_ratio <= 1.0:
        raise ValueError(f"l1_ratio must be in [0, 1], got {l1_ratio}")
    if lam < 0.0:
        raise ValueError(f"penalty must be non-negative, got {lam}")

    standardised, centred, scaling = standardise(design, target)
    n, p = standardised.shape

    beta = np.zeros(p) if warm_start is None else warm_start.astype(float).copy()
    residual = centred - standardised @ beta

    # Every column is unit-variance after standardisation, so the denominator of the
    # coordinate update is 1 + lam * (1 - l1_ratio) for every j — except a constant
    # column, whose norm is zero and which must stay at zero rather than divide by it.
    norms = (standardised**2).sum(axis=0) / n
    l1_amount = lam * l1_ratio
    l2_amount = lam * (1.0 - l1_ratio)

    iterations = 0
    converged = False
    while iterations < MAX_ITERATIONS:
        iterations += 1
        largest_step = 0.0
        for j in range(p):
            if norms[j] <= 0.0:
                continue
            previous = beta[j]
            rho = float(standardised[:, j] @ residual) / n + norms[j] * previous
            updated = _soft_threshold(rho, l1_amount) / (norms[j] + l2_amount)
            if updated != previous:
                residual -= standardised[:, j] * (updated - previous)
                beta[j] = updated
                largest_step = max(largest_step, abs(updated - previous))
        if largest_step < TOLERANCE:
            converged = True
            break

    return ElasticNetFit(
        lam=lam,
        l1_ratio=l1_ratio,
        coefficients=beta,
        raw_coefficients=scaling.raw(beta),
        intercept=scaling.intercept(beta),
        iterations=iterations,
        converged=converged,
    )


def penalty_path(
    design: np.ndarray,
    target: np.ndarray,
    *,
    l1_ratio: float = 0.5,
    steps: int = 60,
) -> np.ndarray:
    """A geometric grid from the all-zero penalty down to ``PATH_EPSILON`` of it.

    Derived from the data rather than fixed, because a grid of absolute penalties is a
    grid whose meaning changes with the units of the target.
    """
    standardised, centred, _ = standardise(design, target)
    top = max_penalty(standardised, centred, l1_ratio=l1_ratio)
    if top <= 0.0:
        return np.zeros(1)
    return np.geomspace(top, top * PATH_EPSILON, num=steps)


def fit_path(
    design: np.ndarray,
    target: np.ndarray,
    penalties: np.ndarray,
    *,
    l1_ratio: float = 0.5,
) -> list[ElasticNetFit]:
    """Fit every penalty in descending order, warm-starting each from the last."""
    fits: list[ElasticNetFit] = []
    warm: np.ndarray | None = None
    for lam in penalties:
        current = fit(design, target, lam=float(lam), l1_ratio=l1_ratio, warm_start=warm)
        warm = current.coefficients
        fits.append(current)
    return fits
