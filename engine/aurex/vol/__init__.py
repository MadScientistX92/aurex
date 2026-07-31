"""Volatility models: three specifications behind one protocol.

* :class:`~aurex.vol.rolling.RollingStd` — the parameter-free baseline.
* :class:`~aurex.vol.garch.GjrGarch` — asymmetric conditional variance, and the only
  one of the three whose recursion gives each simulated path its own variance
  trajectory.
* :class:`~aurex.vol.har.HarRv` — heterogeneous autoregression on measured realised
  variance, which needs an OHLC series to estimate that variance from.

Which of these an asset uses is the asset's declaration, not this package's business:
:class:`~aurex.assets.base.VolConfig` names a model id and :func:`model_for` resolves
it. Nothing here knows what it is modelling, and the leak test in
``tests/test_asset_abstraction.py`` keeps it that way.

Fitting is break-aware. The caller passes the policy discontinuities it knows about;
see :mod:`aurex.vol.base` for what excluding one actually does to the recursion.
"""

from __future__ import annotations

from typing import Any

from aurex.vol.base import (
    FittedVol,
    InsufficientDataError,
    MeanSpec,
    VolatilityModel,
    excluded_mask,
)
from aurex.vol.garch import GjrGarch, GjrGarchFit
from aurex.vol.har import HarRv, HarRvFit, garman_klass_variance, parkinson_variance
from aurex.vol.limits import SessionLimit
from aurex.vol.rolling import RollingStd, RollingStdFit

#: Model id -> constructor, for resolving an asset's declared default.
MODELS: dict[str, type[GjrGarch] | type[RollingStd] | type[HarRv]] = {
    "gjr_garch": GjrGarch,
    "rolling_std": RollingStd,
    "har_rv": HarRv,
}


def model_for(model_id: str, **overrides: Any) -> VolatilityModel:
    """Build the model an asset declares, or fail loudly listing what exists.

    ``overrides`` carry the asset's own settings — a minimum sample size, say —
    without this package needing to know which asset asked for them.
    """
    try:
        constructor = MODELS[model_id]
    except KeyError:
        raise KeyError(
            f"unknown volatility model {model_id!r}; available: {sorted(MODELS)}"
        ) from None
    return constructor(**overrides)


__all__ = [
    "MODELS",
    "FittedVol",
    "GjrGarch",
    "GjrGarchFit",
    "HarRv",
    "HarRvFit",
    "InsufficientDataError",
    "MeanSpec",
    "RollingStd",
    "RollingStdFit",
    "SessionLimit",
    "VolatilityModel",
    "excluded_mask",
    "garman_klass_variance",
    "model_for",
    "parkinson_variance",
]
