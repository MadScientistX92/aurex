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
    DeterministicVarianceError,
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


def require_per_path_variance(model: VolatilityModel, *, leveraged: bool) -> None:
    """Refuse a deterministic-variance model where the path is what is being scored.

    A leveraged position is closed out *on the path* — it meets a margin call before it
    meets the terminal distribution — so barrier and liquidation statistics are path
    statistics. A model that iterates its variance deterministically gives every path
    the same volatility on the same day, which makes those statistics assume constant
    volatility while appearing to have simulated it. The number that comes back is
    wrong in a direction nobody can see from the output.

    Takes a boolean rather than a route, because this package must not know what a
    route or a jurisdiction is. The caller that does know is
    :meth:`aurex.routes.RouteBook.require_model`.
    """
    if leveraged and not model.per_path_variance:
        raise DeterministicVarianceError(
            f"{model.id} iterates its variance deterministically, so every simulated "
            f"path shares one volatility trajectory. A leveraged position is closed "
            f"out on the path, so its barrier and liquidation statistics would assume "
            f"constant volatility while looking as though they had simulated it. Use a "
            f"model whose recursion gives each path its own variance: "
            f"{path_dependent_models()}."
        )


def path_dependent_models() -> list[str]:
    """Model ids whose recursion gives each simulated path its own variance."""
    return sorted(mid for mid, cls in MODELS.items() if cls().per_path_variance)


def model_for(model_id: str, *, leveraged: bool = False, **overrides: Any) -> VolatilityModel:
    """Build the model an asset declares, or fail loudly listing what exists.

    ``overrides`` carry the asset's own settings — a minimum sample size, say —
    without this package needing to know which asset asked for them.

    ``leveraged`` says the caller is about to read the path rather than the horizon.
    Defaulting it to ``False`` is deliberate: an unleveraged distribution from a
    deterministic model is fair game and is scored in step 3a, so the bar is on the
    combination rather than on the model.
    """
    try:
        constructor = MODELS[model_id]
    except KeyError:
        raise KeyError(
            f"unknown volatility model {model_id!r}; available: {sorted(MODELS)}"
        ) from None
    model = constructor(**overrides)
    require_per_path_variance(model, leveraged=leveraged)
    return model


__all__ = [
    "MODELS",
    "DeterministicVarianceError",
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
    "path_dependent_models",
    "require_per_path_variance",
]
