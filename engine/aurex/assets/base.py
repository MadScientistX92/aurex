"""The `Asset` protocol.

Everything asset-specific lives behind this interface. If an asset's name appears as
a literal in ``vol/``, ``dist/``, ``factors/``, ``scenarios/``, ``trade/``, ``score/``
or ``web/``, the abstraction has leaked — enforced by
``tests/test_asset_abstraction.py``, which checks both statically and by running the
whole pipeline over a synthetic asset.

One deviation from the shape §16 sketches: ``price_sources`` is a method taking an
optional cache rather than a bare attribute. Chains need a
:class:`~aurex.data.cache.CacheStore`, and building them at import time would bind
every asset to the default cache directory and make tests share state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from aurex.assets.friction import FrictionProfile
from aurex.assets.lens import CurrencyLens
from aurex.assets.transforms import ReturnTransform
from aurex.data.cache import CacheStore
from aurex.data.chain import SourceChain
from aurex.vol.limits import SessionLimit


@dataclass(frozen=True, slots=True)
class FactorSpec:
    """One regressor in an asset's driver set."""

    id: str
    series_id: str
    #: How the raw series becomes a regressor: ``diff`` | ``pct_change`` | ``level``.
    transform: str
    description: str
    #: Which column of a multi-column series carries the driver. ``None`` falls back to
    #: :func:`~aurex.data.base.price_column`, which is right for a series that carries
    #: one number. It is wrong, and silently so, for a series that carries several: the
    #: report behind this asset's flow proxy also carries two local rates and a
    #: reference fix, and the fallback would have picked whichever sorted first and
    #: fitted a loading on it under the flow factor's name.
    column: str | None = None
    #: Optional factors drop out with a recorded reason when their source is
    #: unavailable, rather than failing the run. Oil's EIA inventory series are the
    #: motivating case: the key is free but Aurex must never require one.
    required: bool = True


@dataclass(frozen=True, slots=True)
class VolConfig:
    """Per-asset volatility defaults.

    Assets sit in genuinely different regimes — an annualised volatility that is
    ordinary for one would be extraordinary for another — so the defaults travel with
    the asset rather than being global constants.
    """

    default_model: str = "gjr_garch"
    annualisation_days: int = 252
    min_observations: int = 500
    #: Fit around the discontinuities in ``policy_breaks.yaml`` rather than letting
    #: a policy step be absorbed as a volatility shock.
    break_aware: bool = True
    #: Settings meaningful only to the chosen model, e.g. a rolling window. Kept here
    #: rather than as named fields so adding a model does not widen this class, and so
    #: an asset can tune one without every other asset inheriting the field.
    model_options: Mapping[str, Any] = field(default_factory=dict)
    #: The venue's daily price limit, where it has one. Simulated paths that gap
    #: through a limit are paths that cannot occur; see :mod:`aurex.vol.limits`.
    session_limit: SessionLimit | None = None


@runtime_checkable
class Asset(Protocol):
    """One tradeable thing Aurex can build a distribution for."""

    # Read-only properties rather than bare attributes: implementations are plain
    # classes with class-level constants, and a bare `id: str` would demand a
    # settable variable.
    @property
    def id(self) -> str: ...

    @property
    def label(self) -> str: ...

    @property
    def quote_currency(self) -> str:
        """Currency the underlying series is quoted in."""
        ...

    @property
    def base_unit(self) -> str:
        """Unit the underlying series is quoted per, e.g. ``troy_ounce``, ``barrel``."""
        ...

    @property
    def price_series_id(self) -> str:
        """Series id of the price used for modelling and for every lens."""
        ...

    @property
    def ohlc_series_id(self) -> str | None:
        """Series carrying true OHLC, where the pricing series is close-only.

        Declared on the asset because only the asset knows whether a second series
        is the same underlying: the realised-variance estimators need a session range,
        and a close-only benchmark cannot give them one. ``None`` where no such series
        exists, in which case range-based models are simply unavailable rather than
        silently fed squared returns.
        """
        ...

    @property
    def reference_rate_series(self) -> str | None:
        """Series carrying an observed local reference rate, if one exists."""
        ...

    @property
    def reference_rate_column(self) -> str | None:
        """Column within that series."""
        ...

    @property
    def friction_profiles(self) -> dict[str, FrictionProfile]: ...

    @property
    def factor_set(self) -> tuple[FactorSpec, ...]: ...

    @property
    def scenario_axes(self) -> tuple[str, ...]: ...

    @property
    def vol_defaults(self) -> VolConfig: ...

    @property
    def return_transform(self) -> ReturnTransform: ...

    @property
    def currency_lenses(self) -> tuple[CurrencyLens, ...]: ...

    def price_sources(self, cache: CacheStore | None = None) -> dict[str, SourceChain]:
        """Every series this asset needs to price itself in all of its lenses."""
        ...

    def describe(self) -> dict[str, Any]:
        """Provenance for the artifact."""
        ...


def lens_by_code(asset: Asset, code: str) -> CurrencyLens:
    """Look up one of an asset's lenses, or fail loudly."""
    for lens in asset.currency_lenses:
        if lens.code == code:
            return lens
    available = [lens.code for lens in asset.currency_lenses]
    raise KeyError(f"{asset.id} has no {code} lens; available: {available}")


def describe_asset(asset: Asset) -> dict[str, Any]:
    """Standard artifact block for any asset."""
    return {
        "id": asset.id,
        "label": asset.label,
        "quote_currency": asset.quote_currency,
        "base_unit": asset.base_unit,
        "price_series_id": asset.price_series_id,
        "ohlc_series_id": asset.ohlc_series_id,
        "return_transform": asset.return_transform.describe(),
        "currency_lenses": [lens.describe() for lens in asset.currency_lenses],
        "friction_profiles": {
            name: profile.describe() for name, profile in asset.friction_profiles.items()
        },
        "factors": [
            {
                "id": f.id,
                "series_id": f.series_id,
                "transform": f.transform,
                "column": f.column,
                "description": f.description,
                "required": f.required,
            }
            for f in asset.factor_set
        ],
        "scenario_axes": list(asset.scenario_axes),
        "vol_defaults": {
            "default_model": asset.vol_defaults.default_model,
            "annualisation_days": asset.vol_defaults.annualisation_days,
            "min_observations": asset.vol_defaults.min_observations,
            "break_aware": asset.vol_defaults.break_aware,
            "model_options": dict(asset.vol_defaults.model_options),
            "session_limit": (
                None
                if asset.vol_defaults.session_limit is None
                else asset.vol_defaults.session_limit.describe()
            ),
        },
    }
