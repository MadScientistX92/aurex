"""Composition: an asset's data, its models, and one distribution per lens.

This is the only module that knows about assets, lenses, volatility models and path
ensembles at the same time. Keeping it out of :mod:`aurex.vol` and :mod:`aurex.dist`
is what lets those packages stay ignorant of what they are modelling; keeping it out
of :mod:`aurex.pipeline` is what stops the orchestrator from growing a statistics
layer.

**One simulation, two views.** The base price is simulated once, and each lens is a
multiplication applied to those same paths. Simulating each lens separately would let
the two views disagree about what the underlying did, which is exactly the drift §15
warns about — and it would waste the copula, whose entire job is to keep the price and
the exchange rate moving together in the same path.

**The tax stack is held at today's rates.** Duty and consumption tax enter the
composition as the constant they are on the last observed day. Aurex records policy
changes as breaks; it does not forecast them, and a simulated duty revision would be
a political prediction wearing a distribution.

**Where there is not enough data, there is no distribution.** A short history raises
:class:`~aurex.vol.base.InsufficientDataError` and the block records why, in the same
spirit as a factor that drops out with a reason rather than vanishing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from aurex.assets.base import Asset
from aurex.assets.lens import CurrencyLens
from aurex.assets.transforms import LogReturn
from aurex.dist.copula import DependenceMode, TCopula, fit_t_copula, joint_shocks
from aurex.dist.fhs import (
    DEMEAN_BY_DEFAULT,
    SimulationSpec,
    bootstrap_shocks,
    residual_pool,
    simulate,
)
from aurex.dist.passage import first_passage
from aurex.dist.paths import PathEnsemble
from aurex.vol import model_for
from aurex.vol.base import FittedVol, InsufficientDataError
from aurex.vol.har import parkinson_variance

log = logging.getLogger(__name__)

#: Exchange rates are strictly positive, so a shift would be meaningless. Fixed here
#: rather than declared per asset because it is a fact about rates, not about assets.
FX_TRANSFORM = LogReturn()


@dataclass(frozen=True, slots=True)
class ForecastRequest:
    """How far ahead to simulate, how hard, and against which adverse moves."""

    #: Trading sessions ahead. A week, a month, a quarter.
    horizons: tuple[int, ...] = (5, 21, 63)
    n_paths: int = 20_000
    block_length: int = 10
    seed: int = 20260731
    #: Adverse moves to report touch-versus-terminal against, as fractions.
    reference_moves: tuple[float, ...] = (0.05, 0.10, 0.20)
    #: Moves to publish an exceedance probability at, as fractions.
    #:
    #: Half-percent steps across ±25%, which covers every round-trip breakeven in the
    #: routes table with room for a leveraged one. The grid is published rather than a
    #: fitted curve so a consumer reads a number the paths actually produced instead of
    #: interpolating a distribution whose shape is the entire point of not assuming.
    exceedance_grid: tuple[float, ...] = tuple(round(-0.25 + 0.005 * i, 4) for i in range(101))
    dependence: DependenceMode = "t_copula"
    #: Resample the residual pool with its mean removed. §0's position on drift, and
    #: the default; see :mod:`aurex.dist.fhs`. Set false to simulate the drift the
    #: sample actually carried, which is a declared choice and lands in the artifact.
    demean_residuals: bool = DEMEAN_BY_DEFAULT

    def spec_for(self, horizon: int) -> SimulationSpec:
        return SimulationSpec(
            horizon_days=horizon,
            n_paths=self.n_paths,
            block_length=self.block_length,
            # Each horizon gets its own stream, so adding a horizon cannot change
            # the numbers published for the others.
            seed=self.seed + horizon,
            demean_residuals=self.demean_residuals,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "horizons": list(self.horizons),
            "n_paths": self.n_paths,
            "block_length": self.block_length,
            "seed": self.seed,
            "reference_moves": list(self.reference_moves),
            "exceedance_grid_steps": len(self.exceedance_grid),
            "dependence": self.dependence,
            "demean_residuals": self.demean_residuals,
        }


@dataclass(frozen=True, slots=True)
class LensForecast:
    """One lens's distribution, or the reason there isn't one."""

    lens_code: str
    ensembles: dict[int, PathEnsemble] = field(default_factory=dict)
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        return bool(self.ensembles)


@dataclass(frozen=True, slots=True)
class AssetForecast:
    asset_id: str
    lenses: dict[str, LensForecast]
    price_fit: FittedVol | None = None
    fx_fit: FittedVol | None = None
    copula: TCopula | None = None
    unavailable_reason: str | None = None


def forecast_asset(
    asset: Asset,
    *,
    base_prices: pd.Series,
    lens_frames: dict[str, pd.DataFrame],
    fx_series: dict[str, pd.Series],
    ohlc: pd.DataFrame | None = None,
    breaks: tuple[pd.Timestamp, ...] = (),
    request: ForecastRequest | None = None,
) -> AssetForecast:
    """Simulate the asset once and present it through each of its lenses."""
    ask = request or ForecastRequest()

    try:
        price_fit = _fit_price(asset, base_prices, ohlc=ohlc, breaks=breaks)
    except InsufficientDataError as exc:
        log.warning("%s: no distribution: %s", asset.id, exc)
        return AssetForecast(asset_id=asset.id, lenses={}, unavailable_reason=str(exc))

    fx_needed = _fx_series_for(asset, lens_frames, fx_series)
    fx_fit: FittedVol | None = None
    copula: TCopula | None = None
    fx_reason: str | None = None

    if fx_needed is not None:
        try:
            fx_fit = _fit_fx(fx_needed, breaks=breaks)
            copula = _fit_copula(price_fit, fx_fit, mode=ask.dependence)
        except (InsufficientDataError, ValueError) as exc:
            log.warning("%s: no currency-converted distribution: %s", asset.id, exc)
            fx_fit, copula, fx_reason = None, None, str(exc)

    lenses: dict[str, LensForecast] = {}
    for lens in asset.currency_lenses:
        frame = lens_frames.get(lens.code)
        if frame is None or frame.empty:
            lenses[lens.code] = LensForecast(
                lens_code=lens.code, unavailable_reason="the lens produced no prices"
            )
            continue
        if lens.requires_fx and fx_fit is None:
            lenses[lens.code] = LensForecast(
                lens_code=lens.code,
                unavailable_reason=fx_reason or "no exchange-rate model is available",
            )
            continue

        lenses[lens.code] = LensForecast(
            lens_code=lens.code,
            ensembles=_simulate_lens(
                lens=lens,
                frame=frame,
                base_prices=base_prices,
                price_fit=price_fit,
                fx_fit=fx_fit,
                fx_anchor=None if fx_needed is None else float(fx_needed.iloc[-1]),
                copula=copula,
                asset=asset,
                request=ask,
            ),
        )

    return AssetForecast(
        asset_id=asset.id,
        lenses=lenses,
        price_fit=price_fit,
        fx_fit=fx_fit,
        copula=copula,
    )


def _fit_price(
    asset: Asset,
    base_prices: pd.Series,
    *,
    ohlc: pd.DataFrame | None,
    breaks: tuple[pd.Timestamp, ...],
) -> FittedVol:
    model = model_for(
        asset.vol_defaults.default_model,
        min_observations=asset.vol_defaults.min_observations,
        **dict(asset.vol_defaults.model_options),
    )
    returns = asset.return_transform.to_returns(base_prices.dropna().astype(float))
    realised = None
    if ohlc is not None and {"high", "low"} <= set(ohlc.columns):
        realised = parkinson_variance(ohlc)

    exclude = breaks if asset.vol_defaults.break_aware else ()
    return model.fit(returns, realised_variance=realised, exclude=exclude)


def _fit_fx(fx: pd.Series, *, breaks: tuple[pd.Timestamp, ...]) -> FittedVol:
    """The exchange rate gets its own conditional variance, not the price's.

    Borrowing the price's variance would make the second currency's distribution a
    rescaling of the first, which is precisely the claim §15 says must be measured.
    """
    model = model_for("gjr_garch")
    return model.fit(FX_TRANSFORM.to_returns(fx.dropna().astype(float)), exclude=breaks)


def _fit_copula(price_fit: FittedVol, fx_fit: FittedVol, *, mode: DependenceMode) -> TCopula | None:
    if mode != "t_copula":
        return None
    return fit_t_copula(price_fit.standardized_residuals, fx_fit.standardized_residuals)


def _fx_series_for(
    asset: Asset, lens_frames: dict[str, pd.DataFrame], fx_series: dict[str, pd.Series]
) -> pd.Series | None:
    """The exchange rate the asset's foreign lenses need, if any lens produced one."""
    for lens in asset.currency_lenses:
        if not lens.requires_fx or lens.fx_series_id is None:
            continue
        frame = lens_frames.get(lens.code)
        if frame is None or frame.empty:
            continue
        series = fx_series.get(lens.fx_series_id)
        if series is not None:
            return series.dropna().astype(float)
    return None


def _simulate_lens(
    *,
    lens: CurrencyLens,
    frame: pd.DataFrame,
    base_prices: pd.Series,
    price_fit: FittedVol,
    fx_fit: FittedVol | None,
    fx_anchor: float | None,
    copula: TCopula | None,
    asset: Asset,
    request: ForecastRequest,
) -> dict[int, PathEnsemble]:
    latest = frame.iloc[-1]
    base_anchor = float(base_prices.dropna().iloc[-1])
    fx_rate = float(latest["fx_rate"])

    # Everything fixed over the horizon in one number: the unit conversion and the
    # rates in force today. Derived from the lens's own output rather than from its
    # internals, so a lens with a different tax stack needs no change here.
    #
    # By construction ``price = base * units_per_base * fx * (1 + duty) * (1 + tax)``,
    # so dividing today's price by today's base and today's rate leaves exactly the
    # part that does not move — and the simulated base and rate supply the parts that
    # do. The native case falls out of the same expression with ``fx_rate == 1``.
    constant = float(latest["price"]) / (base_anchor * fx_rate)

    # One pool per series, built once and reused across horizons: the drift policy is a
    # property of the run, not of how far ahead it happens to be looking.
    price_pool = residual_pool(price_fit.standardized_residuals, demean=request.demean_residuals)
    fx_pool = (
        None
        if fx_fit is None
        else residual_pool(fx_fit.standardized_residuals, demean=request.demean_residuals)
    )

    ensembles: dict[int, PathEnsemble] = {}
    for horizon in request.horizons:
        spec = request.spec_for(horizon)
        rng = np.random.default_rng(spec.seed)

        if (
            lens.requires_fx
            and fx_fit is not None
            and fx_pool is not None
            and fx_anchor is not None
        ):
            price_shocks, fx_shocks = joint_shocks(
                price_pool,
                fx_pool,
                copula=copula,
                mode=request.dependence,
                n_paths=spec.n_paths,
                horizon=spec.horizon_days,
                block_length=spec.block_length,
                rng=rng,
            )
            base = simulate(
                price_fit,
                transform=asset.return_transform,
                anchor=base_anchor,
                spec=spec,
                session_limit=asset.vol_defaults.session_limit,
                shocks=price_shocks,
                pool=price_pool,
            )
            rate = simulate(
                fx_fit,
                transform=FX_TRANSFORM,
                anchor=fx_anchor,
                spec=spec,
                shocks=fx_shocks,
                pool=fx_pool,
            )
            ensembles[horizon] = base.rescaled(rate, factor=constant)
            continue

        shocks = bootstrap_shocks(
            price_pool,
            n_paths=spec.n_paths,
            horizon=spec.horizon_days,
            block_length=spec.block_length,
            rng=rng,
        )
        base = simulate(
            price_fit,
            transform=asset.return_transform,
            anchor=base_anchor,
            spec=spec,
            session_limit=asset.vol_defaults.session_limit,
            shocks=shocks,
            pool=price_pool,
        )
        ensembles[horizon] = base.scaled(constant)

    return ensembles


def describe_forecast(
    forecast: AssetForecast, lens: CurrencyLens, request: ForecastRequest
) -> dict[str, Any]:
    """Artifact block for one lens."""
    lens_forecast = forecast.lenses.get(lens.code)
    if lens_forecast is None or not lens_forecast.available:
        reason = (
            forecast.unavailable_reason
            if lens_forecast is None
            else lens_forecast.unavailable_reason
        )
        return {"available": False, "reason": reason or "no distribution was produced"}

    horizons = {
        str(horizon): _horizon_block(ensemble, request)
        for horizon, ensemble in sorted(lens_forecast.ensembles.items())
    }
    first = next(iter(lens_forecast.ensembles.values()))

    return {
        "available": True,
        "anchor": round(first.anchor, 4),
        "horizons": horizons,
        "simulation": request.describe(),
        "vol_model": forecast.price_fit.describe() if forecast.price_fit else None,
        # Only where the lens actually converts a currency: reporting an exchange-rate
        # model against a native view would imply an exposure it does not carry.
        "fx_vol_model": (
            forecast.fx_fit.describe() if lens.requires_fx and forecast.fx_fit else None
        ),
        "copula": (forecast.copula.describe() if lens.requires_fx and forecast.copula else None),
        "session_limit": _session_limit_block(first),
        "held_constant": (
            "Unit conversion and the duty and consumption-tax rates in force on the "
            "last observed day are held fixed across the horizon. Policy changes are "
            "recorded as breaks, not simulated."
        ),
        "note": (
            "Simulated quantiles, not a forecast of a price. Both currency views come "
            "from one simulation of the underlying, so they cannot disagree about it."
        ),
    }


def _session_limit_block(ensemble: PathEnsemble) -> dict[str, Any] | None:
    """The limit that shaped the paths, from wherever composition left it."""
    diagnostics = ensemble.diagnostics
    if "session_limit" in diagnostics:
        limit: dict[str, Any] | None = diagnostics["session_limit"]
        return limit
    primary = diagnostics.get("primary", {})
    return primary.get("session_limit") if isinstance(primary, dict) else None


def _horizon_block(ensemble: PathEnsemble, request: ForecastRequest) -> dict[str, Any]:
    return {
        "quantiles": {key: round(value, 4) for key, value in ensemble.quantiles().items()},
        "adverse_moves": [_adverse_move_block(ensemble, move) for move in request.reference_moves],
        "exceedance": _exceedance_block(ensemble, request),
        "n_paths": ensemble.n_paths,
    }


def _exceedance_block(ensemble: PathEnsemble, request: ForecastRequest) -> list[dict[str, Any]]:
    """P(move at or beyond x), computed from the paths across a grid of moves.

    Exists so a reader can ask "what are the odds of clearing *this* hurdle" without
    re-running the model. Five published quantiles cannot answer that: a round-trip
    breakeven lands wherever a dealer spread and a tax rate put it, almost never on a
    quantile, and interpolating a CDF from five points in a browser would put an
    invented precision in front of the one number a holder would act on.

    Both readings are kept because they answer different questions and §18 is about the
    gap between them. ``terminal`` is whether the position is above water *at* the
    horizon, which is what a round trip held to term faces. ``touch`` is whether it was
    ever above water on the way, which is what a holder who can exit early faces, and it
    is a floor because paths are monitored at session close.
    """
    anchor = ensemble.anchor
    if anchor <= 0.0:
        return []
    return [
        {
            "move": round(move, 4),
            "level": round(anchor * (1.0 + move), 4),
            "terminal_probability": round(
                ensemble.probability_above(anchor * (1.0 + move), terminal_only=True), 5
            ),
            "touch_probability": round(ensemble.probability_above(anchor * (1.0 + move)), 5),
        }
        for move in request.exceedance_grid
    ]


def _adverse_move_block(ensemble: PathEnsemble, move: float) -> dict[str, Any]:
    """Touch versus terminal at one adverse move — the §18 comparison, per horizon."""
    barrier = ensemble.anchor * (1.0 - move)
    passage = first_passage(ensemble, barrier=barrier, direction="down")
    return {
        "move": move,
        "barrier": round(barrier, 4),
        "touch_probability": round(passage.touch_probability, 4),
        "terminal_probability": round(passage.terminal_probability, 4),
        "path_dependence_ratio": (
            None
            if passage.path_dependence_ratio is None
            else round(passage.path_dependence_ratio, 3)
        ),
        "median_sessions_to_touch": passage.sessions_to_touch.get("q50"),
        "monitoring": "session_close",
    }
