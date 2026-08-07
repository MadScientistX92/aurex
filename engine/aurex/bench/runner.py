"""The shootout: six forecasters, one walk-forward, one set of windows.

**It reuses the harness rather than reimplementing it**, which is the whole reason the
comparison is fair. :func:`~aurex.score.walkforward.walk_forward` already takes a
subject, a required null and any number of further forecasters, and grades all of them
on the same as-of dates with the same realised outcomes. So the shootout is one call:
the engine's own model as the subject, §0's driftless random walk as the required null,
and the four challengers as extra baselines. Every model sees the identical price slice
on the identical date and is scored against the identical outcome, because the harness
gives it no way not to.

That also decides what happens when one model cannot fit. The harness records a skip for
the *date*, not for the model, so a date any model refuses is a date no model is scored
on. That is the conservative choice and the fair one: a challenger that quietly sat out
its hardest windows would post a better mean CRPS for having declined to forecast.

**A run always carries §0's null, here as everywhere.** The random walk is the required
``baseline`` argument rather than one entry in a list of six, so the shootout cannot be
run in a configuration that forgets to include it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from aurex.assets.base import Asset
from aurex.dist.fhs import DEMEAN_BY_DEFAULT
from aurex.score.forecasters import AsOfForecaster, ModelForecaster, RandomWalkForecaster
from aurex.score.shootout import (
    minimum_detectable_effect,
    model_confidence_set,
    superior_predictive_ability,
)
from aurex.score.significance import diebold_mariano
from aurex.score.walkforward import (
    DEFAULT_MIN_OBSERVATIONS,
    WalkForwardRequest,
    WalkForwardResult,
    walk_forward,
)
from aurex.vol import model_for
from aurex.vol.har import parkinson_variance

#: The null every model is measured against. §0's, and not negotiable per run.
BENCHMARK = "random_walk"

#: Challenger ids in the order the README table lists them.
CHALLENGERS = ("gjr_garch", "har_rv", "auto_arima", "nhits", "chronos_t5_small")


def build_forecasters(
    asset: Asset,
    *,
    prices: pd.Series,
    ohlc: pd.DataFrame | None,
    breaks: tuple[pd.Timestamp, ...],
    n_paths: int,
    chronos_paths: int,
    nhits_steps: int,
    include: tuple[str, ...] = CHALLENGERS,
) -> tuple[AsOfForecaster, AsOfForecaster, tuple[AsOfForecaster, ...]]:
    """Assemble the set, sharing every part that is not the model itself.

    The transform, the session limit, the drift policy and the path count come from one
    place for all of them, so the only thing that differs between two entries is the
    forecasting method. Anything else would make a difference in score attributable to
    plumbing.
    """
    from aurex.bench.adapters import AutoArimaForecaster, ChronosForecaster, NhitsForecaster

    defaults = asset.vol_defaults
    shared: dict[str, Any] = {
        "transform": asset.return_transform,
        "session_limit": defaults.session_limit,
        "min_observations": defaults.min_observations,
    }

    realised = None
    if ohlc is not None and {"high", "low"} <= set(ohlc.columns):
        realised = parkinson_variance(ohlc)

    def fhs(model_id: str) -> ModelForecaster:
        return ModelForecaster(
            model=model_for(
                model_id,
                min_observations=defaults.min_observations,
                **(dict(defaults.model_options) if model_id == defaults.default_model else {}),
            ),
            transform=asset.return_transform,
            session_limit=defaults.session_limit,
            n_paths=n_paths,
            breaks=breaks if defaults.break_aware else (),
            realised_variance=realised,
            demean_residuals=DEMEAN_BY_DEFAULT,
        )

    available: dict[str, AsOfForecaster] = {
        "gjr_garch": fhs("gjr_garch"),
        "auto_arima": AutoArimaForecaster(n_paths=n_paths, **shared),
        "nhits": NhitsForecaster(n_paths=n_paths, max_steps=nhits_steps, **shared),
        "chronos_t5_small": ChronosForecaster(n_paths=chronos_paths, **shared),
    }
    # HAR-RV regresses on a *measured* realised variance, which needs an OHLC series. On
    # a close-only asset it is omitted rather than fed squared returns and called
    # realised variance — see aurex.vol.har.
    if realised is not None:
        available["har_rv"] = fhs("har_rv")

    chosen = [name for name in include if name in available]
    if not chosen:
        raise ValueError(f"no runnable models among {include}; available: {sorted(available)}")

    baseline = RandomWalkForecaster(
        n_paths=n_paths,
        transform=asset.return_transform,
        session_limit=defaults.session_limit,
        min_observations=defaults.min_observations,
    )
    subject = available[chosen[0]]
    extras = tuple(available[name] for name in chosen[1:])
    return subject, baseline, extras


@dataclass(frozen=True, slots=True)
class ShootoutRun:
    """The graded records, and the competitors that produced them.

    The harness records the subject's and the null's descriptions but has no reason to
    carry the extra baselines', so they travel back here — the artifact publishes what
    each model says about itself, and a results table naming five labels without them
    would be unauditable.
    """

    result: WalkForwardResult
    competitors: tuple[AsOfForecaster, ...]


def run_shootout(
    asset: Asset,
    *,
    prices: pd.Series,
    ohlc: pd.DataFrame | None = None,
    breaks: tuple[pd.Timestamp, ...] = (),
    request: WalkForwardRequest | None = None,
    n_paths: int = 4_000,
    chronos_paths: int = 200,
    nhits_steps: int = 200,
    include: tuple[str, ...] = CHALLENGERS,
) -> ShootoutRun:
    """One walk-forward carrying every model in the set."""
    defaults = asset.vol_defaults
    ask = request or WalkForwardRequest(
        min_observations=max(DEFAULT_MIN_OBSERVATIONS, defaults.min_observations)
    )
    subject, baseline, extras = build_forecasters(
        asset,
        prices=prices,
        ohlc=ohlc,
        breaks=breaks,
        n_paths=n_paths,
        chronos_paths=chronos_paths,
        nhits_steps=nhits_steps,
        include=include,
    )
    result = walk_forward(
        prices,
        subject=subject,
        baseline=baseline,
        request=ask,
        # The shootout grades distributions, not positions. The hurdle events belong to
        # the calibration artifact and would only add friction arithmetic no model here
        # is being compared on.
        events=(),
        extra_baselines=extras,
    )
    return ShootoutRun(result=result, competitors=(subject, *extras))


def _losses_by_model(records: tuple[Any, ...], subject_label: str) -> dict[str, np.ndarray]:
    """Per-model CRPS at one horizon, on the windows every model was scored on."""
    losses = {subject_label: np.array([r.crps for r in records], dtype=float)}
    for label in sorted(records[0].crps_alternatives):
        losses[label] = np.array([r.crps_alternatives[label] for r in records], dtype=float)
    return losses


def describe_shootout(
    asset: Asset,
    result: WalkForwardResult,
    *,
    competitors: tuple[AsOfForecaster, ...] = (),
    seed: int = 20260807,
) -> dict[str, Any]:
    """The artifact block: every model, every horizon, and the tests that decide it."""
    subject_label = str(result.subject["label"])
    horizons: list[dict[str, Any]] = []

    for horizon in result.request.horizons:
        records = result.for_horizon(horizon)
        if not records:
            continue

        sampling = result.request.sampling_for(horizon)
        losses = _losses_by_model(records, subject_label)
        benchmark = np.array([r.crps_baseline for r in records], dtype=float)
        benchmark_crps = float(np.mean(benchmark))

        entries: list[dict[str, Any]] = []
        for label in sorted(losses):
            model_losses = losses[label]
            crps = float(np.mean(model_losses))
            overlapping = diebold_mariano(
                model_losses, benchmark, sampling=sampling, null=BENCHMARK
            )
            thinned = diebold_mariano(
                sampling.thin(model_losses),
                sampling.thin(benchmark),
                sampling=sampling.independent(),
                null=BENCHMARK,
            )
            entries.append(
                {
                    "model": label,
                    "crps": round(crps, 4),
                    "skill_vs_random_walk": round(
                        1.0 - crps / benchmark_crps if benchmark_crps > 0.0 else 0.0, 5
                    ),
                    "diebold_mariano": {
                        "overlapping_windows_hac": overlapping.describe(),
                        "non_overlapping_subsample": thinned.describe(),
                        "status": (
                            "description, not decision: read the SPA p-value for whether "
                            "anything in this set beat the benchmark"
                        ),
                    },
                    "minimum_detectable_effect": minimum_detectable_effect(
                        model_losses, benchmark, horizon=horizon, sampling=sampling
                    ).describe(),
                }
            )

        # The benchmark competes in the confidence set too. Excluding it would let the
        # set answer "which model is best" while quietly assuming the answer is a model.
        mcs_input = dict(losses) | {BENCHMARK: benchmark}

        horizons.append(
            {
                "horizon_sessions": horizon,
                "observations": len(records),
                "independent_observations": sampling.independent_count(len(records)),
                "sampling": sampling.describe(),
                "benchmark": {"model": BENCHMARK, "crps": round(benchmark_crps, 4)},
                "models": entries,
                "hansen_spa": (
                    None
                    if (
                        spa := superior_predictive_ability(
                            losses, benchmark, benchmark=BENCHMARK, sampling=sampling, seed=seed
                        )
                    )
                    is None
                    else spa.describe()
                ),
                "model_confidence_set": (
                    None
                    if (mcs := model_confidence_set(mcs_input, sampling=sampling, seed=seed))
                    is None
                    else mcs.describe()
                ),
            }
        )

    return {
        "sample": None if result.sample is None else result.sample.describe(),
        "asset": {
            "id": asset.id,
            "label": asset.label,
            "quote_currency": asset.quote_currency,
            "price_series_id": asset.price_series_id,
        },
        "benchmark": result.baseline,
        # Each competitor's own account of itself, so a reader can see what was run
        # rather than inferring it from a label in the results table.
        "models": [entry.describe() for entry in competitors] or [result.subject],
        "walk_forward": result.request.describe(),
        "scored_forecasts": len(result.records),
        "horizons": horizons,
        "skipped": [
            {
                "as_of": entry.as_of.date().isoformat(),
                "horizon": entry.horizon,
                "reason": entry.reason,
            }
            for entry in result.skipped
        ],
        "conventions": {
            "drift": (
                "Every model in this set is driftless: the engine's models and the null "
                "centre the pool they resample, and each challenger's simulated returns "
                "are centred before they are walked to prices. A model carrying drift "
                "scored against one denied it was worth up to +4.6% of spurious CRPS "
                "skill on this asset and that result was withdrawn; this is the guard "
                "against it reappearing six times over."
            ),
            "windows": (
                "One walk-forward, so every model was scored on the same as-of dates "
                "against the same realised outcomes. A date any model could not fit is a "
                "date none of them is scored on."
            ),
            "multiple_comparisons": (
                "Six models against one null at 5% finds a winner whether or not one "
                "exists. Hansen's SPA is the decision; the per-model Diebold-Mariano "
                "statistics are description. The Model Confidence Set answers the "
                "different question of which models cannot be excluded."
            ),
            "lookahead": "none: each fit sees prices up to and including its as-of date",
        },
    }
