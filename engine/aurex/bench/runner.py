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

**Two runs, because CRPS and direction need opposite drift policies.** The CRPS shootout
centres every model, which is §0's position and the only way a distributional comparison
is about conditional variance rather than about eleven years of appreciation. But
centring puts P(up) at about one half for every entrant by construction, so a direction
score computed on that run grades the drift policy and not the models — which is why
step 6 published a CRPS table and left the pre-registration's directional claim
ungraded. The direction run therefore takes ``demean=False``: every competitor, the
benchmark included, carries whatever drift it infers, and the null becomes the
drift-matched walk so the comparison stays like-for-like. What may never happen is a set
that mixes the two, and :func:`_require_one_drift_policy` is where that is enforced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from aurex.assets.base import Asset
from aurex.dist.fhs import DEMEAN_BY_DEFAULT
from aurex.score.events import BinaryEvent
from aurex.score.forecasters import AsOfForecaster, ModelForecaster, RandomWalkForecaster
from aurex.score.reliability import reliability_curve
from aurex.score.shootout import (
    minimum_detectable_effect,
    model_confidence_set,
    resolution_screen,
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
#:
#: A run that asks every model to carry its drift is scored against the drift-matched
#: walk instead, and that label comes off the baseline the run actually built rather than
#: from here — see :func:`_benchmark_label`. What is not negotiable is that *a* random
#: walk is the benchmark; which of the two it is follows from the drift policy, because a
#: comparison across policies is not a comparison.
BENCHMARK = "random_walk"

#: Challenger ids in the order the README table lists them.
CHALLENGERS = ("gjr_garch", "har_rv", "auto_arima", "nhits", "chronos_t5_small")


class MixedDriftPolicyError(ValueError):
    """A set was assembled whose members disagree about carrying drift."""


def _benchmark_label(result: WalkForwardResult) -> str:
    """Which null this run actually carried, read off the run rather than assumed."""
    return str(result.baseline["label"])


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
    demean: bool = DEMEAN_BY_DEFAULT,
) -> tuple[AsOfForecaster, AsOfForecaster, tuple[AsOfForecaster, ...]]:
    """Assemble the set, sharing every part that is not the model itself.

    The transform, the session limit, the drift policy and the path count come from one
    place for all of them, so the only thing that differs between two entries is the
    forecasting method. Anything else would make a difference in score attributable to
    plumbing.

    ``demean`` is that policy, and it reaches the null as well as the models — the walk's
    label changes with it, so an uncentred set is scored against
    ``random_walk_drift_matched`` rather than against §0's driftless one. Centred is the
    default and the CRPS shootout's setting. Uncentred is what grading direction needs,
    because a centred forecast puts P(up) at about one half whatever produced it.
    """
    from aurex.bench.adapters import AutoArimaForecaster, ChronosForecaster, NhitsForecaster

    defaults = asset.vol_defaults
    shared: dict[str, Any] = {
        "transform": asset.return_transform,
        "session_limit": defaults.session_limit,
        "min_observations": defaults.min_observations,
        "demean": demean,
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
            demean_residuals=demean,
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
        demean=demean,
    )
    subject = available[chosen[0]]
    extras = tuple(available[name] for name in chosen[1:])
    assembled = (subject, baseline, *extras)
    _require_one_drift_policy(assembled, demean=demean)
    return subject, baseline, extras


def _require_one_drift_policy(forecasters: tuple[AsOfForecaster, ...], *, demean: bool) -> None:
    """Every member of the set carries drift, or none of them does. Checked, not assumed.

    The failure this guards is the one already withdrawn once: a model carrying drift
    scored against a null denied one posts CRPS skill that is eleven years of
    appreciation and belongs to neither. It is invisible in the output — a skill score
    looks the same whichever way it was earned — so it has to be caught where the set is
    assembled rather than read off the artifact afterwards. The check asks each
    forecaster what it does rather than trusting that a keyword argument reached it,
    which is the difference between a guard and a comment: a challenger that grew its own
    default, or ignored ``demean`` because its constructor never took one, fails here.
    """
    wrong = sorted(entry.label for entry in forecasters if entry.carries_drift is demean)
    if wrong:
        policy = "centred" if demean else "uncentred"
        raise MixedDriftPolicyError(
            f"the run asked for {policy} forecasts and {wrong} disagree. Every model in "
            f"a set must share one drift policy, the null included: a model carrying "
            f"drift scored against one denied it earns skill that belongs to neither, "
            f"and that result has already been published and withdrawn once."
        )


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
    demean: bool = DEMEAN_BY_DEFAULT,
    events: tuple[BinaryEvent, ...] = (),
) -> ShootoutRun:
    """One walk-forward carrying every model in the set.

    ``events`` defaults to none: the CRPS shootout grades distributions, not positions,
    and the hurdle events belong to the calibration artifact where friction arithmetic no
    model here is compared on would only add noise. The direction run passes
    :class:`~aurex.score.events.TerminalAbove` and ``demean=False`` together, and the two
    belong together — direction graded on centred forecasts grades the drift policy.
    """
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
        demean=demean,
    )
    result = walk_forward(
        prices,
        subject=subject,
        baseline=baseline,
        request=ask,
        events=events,
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
    benchmark_label = _benchmark_label(result)
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
                model_losses, benchmark, sampling=sampling, null=benchmark_label
            )
            thinned = diebold_mariano(
                sampling.thin(model_losses),
                sampling.thin(benchmark),
                sampling=sampling.independent(),
                null=benchmark_label,
            )
            entries.append(
                {
                    "model": label,
                    "crps": round(crps, 4),
                    "skill_vs_benchmark": round(
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
        mcs_input = dict(losses) | {benchmark_label: benchmark}

        horizons.append(
            {
                "horizon_sessions": horizon,
                "observations": len(records),
                "independent_observations": sampling.independent_count(len(records)),
                "sampling": sampling.describe(),
                "benchmark": {"model": benchmark_label, "crps": round(benchmark_crps, 4)},
                "models": entries,
                "hansen_spa": (
                    None
                    if (
                        spa := superior_predictive_ability(
                            losses,
                            benchmark,
                            benchmark=benchmark_label,
                            sampling=sampling,
                            seed=seed,
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


def _probabilities_by_model(
    records: tuple[Any, ...], *, subject_label: str, benchmark_label: str, event_id: str
) -> dict[str, np.ndarray]:
    """Per-model forecast probability for one event, on the windows every model saw.

    The benchmark is one entry among the rest here, unlike in the CRPS table where it is
    the thing being compared against. That is the point of the drift-matched run: the
    walk carries whatever drift the history had, so it makes a real directional claim and
    has to be graded on it like everything else. A null that could not lose the
    comparison it defines would not be a competitor.
    """
    series = {subject_label: np.array([r.events[event_id][0] for r in records], dtype=float)}
    series[benchmark_label] = np.array([r.events_baseline[event_id] for r in records], dtype=float)
    for label in sorted(records[0].event_alternatives):
        series[label] = np.array(
            [r.event_alternatives[label][event_id] for r in records], dtype=float
        )
    return series


def describe_direction(
    asset: Asset,
    result: WalkForwardResult,
    *,
    event: BinaryEvent,
    competitors: tuple[AsOfForecaster, ...] = (),
    seed: int = 20260807,
) -> dict[str, Any]:
    """Grade every model on one binary event, with resolution as the headline.

    **Resolution rather than Brier, and the reason is not a preference.** A Brier score
    confounds level with discrimination: a model that knows the base rate and nothing
    else beats a model with real signal and a biased level, which is the opposite of what
    "calls direction" means. Resolution is level-invariant — it asks only whether the
    forecasts separate the windows — so it is the number this block leads with.
    Reliability and the level sit beside it, never summed into one figure.
    """
    subject_label = str(result.subject["label"])
    benchmark_label = _benchmark_label(result)
    horizons: list[dict[str, Any]] = []

    for horizon in result.request.horizons:
        records = result.for_horizon(horizon)
        if not records or not event.applies_at(horizon):
            continue

        sampling = result.request.sampling_for(horizon)
        outcomes = np.array([r.events[event.id][1] for r in records], dtype=float)
        forecasts = _probabilities_by_model(
            records,
            subject_label=subject_label,
            benchmark_label=benchmark_label,
            event_id=event.id,
        )

        curves = {
            label: reliability_curve(values, outcomes, bins=result.request.reliability_bins)
            for label, values in forecasts.items()
        }
        full = resolution_screen(forecasts, outcomes, event=event.id, sampling=sampling, seed=seed)
        thinned = resolution_screen(
            {label: sampling.thin(values) for label, values in forecasts.items()},
            sampling.thin(outcomes),
            event=event.id,
            sampling=sampling.independent(),
            seed=seed,
        )
        # A direction forecast lives near one half, so a model's whole range can sit
        # inside one equal-width bin and score zero resolution whatever it knew. Binning
        # by rank asks the same question without that blind spot, and running it is what
        # keeps "resolution is nil" a measurement rather than a property of the axis.
        by_rank = resolution_screen(
            forecasts,
            outcomes,
            event=event.id,
            sampling=sampling,
            binning="equal_count",
            seed=seed,
        )
        # The equal-count run gets a thinned partner for the same reason the equal-width
        # one does. This was an omission in the first build of this block, found when the
        # equal-count screen produced the only rejection in the table and there was no
        # second run to adjudicate it against — so the table's single interesting number
        # could not be held to the standard every other number here is held to. The fix
        # makes a finding harder to claim rather than easier, which is the safe direction
        # for an omission discovered after seeing a result.
        thinned_by_rank = resolution_screen(
            {label: sampling.thin(values) for label, values in forecasts.items()},
            sampling.thin(outcomes),
            event=event.id,
            sampling=sampling.independent(),
            binning="equal_count",
            seed=seed,
        )

        horizons.append(
            {
                "horizon_sessions": horizon,
                "observations": len(records),
                "independent_observations": sampling.independent_count(len(records)),
                "sampling": sampling.describe(),
                "realised_rate": round(float(np.mean(outcomes)), 5),
                "positive_events": int(np.count_nonzero(outcomes)),
                "models": [{"model": label} | curves[label].describe() for label in sorted(curves)],
                "discrimination": {
                    "full_sample": None if full is None else full.describe(),
                    "non_overlapping_subsample": (None if thinned is None else thinned.describe()),
                    "equal_count_bins_robustness": (
                        None if by_rank is None else by_rank.describe()
                    ),
                    "equal_count_non_overlapping_subsample": (
                        None if thinned_by_rank is None else thinned_by_rank.describe()
                    ),
                    "distinguishable_from_zero": bool(
                        full is not None
                        and thinned is not None
                        and full.rejects
                        and thinned.rejects
                    ),
                    "distinguishable_from_zero_equal_count": bool(
                        by_rank is not None
                        and thinned_by_rank is not None
                        and by_rank.rejects
                        and thinned_by_rank.rejects
                    ),
                    "measurable_at_equal_width": bool(
                        full is not None and any(count > 1 for count in full.occupied_bins)
                    ),
                    "reading": (
                        "Both runs must reject before any model here has been shown to "
                        "discriminate, which is the standard every skill score in this "
                        "repository is held to. The full sample keeps the overlap and "
                        "handles it by shifting the outcomes circularly, so the null "
                        "carries the same persistence the data does; the thinned "
                        "subsample assumes nothing and has far fewer windows to say it "
                        "with — and where the horizon does not overlap the step at all, "
                        "the full sample simply permutes, which the resampling field on "
                        "each run states. The equal-count pair is the same standard "
                        "applied to a different partition: it is the check that a "
                        "resolution of zero is a fact about the model rather than about "
                        "the axis, because a direction forecast near one half can spend "
                        "its whole range inside a single equal-width bin. Where the two "
                        "binnings disagree the equal-count one is the one with power, and "
                        "the disagreement is itself the finding."
                    ),
                },
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
        "event": event.describe(),
        "benchmark": result.baseline,
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
                "Every model in this set carries the drift it infers, the benchmark "
                "included, and the benchmark is therefore the drift-matched random walk "
                "rather than §0's driftless one. That is the whole reason this run is "
                "separate from the CRPS shootout. Centring makes P(up) about one half "
                "for every entrant by construction, so direction graded on centred "
                "forecasts grades the drift policy and not the models — it would return "
                "'no model can call direction' whatever the models did, which is an "
                "answer that cannot be wrong and therefore is not a measurement."
            ),
            "primary_metric": (
                "Resolution. It is level-invariant, so it measures whether a model can "
                "tell one window from another rather than whether it also got the base "
                "rate right. Brier and reliability are reported per model and are not "
                "summed into a single figure of merit."
            ),
            "windows": (
                "One walk-forward, so every model was scored on the same as-of dates "
                "against the same realised outcomes. A date any model could not fit is a "
                "date none of them is scored on."
            ),
            "multiple_comparisons": (
                "Six resolution figures read informally against zero is the leaderboard "
                "problem the CRPS shootout uses Hansen's SPA to avoid. The decision here "
                "is one test over the whole set — the largest studentised resolution "
                "against its own resampled null — and the per-model p-values are "
                "description."
            ),
            "lookahead": "none: each fit sees prices up to and including its as-of date",
        },
    }
