"""The step 6 challengers, behind the same interface the engine's own model uses.

Each class here is an :class:`~aurex.score.forecasters.AsOfForecaster`: it receives a
sliced price history, never learns where it was cut, and returns one
:class:`~aurex.dist.paths.PathEnsemble` per horizon. That is the whole of the fairness
guarantee on the input side, and it is structural rather than procedural — see
:mod:`aurex.score.forecasters`.

**Every heavy import is inside a method.** ``torch``, ``statsforecast`` and
``neuralforecast`` are the ``bench`` extra, about 2.5GB, and importing this module must
stay free for anyone who has not installed it. A module-level import here would put the
whole stack behind ``import aurex``, and the default test suite would start paying for a
shootout it never runs.

**The drift policy is one line, applied identically to all of them.** ``_centre``
removes the mean simulated return from an ensemble, so no challenger carries a drift the
null is denied. This is the same discipline :func:`~aurex.dist.fhs.residual_pool`
applies to the engine's own models, moved to the only place these models expose
something to centre — the simulated returns themselves. It is not a detail. A model
carrying drift scored against one denied it was worth up to +4.6% of CRPS skill on the
first asset this project measured, all of it eleven years of appreciation, and that
result was withdrawn. A shootout is exactly where that error would reappear, six times
over, because three of these models fit a conditional mean by construction and one
infers a trend from context.

**Identically is the invariant; centred is only the default.** A run may instead ask
every model to carry the drift it infers, which is what grading *direction* requires:
centred forecasts put P(up) at about one half for everyone by construction, so a
direction score computed on them measures the drift policy and not the models. What may
never happen is a set that mixes the two, and that is not left to care — every
forecaster declares :attr:`~aurex.score.forecasters.AsOfForecaster.carries_drift` and
:func:`~aurex.bench.runner.build_forecasters` refuses to assemble a set that disagrees,
null included. The null's own label changes with the policy, so an uncentred run is
scored against ``random_walk_drift_matched`` and cannot silently report itself as having
beaten §0's driftless one.

**Two of the three are point forecasters, and CRPS needs a distribution.** AutoARIMA and
NHITS produce a conditional mean path and nothing else. The distribution around it comes
from an iid bootstrap of that model's *own* in-sample one-step residuals, which is the
same idea filtered historical simulation applies to a fitted variance: the model supplies
the location, its residuals supply the shape, and no Gaussian assumption is inserted
anywhere. Chronos needs none of this — it is natively probabilistic and its own sample
paths are used directly.

**The ensemble sizes are not equal, and that is declared rather than hidden.** Chronos
samples autoregressively and 4,000 paths per window would cost more than the rest of the
shootout combined, so it runs at a few hundred. The CRPS estimator here is the fair one,
which removes the ensemble-size bias, so a smaller ensemble is unbiased and noisier
rather than penalised. Noisier means a wider Diebold-Mariano standard error, so the
direction of the remaining effect is *against* finding Chronos significant — stated here
because a reader should not have to work out which way it cuts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aurex.assets.transforms import ReturnTransform
from aurex.dist.fhs import paths_from_returns
from aurex.dist.paths import PathEnsemble
from aurex.vol.base import require_observations
from aurex.vol.limits import SessionLimit

#: Context the foundation model is shown. Its own training context length; a longer
#: window is silently truncated by the tokenizer, so asking for one would misreport
#: what the model actually saw.
CHRONOS_CONTEXT = 512

#: Loaded pipelines, keyed by checkpoint. Loading chronos-t5-small takes about twenty
#: seconds and a walk-forward calls it several hundred times.
_PIPELINES: dict[str, Any] = {}


def _checkpoint_revision(checkpoint: str) -> str | None:
    """The exact commit the weights came from, if it can be recovered.

    A checkpoint named ``amazon/chronos-t5-small`` and nothing more is a moving
    reference. It is the same unreproducibility ``uv.lock`` exists to prevent one layer
    down: a rerun next year resolves that name to whatever is on the branch then, and
    every number in the artifact would silently be a different model's. The commit hash
    is what makes the row re-derivable.

    ``getattr`` rather than a declared interface because this reaches into a third-party
    object's internals, where there is no protocol of ours to add a member to and no
    guarantee across versions. It degrades to ``None`` — recorded as unknown in the
    artifact — rather than raising, because losing the provenance of a run is bad and
    crashing a two-hour job over it is worse.
    """
    pipeline = _PIPELINES.get(checkpoint)
    for holder in (pipeline, getattr(pipeline, "model", None)):
        config = getattr(getattr(holder, "model", None), "config", None) or getattr(
            holder, "config", None
        )
        revision = getattr(config, "_commit_hash", None)
        if revision:
            return str(revision)

    # Fall back to the hub cache's own ref, which is written at download time.
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        ref = Path(HF_HUB_CACHE) / f"models--{checkpoint.replace('/', '--')}" / "refs" / "main"
        return ref.read_text().strip() or None
    except Exception:
        # Provenance is best-effort. An unknown revision is recorded as unknown;
        # it is not worth ending a two-hour run over.
        return None


def _centre(simulated: np.ndarray) -> np.ndarray:
    """Remove the ensemble's mean return. The shared drift policy, in one place.

    A scalar over every path and every step, which is the direct analogue of demeaning
    a residual pool: it takes the drift out without touching the model's dynamics across
    the horizon. Subtracting a per-step mean instead would flatten the conditional mean
    path itself, which is part of what these models forecast and not the part §0 objects
    to.
    """
    return simulated - float(np.mean(simulated))


def _ensemble(
    simulated: np.ndarray,
    *,
    transform: ReturnTransform,
    anchor: float,
    session_limit: SessionLimit | None,
    detail: dict[str, Any],
    demean: bool = True,
) -> PathEnsemble:
    """Apply the run's drift policy, walk to prices, and record what produced it.

    ``demean`` is not a per-model option. It is the whole set's policy, and
    :func:`~aurex.bench.runner.build_forecasters` refuses to assemble a set whose members
    disagree about it — including the null, whose label changes with it. The flag exists
    because grading *direction* on centred forecasts grades the policy rather than the
    models: centring makes P(up) about one half for everyone by construction, so there is
    nothing left for a resolution term to discriminate on. That run needs every
    competitor, benchmark included, to carry whatever drift it infers.
    """
    returns = _centre(simulated) if demean else simulated
    prices, diagnostics = paths_from_returns(
        transform, anchor, returns, session_limit=session_limit
    )
    return PathEnsemble(
        prices=prices,
        anchor=anchor,
        diagnostics=diagnostics
        | detail
        | {
            "transform": transform.describe(),
            "residual_pool": {
                "demeaned": demean,
                "drift": (
                    "none: the simulated returns are centred before they are walked to "
                    "prices, so this forecaster carries the same drift policy as the null"
                    if demean
                    else "the model's own, carried into prices uncentred so direction is "
                    "graded on what the model inferred rather than on the drift policy"
                ),
            },
        },
    )


def _bootstrap_residuals(
    residuals: np.ndarray, *, n_paths: int, horizon: int, rng: np.random.Generator
) -> np.ndarray:
    """iid draws from a model's own one-step errors.

    iid rather than blocked, deliberately: a block bootstrap would import serial
    structure into the dispersion of a model whose conditional mean is supposed to have
    removed it, which would credit the model for dependence it failed to capture.
    """
    usable = residuals[np.isfinite(residuals)]
    if usable.size < 2:
        raise InsufficientResidualsError(
            f"a residual bootstrap needs at least 2 finite residuals, got {usable.size}"
        )
    return rng.choice(usable, size=(n_paths, horizon), replace=True)


class InsufficientResidualsError(ValueError):
    """A fitted model produced too few usable residuals to build a distribution from."""


@dataclass(frozen=True, slots=True)
class AutoArimaForecaster:
    """statsforecast's AutoARIMA on returns, with its own residuals for the shape.

    Order selection runs at every as-of date rather than being fixed once, which is what
    "AutoARIMA" means and is also the only version of it that has no lookahead: an order
    chosen on the whole sample and applied backwards would be a model the forecaster
    could not have had.
    """

    transform: ReturnTransform
    session_limit: SessionLimit | None = None
    n_paths: int = 4_000
    min_observations: int = 750
    #: Carry the model's own drift into prices. See :func:`_ensemble` — the run's policy,
    #: not this model's, and the whole set is checked to agree on it.
    demean: bool = True
    season_length: int = 1

    @property
    def label(self) -> str:
        return "auto_arima"

    @property
    def carries_drift(self) -> bool:
        return not self.demean

    def forecast(
        self, history: pd.Series, *, horizons: tuple[int, ...], seed: int
    ) -> dict[int, PathEnsemble]:
        from statsforecast.models import AutoARIMA

        returns = self.transform.to_returns(history).dropna()
        require_observations(int(returns.size), self.min_observations, self.label)

        values = returns.to_numpy(dtype=float)
        model = AutoARIMA(season_length=self.season_length)
        model.fit(values)

        longest = max(horizons)
        mean_path = np.asarray(model.predict(h=longest)["mean"], dtype=float)
        residuals = np.asarray(model.model_.get("residuals", np.zeros(0)), dtype=float)

        rng = np.random.default_rng(seed)
        drawn = _bootstrap_residuals(residuals, n_paths=self.n_paths, horizon=longest, rng=rng)
        simulated = mean_path[None, :] + drawn

        anchor = float(history.iloc[-1])
        order = model.model_.get("arma")
        return {
            horizon: _ensemble(
                simulated[:, :horizon],
                transform=self.transform,
                anchor=anchor,
                session_limit=self.session_limit,
                demean=self.demean,
                detail={
                    "vol_model": {
                        "model": self.label,
                        "conditional_variance": False,
                        "order": None if order is None else [int(v) for v in order],
                    },
                    "residual_sample_size": int(np.isfinite(residuals).sum()),
                },
            )
            for horizon in horizons
        }

    def describe(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "model": {
                "id": self.label,
                "family": "seasonal ARIMA, order selected per as-of date by AIC",
                "distribution": "iid bootstrap of the fit's own in-sample residuals",
                "conditional_variance": False,
            },
            "transform": self.transform.describe(),
            "n_paths": self.n_paths,
            "carries_drift": self.carries_drift,
        }


@dataclass(frozen=True, slots=True)
class NhitsForecaster:
    """NHITS from neuralforecast, refitted at every as-of date.

    Refitting rather than training once and rolling is the expensive choice and the only
    one comparable with the rest of the set: every other model here is refitted at every
    date, and a network trained once on the earliest window and then applied for eleven
    years would be scored on a fit the others were never given.
    """

    transform: ReturnTransform
    session_limit: SessionLimit | None = None
    n_paths: int = 4_000
    min_observations: int = 750
    #: Carry the model's own drift into prices. See :func:`_ensemble` — the run's policy,
    #: not this model's, and the whole set is checked to agree on it.
    demean: bool = True
    max_steps: int = 200
    input_size: int = 128

    @property
    def label(self) -> str:
        return "nhits"

    @property
    def carries_drift(self) -> bool:
        return not self.demean

    def forecast(
        self, history: pd.Series, *, horizons: tuple[int, ...], seed: int
    ) -> dict[int, PathEnsemble]:
        import logging
        import warnings

        from neuralforecast import NeuralForecast
        from neuralforecast.models import NHITS

        returns = self.transform.to_returns(history).dropna()
        require_observations(int(returns.size), self.min_observations, self.label)

        values = returns.to_numpy(dtype=float)
        longest = max(horizons)
        frame = pd.DataFrame({"unique_id": "series", "ds": np.arange(values.size), "y": values})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
            forecaster = NeuralForecast(
                models=[
                    NHITS(
                        h=longest,
                        input_size=self.input_size,
                        max_steps=self.max_steps,
                        enable_progress_bar=False,
                        logger=False,
                        accelerator="cpu",
                        scaler_type="standard",
                        random_seed=seed % (2**31 - 1),
                    )
                ],
                freq=1,
            )
            forecaster.fit(frame)
            mean_path = np.asarray(forecaster.predict()[self.label.upper()], dtype=float)
            fitted = forecaster.predict_insample(step_size=longest)

        residuals = np.asarray(fitted["y"] - fitted[self.label.upper()], dtype=float)

        rng = np.random.default_rng(seed)
        drawn = _bootstrap_residuals(residuals, n_paths=self.n_paths, horizon=longest, rng=rng)
        simulated = mean_path[None, :] + drawn

        anchor = float(history.iloc[-1])
        return {
            horizon: _ensemble(
                simulated[:, :horizon],
                transform=self.transform,
                anchor=anchor,
                session_limit=self.session_limit,
                demean=self.demean,
                detail={
                    "vol_model": {
                        "model": self.label,
                        "conditional_variance": False,
                        "max_steps": self.max_steps,
                        "input_size": self.input_size,
                    },
                    "residual_sample_size": int(np.isfinite(residuals).sum()),
                },
            )
            for horizon in horizons
        }

    def describe(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "model": {
                "id": self.label,
                "family": "NHITS, neural hierarchical interpolation, refitted per as-of date",
                "distribution": "iid bootstrap of the fit's own in-sample residuals",
                "conditional_variance": False,
                "max_steps": self.max_steps,
                "input_size": self.input_size,
            },
            "transform": self.transform.describe(),
            "n_paths": self.n_paths,
            "carries_drift": self.carries_drift,
        }


@dataclass(frozen=True, slots=True)
class ChronosForecaster:
    """Chronos-T5 zero-shot: no fit, no tuning, no sight of this series in training.

    "Zero-shot" is the claim being tested and it is worth being precise about what it
    means here. The model is never fitted, so there is no order selection and no training
    loop at any as-of date; it is shown the last :data:`CHRONOS_CONTEXT` returns and asked
    for the next ``h``. What it learned, it learned from other series before this
    repository existed, which is a different kind of lookahead question from the one the
    harness rules out — the checkpoint's training corpus is not auditable from here, and
    that is recorded as a limitation rather than resolved.
    """

    transform: ReturnTransform
    session_limit: SessionLimit | None = None
    #: Samples drawn per window. Smaller than the simulation models' path counts by
    #: design; see the module docstring for which way that cuts.
    n_paths: int = 200
    min_observations: int = 750
    #: Carry the model's own drift into prices. See :func:`_ensemble` — the run's policy,
    #: not this model's, and the whole set is checked to agree on it.
    demean: bool = True
    checkpoint: str = "amazon/chronos-t5-small"
    context: int = CHRONOS_CONTEXT
    #: Set once at first use, so a walk-forward does not reload the weights per date.
    _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def label(self) -> str:
        return "chronos_t5_small"

    @property
    def carries_drift(self) -> bool:
        return not self.demean

    def _pipeline(self) -> Any:
        import torch
        from chronos import BaseChronosPipeline

        if self.checkpoint not in _PIPELINES:
            _PIPELINES[self.checkpoint] = BaseChronosPipeline.from_pretrained(
                self.checkpoint, device_map="cpu", torch_dtype=torch.float32
            )
        return _PIPELINES[self.checkpoint]

    def forecast(
        self, history: pd.Series, *, horizons: tuple[int, ...], seed: int
    ) -> dict[int, PathEnsemble]:
        import torch

        returns = self.transform.to_returns(history).dropna()
        require_observations(int(returns.size), self.min_observations, self.label)

        values = returns.to_numpy(dtype=float)[-self.context :]
        longest = max(horizons)

        torch.manual_seed(seed % (2**31 - 1))
        samples = self._pipeline().predict(
            torch.tensor(values, dtype=torch.float32),
            prediction_length=longest,
            num_samples=self.n_paths,
        )
        simulated = np.asarray(samples[0], dtype=float)

        anchor = float(history.iloc[-1])
        return {
            horizon: _ensemble(
                simulated[:, :horizon],
                transform=self.transform,
                anchor=anchor,
                session_limit=self.session_limit,
                demean=self.demean,
                detail={
                    "vol_model": {
                        "model": self.label,
                        "conditional_variance": False,
                        "zero_shot": True,
                        "checkpoint": self.checkpoint,
                        "context_sessions": int(values.size),
                    },
                    "residual_sample_size": int(values.size),
                },
            )
            for horizon in horizons
        }

    def describe(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "model": {
                "id": self.label,
                "family": "Chronos-T5 time-series foundation model, zero-shot",
                "checkpoint": self.checkpoint,
                "checkpoint_revision": _checkpoint_revision(self.checkpoint),
                "distribution": "the model's own sampled paths, centred",
                "conditional_variance": False,
                "context_sessions": self.context,
                "fitted": False,
            },
            "transform": self.transform.describe(),
            "n_paths": self.n_paths,
            "carries_drift": self.carries_drift,
            "caveat": (
                "Zero-shot on this series, but the checkpoint's training corpus is not "
                "auditable from this repository. If it contained gold, this is not a "
                "clean out-of-sample test and nothing here can detect that."
            ),
        }
