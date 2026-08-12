"""Local projections and chain composition, on paths whose answer is known.

The failure modes here are quiet ones. A projection can report a band half the width it
should have, because the overlapping windows it fits by construction were not corrected
for. A compounded chain can report a band that excludes zero because three intervals were
multiplied as if the links were independent. Both produce a finding out of nothing, and
neither looks wrong on the page.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest

from aurex.assets.base import ChainControl, ChainLink, TransmissionChain
from aurex.factors import chain as chain_module
from aurex.factors import projections
from aurex.factors.chain import ChainError
from aurex.factors.projections import ProjectionError

MONTHS = 400
HORIZONS = (0, 1, 3, 6)


def _monthly(values: np.ndarray, start: str = "1995-01-01") -> pd.Series:
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="MS"))


class TestOneLinkWithAKnownAnswer:
    @pytest.fixture
    def known(self) -> tuple[pd.Series, pd.Series]:
        """A response that is exactly 0.5 of the shock, one period later, plus noise."""
        rng = np.random.default_rng(13)
        shock = rng.normal(size=MONTHS)
        response = 0.5 * np.concatenate([[0.0], shock[:-1]]) + rng.normal(scale=0.1, size=MONTHS)
        return _monthly(shock), _monthly(response)

    def test_the_cumulative_response_recovers_the_size(
        self, known: tuple[pd.Series, pd.Series]
    ) -> None:
        """At every horizon past one the cumulative response is the whole 0.5."""
        fitted = projections.project(*known, link_id="known", horizons=HORIZONS)

        assert fitted.at(0) is not None and abs(fitted.at(0).coefficient) < 0.1  # type: ignore[union-attr]
        for horizon in (3, 6):
            response = fitted.at(horizon)
            assert response is not None
            assert response.coefficient == pytest.approx(0.5, abs=0.08), horizon
            assert not response.spans_zero

    def test_a_pure_noise_link_spans_zero_everywhere(self) -> None:
        rng = np.random.default_rng(29)
        fitted = projections.project(
            _monthly(rng.normal(size=MONTHS)),
            _monthly(rng.normal(size=MONTHS)),
            link_id="noise",
            horizons=HORIZONS,
        )
        assert fitted.every_horizon_spans_zero

    def test_the_hac_truncation_is_at_least_the_horizon(
        self, known: tuple[pd.Series, pd.Series]
    ) -> None:
        """The overlap is built in by the cumulative response, so the lag must cover it."""
        fitted = projections.project(*known, link_id="known", horizons=(1, 3, 6, 12))
        for response in fitted.responses:
            assert response.hac_lag >= response.horizon

    def test_ignoring_the_overlap_would_understate_the_band(self) -> None:
        """Why the correction is not optional, measured rather than asserted.

        The shock here is persistent, and that is the whole point rather than incidental
        realism. With an *independent* regressor the score ``x_t * u_t`` is serially
        uncorrelated even though the overlapping residual is not, and the correction
        changes nothing — which is what an earlier version of this test measured and
        mistook for the correction failing. Macro shocks are persistent, the score
        inherits that persistence, and the uncorrected error is then too small.
        """
        rng = np.random.default_rng(13)
        persistent = np.zeros(MONTHS)
        for t in range(1, MONTHS):
            persistent[t] = 0.7 * persistent[t - 1] + rng.normal(scale=0.5)
        response = 0.5 * np.concatenate([[0.0], persistent[:-1]]) + rng.normal(
            scale=0.1, size=MONTHS
        )

        cumulative = _monthly(response).rolling(6).sum().shift(-6)
        block = pd.concat(
            [_monthly(persistent).rename("x"), cumulative.rename("y")], axis=1
        ).dropna()
        design = block[["x"]].to_numpy(dtype=float)
        target = block["y"].to_numpy(dtype=float)

        _, uncorrected = projections.hac_errors(design, target, lag=0)
        _, corrected = projections.hac_errors(design, target, lag=6)

        assert corrected[1] > uncorrected[1] * 1.25, (
            "a six-month overlap on a persistent shock must widen the error materially; "
            "if it does not, the correction is not reaching the covariance"
        )

    def test_no_horizons_raises(self, known: tuple[pd.Series, pd.Series]) -> None:
        with pytest.raises(ProjectionError, match="no horizons"):
            projections.project(*known, link_id="known", horizons=())

    def test_too_short_a_sample_raises_rather_than_returning_an_empty_result(self) -> None:
        short = _monthly(np.arange(8.0))
        with pytest.raises(ProjectionError, match="enough overlapping observations"):
            projections.project(short, short, link_id="short", horizons=(24,))


def _chain_frames(seed: int = 7, *, strength: float = 0.5) -> dict[str, pd.DataFrame]:
    """Two links of known size, and a terminal price that is an exact identity."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("1995-01-01", periods=MONTHS, freq="MS")

    first = np.cumsum(rng.normal(0.0, 0.05, MONTHS)) + 4.0
    first_changes = np.diff(np.log(np.exp(first)), prepend=np.log(np.exp(first[0])))
    middle = np.cumsum(strength * first_changes + rng.normal(0.0, 0.01, MONTHS)) + 3.0
    last = np.cumsum(strength * np.diff(middle, prepend=middle[0]) + rng.normal(0, 0.01, MONTHS))

    def frame(values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({"close": np.exp(values)}, index=index)

    quote = frame(np.cumsum(rng.normal(0.0, 0.03, MONTHS)) + 7.0)
    rate = frame(last + 4.0)
    return {
        "src": frame(first),
        "mid": frame(middle),
        "rate": rate,
        "quote": quote,
        # The identity the terminal arrow asserts: the local price is the quote times
        # the rate, exactly, with no tax step anywhere in this fixture.
        "local_price": pd.DataFrame(
            {"close": quote["close"].to_numpy() * rate["close"].to_numpy()}, index=index
        ),
    }


def _chain(controls: tuple[ChainControl, ...] = ()) -> TransmissionChain:
    return TransmissionChain(
        id="test_chain",
        label="A chain that exists to be estimated",
        links=(
            ChainLink(
                id="first",
                source_series="src",
                target_series="mid",
                source_transform="log_diff",
                target_transform="log_diff",
                description="",
            ),
            ChainLink(
                id="second",
                source_series="mid",
                target_series="rate",
                source_transform="log_diff",
                target_transform="log_diff",
                description="",
            ),
        ),
        terminal_lens="XTT",
        terminal_series_id="local_price",
        direct_source_series="src",
        direct_controls=("quote",),
        controls=controls,
    )


class TestComposingTheChain:
    @pytest.fixture(scope="class")
    @staticmethod
    def estimated() -> chain_module.ChainEstimate:
        return chain_module.estimate(
            _chain(), _chain_frames(), horizons=HORIZONS, draws=200, block=12
        )

    def test_every_link_is_estimated_separately(
        self, estimated: chain_module.ChainEstimate
    ) -> None:
        assert [link.link_id for link in estimated.links] == ["first", "second"]
        for link in estimated.links:
            assert len(link.responses) == len(HORIZONS)

    def test_the_compounded_point_is_the_product_of_the_links(
        self, estimated: chain_module.ChainEstimate
    ) -> None:
        for entry in estimated.compounded:
            assert entry.point == pytest.approx(float(np.prod(entry.contributions)))

    def test_the_band_is_not_the_product_of_the_links_bands(
        self, estimated: chain_module.ChainEstimate
    ) -> None:
        """The mistake this bootstrap exists to avoid, asserted rather than described.

        Multiplying the per-link intervals treats three estimates from overlapping
        windows of one economy as independent. If the joint bootstrap ever collapsed
        into that, the two would coincide.
        """
        entry = next(e for e in estimated.compounded if e.horizon == 3)
        naive_low = float(
            np.prod([link.at(3).interval[0] for link in estimated.links])  # type: ignore[union-attr]
        )
        naive_high = float(
            np.prod([link.at(3).interval[1] for link in estimated.links])  # type: ignore[union-attr]
        )

        assert (entry.interval[0], entry.interval[1]) != (naive_low, naive_high)

    def test_the_direct_estimate_is_published_twice(
        self, estimated: chain_module.ChainEstimate
    ) -> None:
        """Raw, and with the overlapping channel held constant. Never one of the two."""
        assert estimated.direct.controls == ()
        assert estimated.direct_orthogonalised.controls == ("quote",)

    def test_the_terminal_identity_is_checked_not_fitted(
        self, estimated: chain_module.ChainEstimate
    ) -> None:
        """The fixture's local price is exactly quote times rate, so both must be one."""
        check = estimated.mechanical_check
        assert check["available"] is True
        assert check["asserted_elasticity"] == 1.0
        assert check["measured"]["quote"] == pytest.approx(1.0, abs=0.01)
        assert check["measured"]["rate"] == pytest.approx(1.0, abs=0.01)
        assert check["r_squared"] == pytest.approx(1.0, abs=1e-3)

    def test_nothing_in_the_output_sums_the_two_decompositions(
        self, estimated: chain_module.ChainEstimate
    ) -> None:
        """The rule, enforced on the block's shape rather than trusted to its prose.

        Checked on keys rather than on text: the prose is allowed to use the word
        "total" to say there is not one, and a substring search over the whole block
        cannot tell that apart from a field that is one.
        """
        block = estimated.describe()

        def keys(node: object) -> list[str]:
            if isinstance(node, dict):
                return [k for key, value in node.items() for k in (key, *keys(value))]
            if isinstance(node, list):
                return [k for item in node for k in keys(item)]
            return []

        offenders = [
            key for key in keys(block) if any(word in key for word in ("total", "sum", "net_of"))
        ]
        assert not offenders, f"a field named like a total invites the sum: {offenders}"
        assert "never summed" in block["double_counting"]
        assert "compounded" in block and "direct" in block

    def test_the_artifact_block_is_json_safe(self, estimated: chain_module.ChainEstimate) -> None:
        import json

        json.dumps(estimated.describe())


class TestRefusalsAndControls:
    def test_a_chain_with_no_links_raises(self) -> None:
        empty = TransmissionChain(
            id="empty",
            label="",
            links=(),
            terminal_lens="XTT",
            terminal_series_id="local_price",
            direct_source_series="src",
        )
        with pytest.raises(ChainError, match="no links"):
            chain_module.estimate(empty, _chain_frames(), horizons=HORIZONS, draws=10)

    def test_a_missing_series_names_itself(self) -> None:
        frames = _chain_frames()
        del frames["mid"]
        with pytest.raises(ChainError, match="'mid'"):
            chain_module.estimate(_chain(), frames, horizons=HORIZONS, draws=10)

    def test_an_unknown_transform_raises(self) -> None:
        with pytest.raises(ChainError, match="unknown chain transform"):
            chain_module._monthly(_monthly(np.arange(1.0, 50.0)), transform="detrend")

    def test_an_administered_control_is_missing_where_the_schedule_is_silent(self) -> None:
        """The gap must not be filled by carrying the last known level forward."""
        index = pd.date_range("1995-01-01", periods=12, freq="MS")

        def resolver(day: date) -> float | None:
            return None if day.month in (5, 6, 7) else 10.0

        control = ChainControl(
            id="administered",
            description="",
            resolver=resolver,
            provenance=lambda: {},
        )
        resolved = chain_module.control_frame(control, index)

        assert resolved.isna().sum() >= 3, "an unknown level must stay unknown"
        assert not resolved.ffill().equals(resolved), "the gap was filled"

    def test_moves_are_flagged_by_month(self) -> None:
        index = pd.date_range("1995-01-01", periods=6, freq="MS")
        levels = {0: 10.0, 1: 10.0, 2: 12.0, 3: 12.0, 4: 12.0, 5: 9.0}

        def resolver(day: date) -> float | None:
            return levels[(day.year - 1995) * 12 + day.month - 1]

        control = ChainControl(
            id="administered", description="", resolver=resolver, provenance=lambda: {}
        )
        assert chain_module.moves_in(control, index) == ("1995-03-01", "1995-06-01")

    def test_the_local_price_is_never_recomputed_here(self) -> None:
        """A second implementation of the tax stack is how it drifts from its citations."""
        with pytest.raises(NotImplementedError, match="currency lens computes"):
            chain_module.local_price_series(pd.Series(dtype=float), pd.Series(dtype=float))


class TestTheDeclaredChainOnTheRealAsset:
    """What the asset declares, checked without estimating anything."""

    def test_the_registered_asset_declares_a_chain_that_names_its_own_series(self) -> None:
        from aurex.assets import GOLD

        declared = GOLD.transmission_chain
        assert declared is not None
        assert declared.terminal_series_id not in declared.series_ids(), (
            "the terminal is computed by a lens and must not be requested as a source"
        )
        for link in declared.links:
            assert link.source_transform in {"log_diff", "diff"}
            assert link.target_transform in {"log_diff", "diff"}

    def test_the_chains_series_are_resolvable_by_the_registry(self) -> None:
        from aurex.assets import GOLD
        from aurex.data.registry import chains_for

        resolvable = set(chains_for([GOLD]))
        declared = GOLD.transmission_chain
        assert declared is not None
        assert declared.series_ids() <= resolvable, (
            "a chain naming a series nothing can load is a chain that never runs"
        )

    def test_the_administered_control_publishes_its_gaps(self) -> None:
        from aurex.assets import GOLD

        declared = GOLD.transmission_chain
        assert declared is not None
        provenance: dict[str, Any] = declared.controls[0].provenance()

        assert provenance["entries"], "the control must carry its citations"
        assert provenance["gaps"], "and the windows it cannot cite"
        for entry in provenance["entries"]:
            assert entry["source_url"].startswith("http")
            assert entry["source_confidence"] in {"primary", "secondary"}
        for gap in provenance["gaps"]:
            assert gap["source_url"].startswith("http")
            assert gap["reason"]
