"""Distributions through the pipeline, from a cache with enough history to fit.

The seeded cache elsewhere in the suite holds two months of prices, which is the
right fixture for lens arithmetic and the wrong one for a variance model — so this
module builds its own multi-year cache. That split is deliberate: it keeps the
distribution tests honest about how much data they need, and it keeps every other
test fast.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from aurex.assets import GOLD
from aurex.assets.synthetic import SYNTHETIC
from aurex.data.base import LoadedSeries, SeriesMeta
from aurex.data.cache import CacheStore
from aurex.forecast import ForecastRequest, forecast_asset
from aurex.pipeline import run

START, END = date(2020, 1, 1), date(2026, 7, 30)

#: Small but not trivial: enough paths for a stable quantile, few enough to keep the
#: suite quick. Production defaults to twenty thousand.
REQUEST = ForecastRequest(horizons=(5, 21), n_paths=2_000, block_length=5, seed=4242)


def random_walk(series_id: str, *, level: float, sigma: float, seed: int) -> LoadedSeries:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(START, END, name="date")
    prices = level * np.exp(np.cumsum(sigma * rng.standard_normal(len(index))))
    frame = pd.DataFrame({"close": prices}, index=index)
    return LoadedSeries(
        frame=frame,
        meta=SeriesMeta(
            series_id=series_id,
            source_name="test:random-walk",
            source_url="https://example.invalid/series",
            fetched_at=pd.Timestamp("2026-07-31", tz="UTC").to_pydatetime(),
            rows=len(frame),
            start=index[0].date(),
            end=index[-1].date(),
        ),
    )


@pytest.fixture
def long_cache(cache: CacheStore) -> CacheStore:
    index = pd.bdate_range(START, END, name="date")
    cache.write(random_walk("xauusd", level=1_500.0, sigma=0.009, seed=1))
    cache.write(random_walk("usdinr", level=71.0, sigma=0.003, seed=2))
    cache.write(
        LoadedSeries(
            frame=pd.DataFrame(
                {
                    "gold_999_pm": np.full(len(index), 142_000.0),
                    "spdr_gold_tonnes": np.full(len(index), 1_008.73),
                },
                index=index,
            ),
            meta=SeriesMeta(
                series_id="ibja_gold",
                source_name="IBJA:daily-bullion-report",
                source_url="https://www.ibja.co/Upload/x.pdf",
                fetched_at=pd.Timestamp("2026-07-31", tz="UTC").to_pydatetime(),
                rows=len(index),
                start=index[0].date(),
                end=index[-1].date(),
            ),
        )
    )
    return cache


def lenses(cache: CacheStore) -> dict:
    artifact = run(
        offline=True, start=START, end=END, cache=cache, assets=[GOLD], forecast=REQUEST
    ).artifact
    return artifact["assets"]["gold"]["lenses"]


class TestDistributionBlock:
    def test_every_lens_carries_one(self, long_cache: CacheStore) -> None:
        for code, block in lenses(long_cache).items():
            assert block["distribution"]["available"] is True, code

    def test_quantiles_are_ordered_at_every_horizon(self, long_cache: CacheStore) -> None:
        for block in lenses(long_cache).values():
            for horizon in block["distribution"]["horizons"].values():
                quantiles = horizon["quantiles"]
                values = [quantiles[key] for key in sorted(quantiles)]
                assert values == sorted(values)

    def test_the_anchor_is_the_last_observed_price(self, long_cache: CacheStore) -> None:
        """The distribution starts where the data stopped, in the lens's own units.

        Compared to the cent rather than exactly: the published price is rounded for
        display and the anchor keeps two more decimals.
        """
        for block in lenses(long_cache).values():
            assert block["distribution"]["anchor"] == pytest.approx(
                block["latest"]["price"], abs=0.01
            )

    def test_uncertainty_widens_with_horizon(self, long_cache: CacheStore) -> None:
        for block in lenses(long_cache).values():
            horizons = block["distribution"]["horizons"]
            near = horizons["5"]["quantiles"]
            far = horizons["21"]["quantiles"]
            assert far["q95"] - far["q05"] > near["q95"] - near["q05"]

    def test_no_point_estimate_is_published(self, long_cache: CacheStore) -> None:
        """§0: a number without a distribution behind it is a bug."""
        for block in lenses(long_cache).values():
            for horizon in block["distribution"]["horizons"].values():
                assert set(horizon["quantiles"]) == {"q05", "q25", "q50", "q75", "q95"}
                assert "mean" not in horizon
                assert "expected" not in json.dumps(horizon).lower()


class TestPathDependenceReachesTheArtifact:
    def test_touching_is_reported_next_to_ending(self, long_cache: CacheStore) -> None:
        """§18: a terminal-only artifact hides the event a margined holder cares about."""
        moves = lenses(long_cache)["USD"]["distribution"]["horizons"]["21"]["adverse_moves"]
        assert [entry["move"] for entry in moves] == [0.05, 0.10, 0.20]

        for entry in moves:
            assert entry["touch_probability"] >= entry["terminal_probability"]
            assert entry["monitoring"] == "session_close"

    def test_the_gap_is_visible_where_the_barrier_is_reachable(
        self, long_cache: CacheStore
    ) -> None:
        moves = lenses(long_cache)["USD"]["distribution"]["horizons"]["21"]["adverse_moves"]
        nearest = moves[0]

        assert nearest["touch_probability"] > 0.0
        assert nearest["path_dependence_ratio"] is not None
        assert nearest["path_dependence_ratio"] > 1.0

    def test_a_barrier_nothing_reaches_reports_no_ratio(self, long_cache: CacheStore) -> None:
        far = lenses(long_cache)["USD"]["distribution"]["horizons"]["5"]["adverse_moves"][-1]
        assert far["touch_probability"] == 0.0
        assert far["path_dependence_ratio"] is None


class TestTheTwoViewsShareOneSimulation:
    def test_both_lenses_report_the_same_underlying_model(self, long_cache: CacheStore) -> None:
        blocks = lenses(long_cache)
        usd = blocks["USD"]["distribution"]["vol_model"]
        inr = blocks["INR"]["distribution"]["vol_model"]
        assert usd["params"] == inr["params"]

    def test_only_the_converted_lens_reports_an_exchange_rate_model(
        self, long_cache: CacheStore
    ) -> None:
        """A native view carries no FX exposure, so it must not advertise one."""
        blocks = lenses(long_cache)
        assert blocks["USD"]["distribution"]["fx_vol_model"] is None
        assert blocks["USD"]["distribution"]["copula"] is None
        assert blocks["INR"]["distribution"]["fx_vol_model"] is not None

    def test_the_copula_is_fitted_and_described(self, long_cache: CacheStore) -> None:
        copula = lenses(long_cache)["INR"]["distribution"]["copula"]
        assert copula["family"] == "student_t"
        assert -1.0 <= copula["rho"] <= 1.0
        assert copula["tail_dependence"] >= 0.0

    def test_the_converted_view_carries_two_sources_of_uncertainty(
        self, long_cache: CacheStore
    ) -> None:
        """§15: measured, never asserted — so the assertion is the weaker one.

        The rupee view is not required to be wider; it is required to differ, because
        it is the dollar view plus an exchange rate rather than a rescaling of it.
        """
        blocks = lenses(long_cache)
        usd = blocks["USD"]["distribution"]["horizons"]["21"]["quantiles"]
        inr = blocks["INR"]["distribution"]["horizons"]["21"]["quantiles"]

        usd_width = (usd["q95"] - usd["q05"]) / blocks["USD"]["distribution"]["anchor"]
        inr_width = (inr["q95"] - inr["q05"]) / blocks["INR"]["distribution"]["anchor"]
        assert usd_width != pytest.approx(inr_width, rel=1e-6)


class TestReproducibility:
    def test_two_runs_agree_on_every_number(self, long_cache: CacheStore) -> None:
        """A distribution nobody can reproduce cannot be scored in step 3a."""
        first = lenses(long_cache)["INR"]["distribution"]["horizons"]
        second = lenses(long_cache)["INR"]["distribution"]["horizons"]
        assert first == second

    def test_the_seed_is_published(self, long_cache: CacheStore) -> None:
        simulation = lenses(long_cache)["USD"]["distribution"]["simulation"]
        assert simulation["seed"] == REQUEST.seed
        assert simulation["n_paths"] == REQUEST.n_paths

    def test_adding_a_horizon_does_not_move_the_others(self, long_cache: CacheStore) -> None:
        """Each horizon draws its own stream, so publishing more cannot revise less."""
        narrow = run(
            offline=True,
            start=START,
            end=END,
            cache=long_cache,
            assets=[GOLD],
            forecast=ForecastRequest(horizons=(5,), n_paths=2_000, block_length=5, seed=4242),
        ).artifact["assets"]["gold"]["lenses"]["USD"]["distribution"]

        wide = lenses(long_cache)["USD"]["distribution"]
        assert narrow["horizons"]["5"] == wide["horizons"]["5"]


class TestDependenceModes:
    """``synchronised`` is the check on ``t_copula``, so it has to work end to end."""

    def _synchronised(self, cache: CacheStore) -> dict:
        return run(
            offline=True,
            start=START,
            end=END,
            cache=cache,
            assets=[GOLD],
            forecast=ForecastRequest(
                horizons=(21,), n_paths=2_000, block_length=5, seed=4242, dependence="synchronised"
            ),
        ).artifact["assets"]["gold"]["lenses"]["INR"]["distribution"]

    def test_it_produces_a_distribution_without_fitting_a_copula(
        self, long_cache: CacheStore
    ) -> None:
        block = self._synchronised(long_cache)
        assert block["available"] is True
        assert block["copula"] is None
        assert block["fx_vol_model"] is not None

    def test_the_two_modes_disagree_somewhere(self, long_cache: CacheStore) -> None:
        """If they agreed exactly, the copula would be doing no work at all.

        §6's discipline is to publish the disagreement rather than pick a winner, so
        the engine has to be able to produce both from the same inputs.
        """
        parametric = lenses(long_cache)["INR"]["distribution"]["horizons"]["21"]["quantiles"]
        empirical = self._synchronised(long_cache)["horizons"]["21"]["quantiles"]
        assert parametric != empirical


class TestDegradation:
    def test_too_little_history_yields_a_reason_rather_than_a_number(
        self, cache: CacheStore
    ) -> None:
        """The same discipline as a factor that drops out: never silently absent."""
        short_start, short_end = date(2026, 6, 1), date(2026, 7, 30)
        index = pd.bdate_range(short_start, short_end, name="date")
        for series_id, level in (("xauusd", 4_000.0), ("usdinr", 96.5)):
            frame = pd.DataFrame({"close": np.full(len(index), level)}, index=index)
            cache.write(
                LoadedSeries(
                    frame=frame,
                    meta=SeriesMeta(
                        series_id=series_id,
                        source_name="test:flat",
                        source_url="https://example.invalid/series",
                        fetched_at=pd.Timestamp("2026-07-31", tz="UTC").to_pydatetime(),
                        rows=len(frame),
                        start=index[0].date(),
                        end=index[-1].date(),
                    ),
                )
            )

        block = run(
            offline=True, start=short_start, end=short_end, cache=cache, assets=[GOLD]
        ).artifact["assets"]["gold"]["lenses"]["USD"]["distribution"]

        assert block["available"] is False
        assert "500" in block["reason"]

    def test_a_lens_that_produced_no_prices_says_so(self, long_cache: CacheStore) -> None:
        """An empty lens frame must not silently become an absent distribution."""
        loaded = long_cache.read("xauusd")
        assert loaded is not None
        forecast = forecast_asset(
            GOLD,
            base_prices=loaded.frame["close"],
            lens_frames={"USD": pd.DataFrame(), "INR": pd.DataFrame()},
            fx_series={},
            request=REQUEST,
        )

        assert forecast.lenses["USD"].available is False
        assert forecast.lenses["USD"].unavailable_reason == "the lens produced no prices"

    def test_a_missing_exchange_rate_leaves_the_converted_lens_unavailable(
        self, long_cache: CacheStore
    ) -> None:
        """The native view still resolves; only the leg that needs a rate drops out."""
        loaded = long_cache.read("xauusd")
        assert loaded is not None
        prices = loaded.frame["close"]
        usd_frame = pd.DataFrame(
            {
                "price": prices,
                "price_ex_consumption_tax": prices,
                "base_quote_per_unit": prices,
                "fx_rate": 1.0,
                "duty": 0.0,
                "consumption_tax": 0.0,
                "confidence": "high",
            }
        )

        forecast = forecast_asset(
            GOLD,
            base_prices=prices,
            lens_frames={"USD": usd_frame, "INR": usd_frame},
            fx_series={},  # the rate never resolved
            request=REQUEST,
        )

        assert forecast.lenses["USD"].available is True
        assert forecast.lenses["INR"].available is False
        assert forecast.lenses["INR"].unavailable_reason

    def test_the_artifact_stays_json_serialisable(self, long_cache: CacheStore) -> None:
        artifact = run(
            offline=True, start=START, end=END, cache=long_cache, assets=[GOLD], forecast=REQUEST
        ).artifact
        json.dumps(artifact, sort_keys=True)


class TestSessionLimitsReachTheSimulation:
    """The synthetic asset declares a limit, so the limited branch is exercised."""

    def test_a_declared_limit_is_applied_and_reported(self, cache: CacheStore) -> None:
        artifact = run(
            start=date(2026, 5, 1),
            end=date(2026, 7, 30),
            cache=cache,
            assets=[SYNTHETIC],
            forecast=ForecastRequest(horizons=(5,), n_paths=500, block_length=3, seed=7),
        ).artifact

        block = artifact["assets"][SYNTHETIC.id]["lenses"]["XTS"]["distribution"]
        assert block["session_limit"]["fraction"] == 0.05
        assert block["session_limit"]["relaxed_fraction"] == 0.08
        assert block["session_limit"]["share_of_sessions_truncated"] >= 0.0

    def test_no_simulated_session_exceeds_the_declared_cap(self, cache: CacheStore) -> None:
        result = run(
            start=date(2026, 5, 1),
            end=date(2026, 7, 30),
            cache=cache,
            assets=[SYNTHETIC],
            forecast=ForecastRequest(horizons=(5,), n_paths=500, block_length=3, seed=7),
        )
        ensemble = result.forecasts[SYNTHETIC.id].lenses["XTS"].ensembles[5]
        walked = np.hstack([np.full((ensemble.n_paths, 1), ensemble.anchor), ensemble.prices])
        moves = np.abs(np.diff(walked, axis=1) / walked[:, :-1])

        assert moves.max() <= 0.08 + 1e-9

    def test_an_asset_without_a_limit_reports_none(self, long_cache: CacheStore) -> None:
        assert lenses(long_cache)["USD"]["distribution"]["session_limit"] is None
