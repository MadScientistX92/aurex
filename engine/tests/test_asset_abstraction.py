"""§16's leak tests.

Two layers, because they catch different things:

* **The synthetic-asset pipeline run** catches leaks that change behaviour. Per §16,
  if this test needs any change outside the synthetic asset's own definition, the
  abstraction has leaked.
* **The static guard** catches leaks that have not broken anything *yet* — a gold
  literal sitting in ``vol/`` works fine until oil arrives.
"""

from __future__ import annotations

import dataclasses
import re
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from aurex.assets import GOLD, REGISTRY, get, lens_by_code
from aurex.assets.base import Asset
from aurex.assets.friction import FrictionProfile
from aurex.assets.lens import (
    CurrencyLens,
    LensContext,
    NativeLens,
    PriceLinkage,
    TaxedImportLens,
)
from aurex.assets.synthetic import SYNTHETIC
from aurex.assets.transforms import ReturnTransform
from aurex.config import ENGINE_ROOT, REPO_ROOT
from aurex.data.cache import CacheStore
from aurex.pipeline import run

START, END = date(2026, 5, 1), date(2026, 7, 30)


class TestSyntheticAssetRunsTheWholePipeline:
    """The behavioural leak test."""

    def test_pipeline_runs_end_to_end(self, cache: CacheStore) -> None:
        result = run(start=START, end=END, cache=cache, assets=[SYNTHETIC])
        assert SYNTHETIC.id in result.artifact["assets"]

    def test_both_lenses_produce_prices(self, cache: CacheStore) -> None:
        lenses = run(start=START, end=END, cache=cache, assets=[SYNTHETIC]).artifact["assets"][
            SYNTHETIC.id
        ]["lenses"]

        assert set(lenses) == {"XTS", "XTT"}
        for block in lenses.values():
            assert block["latest"]["price"] > 0

    def test_taxed_lens_applies_duty_and_tax(self, cache: CacheStore) -> None:
        latest = run(start=START, end=END, cache=cache, assets=[SYNTHETIC]).artifact["assets"][
            SYNTHETIC.id
        ]["lenses"]["XTT"]["latest"]

        assert latest["duty"] == pytest.approx(0.10)
        assert latest["consumption_tax"] == pytest.approx(0.05)
        assert latest["price"] == pytest.approx(latest["price_ex_consumption_tax"] * 1.05)
        assert latest["unit"] == "crate"

    def test_local_premium_is_measured_for_the_taxed_lens(self, cache: CacheStore) -> None:
        premium = run(start=START, end=END, cache=cache, assets=[SYNTHETIC]).artifact["assets"][
            SYNTHETIC.id
        ]["lenses"]["XTT"]["local_premium"]

        assert premium["observations"] > 0
        assert "latest_bps" in premium

    def test_native_lens_reports_no_local_premium(self, cache: CacheStore) -> None:
        block = run(start=START, end=END, cache=cache, assets=[SYNTHETIC]).artifact["assets"][
            SYNTHETIC.id
        ]["lenses"]["XTS"]
        assert block["local_premium"] is None

    def test_optional_factor_without_a_source_degrades_loudly(self, cache: CacheStore) -> None:
        """The path oil's EIA inventories take when no API key is present."""
        factors = run(start=START, end=END, cache=cache, assets=[SYNTHETIC]).artifact["assets"][
            SYNTHETIC.id
        ]["factors"]
        absent = next(f for f in factors if f["id"] == "absent_driver")

        assert absent["available"] is False
        assert absent["required"] is False
        assert absent["reason"], "a dropped factor must say why"

    def test_the_run_needs_no_network(self, cache: CacheStore) -> None:
        """A self-contained asset must not drag in shared macro series."""
        artifact = run(start=START, end=END, cache=cache, assets=[SYNTHETIC]).artifact
        assert set(artifact["sources"]) == {"widget_price", "widget_fx", "widget_local"}

    def test_shifted_log_transform_is_recorded(self, cache: CacheStore) -> None:
        described = run(start=START, end=END, cache=cache, assets=[SYNTHETIC]).artifact["assets"][
            SYNTHETIC.id
        ]["asset"]["return_transform"]
        assert described["id"] == "shifted_log"
        assert described["shift"] == 25.0


class TestStaticLeakGuard:
    """No asset may be named outside ``assets/``."""

    #: Packages that must stay asset-agnostic.
    GUARDED_DIRS = ("vol", "dist", "factors", "scenarios", "trade", "score")
    #: Modules that compose assets with those packages. They necessarily know what a
    #: lens and an asset *are*, and must still never know which one they are holding.
    GUARDED_MODULES = ("forecast.py", "pipeline.py")
    #: Words that would betray a hardcoded asset. The oil vocabulary is listed while
    #: the guarded packages are still empty: a literal is far cheaper to keep out than
    #: to extract once ``vol/`` and ``factors/`` have been written around it.
    ASSET_LITERALS = ("gold", "xau", "ibja", "brent", "wti", "crude", "oil", "widget")

    def _package_files(self) -> list[Path]:
        files: list[Path] = []
        for name in self.GUARDED_DIRS:
            files.extend((ENGINE_ROOT / "aurex" / name).rglob("*.py"))
        return files

    def _guarded_files(self) -> list[Path]:
        files = self._package_files()
        files.extend(ENGINE_ROOT / "aurex" / name for name in self.GUARDED_MODULES)
        web = REPO_ROOT / "web"
        if web.exists():
            for suffix in ("*.ts", "*.tsx"):
                files.extend(p for p in web.rglob(suffix) if "node_modules" not in p.parts)
        return files

    def test_the_guard_covers_every_downstream_package(self) -> None:
        covered = {p.parent.name for p in self._package_files()}
        assert covered == set(self.GUARDED_DIRS), (
            f"guard missed: {set(self.GUARDED_DIRS) - covered}"
        )

    def test_the_guard_covers_the_composition_modules(self) -> None:
        for name in self.GUARDED_MODULES:
            assert (ENGINE_ROOT / "aurex" / name).exists(), f"{name} is not where the guard looks"

    @pytest.mark.parametrize("literal", ASSET_LITERALS)
    def test_no_asset_literal_downstream(self, literal: str) -> None:
        pattern = re.compile(rf"\b{re.escape(literal)}\b", re.IGNORECASE)
        offenders = [
            f"{path.relative_to(REPO_ROOT)}:{i}"
            for path in self._guarded_files()
            for i, line in enumerate(path.read_text(errors="replace").splitlines(), start=1)
            if pattern.search(line)
        ]
        assert not offenders, f"asset literal {literal!r} leaked into: {offenders}"

    def test_registry_does_not_name_an_asset(self) -> None:
        """Series resolution must be driven by what assets declare."""
        source = (ENGINE_ROOT / "aurex" / "data" / "registry.py").read_text().lower()
        for literal in ("gold", "xau", "ibja", "brent"):
            assert literal not in source, f"registry.py names {literal!r}"


class TestProtocolConformance:
    @pytest.mark.parametrize("asset", [GOLD, SYNTHETIC], ids=lambda a: a.id)
    def test_asset_satisfies_the_protocol(self, asset: Asset) -> None:
        assert isinstance(asset, Asset)
        assert isinstance(asset.return_transform, ReturnTransform)
        assert asset.currency_lenses, "an asset needs at least one lens"
        for lens in asset.currency_lenses:
            assert isinstance(lens, CurrencyLens)
        for profile in asset.friction_profiles.values():
            assert isinstance(profile, FrictionProfile)

    @pytest.mark.parametrize("asset", [GOLD, SYNTHETIC], ids=lambda a: a.id)
    def test_every_foreign_lens_names_its_fx_series(self, asset: Asset) -> None:
        for lens in asset.currency_lenses:
            if lens.requires_fx:
                assert lens.fx_series_id, f"{asset.id}/{lens.code} has no fx_series_id"
                assert lens.fx_series_id in asset.price_sources()

    @pytest.mark.parametrize("asset", [GOLD, SYNTHETIC], ids=lambda a: a.id)
    def test_describe_is_json_safe(self, asset: Asset) -> None:
        import json

        json.dumps(asset.describe())

    @pytest.mark.parametrize("asset", [GOLD, SYNTHETIC], ids=lambda a: a.id)
    def test_every_lens_answers_for_its_provenance(self, asset: Asset) -> None:
        for lens in asset.currency_lenses:
            assert isinstance(lens.provenance_for(date(2026, 7, 30)), dict)

    def test_a_lens_that_misspells_provenance_for_is_not_a_lens(self) -> None:
        """The reason provenance lives on the protocol rather than in a ``getattr``.

        Under duck typing this class reports no citations for rates it applied, and
        nothing fails. As a protocol member it is rejected at the boundary instead.
        """

        class MisspelledLens:
            code = "XTS"
            unit_label = "unit"
            requires_fx = False
            fx_series_id = None
            produces_local_premium = False

            def apply(self, base_prices: pd.Series, ctx: LensContext) -> pd.DataFrame:
                raise NotImplementedError

            def provenence_for(self, day: date) -> dict[str, Any]:  # deliberate typo
                return {"duty": {"source_url": "https://example.invalid"}}

            def describe(self) -> dict[str, Any]:
                return {}

        assert not isinstance(MisspelledLens(), CurrencyLens)

    def test_registry_lookup(self) -> None:
        assert get("gold") is GOLD
        with pytest.raises(KeyError, match="unknown asset"):
            get("plutonium")

    def test_synthetic_is_not_registered_for_production_runs(self) -> None:
        assert SYNTHETIC.id not in REGISTRY


class TestCurrencyLensPolicy:
    def test_gold_offers_both_lenses(self) -> None:
        assert {lens.code for lens in GOLD.currency_lenses} == {"USD", "INR"}

    def test_gold_usd_lens_is_native_and_inr_lens_is_taxed(self) -> None:
        assert isinstance(lens_by_code(GOLD, "USD"), NativeLens)
        assert isinstance(lens_by_code(GOLD, "INR"), TaxedImportLens)

    def test_missing_lens_fails_loudly(self) -> None:
        with pytest.raises(KeyError, match="no EUR lens"):
            lens_by_code(GOLD, "EUR")

    @pytest.mark.parametrize("asset", [GOLD, SYNTHETIC], ids=lambda a: a.id)
    def test_every_lens_price_is_mechanically_linked(self, asset: Asset) -> None:
        assert_price_linkage_is_mechanical(asset)

    def test_the_linkage_guard_rejects_an_administered_price(self) -> None:
        """§18 revises §17.6, which banned a currency where it meant to ban a linkage.

        The superseded rule read "no INR lens for this asset" and would have rejected
        an exchange-settled contract that converts by published formula, while still
        admitting a policy-set retail price in any other currency. The test therefore
        builds an administered lens rather than an INR one.
        """

        administered: PriceLinkage = "administered"

        class PolicyPricedAsset:
            id = "administered_stand_in"
            currency_lenses = (
                dataclasses.replace(
                    NativeLens(code="XTS", unit_label="unit"), price_linkage=administered
                ),
            )

        with pytest.raises(AssertionError, match="administered"):
            assert_price_linkage_is_mechanical(PolicyPricedAsset())  # type: ignore[arg-type]


def assert_price_linkage_is_mechanical(asset: Asset) -> None:
    """Every lens must compute its price from the quote, never report a policy-set one.

    §18: a lens is valid where the buyer's-currency price is a mechanical function of
    the quote — FX, a unit conversion, a statutory rate, a published settlement
    formula — and invalid where it is administered, because presenting it as a view
    on the quote asserts a passthrough nobody has measured.
    """
    for lens in asset.currency_lenses:
        assert lens.price_linkage == "mechanical", (
            f"{asset.id}/{lens.code} declares an administered price: it is set by "
            "policy rather than computed from the quote, so presenting it as a lens "
            "would imply a passthrough that does not exist"
        )
