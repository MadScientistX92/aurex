"""§20's schema test: the routes table, and the rules it is not allowed to break.

Three layers, matching how these fail in practice:

* **The shipped table loads and says what it claims.** Provenance on every entry, and
  the breakeven numbers the README publishes actually coming out of the data.
* **Each invariant is tested against a table that violates it.** Every rule in
  :mod:`aurex.routes` exists because §20 got something wrong first, so each one is
  exercised by building the malformed table it was written to reject. A validator
  tested only against the file that already passes is a validator tested once.
* **The split is asserted structurally.** ``Route`` must not carry a field that varies
  by jurisdiction — checked against the dataclass itself, not against prose, because
  the failure mode is somebody adding one back in good faith.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from typing import Any

import pytest
from typer.testing import CliRunner

from aurex.cli import app
from aurex.config import REPO_ROOT
from aurex.data.schedules.provenance import VALID_CONFIDENCE, ScheduleError
from aurex.routes import (
    JURISDICTION_CODE,
    BenchmarkView,
    Route,
    RouteError,
    build_routes,
    load_routes,
)
from aurex.trade import breakeven_table
from aurex.vol import DeterministicVarianceError, model_for, path_dependent_models

#: A minimal well-formed table. Every invariant test starts here and breaks one thing,
#: so a failure names the rule rather than an unrelated missing key.
VALID: dict[str, Any] = {
    "jurisdictions": {
        "AAA": {
            "label": "Alphaland",
            "bloc": None,
            "source_url": "https://example.invalid/aaa",
            "source_confidence": "primary",
        },
        "BBB": {
            "label": "Betastan",
            "bloc": "XB",
            "source_url": "https://example.invalid/bbb-national-law",
            "source_confidence": "primary",
        },
    },
    "blocs": {
        "XB": {
            "label": "Example bloc",
            "members": ["BBB"],
            "instrument": "Example Directive, implemented nationally",
            "source_url": "https://example.invalid/directive",
            "source_confidence": "primary",
        }
    },
    "routes": {
        "spot": {
            "asset_id": "widget",
            "venue": "over_the_counter",
            "instrument": "physical",
            "quote_currency": "XTS",
            "available_in": ["AAA", "BBB"],
            "source_url": "https://example.invalid/route",
            "source_confidence": "primary",
        }
    },
    "terms": [
        {
            "route_id": "spot",
            "jurisdiction": "AAA",
            "source_url": "https://example.invalid/aaa-tax",
            "source_confidence": "primary",
            "max_leverage": None,
            "friction": {
                "kind": "physical",
                "label": "Alphaland retail",
                "dealer_premium": 0.02,
                "consumption_tax": 0.05,
                "buyback_discount": 0.02,
                "spread_basis": "representative dealer quotes, user-editable",
            },
        },
        {
            "route_id": "spot",
            "jurisdiction": "BBB",
            "source_url": "https://example.invalid/bbb-tax",
            "source_confidence": "primary",
            "max_leverage": None,
            "friction": {
                "kind": "physical",
                "label": "Betastan retail",
                "dealer_premium": 0.0,
                "consumption_tax": 0.0,
                "buyback_discount": 0.0,
            },
        },
    ],
}


def broken(**mutations: Any) -> dict[str, Any]:
    """A deep copy of :data:`VALID` with top-level keys replaced."""
    table = copy.deepcopy(VALID)
    table.update(copy.deepcopy(mutations))
    return table


class TestTheShippedTable:
    def test_it_loads(self) -> None:
        book = load_routes()

        assert book.routes
        assert book.jurisdictions
        assert book.terms

    def test_every_entry_carries_its_own_provenance(self) -> None:
        """The rule duty.yaml is held to, applied by the same function."""
        book = load_routes()
        carriers = [*book.routes, *book.jurisdictions, *book.blocs, *book.terms]

        for entry in carriers:
            assert entry.source_url.startswith("http"), f"{entry} has no usable source_url"
            assert entry.source_confidence in VALID_CONFIDENCE

    def test_no_jurisdiction_is_the_default(self) -> None:
        """An unset jurisdiction is a supported state, not a missing input."""
        view = load_routes().benchmark("physical_retail")

        assert isinstance(view, BenchmarkView)
        assert view.friction_excluded
        assert view.describe()["jurisdiction"] is None
        assert view.describe()["friction"] is None
        assert "friction excluded" in view.describe()["reading"]

    def test_the_benchmark_view_cannot_be_mistaken_for_terms(self) -> None:
        """It has no friction attribute at all, so no caller can reach for a hurdle."""
        assert not hasattr(load_routes().benchmark("physical_retail"), "friction")

    def test_the_same_route_has_different_terms_in_different_jurisdictions(self) -> None:
        """The reason terms are keyed by both. If this collapses, §20's bug is back."""
        book = load_routes()
        hurdles = {
            code: book.terms_for("physical_retail", code).friction.quote(21).breakeven_pct
            for code in ("IND", "USA")
        }

        assert hurdles["IND"] > hurdles["USA"]

    def test_restricted_in_is_derived_and_complements_available_in(self) -> None:
        book = load_routes()
        route = book.route("cfd")
        derived = book.restricted_in("cfd")

        assert set(derived).isdisjoint(route.available_in)
        assert set(derived) | set(route.available_in) == {j.code for j in book.jurisdictions}

    def test_an_unrecorded_cell_fails_without_advising(self) -> None:
        """Absence of data must not read as a statement about what a reader may hold."""
        with pytest.raises(KeyError, match="not a statement about whether"):
            load_routes().terms_for("cfd", "IND")

    def test_the_leveraged_route_cites_the_national_regulator(self) -> None:
        leveraged = load_routes().leveraged_terms()

        assert leveraged, "the HAR-RV bar needs at least one leveraged route to bar"
        for entry in leveraged:
            assert "esma.europa.eu" not in entry.source_url

    def test_a_bloc_member_cites_its_own_national_instrument(self) -> None:
        """VAT exemption on investment gold is a Directive implemented per member state."""
        book = load_routes()
        directives = {bloc.source_url for bloc in book.blocs}
        members = {j.code for j in book.jurisdictions if j.bloc is not None}

        for entry in book.terms:
            if entry.jurisdiction in members:
                assert entry.source_url not in directives

    def test_the_table_is_json_safe(self) -> None:
        json.dumps(load_routes().describe())


class TestRouteHoldsOnlyWhatIsJurisdictionInvariant:
    """The correction §20 needed. Asserted against the class, not against the docstring."""

    def test_the_route_dataclass_has_no_jurisdiction_varying_field(self) -> None:
        fields = {f.name for f in dataclasses.fields(Route)}

        assert "friction" not in fields
        assert "max_leverage" not in fields

    @pytest.mark.parametrize("field", ["friction", "max_leverage"])
    def test_declaring_one_on_a_route_is_refused(self, field: str) -> None:
        routes = copy.deepcopy(VALID["routes"])
        routes["spot"][field] = 5.0

        with pytest.raises(RouteError, match="varies by jurisdiction"):
            build_routes(broken(routes=routes))


class TestProvenanceIsEnforcedEverywhere:
    @pytest.mark.parametrize("section", ["jurisdictions", "blocs", "routes"])
    def test_a_keyed_entry_without_a_source_is_refused(self, section: str) -> None:
        table = copy.deepcopy(VALID)
        key = next(iter(table[section]))
        del table[section][key]["source_url"]

        with pytest.raises(ScheduleError, match="missing source_url"):
            build_routes(table)

    def test_a_terms_entry_without_a_source_is_refused(self) -> None:
        terms = copy.deepcopy(VALID["terms"])
        del terms[0]["source_url"]

        with pytest.raises(ScheduleError, match="missing source_url"):
            build_routes(broken(terms=terms))

    def test_an_invented_confidence_level_is_refused(self) -> None:
        terms = copy.deepcopy(VALID["terms"])
        terms[0]["source_confidence"] = "pretty sure"

        with pytest.raises(ScheduleError, match="not in"):
            build_routes(broken(terms=terms))


class TestJurisdictionCodes:
    def test_the_shipped_codes_are_uppercase_alpha_3(self) -> None:
        for entry in load_routes().jurisdictions:
            assert JURISDICTION_CODE.match(entry.code)

    @pytest.mark.parametrize("code", ["IN", "ind", "INDIA", "In"])
    def test_anything_but_uppercase_alpha_3_is_refused(self, code: str) -> None:
        jurisdictions = {code: copy.deepcopy(VALID["jurisdictions"]["AAA"])}
        routes = copy.deepcopy(VALID["routes"])
        routes["spot"]["available_in"] = [code]

        with pytest.raises(RouteError, match="uppercase ISO-3166 alpha-3"):
            build_routes(broken(jurisdictions=jurisdictions, routes=routes, blocs={}, terms=[]))


class TestRestrictedInIsNeverStored:
    def test_writing_it_by_hand_is_refused(self) -> None:
        """Two hand-maintained lists eventually disagree, and nobody reads the wrong one."""
        routes = copy.deepcopy(VALID["routes"])
        routes["spot"]["restricted_in"] = ["BBB"]

        with pytest.raises(RouteError, match="derived from available_in"):
            build_routes(broken(routes=routes))


class TestAvailabilityAndTermsAreCheckedAgainstEachOther:
    def test_a_route_available_somewhere_with_no_terms_is_refused(self) -> None:
        """A missing cell would silently mean a hurdle of zero."""
        terms = [copy.deepcopy(VALID["terms"][0])]

        with pytest.raises(RouteError, match="claim availability with no terms"):
            build_routes(broken(terms=terms))

    def test_terms_for_a_jurisdiction_the_route_is_not_available_in_are_refused(self) -> None:
        routes = copy.deepcopy(VALID["routes"])
        routes["spot"]["available_in"] = ["AAA"]

        with pytest.raises(RouteError, match="not available_in"):
            build_routes(broken(routes=routes))

    def test_a_duplicated_cell_is_refused(self) -> None:
        terms = [*copy.deepcopy(VALID["terms"]), copy.deepcopy(VALID["terms"][0])]

        with pytest.raises(RouteError, match="duplicates the cell"):
            build_routes(broken(terms=terms))

    def test_an_unknown_jurisdiction_in_available_in_is_refused(self) -> None:
        routes = copy.deepcopy(VALID["routes"])
        routes["spot"]["available_in"] = ["AAA", "BBB", "ZZZ"]

        with pytest.raises(RouteError, match="unknown jurisdictions"):
            build_routes(broken(routes=routes))


class TestLeverageCaps:
    def test_a_cap_sourced_to_a_coordinating_body_is_refused(self) -> None:
        """The national regulator publishes the binding number; cite whoever a reader
        would have to argue with."""
        terms = copy.deepcopy(VALID["terms"])
        terms[0]["max_leverage"] = 20.0
        terms[0]["source_url"] = "https://www.esma.europa.eu/some-opinion"

        with pytest.raises(RouteError, match="coordinates rather than sets"):
            build_routes(broken(terms=terms))

    def test_a_national_regulator_is_accepted(self) -> None:
        terms = copy.deepcopy(VALID["terms"])
        terms[0]["max_leverage"] = 20.0

        book = build_routes(broken(terms=terms))
        assert book.terms_for("spot", "AAA").leveraged

    @pytest.mark.parametrize("cap", [1.0, 0.5, -2.0])
    def test_a_cap_that_is_not_leverage_is_refused(self, cap: float) -> None:
        terms = copy.deepcopy(VALID["terms"])
        terms[0]["max_leverage"] = cap

        with pytest.raises(RouteError, match="not leverage"):
            build_routes(broken(terms=terms))

    def test_an_unleveraged_route_reports_so(self) -> None:
        assert not build_routes(VALID).terms_for("spot", "AAA").leveraged


class TestBlocs:
    def test_a_bloc_may_not_also_be_a_jurisdiction(self) -> None:
        jurisdictions = copy.deepcopy(VALID["jurisdictions"])
        jurisdictions["XBX"] = {
            "label": "Bloc pretending to be a country",
            "bloc": None,
            "source_url": "https://example.invalid/x",
            "source_confidence": "primary",
        }
        blocs = {"XBX": {**copy.deepcopy(VALID["blocs"]["XB"]), "members": ["AAA"]}}

        with pytest.raises(RouteError, match="both a bloc and a jurisdiction"):
            build_routes(broken(jurisdictions=jurisdictions, blocs=blocs))

    def test_a_bloc_with_no_members_is_refused(self) -> None:
        blocs = copy.deepcopy(VALID["blocs"])
        blocs["XB"]["members"] = []
        jurisdictions = copy.deepcopy(VALID["jurisdictions"])
        jurisdictions["BBB"]["bloc"] = None

        with pytest.raises(RouteError, match="no members"):
            build_routes(broken(blocs=blocs, jurisdictions=jurisdictions))

    def test_membership_must_be_stated_on_both_sides(self) -> None:
        jurisdictions = copy.deepcopy(VALID["jurisdictions"])
        jurisdictions["BBB"]["bloc"] = None

        with pytest.raises(RouteError, match="do not declare that membership"):
            build_routes(broken(jurisdictions=jurisdictions))

    def test_a_member_citing_the_directive_instead_of_its_own_law_is_refused(self) -> None:
        """The substantive reason blocs are not jurisdictions here."""
        terms = copy.deepcopy(VALID["terms"])
        terms[1]["source_url"] = VALID["blocs"]["XB"]["source_url"]

        with pytest.raises(RouteError, match="implemented in national law"):
            build_routes(broken(terms=terms))


class TestFrictionShapeIsDeclared:
    def test_an_undeclared_shape_is_refused(self) -> None:
        terms = copy.deepcopy(VALID["terms"])
        del terms[0]["friction"]["kind"]

        with pytest.raises(RouteError, match="not one of 'physical' or 'roll'"):
            build_routes(broken(terms=terms))

    def test_a_roll_shape_is_horizon_dependent_and_a_physical_one_is_not(self) -> None:
        book = load_routes()
        physical = book.terms_for("physical_retail", "IND").friction
        roll = book.terms_for("cfd", "GBR").friction

        assert physical.quote(5).breakeven_multiple == physical.quote(252).breakeven_multiple
        assert roll.quote(252).breakeven_multiple > roll.quote(5).breakeven_multiple


class TestMarketSpreadsMayNotBorrowARegulatorsCitation:
    """A dealer premium is not a published rule and must not sit under one's URL."""

    def test_a_non_zero_spread_without_a_basis_is_refused(self) -> None:
        terms = copy.deepcopy(VALID["terms"])
        del terms[0]["friction"]["spread_basis"]

        with pytest.raises(RouteError, match="no spread_basis is declared"):
            build_routes(broken(terms=terms))

    def test_all_zero_spreads_need_no_basis(self) -> None:
        """Nothing to source when nothing is charged."""
        assert build_routes(VALID).terms_for("spot", "BBB").friction.quote(21)

    def test_the_basis_travels_into_the_quote(self) -> None:
        notes = build_routes(VALID).terms_for("spot", "AAA").friction.quote(21).notes

        assert any("representative dealer quotes" in note for note in notes)


class TestHarRvIsBarredOnALeveragedRoute:
    """§8's constraint, which needed Route to exist before it had anything to hang on.

    The bar is on the *combination*, not on the model: an unleveraged distribution
    from a deterministic-variance model is a legitimate thing to publish, and step 3a
    scores one. What is refused is reading a path statistic off an ensemble whose
    paths all share one variance trajectory.
    """

    def test_the_leveraged_cell_refuses_the_deterministic_model(self) -> None:
        book = load_routes()

        with pytest.raises(DeterministicVarianceError, match="shares one volatility"):
            book.require_model(model_for("har_rv"), "cfd", "GBR")

    def test_the_same_model_is_allowed_where_there_is_no_leverage(self) -> None:
        load_routes().require_model(model_for("har_rv"), "physical_retail", "IND")

    def test_the_path_dependent_model_is_allowed_everywhere(self) -> None:
        book = load_routes()

        book.require_model(model_for("gjr_garch"), "cfd", "GBR")
        book.require_model(model_for("gjr_garch"), "physical_retail", "IND")

    def test_the_refusal_names_a_model_that_would_work(self) -> None:
        """An error that only says no makes the reader guess."""
        with pytest.raises(DeterministicVarianceError, match="gjr_garch"):
            model_for("har_rv", leveraged=True)

    def test_exactly_one_shipped_model_carries_per_path_variance(self) -> None:
        """If a second ever does, this test is where that gets noticed and written down."""
        assert path_dependent_models() == ["gjr_garch"]

    @pytest.mark.parametrize("model_id", ["har_rv", "rolling_std"])
    def test_the_deterministic_models_declare_themselves(self, model_id: str) -> None:
        assert not model_for(model_id).per_path_variance


class TestTheGeneratedBreakevenTable:
    def test_it_renders_a_row_per_cell(self) -> None:
        rendered = breakeven_table(load_routes().table_rows("gold"))

        assert rendered.count("\n") == len(load_routes().table_rows("gold")) + 1

    def test_horizon_invariant_friction_is_flat_across_the_columns(self) -> None:
        row = next(
            line
            for line in breakeven_table(load_routes().table_rows("gold")).splitlines()
            if "India" in line
        )
        cells = [cell.strip() for cell in row.split("|")[2:-2]]

        assert len(set(cells)) == 1
        assert row.strip().endswith("no |")

    def test_accruing_friction_grows_with_the_horizon(self) -> None:
        row = next(
            line
            for line in breakeven_table(load_routes().table_rows("gold")).splitlines()
            if line.startswith("| cfd")
        )
        cells = [float(cell.strip().rstrip("%")) for cell in row.split("|")[2:-2]]

        assert cells == sorted(cells)
        assert cells[0] < cells[-1]
        assert row.strip().endswith("yes |")

    def test_the_friction_thesis_survives_being_generated(self) -> None:
        """The high-friction physical route's hurdle is an order above the CFD's.

        Which is the comparison the whole step exists to make, and it comes out of the
        data rather than out of a sentence somebody typed.
        """
        book = load_routes()
        physical = book.terms_for("physical_retail", "IND").friction.quote(21)
        cfd = book.terms_for("cfd", "GBR").friction.quote(21)

        assert physical.breakeven_pct > 10.0 * cfd.breakeven_pct


class TestTheReadmeTableIsGeneratedNotTyped:
    """§20: the published friction table must come from the routes data.

    A hand-written table drifts from its source the first time a rate changes, and the
    copy a reader sees is the one that is wrong. This test is what makes the README's
    copy a cache rather than a second source of truth.
    """

    MARKERS = ("<!-- BEGIN GENERATED breakeven-table -->", "<!-- END GENERATED breakeven-table -->")

    def _published(self) -> str:
        text = (REPO_ROOT / "README.md").read_text()
        start, end = self.MARKERS
        assert start in text and end in text, "the generated-table markers are missing"
        return text.split(start, 1)[1].split(end, 1)[0].strip()

    def test_the_published_table_matches_the_data(self) -> None:
        assert self._published() == breakeven_table(load_routes().table_rows("gold"))

    def test_the_cli_emits_exactly_what_the_readme_carries(self) -> None:
        result = CliRunner().invoke(app, ["routes", "--asset", "gold", "--markdown"])

        assert result.exit_code == 0
        assert result.stdout.strip() == self._published()

    def test_an_unknown_asset_fails_loudly_rather_than_printing_nothing(self) -> None:
        result = CliRunner().invoke(app, ["routes", "--asset", "plutonium", "--markdown"])

        assert result.exit_code == 1

    def test_the_plain_listing_states_that_nothing_is_the_default(self) -> None:
        result = CliRunner().invoke(app, ["routes"])

        assert result.exit_code == 0
        assert "No jurisdiction is the default" in result.stdout
        assert "never advises" in result.stdout
