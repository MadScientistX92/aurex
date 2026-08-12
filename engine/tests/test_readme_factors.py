"""Every figure the README's attribution section quotes must come from the artifact.

The same mechanism as ``test_readme_direction.py`` and for the same reason. That guard
exists because the direction section's strongest claims were once computed in a scratch
file, quoted in prose, and reproducible by nothing in the repository — the scratch file
was deleted and the numbers stayed. It has since caught a real disagreement between a
README figure and the run that produced it, which is why this section does not get to opt
out of it.

Deliberately literal. Expected strings are *formatted from the artifact* and then looked
for in the README, so a figure that drifts fails rather than passing on a rounding rule
nobody chose. Where a table displays fewer decimals than the artifact stores, the
comparison is against the artifact rounded to that displayed precision and the choice is
named at the assertion.

The direction that matters more is the second one: an artifact regenerated into
disagreeing with the README fails here too, because that is what a re-run with changed
code looks like.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, ClassVar

import pytest

REPO = Path(__file__).resolve().parents[2]
README = REPO / "README.md"
ARTIFACT = REPO / "public-data" / "factors.json"

#: The chain horizon the prose singles out — the one cell in the section that rejects.
FOCUS = 6

pytestmark = pytest.mark.skipif(
    not ARTIFACT.exists() or not README.exists(),
    reason="needs both the committed artifact and the README",
)


@pytest.fixture(scope="module")
def artifact() -> dict[str, Any]:
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text()


@pytest.fixture(scope="module")
def attribution(artifact: dict[str, Any]) -> dict[str, Any]:
    return artifact["attribution"]


@pytest.fixture(scope="module")
def chain(artifact: dict[str, Any]) -> dict[str, Any]:
    block = artifact["chain"]
    assert block is not None, "the README publishes a chain; the artifact has none"
    return block


def _tables(text: str) -> list[list[list[str]]]:
    """Every markdown table in the document, as rows of stripped cells."""
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [
                cell.strip().replace("*", "").replace("`", "")
                for cell in stripped.strip("|").split("|")
            ]
            if not all(set(cell) <= set("-: ") for cell in cells):
                current.append(cells)
            continue
        if current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return tables


def _table_with(text: str, *required: str) -> list[list[str]]:
    """The one table carrying all of ``required`` as headers.

    Matched on the whole header rather than the first cell: this README has several
    tables whose first column is "Horizon", and picking the first would check the chain's
    claims against the CRPS shootout's numbers.
    """
    found = [table for table in _tables(text) if table and set(required) <= set(table[0])]
    assert found, f"no table in the README carries the headers {required}"
    assert len(found) == 1, f"{len(found)} tables carry the headers {required}; expected one"
    return found[0]


def _quotes(readme: str, expected: str, what: str) -> None:
    assert expected in readme, (
        f"the README no longer quotes the artifact's {what}: expected to find "
        f"{expected!r}. Either the artifact was regenerated and the README was not "
        f"updated, or the README was edited away from the run that produced it."
    )


def _interval(values: list[float]) -> str:
    return f"[{values[0]}, {values[1]}]"


class TestTheLoadingsTable:
    """Six drivers, cell by cell, against the artifact."""

    def test_every_cell_is_the_artifacts(self, readme: str, attribution: dict[str, Any]) -> None:
        table = _table_with(readme, "Driver", "Loading", "OLS p", "R² without it")
        header, rows = table[0], table[1:]
        published = {row[0]: dict(zip(header, row, strict=True)) for row in rows}

        loadings = {entry["factor"]: entry for entry in attribution["loadings"]}
        stability = {entry["factor"]: entry for entry in attribution["stability"]["factors"]}
        withheld = {
            entry["withheld"]: entry for entry in attribution["omitted_variable_check"]["drivers"]
        }

        assert set(published) == set(loadings), (
            "the README table and the artifact disagree on which drivers were fitted"
        )

        for name, cells in published.items():
            loading = loadings[name]
            assert float(cells["Loading"]) == pytest.approx(loading["standardised"], abs=1e-9), name
            assert cells["95% interval (OLS)"] == _interval(loading["ols_interval_95"]), name
            assert float(cells["OLS p"]) == pytest.approx(
                round(loading["ols_p_value"], 4), abs=1e-9
            ), name
            # Displayed at three decimals, so compared against the artifact rounded to
            # three. Named here rather than hidden in a tolerance, because a silent
            # tolerance is how a real disagreement passes for a rounding difference.
            assert float(cells["Selected"]) == pytest.approx(
                round(loading["selection_rate"], 3), abs=1e-9
            ), name
            assert int(cells["Sign flips"]) == stability[name]["sign_changes"], name
            assert float(cells["R² without it"]) == pytest.approx(
                withheld[name]["r_squared_oos_without"], abs=1e-9
            ), name

    def test_the_table_is_ordered_by_what_it_claims_to_be(
        self, readme: str, attribution: dict[str, Any]
    ) -> None:
        """Largest loading first. A table sorted by nothing invites reading it as sorted."""
        table = _table_with(readme, "Driver", "Loading", "OLS p", "R² without it")
        sizes = [abs(float(row[1])) for row in table[1:]]
        assert sizes == sorted(sizes, reverse=True)

    def test_the_dropped_driver_is_named_with_the_sample_it_would_have_cost(
        self, readme: str, attribution: dict[str, Any]
    ) -> None:
        dropped = attribution["design"]["dropped"]
        assert dropped, "the artifact dropped nothing; the README says a driver was dropped"
        for name in dropped:
            _quotes(readme, name, "the dropped driver's name")
        _quotes(readme, str(attribution["design"]["observations"]), "the surviving sample")


class TestTheOutOfSampleNumbers:
    """The two the whole pre-registration turns on."""

    def test_both_r_squareds_are_the_artifacts(
        self, readme: str, attribution: dict[str, Any]
    ) -> None:
        for kind in ("contemporaneous", "predictive"):
            scored = attribution["out_of_sample"][kind]
            assert scored is not None, f"the README grades {kind}; the artifact has none"
            _quotes(readme, f"{scored['r_squared_oos']:.5f}", f"{kind} out-of-sample R2")
            _quotes(readme, f"{scored['dm_p_value']:.4f}", f"{kind} DM p-value")
            _quotes(readme, str(scored["observations"]), f"{kind} observation count")

    def test_the_in_sample_figure_is_the_artifacts(
        self, readme: str, attribution: dict[str, Any]
    ) -> None:
        _quotes(readme, f"{attribution['in_sample_r_squared']:.5f}", "in-sample R2")

    def test_the_two_that_hurt_out_of_sample_are_the_artifacts(
        self, readme: str, attribution: dict[str, Any]
    ) -> None:
        """The README names them; the artifact decides which they are."""
        scored = attribution["out_of_sample"]["contemporaneous"]["r_squared_oos"]
        withheld = attribution["omitted_variable_check"]["drivers"]
        harmful = [
            entry["withheld"]
            for entry in withheld
            if entry["r_squared_oos_without"] is not None
            and entry["r_squared_oos_without"] > scored
        ]
        assert harmful, "the README claims two drivers hurt; the artifact shows none"
        for name in harmful:
            _quotes(readme, name, "a driver the artifact shows is net negative")

    def test_the_sample_bounds_are_the_artifacts(
        self, readme: str, attribution: dict[str, Any]
    ) -> None:
        design = attribution["design"]
        _quotes(readme, design["first_week"], "first week")
        _quotes(readme, design["last_week"], "last week")


class TestTheClaimAboutTheIndexThatHadToBeWiredFirst:
    """The section's own thesis, checked against the number that decides it."""

    #: The driver whose omission the whole prerequisite was about.
    TRACKED = "d_geopolitical_risk"

    def test_its_withheld_shift_is_the_artifacts(
        self, readme: str, attribution: dict[str, Any]
    ) -> None:
        entry = next(
            x
            for x in attribution["omitted_variable_check"]["drivers"]
            if x["withheld"] == self.TRACKED
        )
        _quotes(readme, f"{entry['largest_loading_shift']:.8f}", "the withheld shift")
        _quotes(readme, f"{entry['r_squared_oos_without']:.5f}", "R2 without the index")

    def test_the_readme_does_not_claim_a_sign_flip_the_artifact_does_not_show(
        self, readme: str, attribution: dict[str, Any]
    ) -> None:
        """The claim the section would most like to be able to make, held to the data."""
        flips = [
            entry["withheld"]
            for entry in attribution["omitted_variable_check"]["drivers"]
            if entry["sign_flips"]
        ]
        if not flips:
            assert "did not materialise in the loadings" in readme, (
                "no driver flips sign when another is withheld, so the README must say "
                "the bias did not materialise rather than implying it was averted"
            )

    def test_the_loading_is_reported_as_indistinguishable_from_zero(
        self, attribution: dict[str, Any]
    ) -> None:
        entry = next(x for x in attribution["loadings"] if x["factor"] == self.TRACKED)
        assert not entry["ols_excludes_zero"], (
            "the README says this loading is not distinguishable from zero; the "
            "artifact now disagrees and the prose has to be rewritten, not the test"
        )
        assert entry["standardised"] > 0.0, "the README says the sign is positive"


class TestTheChainTable:
    """Five horizons of compounded and direct responses, cell by cell."""

    COLUMNS: ClassVar[dict[str, str]] = {
        "Direct": "raw",
        "Orthogonalised": "orthogonalised",
    }

    def test_every_cell_is_the_artifacts(self, readme: str, chain: dict[str, Any]) -> None:
        table = _table_with(readme, "Horizon (months)", "Compounded", "95% band")
        header, rows = table[0], table[1:]
        published = {int(row[0]): dict(zip(header, row, strict=True)) for row in rows}

        compounded = {entry["horizon_months"]: entry for entry in chain["compounded"]["horizons"]}
        assert set(published) == set(compounded), (
            "the README table and the artifact disagree on horizons"
        )

        for horizon, cells in published.items():
            entry = compounded[horizon]
            assert float(cells["Compounded"]) == pytest.approx(
                round(entry["compounded"], 6), abs=1e-9
            ), horizon
            low, high = entry["interval_95"]
            assert cells["95% band"] == f"[{low:+.6f}, {high:+.6f}]", horizon

            for column, key in self.COLUMNS.items():
                response = next(
                    r for r in chain["direct"][key]["responses"] if r["horizon_months"] == horizon
                )
                assert float(cells[column]) == pytest.approx(
                    round(response["coefficient"], 5), abs=1e-9
                ), f"{horizon} {column}"

    def test_the_band_spans_zero_everywhere_as_the_prose_says(
        self, readme: str, chain: dict[str, Any]
    ) -> None:
        assert chain["compounded"]["every_horizon_spans_zero"] is True, (
            "the README reports the pre-registered prediction as hit; the artifact "
            "no longer supports it"
        )
        assert "spans zero at every horizon** — pre-registered, and **hit**" in readme

    def test_the_one_cell_that_rejects_is_the_one_the_prose_names(
        self, readme: str, chain: dict[str, Any]
    ) -> None:
        rejecting = [
            (key, response["horizon_months"])
            for key in ("raw", "orthogonalised")
            for response in chain["direct"][key]["responses"]
            if not response["spans_zero"]
        ]
        assert rejecting, "the README says one cell excludes zero; the artifact has none"
        assert {horizon for _, horizon in rejecting} == {FOCUS}, (
            f"the README singles out {FOCUS} months; the artifact rejects elsewhere too"
        )

        focus = next(r for r in chain["direct"]["raw"]["responses"] if r["horizon_months"] == FOCUS)
        _quotes(readme, _interval(focus["interval_95"]), "the rejecting band")

    def test_the_middle_link_is_the_one_the_prose_blames(self, chain: dict[str, Any]) -> None:
        """The README says the middle link is why the product spans zero everywhere."""
        middle = chain["links"][1]
        assert middle["every_horizon_spans_zero"] is True
        impact = middle["responses"][0]["coefficient"]
        one_month = middle["responses"][1]["coefficient"]
        assert impact * one_month < 0.0, (
            "the README says the middle link changes sign between impact and one month"
        )


class TestTheAdministeredControl:
    def test_the_coverage_and_the_moves_are_the_artifacts(
        self, readme: str, chain: dict[str, Any]
    ) -> None:
        control = chain["administered_control"]
        _quotes(readme, str(control["months_covered"]), "months the control covers")
        for month in control["moves_in_sample"]:
            _quotes(readme, month[:7], "a month the administered rate moved")

    def test_the_gap_is_declared_rather_than_filled(self, artifact: dict[str, Any]) -> None:
        control = artifact["asset"]["transmission_chain"]["controls"][0]
        assert control["gaps"], "the README says the schedule has a declared hole"
        for gap in control["gaps"]:
            assert gap["reason"] and gap["source_url"].startswith("http")
            assert gap["source_confidence"] in {"primary", "secondary"}

    def test_every_cited_entry_carries_its_own_provenance(self, artifact: dict[str, Any]) -> None:
        control = artifact["asset"]["transmission_chain"]["controls"][0]
        for entry in control["entries"]:
            assert entry["source_url"].startswith("http")
            assert entry["source_confidence"] in {"primary", "secondary"}

    def test_the_mechanical_check_figures_are_the_artifacts(
        self, readme: str, chain: dict[str, Any]
    ) -> None:
        check = chain["mechanical_link"]
        assert check["available"] is True
        for value in check["measured"].values():
            _quotes(readme, f"{value}", "a measured elasticity of the terminal identity")
        _quotes(readme, f"{check['r_squared']}", "the identity check's R2")


class TestTheArtifactCarriesWhatTheSectionRestsOn:
    """Absence checks: fields the README's claims cannot be read without."""

    def test_the_reproducing_command_bounds_the_sample(self, artifact: dict[str, Any]) -> None:
        """An artifact whose own command would not regenerate it is not a reproducer."""
        assert "--to " in artifact["reproduce"]
        assert artifact["attribution"]["design"]["last_week"] in artifact["reproduce"]

    def test_every_input_series_carries_a_citation_and_a_confidence(
        self, artifact: dict[str, Any]
    ) -> None:
        assert artifact["sources"], "an artifact with no sources cites nothing"
        for series_id, meta in artifact["sources"].items():
            citation = meta["citation"]
            assert citation["source_url"].startswith("http"), series_id
            assert citation["source_confidence"] in {"primary", "secondary"}, series_id

    def test_the_index_the_step_waited_on_is_a_required_driver(
        self, artifact: dict[str, Any]
    ) -> None:
        factors = {entry["id"]: entry for entry in artifact["asset"]["factors"]}
        tracked = factors["d_geopolitical_risk"]
        assert tracked["required"] is True
        assert tracked["transform"] == "diff", (
            "the README pre-registers that it enters as a change, not a level"
        )

    def test_nothing_in_the_artifact_is_named_like_a_total(self, artifact: dict[str, Any]) -> None:
        """The no-double-counting rule, enforced on the published shape."""

        def keys(node: object) -> list[str]:
            if isinstance(node, dict):
                return [k for key, value in node.items() for k in (key, *keys(value))]
            if isinstance(node, list):
                return [k for item in node for k in keys(item)]
            return []

        offenders = [
            key
            for key in keys(artifact)
            if any(word in key for word in ("total", "combined_effect", "net_effect"))
        ]
        assert not offenders, f"a field named like a total invites the sum: {offenders}"

    def test_the_out_of_sample_block_keeps_both_forms(self, artifact: dict[str, Any]) -> None:
        """Collapsing the two would make the pre-registration ungradeable after the fact."""
        block = artifact["attribution"]["out_of_sample"]
        assert set(block) >= {"contemporaneous", "predictive"}
        assert block["contemporaneous"]["kind"] == "contemporaneous"
        assert block["predictive"]["kind"] == "predictive"


def test_the_readme_grades_every_prediction_it_registered(readme: str) -> None:
    """Four pre-registered claims, and each must be marked hit or missed in the prose."""
    section = readme.split("### The loadings, against that prediction")[1]
    verdicts = re.findall(r"\*\*(hit|missed[^.*]*)\.?\*\*", section, re.IGNORECASE)
    assert len(verdicts) >= 4, (
        f"only {len(verdicts)} graded verdicts found after the pre-registration; every "
        f"registered prediction has to be marked hit or missed, including the ones that "
        f"missed: {verdicts}"
    )
    assert any(v.lower().startswith("missed") for v in verdicts), (
        "a pre-registration where everything hit is a pre-registration written afterwards"
    )
