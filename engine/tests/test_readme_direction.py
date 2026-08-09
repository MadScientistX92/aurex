"""Every figure the README's direction section quotes must come from the artifact.

The rest of this repository ties a published number to the run that produced it: the
artifact carries its own reproducing command, ``--to`` bounds the sample so the command
is not a lie, and the workflow refuses an artifact whose sample is unbounded. The
direction section broke that chain for a while. Its strongest claims — the calendar
placebo, the two trend-preserving nulls, leave-one-year-out — were computed in a scratch
file, quoted in prose, and reproducible by nothing in the repository. The scratch file
was deleted. The numbers stayed.

So this is the missing link and not a style check. It parses the two tables in the
direction section and every loose figure in its prose, and asserts each against
``public-data/direction.json``. A README edited away from the artifact fails here; so
does an artifact regenerated into disagreeing with the README, which is the direction
that matters more, because that is what a re-run with changed code looks like.

Deliberately literal. There is no tolerance and no fuzzy matching: the expected string
is *formatted from the artifact* and then looked for in the README, so a figure that
drifts in the fourth decimal fails rather than passing on a rounding rule nobody chose.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, ClassVar

import pytest

REPO = Path(__file__).resolve().parents[2]
README = REPO / "README.md"
ARTIFACT = REPO / "public-data" / "direction.json"

#: The horizon the controls are reported at. The one cell in the table that rejects, and
#: therefore the only one whose supporting figures the README spells out.
FOCUS = 10

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
def horizon(artifact: dict[str, Any]) -> dict[str, Any]:
    for entry in artifact["horizons"]:
        if entry["horizon_sessions"] == FOCUS:
            return entry
    raise AssertionError(f"the artifact has no {FOCUS}-session horizon")


def _tables(text: str) -> list[list[list[str]]]:
    """Every markdown table in the document, as rows of stripped cells."""
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [cell.strip().replace("*", "") for cell in stripped.strip("|").split("|")]
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
    tables whose first column is "Horizon", and picking the first would silently check
    the direction claims against the CRPS shootout's numbers.
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


class TestTheScreenTable:
    """The five-horizon table of p-values, cell by cell against the artifact."""

    SCREENS: ClassVar[dict[str, str]] = {
        "Equal-width, full": "full_sample",
        "Equal-width, thinned": "non_overlapping_subsample",
        "Equal-count, full": "equal_count_bins_robustness",
        "Equal-count, thinned": "equal_count_non_overlapping_subsample",
    }

    def test_every_p_value_in_it_is_the_artifacts(
        self, readme: str, artifact: dict[str, Any]
    ) -> None:
        table = _table_with(readme, "Horizon", "Windows", "Equal-count, full")
        header, rows = table[0], table[1:]
        by_horizon = {entry["horizon_sessions"]: entry for entry in artifact["horizons"]}
        assert len(rows) == len(by_horizon), "the README table and the artifact disagree on rows"

        for row in rows:
            cells = dict(zip(header, row, strict=True))
            entry = by_horizon[int(cells["Horizon"])]
            assert int(cells["Windows"]) == entry["observations"]
            assert int(cells["Independent"]) == entry["independent_observations"]
            for column, key in self.SCREENS.items():
                screen = entry["discrimination"][key]
                assert screen is not None, f"{key} missing at horizon {cells['Horizon']}"
                assert float(cells[column]) == pytest.approx(
                    round(screen["p_value"], 3), abs=1e-9
                ), f"horizon {cells['Horizon']}, {column}"


class TestTheControlsTheProseQuotes:
    """The placebo, the trend-preserving nulls, and the studentised figures."""

    def test_the_calendar_placebo_figures_are_the_artifacts(
        self, readme: str, horizon: dict[str, Any]
    ) -> None:
        controls = horizon["discrimination"]["controls"]
        for key, label in (
            ("calendar_placebo", "full sample"),
            ("calendar_placebo_non_overlapping_subsample", "thinned"),
        ):
            placebo = controls[key]
            assert placebo is not None, f"the {label} placebo is missing from the artifact"
            _quotes(readme, f"{placebo['resolution']:.5f}", f"{label} placebo resolution")
            _quotes(readme, f"{placebo['resolution_under_null']:.5f}", f"{label} placebo null")
        _quotes(readme, f"{controls['calendar_placebo']['p_value']:.3f}", "placebo marginal p")
        assert controls["calendar_placebo"]["rank"] == controls["calendar_placebo"]["entrants"], (
            "the README says the placebo ranks last of seven; the artifact disagrees"
        )

    def test_adding_the_placebo_does_not_move_the_published_p(
        self, readme: str, horizon: dict[str, Any]
    ) -> None:
        controls = horizon["discrimination"]["controls"]
        assert controls["set_p_value_without_placebo"] == pytest.approx(
            horizon["discrimination"]["equal_count_bins_robustness"]["p_value"]
        )
        _quotes(readme, f"{controls['calendar_placebo']['set_p_value_with_placebo']:.3f}", "set p")

    def test_both_trend_preserving_nulls_are_the_artifacts(
        self, readme: str, horizon: dict[str, Any]
    ) -> None:
        controls = horizon["discrimination"]["controls"]
        for key, label in (
            ("trend_preserving_nulls", "full sample"),
            ("trend_preserving_nulls_non_overlapping_subsample", "thinned"),
        ):
            for name in ("within_group_permutation", "within_group_cyclic_shift"):
                screen = controls[key][name]
                assert screen is not None, f"{label} {name} missing from the artifact"
                assert screen["resampling"] == name
        # The README quotes three of the four: both full-sample p-values and the thinned
        # within-year permutation. The thinned shift is emitted and not quoted, which is
        # the allowed direction — the artifact may carry more than the prose does.
        full = controls["trend_preserving_nulls"]
        thinned = controls["trend_preserving_nulls_non_overlapping_subsample"]
        for screen, what in (
            (full["within_group_permutation"], "within-year permutation, full"),
            (full["within_group_cyclic_shift"], "within-year cyclic shift, full"),
            (thinned["within_group_permutation"], "within-year permutation, thinned"),
        ):
            _quotes(readme, f"{screen['p_value']:.4f}", f"{what} set p")

    def test_har_rvs_marginals_under_the_grouped_nulls_are_the_artifacts(
        self, readme: str, horizon: dict[str, Any]
    ) -> None:
        """The two figures the prose leans on hardest, since they are the mechanism."""
        full = horizon["discrimination"]["controls"]["trend_preserving_nulls"]
        tracked = horizon["discrimination"]["controls"]["leave_one_year_out"]["model"]
        for name in ("within_group_permutation", "within_group_cyclic_shift"):
            row = full[name]["models"][tracked]
            _quotes(readme, f"{row['p_value']:.4f}", f"{tracked} marginal p under {name}")
        # The prose quotes one null mean, not two: the point it makes is that holding the
        # year structure fixed leaves this model's null where the shift put it, and one
        # of the pair carries that. The other is in the artifact and unquoted, which is
        # the direction this guard permits.
        permutation = full["within_group_permutation"]["models"][tracked]
        _quotes(readme, f"{permutation['resolution_under_null']:.5f}", f"{tracked} grouped null")


class TestTheLeaveOneYearOutTable:
    """Twelve refits, published as a table, checked row by row."""

    #: The table names models the way the prose does; the artifact names them the way the
    #: registry does. The mapping is here rather than in the README because a reader
    #: should not have to learn model ids to read a table.
    DISPLAY: ClassVar[dict[str, str]] = {
        "HAR-RV": "har_rv",
        "NHITS": "nhits",
        "GJR-GARCH": "gjr_garch",
        "AutoARIMA": "auto_arima",
        "walk": "random_walk_drift_matched",
    }

    def test_every_row_is_the_artifacts(self, readme: str, horizon: dict[str, Any]) -> None:
        withheld = horizon["discrimination"]["controls"]["leave_one_year_out"]
        assert withheld is not None, "the README publishes the table; the artifact has none"

        table = _table_with(readme, "Dropped", "Marginal p", "Set p")
        header, rows = table[0], table[1:]
        published = {row[0]: dict(zip(header, row, strict=True)) for row in rows}

        complete = withheld["complete_sample"]
        assert "—" in published, "the README table has no whole-sample row to read against"
        assert int(published["—"]["n"]) == complete["observations"]

        for year, row in withheld["without"].items():
            assert year in published, f"the artifact refits without {year}; the README omits it"
            cells = published[year]
            assert row is not None, f"the README publishes a row for {year}; the refit is None"
            assert int(cells["n"]) == row["observations"], year
            assert float(cells["Marginal p"]) == pytest.approx(row["p_value"], abs=5e-5), year
            # Compared at the artifact's own precision. Rounding the artifact down to the
            # README's display precision would let a real disagreement pass whenever the
            # two happened to round together — which is exactly what this caught on its
            # first run, where the README quoted 0.010 against an artifact 0.0095.
            assert float(cells["Set p"]) == pytest.approx(row["set_p_value"], abs=1e-9), year
            assert float(cells["That year's up-rate"]) == pytest.approx(
                round(row["withheld_group_rate"], 3), abs=1e-9
            ), year
            assert float(cells["HAR-RV resolution"]) == pytest.approx(
                row["resolution"], abs=6e-6
            ), year
            assert float(cells["Its null"]) == pytest.approx(
                row["resolution_under_null"], abs=6e-6
            ), year
            assert self.DISPLAY[cells["Best"]] == row["set_best_model"], year

    #: The README writes these counts as words, so the guard has to read words.
    NUMERALS: ClassVar[dict[int, str]] = {
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
        11: "eleven",
        12: "twelve",
    }

    def test_the_counts_the_prose_states_are_the_artifacts(
        self, readme: str, horizon: dict[str, Any]
    ) -> None:
        """The sentence "six of twelve refits still reject" is a claim about the run."""
        withheld = horizon["discrimination"]["controls"]["leave_one_year_out"]
        sentence = re.search(
            r"(\w+) of (\w+) refits still reject marginally, (\w+) of (\w+) at set level",
            readme,
        )
        assert sentence is not None, "the README no longer states the leave-one-out counts"
        marginal, refits, at_set, refits_again = (word.lower() for word in sentence.groups())
        assert marginal == self.NUMERALS[withheld["refits_rejecting_marginally"]]
        assert at_set == self.NUMERALS[withheld["refits_rejecting_at_set_level"]]
        assert refits == refits_again == self.NUMERALS[withheld["refits"]]


class TestTheArtifactCarriesWhatTheSectionRestsOn:
    """Absence checks: the fields the README's claims cannot be read without."""

    def test_the_controls_ran_at_every_horizon(self, artifact: dict[str, Any]) -> None:
        for entry in artifact["horizons"]:
            controls = entry["discrimination"]["controls"]
            assert controls is not None, (
                f"horizon {entry['horizon_sessions']} has no controls: this artifact was "
                f"produced with --no-controls and cannot support the README"
            )
            assert controls["grouping"] == "calendar_year"

    def test_the_reproducing_command_runs_the_controls(self, artifact: dict[str, Any]) -> None:
        """An artifact whose own command would not regenerate it is not a reproducer."""
        assert "--no-controls" not in artifact["reproduce"]
        assert "--controls" in artifact["reproduce"]

    def test_every_screen_names_the_null_it_used(self, artifact: dict[str, Any]) -> None:
        """Which null is load-bearing now, so no screen may leave it to be inferred."""
        for entry in artifact["horizons"]:
            discrimination = entry["discrimination"]
            for key in (
                "full_sample",
                "non_overlapping_subsample",
                "equal_count_bins_robustness",
                "equal_count_non_overlapping_subsample",
            ):
                assert discrimination[key]["resampling"] in {"cyclic_shift", "permutation"}


def test_the_readme_quotes_no_stale_key_name(readme: str) -> None:
    """The rename has to reach the prose, not only the emitting code."""
    assert "`distinguishable_from_zero_equal_count` is" not in readme
    assert "both_screens_reject_equal_count" in readme
