"""CLI surface. Everything here runs offline."""

from __future__ import annotations

import hashlib
import json

import pytest
from typer.testing import CliRunner

from aurex import __version__
from aurex.cli import app

runner = CliRunner()


class TestPipelineCommand:
    def test_dry_run_prints_an_artifact_and_writes_nothing(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("aurex.config.PUBLIC_DATA_DIR", tmp_path / "public-data")
        result = runner.invoke(app, ["pipeline", "--dry-run"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["mode"] == "offline"
        assert "policy_breaks" in payload
        assert not (tmp_path / "public-data").exists(), "--dry-run must not write"

    def test_offline_write_produces_latest_json(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "public-data"
        monkeypatch.setattr("aurex.config.PUBLIC_DATA_DIR", target)
        # The seed cache this runs from is committed and therefore always old; the
        # freshness guard is exercised on its own terms in test_freshness.py.
        result = runner.invoke(app, ["pipeline", "--offline", "--allow-stale"])

        assert result.exit_code == 0, result.output
        written = json.loads((target / "latest.json").read_text())
        assert written["schema_version"] == 4

    def test_dry_run_output_contains_no_point_forecast_keys(self) -> None:
        """Structural guard: step 1 emits data and parity, never a predicted level."""
        result = runner.invoke(app, ["pipeline", "--dry-run"])
        payload = json.loads(result.stdout)
        forbidden = {"forecast_price", "predicted_price", "target", "point_forecast"}
        assert forbidden.isdisjoint(payload.keys())


class TestTheNightlyWritesOnlyPublicData:
    """A job that edits prose can trip the §0 guard at 02:00 UTC with nobody watching.

    The workflow enforces this too, by only ever committing ``public-data/``. This is
    the same rule one layer down: if the command itself never writes anything else,
    the workflow's restriction cannot be defeated by a future change to the command.
    """

    def test_a_published_run_leaves_the_readme_untouched(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aurex.config import REPO_ROOT

        readme = REPO_ROOT / "README.md"
        before = hashlib.sha256(readme.read_bytes()).hexdigest()
        monkeypatch.setattr("aurex.config.PUBLIC_DATA_DIR", tmp_path / "public-data")

        runner.invoke(app, ["pipeline", "--offline", "--allow-stale"])

        assert hashlib.sha256(readme.read_bytes()).hexdigest() == before

    def test_a_refused_run_leaves_the_readme_untouched(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The failure path writes a skip record, and that is all it writes."""
        from aurex.config import REPO_ROOT

        readme = REPO_ROOT / "README.md"
        before = hashlib.sha256(readme.read_bytes()).hexdigest()
        monkeypatch.setattr("aurex.config.PUBLIC_DATA_DIR", tmp_path / "public-data")

        result = runner.invoke(app, ["pipeline", "--offline"])

        assert result.exit_code == 1
        assert hashlib.sha256(readme.read_bytes()).hexdigest() == before

    def test_everything_written_lands_under_public_data(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "public-data"
        monkeypatch.setattr("aurex.config.PUBLIC_DATA_DIR", target)

        runner.invoke(app, ["pipeline", "--offline", "--allow-stale"])

        written = sorted(p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file())
        assert written, "the run must have written something"
        assert all(name == "latest.json" or name.startswith("forecasts/") for name in written), (
            written
        )
        assert "forecasts/index.json" in written


class TestDutyCommand:
    def test_reports_the_rate_and_its_provenance(self) -> None:
        result = runner.invoke(app, ["duty", "2026-07-29"])
        assert result.exit_code == 0
        assert "15.00%" in result.output
        assert "secondary" in result.output
        assert "http" in result.output

    def test_reports_the_pre_2024_regime(self) -> None:
        result = runner.invoke(app, ["duty", "2025-01-15"])
        assert "6.00%" in result.output

    def test_pre_ad_valorem_says_so_rather_than_inventing_a_rate(self) -> None:
        result = runner.invoke(app, ["duty", "2010-01-01"])
        assert result.exit_code == 0
        assert "no rate defined" in result.output

    def test_pre_gst_flags_low_confidence(self) -> None:
        result = runner.invoke(app, ["duty", "2015-06-01"])
        assert "pre-GST" in result.output
        assert "confidence low" in result.output

    def test_rejects_a_malformed_date(self) -> None:
        assert runner.invoke(app, ["duty", "not-a-date"]).exit_code != 0


class TestScheduleCommand:
    def test_lists_every_entry_with_confidence_and_url(self) -> None:
        result = runner.invoke(app, ["schedule"])
        assert result.exit_code == 0

        lines = [ln for ln in result.output.splitlines() if ln.strip()]
        assert len(lines) >= 10
        for line in lines:
            assert "primary" in line or "secondary" in line
            assert "http" in line

    def test_includes_the_2026_change(self) -> None:
        result = runner.invoke(app, ["schedule"])
        assert "2026-05-13" in result.output
        assert "15.00%" in result.output


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_verbose_flag_is_accepted() -> None:
    assert runner.invoke(app, ["-v", "version"]).exit_code == 0
