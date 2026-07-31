"""CLI surface. Everything here runs offline."""

from __future__ import annotations

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
        monkeypatch.setattr("aurex.pipeline.PUBLIC_DATA_DIR", tmp_path / "public-data")
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
        monkeypatch.setattr("aurex.pipeline.PUBLIC_DATA_DIR", target)
        result = runner.invoke(app, ["pipeline", "--offline"])

        assert result.exit_code == 0, result.output
        written = json.loads((target / "latest.json").read_text())
        assert written["schema_version"] == 3

    def test_dry_run_output_contains_no_point_forecast_keys(self) -> None:
        """Structural guard: step 1 emits data and parity, never a predicted level."""
        result = runner.invoke(app, ["pipeline", "--dry-run"])
        payload = json.loads(result.stdout)
        forbidden = {"forecast_price", "predicted_price", "target", "point_forecast"}
        assert forbidden.isdisjoint(payload.keys())


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
