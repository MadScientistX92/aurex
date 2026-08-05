"""Which code produced an artifact, and the honesty of saying when we don't know."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from aurex.provenance import code_provenance


class TestIdentifyingTheCode:
    def test_ci_environment_is_preferred_over_shelling_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CI checks out the commit it reports, and does it in a detached HEAD where
        the branch name is empty but the SHA is exactly right."""
        monkeypatch.setenv("GITHUB_SHA", "1234567890abcdef")
        monkeypatch.setenv("GITHUB_REF_NAME", "main")

        provenance = code_provenance()

        assert provenance.commit == "1234567890abcdef"
        assert provenance.ref == "main"
        assert provenance.dirty is False
        assert provenance.source == "environment:GITHUB_SHA"

    def test_a_local_checkout_reports_its_head(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in ("GITHUB_SHA", "GIT_COMMIT", "CI_COMMIT_SHA"):
            monkeypatch.delenv(name, raising=False)

        provenance = code_provenance()

        assert provenance.source == "git"
        assert provenance.commit is not None
        assert len(provenance.commit) == 40

    def test_outside_a_checkout_it_degrades_rather_than_raising(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run that succeeded must not be failed by an inability to describe itself."""
        for name in ("GITHUB_SHA", "GIT_COMMIT", "CI_COMMIT_SHA"):
            monkeypatch.delenv(name, raising=False)

        provenance = code_provenance(root=tmp_path)

        assert provenance.commit is None
        assert provenance.dirty is None
        assert "not a git checkout" in provenance.source

    def test_a_missing_git_binary_degrades_rather_than_raising(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in ("GITHUB_SHA", "GIT_COMMIT", "CI_COMMIT_SHA"):
            monkeypatch.delenv(name, raising=False)

        def explode(*args: object, **kwargs: object) -> None:
            raise OSError("git: command not found")

        monkeypatch.setattr(subprocess, "run", explode)

        assert code_provenance().commit is None


class TestWhatItPublishes:
    def test_the_block_says_the_version_identifies_nothing(self) -> None:
        """engine_version is a static 0.1.0 carried by every forecast ever published,
        including the withdrawn ones. Publishing it beside the SHA without saying so
        would invite a reader to treat it as a build identifier."""
        block = code_provenance().describe()

        assert block["engine_version"] == "0.1.0"
        assert "identifies nothing" in block["note"]

    def test_a_dirty_tree_is_published_as_dirty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A SHA that does not describe the working tree is worse than no SHA, so the
        artifact says which it is rather than implying reproducibility it lacks."""
        for name in ("GITHUB_SHA", "GIT_COMMIT", "CI_COMMIT_SHA"):
            monkeypatch.delenv(name, raising=False)
        real = subprocess.run

        def fake(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            if args[1] == "status":
                return subprocess.CompletedProcess(args, 0, " M engine/aurex/cli.py\n", "")
            return real(args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(subprocess, "run", fake)

        assert code_provenance().dirty is True
