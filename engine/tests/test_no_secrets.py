"""No API key is required to run Aurex, and it must stay that way.

This is a stated feature in the README, which means it is a claim, which means it needs
a guard. It will not break loudly: it breaks the day someone adds a source that reads a
key and works fine on their machine because their machine has one. Every run after that
degrades silently for everybody else, and the artifact records a fallback rather than
an error.

Two layers. Here, the *code-level* guard: nothing may read an environment variable
without declaring it, and nothing may require one. In CI, the same suite runs again in
a job with no secrets configured at all. In production, the nightly job is the
*source-level* guard — it runs keyless against live endpoints, so a source that starts
demanding authentication fails a real run rather than going unnoticed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from aurex.config import ENGINE_ROOT, fred_api_key
from aurex.data.sources.fred import CSV_ENDPOINT, FredLoader

#: Every environment variable the engine is permitted to read, and why.
#:
#: Adding a name here is the deliberate act this test exists to force. A new entry that
#: is a credential should not be added at all: it would move Aurex from "no API key is
#: required" to "no API key is required for the parts you happen to exercise".
DECLARED_ENV = {
    "AUREX_CACHE_DIR": "cache location; a path, not a credential",
    "FRED_API_KEY": "optional and unused by the loader — see the test below",
    "GITHUB_SHA": "CI-supplied commit id for artifact provenance",
    "GITHUB_REF_NAME": "CI-supplied branch or tag name",
    "GIT_COMMIT": "CI-supplied commit id, alternate spelling",
    "CI_COMMIT_SHA": "CI-supplied commit id, alternate spelling",
    "CI_COMMIT_REF_NAME": "CI-supplied ref name, alternate spelling",
}

#: Substrings that mark a variable as a credential rather than configuration.
SECRET_SHAPED = ("key", "token", "secret", "password", "credential")

ENV_READ = re.compile(
    r"""os\.environ\.get\(\s*["']([A-Z0-9_]+)["']|"""
    r"""os\.environ\[\s*["']([A-Z0-9_]+)["']|"""
    r"""os\.getenv\(\s*["']([A-Z0-9_]+)["']""",
)


def env_reads() -> dict[str, list[str]]:
    """Every environment variable the package reads, and where."""
    found: dict[str, list[str]] = {}
    for path in sorted(ENGINE_ROOT.rglob("*.py")):
        if any(part in {".venv", "__pycache__", "tests"} for part in path.parts):
            continue
        for i, line in enumerate(path.read_text().splitlines(), start=1):
            for match in ENV_READ.finditer(line):
                name = next(group for group in match.groups() if group)
                found.setdefault(name, []).append(f"{path.relative_to(ENGINE_ROOT)}:{i}")
    return found


class TestNothingReadsAnUndeclaredVariable:
    def test_every_environment_read_is_declared(self) -> None:
        undeclared = {
            name: sites for name, sites in env_reads().items() if name not in DECLARED_ENV
        }

        assert not undeclared, (
            f"undeclared environment reads: {undeclared}. Add the name to DECLARED_ENV "
            f"with a reason — and if it is a credential, do not add it at all."
        )

    def test_the_guard_actually_finds_the_reads_it_should(self) -> None:
        """A scanner that matches nothing passes vacuously and protects nothing."""
        found = env_reads()

        assert "AUREX_CACHE_DIR" in found
        assert "FRED_API_KEY" in found

    def test_no_declared_variable_is_a_required_credential(self) -> None:
        """A secret-shaped name may be read, but only as an optional enhancement.

        ``FRED_API_KEY`` is the one that exists and the docstring beside it in
        ``config.py`` records why it is optional. Any future entry has to survive the
        same reading.
        """
        secret_shaped = [
            name for name in DECLARED_ENV if any(marker in name.lower() for marker in SECRET_SHAPED)
        ]

        assert secret_shaped == ["FRED_API_KEY"], (
            f"credential-shaped variables in the declared set: {secret_shaped}. "
            f"Aurex claims no API key is required; each of these has to be optional."
        )


class TestTheKeylessPath:
    def test_no_key_configured_reads_as_no_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        assert fred_api_key() is None

    def test_an_empty_key_is_not_a_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A blank secret in CI is the normal shape of an unset one."""
        monkeypatch.setenv("FRED_API_KEY", "   ")
        assert fred_api_key() is None

    def test_the_fred_url_carries_no_credential(self) -> None:
        """The strongest form of the claim: the request that goes out has no key in it.

        ``fredgraph.csv`` serves the same observations as the authenticated API, so the
        loader never had a reason to hold a credential. Asserting on the URL the loader
        actually builds catches the realistic regression — someone appending a key when
        one happens to be set — which reading the module text would not, because the
        module text legitimately discusses the variable in its docstring.
        """
        url = FredLoader("wti", "DCOILWTICO", "wti").url

        assert url == f"{CSV_ENDPOINT}?id=DCOILWTICO"
        assert "api_key" not in url.lower()

    def test_the_url_is_unchanged_when_a_key_is_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A key in the environment must not change what gets requested.

        This is the guard that fails if someone wires the optional key into the loader:
        it would pass on a machine with no key and break the zero-key property for
        everyone who has one, which is the wrong way round and easy to miss.
        """
        keyless = FredLoader("wti", "DCOILWTICO", "wti").url
        monkeypatch.setenv("FRED_API_KEY", "0123456789abcdef0123456789abcdef")

        assert FredLoader("wti", "DCOILWTICO", "wti").url == keyless

    def test_the_offline_pipeline_runs_with_every_credential_stripped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """End to end with a scrubbed environment — the CI no-secrets job in miniature."""
        from aurex.pipeline import run

        for name in DECLARED_ENV:
            if any(marker in name.lower() for marker in SECRET_SHAPED):
                monkeypatch.delenv(name, raising=False)

        result = run(offline=True)

        assert result.artifact["schema_version"] == 4
        assert result.freshness is not None
