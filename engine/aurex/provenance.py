"""Which code produced this artifact.

``engine_version`` is a static ``0.1.0`` written into ``__init__.py`` and bumped by
hand approximately never. It identifies nothing: every forecast this repository has
ever published carries it, including the ones produced by the drift-carrying
simulation that was later withdrawn. A reader who wants to know what produced a given
number cannot get there from a version string that never changes.

The commit SHA can. Paired with a lockfile CI installs with ``--frozen``, it names the
exact code and the exact dependency set behind a published distribution — which is
what §0's reproducibility claim actually requires, and what a recorded seed does not
provide on its own. A seed reproduces nothing if SciPy's optimiser moved underneath it.

Nothing here may raise. Provenance is a description of a run, and a run that succeeded
must not be failed by an inability to describe itself; every field degrades to ``None``
with the reason recorded instead.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aurex import __version__
from aurex.config import REPO_ROOT

#: CI sets these; trusting them avoids shelling out and is correct in a detached-HEAD
#: checkout, where `git rev-parse HEAD` is right but `git branch --show-current` is empty.
_SHA_ENV = ("GITHUB_SHA", "GIT_COMMIT", "CI_COMMIT_SHA")
_REF_ENV = ("GITHUB_REF_NAME", "CI_COMMIT_REF_NAME")


def _git(*args: str, cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


@dataclass(frozen=True, slots=True)
class CodeProvenance:
    """The commit a run came from, and whether the tree matched it."""

    engine_version: str
    commit: str | None
    ref: str | None
    #: True when tracked files differed from the commit. A dirty artifact is still
    #: published — refusing would make local development impossible — but it says so,
    #: because a SHA that does not describe the working tree is worse than no SHA.
    dirty: bool | None
    source: str

    def describe(self) -> dict[str, Any]:
        return {
            "engine_version": self.engine_version,
            "commit": self.commit,
            "ref": self.ref,
            "dirty": self.dirty,
            "source": self.source,
            "note": (
                "engine_version is static and identifies nothing on its own; commit is "
                "what names the code behind this artifact. dirty=true means tracked "
                "files differed from that commit when the run started, so the SHA "
                "locates the run but does not fully reproduce it. dirty=null means the "
                "run could not tell."
            ),
        }


def code_provenance(root: Path | None = None) -> CodeProvenance:
    """Identify the code behind this run, preferring CI's own record of it."""
    cwd = root or REPO_ROOT

    for name in _SHA_ENV:
        sha = os.environ.get(name, "").strip()
        if sha:
            ref = next((os.environ[r] for r in _REF_ENV if os.environ.get(r, "").strip()), None)
            # CI checks out the commit it reports, so there is nothing to be dirty
            # against; claiming otherwise would need a diff we have no reason to run.
            return CodeProvenance(
                engine_version=__version__,
                commit=sha,
                ref=ref,
                dirty=False,
                source=f"environment:{name}",
            )

    commit = _git("rev-parse", "HEAD", cwd=cwd)
    if commit is None:
        return CodeProvenance(
            engine_version=__version__,
            commit=None,
            ref=None,
            dirty=None,
            source="unavailable: not a git checkout, or git is not installed",
        )

    status = _git("status", "--porcelain", "--untracked-files=no", cwd=cwd)
    return CodeProvenance(
        engine_version=__version__,
        commit=commit,
        ref=_git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd),
        dirty=None if status is None else bool(status),
        source="git",
    )
