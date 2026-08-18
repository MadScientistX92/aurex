"""An artifact's own ``reproduce`` string must reproduce the sample it published.

Nothing else in this suite asks that question. ``tests/test_readme_direction.py`` and
``tests/test_readme_factors.py`` compare the README against the *committed* JSON, cell
by cell, and never regenerate anything — so a change that moves what the published
command resolves leaves both of them green while every published figure moves. They are
blind to it by construction: they check that the prose matches the artifact, not that
the artifact matches its own instructions.

The failure they could not see was real. ``SourceChain.load`` returned the cache's whole
union and let ``start`` fall on the floor, so the sample was a property of the machine's
cache history rather than of the command: one published ``aurex score`` command returned
5,009 observations from 2006-08-04 on the Mac that published it, 6,662 from 2000-01-04 on
the same Mac after its cache was extended backwards, and 5,408 from 2005-01-04 on a
cacheless runner — which is the only one that got the window the code asked for.

The asset here is the synthetic one, whose loader generates in process, so the test needs
no network and no fixture data. The mechanism under test is the data layer's, not gold's.
"""

from __future__ import annotations

import json
import shlex
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from aurex.assets import REGISTRY
from aurex.assets.synthetic import SYNTHETIC, GeneratedLoader
from aurex.cli import app
from aurex.data.cache import CacheStore

runner = CliRunner()

#: Two years of forecasts is enough to produce a sample block and short enough that the
#: whole test is a few seconds. The lookback the command applies behind ``--from`` is
#: what the clip actually bites on, and that is fixed by the CLI, not by this number.
FIRST_FORECAST = (date.today() - timedelta(days=730)).isoformat()


def _run(argv: list[str], *, expect_ok: bool = True) -> dict[str, Any]:
    result = runner.invoke(app, argv)
    if expect_ok:
        assert result.exit_code == 0, result.output
    return json.loads(result.stdout)  # type: ignore[no-any-return]


def _reproducing_argv(artifact: dict[str, Any]) -> list[str]:
    """Turn the artifact's own published command into arguments for this CLI.

    Parsed rather than reconstructed. A test that rebuilt the argument list from the
    same values the command was invoked with would pass while the ``reproduce`` string
    said something else entirely, which is the thing being checked.
    """
    command = artifact["reproduce"]
    assert command.startswith("uv run aurex "), command
    return [*shlex.split(command)[3:], "--dry-run"]


def _deepen_the_cache(root: Path, *, back_to: date) -> None:
    """Give the cache more history than the published command asked for.

    This is what a nightly with a longer lookback does, what another contributor's
    machine looks like, and what happened to the machine that published
    ``calibration-gold.json``. It must not change what the command resolves.
    """
    loader = GeneratedLoader("widget_price", 100.0, seed=1)
    CacheStore(root).merge_write(loader.fetch(back_to, datetime.now(UTC).date()))


class TestScoreReproducesItsOwnSample:
    @pytest.fixture(autouse=True)
    def _isolated_cache(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        root = tmp_path / "cache"
        root.mkdir()
        monkeypatch.setattr("aurex.config.CACHE_DIR", root)
        # The synthetic asset is deliberately absent from the production registry; the
        # CLI resolves assets through it, so it is added for the duration of the test
        # rather than registered for real.
        monkeypatch.setitem(REGISTRY, SYNTHETIC.id, SYNTHETIC)
        return root

    def test_the_published_command_resolves_the_published_sample(
        self, _isolated_cache: Path
    ) -> None:
        """Same command, deeper cache, same sample — bounds and count alike."""
        published = _run(
            [
                "score",
                "--asset",
                SYNTHETIC.id,
                "--from",
                FIRST_FORECAST,
                "--step",
                "5",
                "--horizons",
                "5",
                "--paths",
                "200",
                "--dry-run",
            ]
        )

        _deepen_the_cache(_isolated_cache, back_to=date(1995, 1, 2))
        rerun = _run(_reproducing_argv(published))

        assert rerun["sample"]["resolved"] == published["sample"]["resolved"], (
            "the published command resolved a different window on a deeper cache"
        )
        assert rerun["sample"]["observations"] == published["sample"]["observations"], (
            "same bounds, different observation count: the window is not what was clipped"
        )

    def test_the_numbers_move_with_the_sample_and_are_checked_too(
        self, _isolated_cache: Path
    ) -> None:
        """Bounds are the diagnosis; the graded figures are the claim.

        Every expanding-window fit sees the resolved history, so a sample that moves
        moves the published numbers with it — that is how a headline CRPS skill halved
        on a rerun of its own command. Asserting the bounds alone would pass a
        hypothetical clip that got the bounds right and the contents wrong.
        """
        published = _run(
            [
                "score",
                "--asset",
                SYNTHETIC.id,
                "--from",
                FIRST_FORECAST,
                "--step",
                "5",
                "--horizons",
                "5",
                "--paths",
                "200",
                "--dry-run",
            ]
        )

        _deepen_the_cache(_isolated_cache, back_to=date(1995, 1, 2))
        rerun = _run(_reproducing_argv(published))

        assert rerun["calibration"] == published["calibration"]

    def test_the_deeper_cache_would_otherwise_have_moved_it(self, _isolated_cache: Path) -> None:
        """The control: without the clip there is a different sample to be had.

        A test whose fixture cannot produce the failure it guards against passes
        forever and guards nothing. This asserts the deepened cache really does hold
        history outside the requested window — so if the clip is removed, the two
        assertions above have something to catch.
        """
        published = _run(
            [
                "score",
                "--asset",
                SYNTHETIC.id,
                "--from",
                FIRST_FORECAST,
                "--step",
                "5",
                "--horizons",
                "5",
                "--paths",
                "200",
                "--dry-run",
            ]
        )
        _deepen_the_cache(_isolated_cache, back_to=date(1995, 1, 2))

        cached = CacheStore(_isolated_cache).read("widget_price")
        assert cached is not None
        resolved_from = date.fromisoformat(published["sample"]["resolved"]["from"])
        assert cached.meta.start is not None and cached.meta.start < resolved_from, (
            "the cache does not reach behind the resolved sample, so this fixture "
            "cannot distinguish a clipped load from an unclipped one"
        )
        assert cached.meta.rows > published["sample"]["observations"]
