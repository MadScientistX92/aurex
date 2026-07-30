"""Aurex command line."""

from __future__ import annotations

import json
import logging

import typer

from aurex import __version__
from aurex.data.schedules import duty_on, gst_on, load_duty_schedule
from aurex.pipeline import run, write_artifact

app = typer.Typer(
    add_completion=False,
    help="Aurex — calibrated uncertainty for gold, in INR. Distributions, not forecasts.",
)


@app.callback()
def _root(verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging.")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


@app.command()
def pipeline(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run offline from cache and print the artifact without writing it.",
    ),
    offline: bool = typer.Option(False, "--offline", help="Never touch the network."),
) -> None:
    """Resolve data, compute parity, and emit the artifact."""
    result = run(offline=offline or dry_run)

    if dry_run:
        typer.echo(json.dumps(result.artifact, indent=2, sort_keys=True))
        resolved = len(result.artifact["sources"])
        missing = len(result.artifact["unavailable"])
        typer.echo(f"\ndry run: {resolved} series resolved, {missing} unavailable", err=True)
        return

    path = write_artifact(result.artifact)
    typer.echo(f"wrote {path}")


@app.command()
def duty(on: str = typer.Argument(..., help="Date as YYYY-MM-DD.")) -> None:
    """Show the duty and GST in force on a date, with provenance."""
    from datetime import date

    when = date.fromisoformat(on)
    duty_entry = duty_on(when)
    gst_entry = gst_on(when)

    if duty_entry is None:
        typer.echo(f"{on}: before the ad valorem duty regime; no rate defined.")
    else:
        typer.echo(f"{on}: duty {duty_entry.total:.2%}  components={duty_entry.components}")
        typer.echo(f"      since {duty_entry.effective_from}  [{duty_entry.source_confidence}]")
        typer.echo(f"      {duty_entry.source_url}")

    if gst_entry is None:
        typer.echo(f"{on}: pre-GST regime (state VAT + excise); parity confidence low.")
    else:
        typer.echo(f"{on}: GST metal {gst_entry.metal:.2%}  making {gst_entry.making_charges:.2%}")


@app.command()
def schedule() -> None:
    """Print the full duty schedule with per-entry provenance."""
    for entry in load_duty_schedule():
        typer.echo(
            f"{entry.effective_from}  {entry.total:>7.2%}  "
            f"[{entry.source_confidence:<9}]  {entry.source_url}"
        )


@app.command()
def version() -> None:
    """Print the engine version."""
    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
