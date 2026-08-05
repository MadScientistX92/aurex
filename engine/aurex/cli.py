"""Aurex command line."""

from __future__ import annotations

import json
import logging

import typer

from aurex import __version__
from aurex.data.schedules import duty_on, gst_on, load_duty_schedule
from aurex.pipeline import run, write_artifact, write_forecast_log

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
    logged = write_forecast_log(result.artifact)
    typer.echo(f"wrote {path}")
    typer.echo(f"logged {logged}")


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
def score(
    asset_id: str = typer.Option("gold", "--asset", help="Which registered asset to grade."),
    since: str = typer.Option("2015-01-01", "--from", help="First forecast date, YYYY-MM-DD."),
    step: int = typer.Option(5, "--step", help="Sessions between forecasts."),
    horizons: str = typer.Option("5,21,63", "--horizons", help="Comma-separated session counts."),
    paths: int = typer.Option(4_000, "--paths", help="Simulated paths per forecast."),
    model: str = typer.Option("", "--model", help="Override the asset's declared model."),
    offline: bool = typer.Option(False, "--offline", help="Never touch the network."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print without writing."),
) -> None:
    """Walk the engine forward over history and grade what it would have said."""
    import json as _json
    from datetime import UTC, date, datetime, timedelta

    import pandas as pd

    from aurex.assets import get
    from aurex.backtest import backtest_asset, describe_backtest
    from aurex.config import PUBLIC_DATA_DIR
    from aurex.data.schedules import load_policy_breaks
    from aurex.pipeline import DEFAULT_LOOKBACK_DAYS, _price_column, resolve_series
    from aurex.routes import load_routes
    from aurex.score import ClearsHurdle, WalkForwardRequest
    from aurex.trade import hurdle_for

    asset = get(asset_id)
    start_date = date.fromisoformat(since)
    end = datetime.now(UTC).date()

    # History must begin well before the first forecast, or the earliest fits are
    # short and the run silently grades a different model than the recent ones.
    series, unavailable = resolve_series(
        [asset],
        start=min(start_date - timedelta(days=DEFAULT_LOOKBACK_DAYS // 2), start_date),
        end=end,
        offline=offline,
    )
    price = series.get(asset.price_series_id)
    if price is None:
        typer.echo(
            f"cannot score {asset.id}: {asset.price_series_id} is unavailable "
            f"({unavailable.get(asset.price_series_id, 'no reason recorded')})",
            err=True,
        )
        raise typer.Exit(code=1)

    ohlc = None
    if asset.ohlc_series_id is not None and asset.ohlc_series_id in series:
        ohlc = series[asset.ohlc_series_id].frame

    request = WalkForwardRequest(
        horizons=tuple(int(h) for h in horizons.split(",") if h.strip()),
        step=step,
        start=pd.Timestamp(start_date),
    )

    # The composition §20 asks for, and the only place in the engine that holds an
    # asset, a route and a jurisdiction at the same time. Each cell becomes one binary
    # event carrying a number; the scorer never learns where the number came from.
    #
    # Horizon-dependent friction gets one event per horizon because its hurdle moves
    # with the holding period; a single event would score a quarterly hold against a
    # weekly cost. Horizon-invariant friction gets one event, because a second would be
    # the same event scored twice under different names.
    book = load_routes()
    hurdles: list[ClearsHurdle] = []
    for terms in book.terms:
        route = book.route(terms.route_id)
        if route.asset_id != asset.id:
            continue
        label = f"{route.instrument}_{terms.jurisdiction}"
        for held_for in request.horizons:
            quote = hurdle_for(terms.friction, horizon_days=held_for, label=label)
            hurdles.append(
                ClearsHurdle(
                    multiple=quote.multiple,
                    label=label,
                    # Horizon-invariant friction gets one event scored at every
                    # horizon; accruing friction gets one per horizon, each scored only
                    # at the horizon it was priced for.
                    horizon=held_for if quote.horizon_dependent else None,
                )
            )
            if not quote.horizon_dependent:
                break

    result = backtest_asset(
        asset,
        prices=_price_column(price.frame),
        ohlc=ohlc,
        breaks=tuple(pd.Timestamp(b.date) for b in load_policy_breaks()),
        request=request,
        model_id=model or None,
        n_paths=paths,
        extra_events=tuple(hurdles),
    )

    block = describe_backtest(asset, result) | {
        "generated_at": datetime.now(UTC).isoformat(),
        "engine_version": __version__,
        "source": price.meta.to_dict(),
    }

    if dry_run:
        typer.echo(_json.dumps(block, indent=2, sort_keys=True))
        return

    PUBLIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = PUBLIC_DATA_DIR / f"calibration-{asset.id}.json"
    out.write_text(_json.dumps(block, indent=2, sort_keys=True) + "\n")
    typer.echo(f"wrote {out}")

    def _p(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    calibration = result.calibration()
    for horizon in calibration.horizons:
        uniformity = horizon.pit_uniformity
        test = horizon.baseline_test
        typer.echo(
            f"  {horizon.horizon:>3} sessions: "
            f"n={horizon.n:<5} independent={horizon.n_independent:<5} "
            f"CRPS skill {horizon.skill:+.4f}  "
            f"DM p={_p(test.overlapping.p_value)}/{_p(test.thinned.p_value)}  "
            f"PIT KS p={'n/a' if uniformity is None else f'{uniformity.p_value:.4f}'}"
        )

    decay = calibration.skill_decay
    if decay is not None:
        typer.echo(
            f"  skill vs log-horizon: slope {decay.slope:+.5f} "
            f"(bootstrap p={decay.p_value:.3f}, R²={decay.r_squared:.2f})"
        )


@app.command()
def routes(
    asset_id: str = typer.Option("", "--asset", help="Restrict to one registered asset."),
    markdown: bool = typer.Option(
        False, "--markdown", help="Emit the generated breakeven table for the README."
    ),
) -> None:
    """Print the route x jurisdiction table, or the breakeven markdown it generates."""
    from aurex.routes import load_routes
    from aurex.trade import breakeven_table

    book = load_routes()
    rows = book.table_rows(asset_id or None)
    if not rows:
        typer.echo(f"no routes recorded for asset {asset_id!r}", err=True)
        raise typer.Exit(code=1)

    if markdown:
        typer.echo(breakeven_table(rows))
        return

    for terms in book.terms:
        route = book.route(terms.route_id)
        if asset_id and route.asset_id != asset_id:
            continue
        cap = "none" if terms.max_leverage is None else f"{terms.max_leverage:g}:1"
        typer.echo(
            f"{route.id:<16} {terms.jurisdiction}  "
            f"breakeven {terms.friction.quote(21).breakeven_pct:>6.2f}%  "
            f"leverage {cap:<6}  [{terms.source_confidence}]  {terms.source_url}"
        )
    typer.echo(
        "\nNo jurisdiction is the default. Availability is informational: Aurex states "
        "what a regulator publishes and links it, and never advises."
    )


@app.command()
def version() -> None:
    """Print the engine version."""
    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
