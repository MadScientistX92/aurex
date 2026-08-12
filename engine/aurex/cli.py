"""Aurex command line."""

from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

import typer

from aurex import __version__
from aurex.data.schedules import duty_on, gst_on, load_duty_schedule
from aurex.pipeline import run, write_artifact, write_forecast_log
from aurex.provenance import code_provenance

if TYPE_CHECKING:
    from aurex.routes import RouteBook

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
    require_fresh: bool = typer.Option(
        True,
        "--require-fresh/--allow-stale",
        help=(
            "Refuse to publish when a blocking series does not reach the run date "
            "within its declared tolerance. Never turn this off in the nightly job."
        ),
    ),
) -> None:
    """Resolve data, compute parity, and emit the artifact.

    Exits non-zero without writing anything when the price series does not reach the
    run date. That refusal is the point of the command in an unattended context: a
    missing night is a hole anybody can see in the index, and a night built on
    week-old prices is a fabrication nobody can see at all.
    """
    from aurex.data.freshness import StaleDataError
    from aurex.record import write_index, write_skip_record

    result = run(offline=offline or dry_run)
    verdict = result.freshness

    if dry_run:
        typer.echo(json.dumps(result.artifact, indent=2, sort_keys=True))
        resolved = len(result.artifact["sources"])
        missing = len(result.artifact["unavailable"])
        typer.echo(f"\ndry run: {resolved} series resolved, {missing} unavailable", err=True)
        if verdict is not None and not verdict.publishable:
            # Reported, never enforced: a dry run reads whatever cache it was pointed
            # at, and the committed seed cache is deliberately old.
            typer.echo(
                f"dry run: {len(verdict.failures)} blocking series would refuse "
                f"publication; --dry-run does not enforce",
                err=True,
            )
        return

    if verdict is not None and require_fresh and not verdict.publishable:
        when = date.fromisoformat(str(result.artifact["generated_at"])[:10])
        record = write_skip_record(
            when=when,
            reason="price series did not reach the run date within its declared tolerance",
            detail=verdict.describe(),
        )
        write_index()
        try:
            verdict.raise_if_stale()
        except StaleDataError as exc:
            typer.echo(str(exc), err=True)
        typer.echo(f"recorded the skip at {record}", err=True)
        raise typer.Exit(code=1)

    path = write_artifact(result.artifact)
    logged = write_forecast_log(result.artifact)
    index = write_index()
    typer.echo(f"wrote {path}")
    typer.echo(f"logged {logged}")
    typer.echo(f"indexed {index}")


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


def _score_command(
    *,
    asset_id: str,
    since: str,
    until: str,
    step: int,
    horizons: str,
    paths: int,
    model: str,
) -> str:
    """The command that reproduces this artifact, spelled out in full.

    Every option is written even where it matches the default. A reader running the
    short form gets whatever the defaults are on the day they run it, and a default
    that moved between then and now is exactly the kind of silent difference a
    reproduction instruction exists to rule out.
    """
    parts = [
        "uv run aurex score",
        f"--asset {asset_id}",
        f"--from {since}",
        f"--to {until}",
        f"--step {step}",
        f"--horizons {horizons}",
        f"--paths {paths}",
    ]
    if model:
        parts.append(f"--model {model}")
    return " ".join(parts)


@app.command()
def score(
    asset_id: str = typer.Option("gold", "--asset", help="Which registered asset to grade."),
    since: str = typer.Option("2015-01-01", "--from", help="First forecast date, YYYY-MM-DD."),
    until: str = typer.Option(
        "",
        "--to",
        help=(
            "Last observation the run may see, YYYY-MM-DD. Defaults to the series' own "
            "last observation, never to today. Pass the resolved value from a previous "
            "artifact to reproduce its numbers exactly."
        ),
    ),
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

    from aurex import config
    from aurex.assets import get
    from aurex.backtest import backtest_asset, describe_backtest
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

    # Named for the series rather than the column it came out of, so the sample window
    # in the artifact says `xauusd` rather than `close`.
    prices = _price_column(price.frame).rename(asset.price_series_id)

    # Defaults to the series' last observation, never to today. Those differ whenever
    # the market was shut or a source lagged, and defaulting to today would silently
    # make the sample bound depend on when the command was typed rather than on what
    # data existed — which is the whole reason a reader could not reproduce the
    # published table before this flag existed.
    if until.strip():
        end_stamp = pd.Timestamp(date.fromisoformat(until.strip()))
    else:
        end_stamp = pd.Timestamp(prices.dropna().index[-1])

    request = WalkForwardRequest(
        horizons=tuple(int(h) for h in horizons.split(",") if h.strip()),
        step=step,
        start=pd.Timestamp(start_date),
        end=end_stamp,
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
        prices=prices,
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
        "code": code_provenance().describe(),
        "source": price.meta.to_dict(),
        # Built from the *resolved* window rather than from what was typed, so the
        # command in the artifact reproduces the artifact even when --to was omitted.
        "reproduce": _score_command(
            asset_id=asset.id,
            since=start_date.isoformat(),
            until=(
                result.sample.resolved_end.date().isoformat()
                if result.sample is not None
                else end_stamp.date().isoformat()
            ),
            step=step,
            horizons=horizons,
            paths=paths,
            model=model,
        ),
    }

    if dry_run:
        typer.echo(_json.dumps(block, indent=2, sort_keys=True))
        return

    config.PUBLIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = config.PUBLIC_DATA_DIR / f"calibration-{asset.id}.json"
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


def _bench_command(
    *,
    asset_id: str,
    since: str,
    until: str,
    step: int,
    horizons: str,
    paths: int,
    chronos_paths: int,
    nhits_steps: int,
    models: str,
) -> str:
    """The command that reproduces the shootout, every option spelled out.

    Same reasoning as :func:`_score_command`: a reader running the short form gets the
    defaults of the day they run it. It matters more here, because two of these models
    are stochastic in their fitting as well as their sampling and a moved default would
    be invisible in the output.
    """
    return " ".join(
        [
            "uv run aurex bench",
            f"--asset {asset_id}",
            f"--from {since}",
            f"--to {until}",
            f"--step {step}",
            f"--horizons {horizons}",
            f"--paths {paths}",
            f"--chronos-paths {chronos_paths}",
            f"--nhits-steps {nhits_steps}",
            f"--models {models}",
        ]
    )


@app.command()
def bench(
    asset_id: str = typer.Option("gold", "--asset", help="Which registered asset to grade."),
    since: str = typer.Option("2015-01-01", "--from", help="First forecast date, YYYY-MM-DD."),
    until: str = typer.Option(
        "",
        "--to",
        help=(
            "Last observation the run may see, YYYY-MM-DD. Defaults to the series' own "
            "last observation, never to today. Pass the resolved value back to reproduce."
        ),
    ),
    step: int = typer.Option(5, "--step", help="Sessions between forecasts."),
    horizons: str = typer.Option("5,10,21,42,63", "--horizons", help="Comma-separated."),
    paths: int = typer.Option(4_000, "--paths", help="Simulated paths per simulation model."),
    chronos_paths: int = typer.Option(
        200, "--chronos-paths", help="Sample paths drawn from Chronos per window."
    ),
    nhits_steps: int = typer.Option(200, "--nhits-steps", help="NHITS training steps per fit."),
    models: str = typer.Option(
        "",
        "--models",
        help="Comma-separated subset of the challenger set. Empty runs all of them.",
    ),
    offline: bool = typer.Option(False, "--offline", help="Never touch the network."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print without writing."),
) -> None:
    """Run the step 6 shootout: every model against the driftless random walk.

    Needs the ``bench`` extra (``uv sync --extra bench``), which is deliberately absent
    from default CI. This is a long job — every model is refitted at every as-of date —
    and it writes ``public-data/benchmarks.json``.
    """
    import json as _json
    from datetime import UTC, date, datetime, timedelta

    import pandas as pd

    from aurex import config
    from aurex.assets import get
    from aurex.bench import CHALLENGERS, describe_shootout, run_shootout
    from aurex.data.schedules import load_policy_breaks
    from aurex.pipeline import DEFAULT_LOOKBACK_DAYS, _price_column, resolve_series
    from aurex.score import WalkForwardRequest

    asset = get(asset_id)
    start_date = date.fromisoformat(since)

    series, unavailable = resolve_series(
        [asset],
        start=min(start_date - timedelta(days=DEFAULT_LOOKBACK_DAYS // 2), start_date),
        end=datetime.now(UTC).date(),
        offline=offline,
    )
    price = series.get(asset.price_series_id)
    if price is None:
        typer.echo(
            f"cannot bench {asset.id}: {asset.price_series_id} is unavailable "
            f"({unavailable.get(asset.price_series_id, 'no reason recorded')})",
            err=True,
        )
        raise typer.Exit(code=1)

    ohlc = None
    if asset.ohlc_series_id is not None and asset.ohlc_series_id in series:
        ohlc = series[asset.ohlc_series_id].frame

    prices = _price_column(price.frame).rename(asset.price_series_id)
    if until.strip():
        end_stamp = pd.Timestamp(date.fromisoformat(until.strip()))
    else:
        end_stamp = pd.Timestamp(prices.dropna().index[-1])

    request = WalkForwardRequest(
        horizons=tuple(int(h) for h in horizons.split(",") if h.strip()),
        step=step,
        start=pd.Timestamp(start_date),
        end=end_stamp,
    )
    include = tuple(m.strip() for m in models.split(",") if m.strip()) or CHALLENGERS

    run = run_shootout(
        asset,
        prices=prices,
        ohlc=ohlc,
        breaks=tuple(pd.Timestamp(b.date) for b in load_policy_breaks()),
        request=request,
        n_paths=paths,
        chronos_paths=chronos_paths,
        nhits_steps=nhits_steps,
        include=include,
    )

    block = describe_shootout(asset, run.result, competitors=run.competitors) | {
        "generated_at": datetime.now(UTC).isoformat(),
        "engine_version": __version__,
        "code": code_provenance().describe(),
        "source": price.meta.to_dict(),
        "reproduce": _bench_command(
            asset_id=asset.id,
            since=start_date.isoformat(),
            until=(
                run.result.sample.resolved_end.date().isoformat()
                if run.result.sample is not None
                else end_stamp.date().isoformat()
            ),
            step=step,
            horizons=horizons,
            paths=paths,
            chronos_paths=chronos_paths,
            nhits_steps=nhits_steps,
            models=",".join(include),
        ),
        "pre_registration": (
            "Stated before this ran, and kept as written whatever the result: the GARCH "
            "family should win on volatility and distribution shape, and no model — "
            "including the time-series foundation models — is expected to beat the "
            "random walk on 10-day directional accuracy."
        ),
    }

    if dry_run:
        typer.echo(_json.dumps(block, indent=2, sort_keys=True))
        return

    config.PUBLIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = config.PUBLIC_DATA_DIR / "benchmarks.json"
    out.write_text(_json.dumps(block, indent=2, sort_keys=True) + "\n")
    typer.echo(f"wrote {out}")

    for horizon in block["horizons"]:
        spa = horizon["hansen_spa"]
        typer.echo(f"  {horizon['horizon_sessions']:>3} sessions (n={horizon['observations']}):")
        for entry in horizon["models"]:
            mde = entry["minimum_detectable_effect"]["detectable_skill_pct"]
            typer.echo(
                f"      {entry['model']:<18} skill {entry['skill_vs_benchmark']:+.4f}  "
                f"MDE {'n/a' if mde is None else f'{mde:.2f}%'}"
            )
        if spa is not None:
            typer.echo(
                f"      SPA p={spa['p_value']:.3f} (best: {spa['best_model']})  "
                f"MCS keeps {len(horizon['model_confidence_set']['included'])}"
            )


def _direction_command(
    *,
    asset_id: str,
    since: str,
    until: str,
    step: int,
    horizons: str,
    paths: int,
    chronos_paths: int,
    nhits_steps: int,
    models: str,
    controls: bool,
) -> str:
    """The command that reproduces the direction run, every option spelled out."""
    return " ".join(
        [
            "uv run aurex direction",
            f"--asset {asset_id}",
            f"--from {since}",
            f"--to {until}",
            f"--step {step}",
            f"--horizons {horizons}",
            f"--paths {paths}",
            f"--chronos-paths {chronos_paths}",
            f"--nhits-steps {nhits_steps}",
            f"--models {models}",
            "--controls" if controls else "--no-controls",
        ]
    )


@app.command()
def direction(
    asset_id: str = typer.Option("gold", "--asset", help="Which registered asset to grade."),
    since: str = typer.Option("2015-01-01", "--from", help="First forecast date, YYYY-MM-DD."),
    until: str = typer.Option(
        "",
        "--to",
        help=(
            "Last observation the run may see, YYYY-MM-DD. Defaults to the series' own "
            "last observation, never to today. Pass the resolved value back to reproduce."
        ),
    ),
    step: int = typer.Option(5, "--step", help="Sessions between forecasts."),
    horizons: str = typer.Option("5,10,21,42,63", "--horizons", help="Comma-separated."),
    paths: int = typer.Option(4_000, "--paths", help="Simulated paths per simulation model."),
    chronos_paths: int = typer.Option(
        200, "--chronos-paths", help="Sample paths drawn from Chronos per window."
    ),
    nhits_steps: int = typer.Option(200, "--nhits-steps", help="NHITS training steps per fit."),
    models: str = typer.Option(
        "",
        "--models",
        help="Comma-separated subset of the challenger set. Empty runs all of them.",
    ),
    controls: bool = typer.Option(
        True,
        "--controls/--no-controls",
        help=(
            "Run the calendar placebo, the two trend-preserving nulls and "
            "leave-one-year-out beside the screens. On by default; costs a tenth of a "
            "second against a run measured in hours."
        ),
    ),
    offline: bool = typer.Option(False, "--offline", help="Never touch the network."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print without writing."),
) -> None:
    """Grade every model on direction, uncentred, against the drift-matched random walk.

    A separate run from ``aurex bench`` and not a flag on it, because the two need
    opposite drift policies. The CRPS shootout centres every model, which puts P(up) at
    about one half for all of them by construction — so direction measured there grades
    the drift policy rather than the models. Here every competitor including the
    benchmark carries whatever drift it infers, which is the like-for-like comparison.

    Needs the ``bench`` extra (``uv sync --extra bench``). Long job: every model is
    refitted at every as-of date. Writes ``public-data/direction.json``.
    """
    import json as _json
    from datetime import UTC, date, datetime, timedelta

    import pandas as pd

    from aurex import config
    from aurex.assets import get
    from aurex.bench import CHALLENGERS, describe_direction, run_shootout
    from aurex.data.schedules import load_policy_breaks
    from aurex.pipeline import DEFAULT_LOOKBACK_DAYS, _price_column, resolve_series
    from aurex.score import TerminalAbove, WalkForwardRequest

    asset = get(asset_id)
    start_date = date.fromisoformat(since)

    series, unavailable = resolve_series(
        [asset],
        start=min(start_date - timedelta(days=DEFAULT_LOOKBACK_DAYS // 2), start_date),
        end=datetime.now(UTC).date(),
        offline=offline,
    )
    price = series.get(asset.price_series_id)
    if price is None:
        typer.echo(
            f"cannot grade {asset.id}: {asset.price_series_id} is unavailable "
            f"({unavailable.get(asset.price_series_id, 'no reason recorded')})",
            err=True,
        )
        raise typer.Exit(code=1)

    ohlc = None
    if asset.ohlc_series_id is not None and asset.ohlc_series_id in series:
        ohlc = series[asset.ohlc_series_id].frame

    prices = _price_column(price.frame).rename(asset.price_series_id)
    if until.strip():
        end_stamp = pd.Timestamp(date.fromisoformat(until.strip()))
    else:
        end_stamp = pd.Timestamp(prices.dropna().index[-1])

    request = WalkForwardRequest(
        horizons=tuple(int(h) for h in horizons.split(",") if h.strip()),
        step=step,
        start=pd.Timestamp(start_date),
        end=end_stamp,
    )
    include = tuple(m.strip() for m in models.split(",") if m.strip()) or CHALLENGERS
    event = TerminalAbove()

    run = run_shootout(
        asset,
        prices=prices,
        ohlc=ohlc,
        breaks=tuple(pd.Timestamp(b.date) for b in load_policy_breaks()),
        request=request,
        n_paths=paths,
        chronos_paths=chronos_paths,
        nhits_steps=nhits_steps,
        include=include,
        # The two belong together and are passed together: uncentred forecasts are what
        # make the event worth grading, and the event is why the run is uncentred.
        demean=False,
        events=(event,),
    )

    block = describe_direction(
        asset, run.result, event=event, competitors=run.competitors, controls=controls
    ) | {
        "generated_at": datetime.now(UTC).isoformat(),
        "engine_version": __version__,
        "code": code_provenance().describe(),
        "source": price.meta.to_dict(),
        "reproduce": _direction_command(
            asset_id=asset.id,
            since=start_date.isoformat(),
            until=(
                run.result.sample.resolved_end.date().isoformat()
                if run.result.sample is not None
                else end_stamp.date().isoformat()
            ),
            step=step,
            horizons=horizons,
            paths=paths,
            chronos_paths=chronos_paths,
            nhits_steps=nhits_steps,
            models=",".join(include),
            controls=controls,
        ),
        "pre_registration": (
            "Stated before this ran, and kept as written whatever the result: resolution "
            "at or near zero for every model at every horizon, including Chronos. The "
            "foundation model has the most room to surprise here — it infers a trend "
            "from its context window rather than fitting a mean, so it is the one "
            "entrant that could in principle discriminate. If it does, that is a finding "
            "and it gets published as one."
        ),
    }

    if dry_run:
        typer.echo(_json.dumps(block, indent=2, sort_keys=True))
        return

    config.PUBLIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = config.PUBLIC_DATA_DIR / "direction.json"
    out.write_text(_json.dumps(block, indent=2, sort_keys=True) + "\n")
    typer.echo(f"wrote {out}")

    for horizon in block["horizons"]:
        screen = horizon["discrimination"]["full_sample"]
        typer.echo(
            f"  {horizon['horizon_sessions']:>3} sessions (n={horizon['observations']}, "
            f"realised {horizon['realised_rate']:.3f}):"
        )
        for entry in horizon["models"]:
            typer.echo(
                f"      {entry['model']:<26} resolution {entry['decomposition']['resolution']:.5f}"
                f"  mean p {entry['mean_forecast']:.3f}  brier {entry['brier']:.4f}"
            )
        if screen is not None:
            typer.echo(
                f"      max-resolution p={screen['p_value']:.3f} "
                f"(best: {screen['best_model']}), both runs reject: "
                f"{horizon['discrimination']['distinguishable_from_zero']}"
            )


def _routes_export(book: RouteBook, *, horizons: tuple[int, ...]) -> dict[str, Any]:
    """The route table as published JSON, with the breakeven each cell implies.

    Written to ``public-data/`` because the dashboard reads committed JSON and nothing
    else — no engine on the server, no model execution behind the page. The breakeven
    is resolved here, per horizon, rather than left for a client to recompute from
    friction components: the arithmetic is `(1 + premium)(1 + tax) / (1 - buyback)`
    plus accrual, and a second implementation of it in TypeScript is a second place for
    it to be wrong. The client looks the number up; it never derives one.
    """
    from aurex.trade import hurdle_for

    cells: list[dict[str, Any]] = []
    for terms in book.terms:
        route = book.route(terms.route_id)
        quotes = {}
        for horizon in horizons:
            quote = hurdle_for(
                terms.friction,
                horizon_days=horizon,
                label=f"{route.instrument}_{terms.jurisdiction}",
            )
            quotes[str(horizon)] = {
                "required_move": round(quote.required_move, 8),
                "required_move_pct": round(quote.required_move_pct, 6),
                "multiple": round(quote.multiple, 8),
                "horizon_dependent": quote.horizon_dependent,
                "components": {k: round(v, 8) for k, v in quote.components.items()},
            }
        cells.append(
            terms.describe()
            | {
                "route": route.describe(),
                "breakeven": quotes,
            }
        )

    return book.describe() | {
        "generated_at": datetime.now(UTC).isoformat(),
        "engine_version": __version__,
        "code": code_provenance().describe(),
        "horizons": list(horizons),
        "cells": cells,
        "reading": (
            "One entry per (route, jurisdiction). breakeven is the round-trip move "
            "required before a position is above water, resolved per horizon because "
            "carry accrues and a door charge does not. There is no default "
            "jurisdiction: with none set the correct view is the quote-currency "
            "benchmark with friction excluded and labelled, never one country's stack "
            "applied silently."
        ),
    }


@app.command()
def routes(
    asset_id: str = typer.Option("", "--asset", help="Restrict to one registered asset."),
    markdown: bool = typer.Option(
        False, "--markdown", help="Emit the generated breakeven table for the README."
    ),
    to_json: bool = typer.Option(
        False, "--json", help="Write public-data/routes.json for the dashboard."
    ),
    horizons: str = typer.Option(
        "5,10,21,42,63,252", "--horizons", help="Horizons to resolve breakeven at."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print without writing."),
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

    if to_json:
        import json as _json

        from aurex import config

        block = _routes_export(
            book, horizons=tuple(int(h) for h in horizons.split(",") if h.strip())
        )
        if dry_run:
            typer.echo(_json.dumps(block, indent=2, sort_keys=True))
            return
        config.PUBLIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
        out = config.PUBLIC_DATA_DIR / "routes.json"
        out.write_text(_json.dumps(block, indent=2, sort_keys=True) + "\n")
        typer.echo(f"wrote {out}")
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
def livelog(
    offline: bool = typer.Option(False, "--offline", help="Never touch the network."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print without writing."),
) -> None:
    """Score the forecasts this engine actually published, before their outcomes existed.

    Reported separately from the walk-forward and never pooled with it. Where the count
    cannot support a test, this says so and reports the count rather than waiting until
    the number looks like something.
    """
    import json as _json

    from aurex.livelog import collect, summarise, write_live_log
    from aurex.pipeline import run

    result = run(offline=offline)
    realised = {
        asset_id: {code: lens.frame["price"] for code, lens in asset.lenses.items()}
        for asset_id, asset in result.assets.items()
    }
    observations = collect(realised=realised)

    if dry_run:
        typer.echo(_json.dumps(summarise(observations), indent=2, sort_keys=True))
        return

    path = write_live_log(observations)
    summary = summarise(observations)
    typer.echo(f"wrote {path}")
    typer.echo(f"live forecasts scored: {summary['total_observations']}")
    for horizon in summary["horizons"]:
        typer.echo(
            f"  {horizon['horizon_sessions']:>3} sessions: "
            f"n={horizon['observations']} "
            f"independent={horizon['independent_observations']}  "
            f"{horizon['test_note']}"
        )
    if not summary["horizons"]:
        typer.echo(
            "  no horizon has elapsed yet. The live log starts empty by construction: "
            "it can only contain forecasts published before their outcome existed."
        )


@app.command()
def index(
    dry_run: bool = typer.Option(False, "--dry-run", help="Print without writing."),
) -> None:
    """Rebuild the forecast index, including the dates expected but missing."""
    import json as _json

    from aurex.record import build_index, write_index

    if dry_run:
        typer.echo(_json.dumps(build_index(), indent=2, sort_keys=True))
        return

    path = write_index()
    payload = build_index()
    counts = payload["counts"]
    typer.echo(f"wrote {path}")
    typer.echo(
        f"published={counts['published']}  gaps={counts['gaps']} "
        f"(explained={counts['gaps_explained']}, unexplained={counts['gaps_unexplained']})"
    )


def _factors_command(
    *, asset_id: str, since: str, until: str, window: int, draws: int, horizons: str
) -> str:
    """The command that reproduces the attribution run, every option spelled out."""
    return " ".join(
        [
            "uv run aurex factors",
            f"--asset {asset_id}",
            f"--from {since}",
            f"--to {until}",
            f"--window {window}",
            f"--draws {draws}",
            f"--horizons {horizons}",
        ]
    )


@app.command()
def factors(
    asset_id: str = typer.Option("gold", "--asset", help="Which registered asset to attribute."),
    since: str = typer.Option("2000-01-01", "--from", help="Earliest observation, YYYY-MM-DD."),
    until: str = typer.Option(
        "",
        "--to",
        help=(
            "Last observation the run may see, YYYY-MM-DD. Defaults to the price "
            "series' own last observation, never to today. Pass the resolved value "
            "back to reproduce the run byte for byte."
        ),
    ),
    window: int = typer.Option(156, "--window", help="Rolling window, in weeks."),
    draws: int = typer.Option(2_000, "--draws", help="Bootstrap replicates."),
    horizons: str = typer.Option("0,1,3,6,12", "--horizons", help="Chain horizons, in months."),
    offline: bool = typer.Option(False, "--offline", help="Never touch the network."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print without writing."),
) -> None:
    """Estimate driver loadings and the declared transmission chain.

    Attribution, never direction forecasting. Loadings are published with bootstrap
    intervals and an out-of-sample R-squared reported at whatever it is; the chain is
    estimated by local projections with a band per horizon and no identifying
    assumptions imposed. Writes ``public-data/factors.json``.

    Refuses rather than degrades when a *required* driver is missing. The geopolitical
    risk index is required for exactly that reason: without it the loadings reach
    "escalation is bearish" through the real-yield channel alone, honestly estimated and
    with the wrong sign.
    """
    import json as _json
    from datetime import UTC, date, datetime

    import pandas as pd

    from aurex import config
    from aurex import factors as factors_module
    from aurex.assets import get, lens_by_code
    from aurex.assets.lens import LensContext
    from aurex.pipeline import _price_column, resolve_series

    asset = get(asset_id)
    start_date = date.fromisoformat(since)

    series, unavailable = resolve_series(
        [asset], start=start_date, end=datetime.now(UTC).date(), offline=offline
    )
    price = series.get(asset.price_series_id)
    if price is None:
        typer.echo(
            f"cannot attribute {asset.id}: {asset.price_series_id} is unavailable "
            f"({unavailable.get(asset.price_series_id, 'no reason recorded')})",
            err=True,
        )
        raise typer.Exit(code=1)

    last = pd.Timestamp(_price_column(price.frame).dropna().index[-1])
    end_stamp = pd.Timestamp(date.fromisoformat(until.strip())) if until.strip() else last

    # Bounded at both ends before anything is estimated, so the reproducing command in
    # the artifact is a command and not a description of one. §0's reproducibility claim
    # is what makes this mandatory rather than tidy.
    frames = {sid: loaded.frame.loc[:end_stamp] for sid, loaded in series.items()}

    declared = asset.transmission_chain
    if declared is not None:
        lens = lens_by_code(asset, declared.terminal_lens)
        fx = None
        if lens.fx_series_id is not None and lens.fx_series_id in frames:
            fx = _price_column(frames[lens.fx_series_id])
        local = lens.apply(_price_column(frames[asset.price_series_id]), LensContext(fx=fx))
        # The lens computes the local price; the chain never recomputes one. There is one
        # implementation of the tax stack and it is the one carrying the citations.
        frames[declared.terminal_series_id] = local[["price"]].rename(columns={"price": "close"})

    try:
        attribution = factors_module.estimate_loadings(
            asset, frames, unavailable=unavailable, window=window, draws=draws
        )
    except factors_module.DesignError as exc:
        typer.echo(f"refusing to publish loadings: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    chain = None
    chain_reason = None
    try:
        chain = factors_module.estimate_chain(
            asset,
            frames,
            horizons=tuple(int(h) for h in horizons.split(",") if h.strip()),
            draws=draws,
        )
    except factors_module.ChainError as exc:
        # The chain degrades where the loadings refuse, and the asymmetry is deliberate:
        # a missing chain leaves the loadings correct, while a missing required driver
        # leaves them confidently wrong.
        chain_reason = str(exc)
        typer.echo(f"chain not estimated: {exc}", err=True)

    block = factors_module.describe(asset, attribution, chain) | {
        "generated_at": datetime.now(UTC).isoformat(),
        "engine_version": __version__,
        "code": code_provenance().describe(),
        "sources": {sid: loaded.meta.to_dict() for sid, loaded in sorted(series.items())},
        "chain_unavailable_reason": chain_reason,
        "reproduce": _factors_command(
            asset_id=asset.id,
            since=start_date.isoformat(),
            until=end_stamp.date().isoformat(),
            window=window,
            draws=draws,
            horizons=horizons,
        ),
        "pre_registration": (
            "Stated in the README before this ran and left as written: out-of-sample "
            "R-squared low and plausibly indistinguishable from zero, with the "
            "predictive form held to be indistinguishable from zero with confidence and "
            "the contemporaneous form expected positive and material at 0.2 to 0.45; "
            "rolling loading stability poor; and the compounded chain band spanning "
            "zero at every horizon."
        ),
    }

    if dry_run:
        typer.echo(_json.dumps(block, indent=2, sort_keys=True))
        return

    config.PUBLIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = config.PUBLIC_DATA_DIR / "factors.json"
    out.write_text(_json.dumps(block, indent=2, sort_keys=True) + "\n")
    typer.echo(f"wrote {out}")

    design = block["attribution"]["design"]
    typer.echo(f"  {design['observations']} weeks, {design['first_week']}..{design['last_week']}")
    for loading in block["attribution"]["loadings"]:
        typer.echo(
            f"      {loading['factor']:<22} {loading['standardised']:+.5f}  "
            f"selected {loading['selection_rate']}  OLS p {loading['ols_p_value']}"
        )
    for kind in ("contemporaneous", "predictive"):
        scored = block["attribution"]["out_of_sample"][kind]
        if scored is not None:
            typer.echo(
                f"      R2 {kind:<16} {scored['r_squared_oos']:+.4f}  DM p {scored['dm_p_value']}"
            )
    if chain is not None:
        for entry in block["chain"]["compounded"]["horizons"]:
            low, high = entry["interval_95"]
            typer.echo(
                f"      chain h={entry['horizon_months']:>2}  {entry['compounded']:+.6f} "
                f"[{low:+.6f}, {high:+.6f}]  spans zero: {entry['spans_zero']}"
            )


@app.command()
def version() -> None:
    """Print the engine version."""
    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
