"""First passage: the statistic a terminal distribution cannot produce.

A leveraged position does not survive to the horizon and then settle. It is closed
out the moment the equity behind it is exhausted, and that moment is a property of
the path, not of the endpoint. §18 puts the size of the gap plainly: for a driftless
walk, the chance of touching a level before the horizon is close to twice the chance
of finishing beyond it. Reporting only the terminal figure halves the number that
matters to a margined holder.

Three quantities come out of the same ensemble:

* **Touch probability** — the share of paths that reach the barrier at any point.
* **Time to the barrier** — conditional on reaching it, because an unconditional
  average over paths that never got there is not a duration of anything.
* **Terminal distribution conditional on surviving** — what the position that was
  *not* closed out is actually holding, which is a different distribution from the
  unconditional one and is usually the misleading one to quote.

The barrier is a price. Converting a leverage ratio into one is
:func:`margin_call_barrier`, kept separate so nothing here needs to know whether the
level came from margin, a stop, or a reader's own question.

**Monitoring is at session close, and that biases the answer downward.** A simulated
path has one price per session, so a barrier breached and recovered within a session
is not counted. Daily mark-to-market is the convention a margin call actually follows,
which is why this is the right default — but venues do liquidate intraday, so the
touch probability reported here is a floor rather than an estimate. It is also why
the ratio to the terminal probability comes out below the continuous-monitoring
factor of about two that §18 quotes: that factor assumes the barrier is watched
without gaps. The gap between the two is a modelling choice, stated, not a
discrepancy to tune away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from aurex.dist.paths import DEFAULT_QUANTILES, PathEnsemble

#: Which way the barrier is breached. A long position is stopped out downward.
Direction = Literal["down", "up"]

#: What the headline number must be, per §7 as extended by §18.
Headline = Literal["liquidation_probability", "expected_loss", "distribution"]

#: Quantiles reported for the time-to-barrier distribution.
DURATION_QUANTILES = (0.1, 0.5, 0.9)


@dataclass(frozen=True, slots=True)
class FirstPassage:
    """Barrier statistics from a retained path ensemble."""

    barrier: float
    direction: Direction
    #: Share of paths touching the barrier at any session in the horizon.
    touch_probability: float
    #: Share of paths beyond the barrier at the horizon — what a terminal-only
    #: distribution would have reported, kept alongside so the gap is visible.
    terminal_probability: float
    n_paths: int
    n_touched: int
    #: Sessions until first touch, conditional on touching.
    sessions_to_touch: dict[str, float]
    mean_sessions_to_touch: float | None
    #: Terminal prices of the paths that never touched.
    survivor_terminal_quantiles: dict[str, float]

    @property
    def path_dependence_ratio(self) -> float | None:
        """Touch probability over terminal probability. ``None`` when nothing lands.

        Around two for a driftless walk against a barrier it can reach; higher for a
        near barrier, and the number is the cost of having reported only terminals.
        """
        if self.terminal_probability <= 0.0:
            return None
        return self.touch_probability / self.terminal_probability

    def describe(self) -> dict[str, Any]:
        return {
            "barrier": self.barrier,
            "direction": self.direction,
            "touch_probability": self.touch_probability,
            "terminal_probability": self.terminal_probability,
            "path_dependence_ratio": self.path_dependence_ratio,
            "paths_touched": self.n_touched,
            "n_paths": self.n_paths,
            "sessions_to_touch": dict(self.sessions_to_touch),
            "mean_sessions_to_touch": self.mean_sessions_to_touch,
            "survivor_terminal_quantiles": dict(self.survivor_terminal_quantiles),
            "monitoring": "session_close",
            "note": (
                "Touch probability counts paths reaching the barrier at any session; "
                "terminal probability counts only those beyond it at the horizon. "
                "Survivor quantiles condition on never touching and are therefore "
                "not the unconditional distribution. Monitoring is at session close, "
                "so a barrier breached and recovered within one session is not "
                "counted and this probability is a floor."
            ),
        }


def first_passage(
    ensemble: PathEnsemble, *, barrier: float, direction: Direction = "down"
) -> FirstPassage:
    """Barrier statistics for one level."""
    if direction not in ("down", "up"):
        raise ValueError(f"direction must be 'down' or 'up', got {direction!r}")

    prices = ensemble.prices
    breached = prices <= barrier if direction == "down" else prices >= barrier

    touched = breached.any(axis=1)
    n_touched = int(np.count_nonzero(touched))
    # argmax on a boolean row returns the first True; rows with no True are masked
    # out by `touched` rather than silently reporting session zero.
    first_index = np.argmax(breached, axis=1)
    sessions = (first_index[touched] + 1).astype(float)

    terminal = ensemble.terminal()
    terminal_breach = terminal <= barrier if direction == "down" else terminal >= barrier

    survivors = terminal[~touched]
    survivor_quantiles: dict[str, float] = {}
    if survivors.size:
        values = np.quantile(survivors, DEFAULT_QUANTILES)
        survivor_quantiles = {
            f"q{int(p * 100):02d}": float(v)
            for p, v in zip(DEFAULT_QUANTILES, values, strict=True)
        }

    duration: dict[str, float] = {}
    if sessions.size:
        values = np.quantile(sessions, DURATION_QUANTILES)
        duration = {
            f"q{int(p * 100):02d}": float(v)
            for p, v in zip(DURATION_QUANTILES, values, strict=True)
        }

    return FirstPassage(
        barrier=float(barrier),
        direction=direction,
        touch_probability=float(np.mean(touched)),
        terminal_probability=float(np.mean(terminal_breach)),
        n_paths=ensemble.n_paths,
        n_touched=n_touched,
        sessions_to_touch=duration,
        mean_sessions_to_touch=float(sessions.mean()) if sessions.size else None,
        survivor_terminal_quantiles=survivor_quantiles,
    )


def margin_call_barrier(
    anchor: float,
    *,
    leverage: float,
    maintenance_fraction: float = 0.0,
    side: Literal["long", "short"] = "long",
) -> float:
    """The price at which a leveraged position's equity is exhausted.

    At ``leverage`` L the position moves L times the margin behind it, so an adverse
    move of ``1 / L`` wipes it out: ten times leverage liquidates on a ten percent
    move. ``maintenance_fraction`` is the share of the initial margin that must
    remain, so a venue closing positions at 50% of initial margin liquidates on half
    that move.

    Financing and the futures basis are not in here. §18 is explicit that carry is
    already implicit in the basis and must not be charged twice; the friction layer
    owns it.
    """
    if leverage <= 0.0:
        raise ValueError(f"leverage must be positive, got {leverage}")
    if not 0.0 <= maintenance_fraction < 1.0:
        raise ValueError(f"maintenance_fraction must be in [0, 1), got {maintenance_fraction}")

    adverse_move = (1.0 - maintenance_fraction) / leverage
    if adverse_move >= 1.0 and side == "long":
        # Below 1x nothing liquidates a long: the price would have to go negative.
        return -np.inf
    return anchor * (1.0 - adverse_move) if side == "long" else anchor * (1.0 + adverse_move)


def headline_statistic(
    *, profit_probability: float, liquidation_probability: float | None = None
) -> Headline:
    """Which number the interface must lead with.

    §7 requires that a position more likely to lose than win leads with the loss.
    §18 extends it: where liquidation is likelier than profit, liquidation leads,
    because a position that is closed out never reaches the distribution being shown.
    """
    if liquidation_probability is not None and liquidation_probability > profit_probability:
        return "liquidation_probability"
    if profit_probability < 0.5:
        return "expected_loss"
    return "distribution"
