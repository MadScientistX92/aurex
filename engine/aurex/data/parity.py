"""Import parity for Indian gold, and the domestic premium residual.

Parity is what a gram of gold *should* cost in India given the world price, the
rupee, and the tax stack::

    parity_ex_gst   = XAUUSD / 31.1034768 * USDINR * (1 + duty)
    parity_incl_gst = parity_ex_gst * (1 + gst_metal)

Two things are easy to get wrong here and both are guarded:

**Compare like with like.** IBJA publishes its 999 rate *exclusive* of GST, so the
premium is measured against ``parity_ex_gst``. Measuring it against the GST-inclusive
figure would print a spurious premium of roughly -300bps at all times.

**Never substitute parity for an observation.** ``local_premium_bps`` is
``observed / parity - 1``. Where no observed IBJA rate exists the premium is NaN, not
zero — filling it with parity would make the residual identically zero and fabricate
the very signal the premium is meant to measure.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from aurex.config import GRAMS_PER_TROY_OUNCE
from aurex.data.schedules import duty_on, gst_on

#: GST replaced state-varying VAT and excise on this date. Before it, no single
#: national rate applies, so parity is indicative only.
GST_REGIME_START = date(2017, 7, 1)

#: IBJA quotes rupees per 10 grams; matching that avoids a unit mismatch at the seam.
DEFAULT_QUOTE_GRAMS = 10.0


@dataclass(frozen=True, slots=True)
class ParityInputs:
    """Aligned inputs for one parity computation."""

    xauusd: pd.Series
    usdinr: pd.Series

    def aligned(self) -> pd.DataFrame:
        frame = pd.concat(
            {"xauusd": self.xauusd, "usdinr": self.usdinr}, axis=1, join="inner"
        ).sort_index()
        return frame.dropna()


def compute_parity(
    xauusd: pd.Series,
    usdinr: pd.Series,
    *,
    quote_grams: float = DEFAULT_QUOTE_GRAMS,
) -> pd.DataFrame:
    """Compute duty-inclusive parity per ``quote_grams`` grams.

    Returns a frame indexed by date with columns:

    ``parity_ex_gst``
        Landed cost including import duty, excluding GST. Compare IBJA against this.
    ``parity_incl_gst``
        What a retail buyer pays in tax terms, before dealer premium and making.
    ``duty_total`` / ``gst_metal``
        The rates actually applied, so any number can be traced to a schedule entry.
    ``confidence``
        ``high`` from 2017-07-01 (national GST regime), ``low`` before it.

    Dates before the ad valorem duty regime (2012-01-17) are dropped: the duty was a
    specific levy with no well-defined percentage, and inventing one would be worse
    than having no number.
    """
    frame = ParityInputs(xauusd, usdinr).aligned()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "parity_ex_gst",
                "parity_incl_gst",
                "duty_total",
                "gst_metal",
                "confidence",
            ]
        )

    duties: list[float] = []
    gsts: list[float] = []
    confidences: list[str] = []

    for stamp in frame.index:
        day = stamp.date()
        duty = duty_on(day)
        gst = gst_on(day)
        duties.append(duty.total if duty else np.nan)
        gsts.append(gst.metal if gst else 0.0)
        confidences.append("high" if day >= GST_REGIME_START else "low")

    per_gram_usd = frame["xauusd"] / GRAMS_PER_TROY_OUNCE
    base_inr = per_gram_usd * frame["usdinr"] * quote_grams

    duty_series = pd.Series(duties, index=frame.index, name="duty_total")
    gst_series = pd.Series(gsts, index=frame.index, name="gst_metal")

    parity_ex_gst = base_inr * (1.0 + duty_series)
    out = pd.DataFrame(
        {
            "parity_ex_gst": parity_ex_gst,
            "parity_incl_gst": parity_ex_gst * (1.0 + gst_series),
            "duty_total": duty_series,
            "gst_metal": gst_series,
            "confidence": pd.Series(confidences, index=frame.index),
        }
    )
    # Drop the pre-ad-valorem era rather than emit a parity built on a guessed rate.
    return out[out["duty_total"].notna()]


def local_premium_bps(observed: pd.Series, parity_ex_gst: pd.Series) -> pd.Series:
    """Domestic premium in basis points: ``observed / parity_ex_gst - 1``.

    ``observed`` must be a GST-exclusive IBJA quote in the same units as
    ``parity_ex_gst``. The result is NaN wherever there is no observation — the
    premium is a measurement, and an unmeasured day has no value.
    """
    # sort=False is safe and explicit: the reindex on the next line imposes the
    # parity index regardless, and pandas 4 will stop sorting by default here.
    aligned = pd.concat(
        {"observed": observed, "parity": parity_ex_gst}, axis=1, join="outer", sort=False
    )
    aligned = aligned.reindex(parity_ex_gst.index)

    valid = aligned["observed"].notna() & aligned["parity"].notna() & (aligned["parity"] != 0)
    premium = pd.Series(np.nan, index=aligned.index, name="local_premium_bps")
    ratio = aligned.loc[valid, "observed"] / aligned.loc[valid, "parity"]
    premium[valid] = (ratio - 1.0) * 10_000.0
    return premium


def passthrough_diagnostic(
    premium: pd.Series,
    break_date: date,
    *,
    window: int = 10,
) -> dict[str, float | int | None]:
    """Measure how the premium behaved across a policy break.

    This is a **diagnostic, not an assertion**. Parity steps mechanically with the
    duty schedule; whether retail prices follow is a market outcome Aurex reports
    rather than presumes. Complete passthrough leaves the premium flat; incomplete
    passthrough moves it. Both are legitimate observations.

    Returns means either side of the break, their difference in bps, and the sample
    sizes — ``None`` where a side has no observations.
    """
    stamp = pd.Timestamp(break_date)
    before = premium.loc[premium.index < stamp].dropna().tail(window)
    after = premium.loc[premium.index >= stamp].dropna().head(window)

    mean_before = float(before.mean()) if len(before) else None
    mean_after = float(after.mean()) if len(after) else None
    shift = mean_after - mean_before if mean_before is not None and mean_after is not None else None
    return {
        "mean_bps_before": mean_before,
        "mean_bps_after": mean_after,
        "shift_bps": shift,
        "n_before": len(before),
        "n_after": len(after),
    }
