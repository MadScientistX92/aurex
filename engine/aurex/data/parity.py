"""Residuals: observed price versus modelled price.

The parity *computation* now lives in :mod:`aurex.assets.lens`, because what a price
becomes in a buyer's currency is a property of the lens, not of this module. What
stays here is asset-neutral: given an observed reference rate and a modelled price,
measure the gap and describe how it behaved across a structural break.

Two rules hold whatever the asset:

**Never substitute the model for a missing observation.** The residual is
``observed / modelled - 1``. Filling a gap with the modelled value drives the
residual to zero and fabricates the signal it exists to measure. No observation, no
number.

**Compare like with like.** The observed rate and the modelled price must be on the
same tax basis and in the same units. IBJA publishes its gold rate exclusive of GST,
so it is differenced against ``price_ex_consumption_tax``; against the tax-inclusive
column it would print a spurious premium of roughly -291bps at all times.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def local_premium_bps(observed: pd.Series, modelled: pd.Series) -> pd.Series:
    """Residual in basis points: ``observed / modelled - 1``.

    Both series must be in the same currency, unit and tax basis. The result is NaN
    wherever there is no observation — an unmeasured day has no value.
    """
    # sort=False is safe and explicit: the reindex below imposes the modelled index
    # regardless, and pandas 4 will stop sorting by default here.
    aligned = pd.concat(
        {"observed": observed, "modelled": modelled}, axis=1, join="outer", sort=False
    )
    aligned = aligned.reindex(modelled.index)

    valid = aligned["observed"].notna() & aligned["modelled"].notna() & (aligned["modelled"] != 0)
    premium = pd.Series(np.nan, index=aligned.index, name="local_premium_bps")
    ratio = aligned.loc[valid, "observed"] / aligned.loc[valid, "modelled"]
    premium[valid] = (ratio - 1.0) * 10_000.0
    return premium


def passthrough_diagnostic(
    premium: pd.Series,
    break_date: date,
    *,
    window: int = 10,
) -> dict[str, float | int | None]:
    """Measure how the residual behaved across a policy break.

    A **diagnostic, not an assertion**. The modelled price steps mechanically with
    the duty schedule; whether retail prices follow is a market outcome Aurex reports
    rather than presumes. Complete passthrough leaves the residual flat, incomplete
    passthrough moves it, and both are legitimate observations.

    Returns means either side of the break, their difference in bps, and sample
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
