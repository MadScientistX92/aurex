"""Shared macro series.

These are global economic series rather than any one asset's property — the dollar,
real yields and equity volatility drive gold and crude alike. Assets reference them
by ``series_id`` in their ``factor_set``; they are loaded once per run and shared.

Crude appears here as a *factor* (an inflation-expectations proxy for gold) and will
also appear as an *asset* when the oil module lands. That is not a conflict: the same
series can be a driver of one thing and the subject of another. The asset registry
and this table are merged by ``series_id``, so it is loaded once either way.
"""

from __future__ import annotations

from collections.abc import Sequence

from aurex.data.base import Loader
from aurex.data.cache import CacheStore
from aurex.data.chain import SourceChain
from aurex.data.freshness import SeriesFreshness
from aurex.data.sources import FredLoader, GprDailyLoader, YahooLoader


def _spec() -> dict[str, Sequence[Loader]]:
    return {
        "vix": (
            YahooLoader("vix", "^VIX"),
            FredLoader("vix", "VIXCLS", "close"),
        ),
        "real_yield_10y": (FredLoader("real_yield_10y", "DFII10", "real_yield"),),
        "dxy": (FredLoader("dxy", "DTWEXBGS", "dxy"),),
        "wti": (FredLoader("wti", "DCOILWTICO", "wti"),),
        # Single-source deliberately. There is no second publisher of this index —
        # a mirror would be a copy of the same file with a worse citation, and the
        # chain's fallback to cache is the right degradation for a weekly series.
        "gpr": (GprDailyLoader("gpr"),),
        # The importing economy's side of the transmission chain. Monthly, and both
        # reached through FRED, which is why both are `secondary` in the artifact.
        "local_cpi": (FredLoader("local_cpi", "INDCPIALLMINMEI", "cpi"),),
        "local_policy_rate": (FredLoader("local_policy_rate", "IRSTCI01INM156N", "rate"),),
    }


#: How stale each macro series may be. None of these blocks publication — they are
#: factor inputs, not prices — but they are declared and measured so a series that
#: quietly stops updating is visible in the artifact rather than only in a loading.
#:
#: The numbers come from observed FRED publication lag rather than from a stated SLA,
#: which is why each carries the calendar it was read off. FRED restates: a series can
#: reach yesterday one day and the previous week the next, so these are deliberately
#: looser than the market calendar alone would suggest.
_FRESHNESS: dict[str, SeriesFreshness] = {
    "vix": SeriesFreshness(
        max_lag_days=5,
        calendar="US equity market days",
        rationale=(
            "Yahoo carries the previous session by the following morning; the FRED "
            "fallback lags a day further. Five days covers a long weekend plus the "
            "fallback's own lag without hiding a genuine multi-day outage."
        ),
    ),
    "real_yield_10y": SeriesFreshness(
        max_lag_days=5,
        calendar="US business days",
        rationale=(
            "DFII10 is published each business day with about a one-day lag. Five days "
            "absorbs a Monday holiday landing on top of that lag."
        ),
    ),
    "dxy": SeriesFreshness(
        max_lag_days=10,
        calendar="US business days, published in arrears",
        rationale=(
            "DTWEXBGS runs materially further behind than the other FRED series — six "
            "days behind on the 2026-07-30 run, which was an ordinary day. Ten days is "
            "set from that observed lag rather than from the market calendar; a tighter "
            "number would fire on healthy runs and train the reader to ignore it."
        ),
    ),
    "wti": SeriesFreshness(
        max_lag_days=7,
        calendar="US business days, published in arrears",
        rationale=(
            "DCOILWTICO was three days behind on the 2026-07-30 run. Seven days covers "
            "that lag plus a holiday week."
        ),
    ),
    "gpr": SeriesFreshness(
        max_lag_days=10,
        calendar="calendar days; the daily file is refreshed every Monday",
        rationale=(
            "The index itself is daily and covers weekends, but the workbook carrying "
            "it is rebuilt weekly — the authors state Monday, moving to the next "
            "business day when Monday is a federal holiday. So the last observation "
            "sits up to six days behind by Sunday, seven with the holiday shift. Ten "
            "days leaves room for one skipped update to be visible without a healthy "
            "Saturday run reading as a fault."
        ),
    ),
    "local_cpi": SeriesFreshness(
        max_lag_days=45,
        calendar="monthly, published in arrears",
        rationale=(
            "A monthly index released a few weeks after the month it measures, so 45 "
            "days is the ordinary lag rather than a fault. THIS SERIES IS EXPECTED TO "
            "READ STALE: the OECD discontinued it upstream and its last observation is "
            "2025-03. That is deliberately not papered over by widening the tolerance "
            "to swallow it — the chain is estimated to the last observation the "
            "publisher made, and the artifact should say so on every run rather than "
            "let a dead series look alive."
        ),
    ),
    "local_policy_rate": SeriesFreshness(
        max_lag_days=75,
        calendar="monthly, published in arrears",
        rationale=(
            "The immediate-rate series runs a full month or more behind: on the "
            "2026-08-12 read its last observation was 2026-05. Seventy-five days is "
            "set from that observed lag rather than from the monthly calendar alone, "
            "which would fire on every ordinary run."
        ),
    ),
}

#: Series ids this module can serve.
MACRO_SERIES_IDS: tuple[str, ...] = tuple(_spec())


def macro_freshness() -> dict[str, SeriesFreshness]:
    """Declared staleness tolerance per macro series."""
    return dict(_FRESHNESS)


def macro_chains(cache: CacheStore | None = None) -> dict[str, SourceChain]:
    """Resolution chains for every shared macro series."""
    store = cache or CacheStore()
    return {
        sid: SourceChain(sid, loaders, store, freshness=_FRESHNESS.get(sid))
        for sid, loaders in _spec().items()
    }
