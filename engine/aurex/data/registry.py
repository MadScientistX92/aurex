"""Which sources serve which series, in preference order.

One place to see the whole data topology. The order here is the order the spec asks
for; the fallbacks are what keeps the pipeline reproducible when the preferred source
declines.
"""

from __future__ import annotations

from collections.abc import Sequence

from aurex.data.base import Loader
from aurex.data.cache import CacheStore
from aurex.data.chain import SourceChain
from aurex.data.sources import FredLoader, IbjaReportLoader, LbmaGoldLoader, YahooLoader

#: Series feeding the parity calculation and, later, the vol and factor layers.
SERIES_IDS = (
    "xauusd",
    "xau_futures",
    "usdinr",
    "vix",
    "real_yield_10y",
    "dxy",
    "wti",
    "ibja_gold",
)


def _loaders_for(series_id: str) -> Sequence[Loader]:
    match series_id:
        case "xauusd":
            # SPOT, and deliberately not GC=F. The spec names `GC=F` for XAU/USD, but
            # that is the COMEX front-month *future*, which carries a cost-of-carry
            # basis over spot — measured at +2.40% against the London PM fix on
            # 2026-07-29. Parity built on futures pushes that entire basis into
            # `local_premium_bps`, and because the basis moves with rates and time to
            # expiry it would inject a spurious time-varying signal into the one
            # number §2 calls the real signal. Yahoo publishes no spot gold ticker
            # (`XAUUSD=X` returns 404), so the London fix is the source. It is the
            # same benchmark IBJA prints in its own daily report, which makes the
            # comparison like-for-like.
            return (LbmaGoldLoader("xauusd"),)
        case "xau_futures":
            # Kept separately for step 2: futures give true OHLC, which the
            # realised-volatility estimators want and the close-only fix cannot
            # provide. Never used for parity.
            return (YahooLoader("xau_futures", "GC=F"),)
        case "usdinr":
            return (YahooLoader("usdinr", "INR=X"), FredLoader("usdinr", "DEXINUS", "close"))
        case "vix":
            return (YahooLoader("vix", "^VIX"), FredLoader("vix", "VIXCLS", "close"))
        case "real_yield_10y":
            return (FredLoader("real_yield_10y", "DFII10", "real_yield"),)
        case "dxy":
            return (FredLoader("dxy", "DTWEXBGS", "dxy"),)
        case "wti":
            return (FredLoader("wti", "DCOILWTICO", "wti"),)
        case "ibja_gold":
            return (IbjaReportLoader("ibja_gold"),)
        case _:
            raise KeyError(f"unknown series {series_id!r}")


def chain_for(series_id: str, cache: CacheStore | None = None) -> SourceChain:
    """Build the resolution chain for one series."""
    return SourceChain(series_id, _loaders_for(series_id), cache=cache)


def all_chains(cache: CacheStore | None = None) -> dict[str, SourceChain]:
    """Every series Aurex knows how to load."""
    store = cache or CacheStore()
    return {series_id: chain_for(series_id, store) for series_id in SERIES_IDS}
