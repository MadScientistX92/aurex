"""Concrete data sources. Each exposes the :class:`~aurex.data.base.Loader` protocol."""

from aurex.data.sources.fred import FredLoader
from aurex.data.sources.ibja import IbjaReportLoader
from aurex.data.sources.lbma import LbmaGoldLoader
from aurex.data.sources.yahoo import YahooLoader

__all__ = ["FredLoader", "IbjaReportLoader", "LbmaGoldLoader", "YahooLoader"]
