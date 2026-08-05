"""Aurex — a calibrated uncertainty engine for gold.

Aurex does not forecast price direction. It produces probability distributions and
scores them in public. See README.md §0 for the reasoning.

**No jurisdiction is the default.** This line used to end "priced in INR", which was
true of the first prototype and became a contradiction the moment §20 said that friction
and leverage are set per country. An unset jurisdiction gets the quote-currency
benchmark with friction excluded and labelled; see :mod:`aurex.routes`.
"""

__version__ = "0.1.0"
