"""Event scenario engine. Populated in step 4.

Priors come from markets, not from us. Every branch probability records its
``prior_source``; where no market exists, the historical base rate is used and
labelled as such. A directional tilt not traceable to a market-implied probability
is a bug, not a view.
"""
