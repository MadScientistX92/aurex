"""Return distributions. Populated in step 2.

Filtered historical simulation over GJR-GARCH standardised residuals,
block-bootstrapped to retain short-run dependence, plus a bivariate t-copula on the
price and FX residual pair so tail co-movement survives into the buyer's-currency
path.

Which series those are is the asset's business, not this module's — the pair comes
from the asset's price series and its lens's FX series. See :mod:`aurex.assets`.
"""
