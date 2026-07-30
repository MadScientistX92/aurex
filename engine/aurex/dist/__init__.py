"""Return distributions. Populated in step 2.

Filtered historical simulation over GJR-GARCH standardised residuals, block-bootstrapped
to retain short-run dependence, plus a bivariate t-copula on the XAU/USDINR residual
pair so tail co-movement survives into the INR-per-gram path.
"""
