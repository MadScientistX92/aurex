"""Volatility models. Populated in step 2: RollingStd, GJRGarch, HarRv.

All three implement one ``VolModel`` protocol and produce an h-day-ahead conditional
volatility forecast with a confidence band. Fitting is break-aware — the policy
discontinuities in ``data/schedules/policy_breaks.yaml`` must not be absorbed as
volatility shocks.
"""
