"""Friction and P&L. Populated in step 3b.

Consumes the friction profiles an asset declares — spread-and-tax shapes and
carry/roll shapes alike — and produces the breakeven hurdle, P(profit), expected
P&L, percentiles and P(loss > 10%).

Carries the rule that when P(profit) < 50% the headline states the expected loss
rather than the win probability. The interface must never assume friction is
horizon-independent; see :mod:`aurex.assets.friction`.
"""
