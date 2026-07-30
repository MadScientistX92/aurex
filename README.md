# Aurex

**A calibrated uncertainty engine for gold. Distributions, not predictions.**

[![CI](https://github.com/MadScientistX92/aurex/actions/workflows/ci.yml/badge.svg)](https://github.com/MadScientistX92/aurex/actions)
[![Nightly](https://github.com/MadScientistX92/aurex/actions/workflows/nightly.yml/badge.svg)](https://github.com/MadScientistX92/aurex/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)

---

## The thesis

Search for a gold forecasting tool and you will find hundreds that output a price and an arrow. Almost none of them tell you how often they have been right, because almost none of them keep score.

Aurex takes the opposite position:

> Short-horizon price **direction** is not reliably forecastable. Short-horizon **volatility** partly is. So Aurex never predicts a price — it produces a probability distribution, and then publicly grades how well-calibrated that distribution turned out to be.

Every forecast is timestamped and committed to this repository. Once its horizon elapses, it is scored automatically. The git history is the track record, and it cannot be quietly edited after the fact.

## What it does

- Fits a **GJR-GARCH(1,1)** volatility model with skewed-t innovations to XAU/USD and USD/INR
- Generates return distributions via **filtered historical simulation** — no assumed parametric shape
- Couples the two exposures with a **t-copula** so tail co-movement survives into the INR price
- Enumerates **event scenarios** (geopolitics, CPI, payrolls) with probabilities sourced from prediction markets, and propagates them as a mixture distribution
- Models the **real cost of physical gold in India** — dealer premium, GST, buyback spread — and reports the move required to break even
- Scores itself with **PIT histograms, CRPS, Kupiec/Christoffersen VaR tests, and reliability diagrams**

## What it does not do

- It does not tell you where gold is going.
- It does not tell you to buy or sell anything.
- It does not claim to beat the market, and the benchmark table below reports the cases where it loses.

## Quick start

```bash
git clone https://github.com/MadScientistX92/aurex.git
cd aurex/engine
uv sync
uv run aurex pipeline --dry-run     # runs offline from cached fixtures
uv run aurex forecast --horizon 10  # 10-trading-day distribution
```

Dashboard:

```bash
cd web && pnpm install && pnpm dev
```

## Methodology

### Volatility

Gold's volatility clusters — turbulent days follow turbulent days — and responds asymmetrically to shocks. GJR-GARCH captures both:

```
σ²ₜ = ω + (α + γ·I[rₜ₋₁ < 0])·r²ₜ₋₁ + β·σ²ₜ₋₁
```

The `γ` term lets negative and positive shocks move volatility differently. Two alternative estimators (21-day realised, HAR-RV) are implemented for comparison and are selectable at runtime.

### Return distribution

Aurex does **not** sample from a fitted normal or Student-t. It uses filtered historical simulation, which is standard practice on institutional risk desks:

1. Fit GJR-GARCH, extract standardised residuals `zₜ = rₜ / σₜ`
2. Block-bootstrap from the empirical `z` pool (block length ≈ 5, preserving short-run dependence)
3. Rescale by the h-day-ahead volatility forecast and accumulate

The result inherits gold's actual fat tails and skew rather than a convenient assumption about them. This is the single most defensible technical choice in the project.

### Joint INR exposure

An Indian buyer holds two risks: the dollar gold price and the rupee. Modelling them independently understates joint tail events. Aurex fits a **t-copula** to the standardised residuals of both series and samples jointly, then composes the INR-per-gram path.

It also computes a **local premium** — the residual between the observed Indian retail rate and import parity:

```
parity = (XAUUSD / 31.1035) × USDINR × (1 + duty) × (1 + gst)
local_premium_bps = (observed / parity − 1) × 10⁴
```

This captures domestic demand pressure and import friction, and it is the part a retail buyer in India actually pays.

### Factor attribution

An elastic-net regression of weekly returns on real yields, the dollar index, oil, VIX, ETF flows, and lagged momentum, on a rolling three-year window.

**This is used for attribution and scenario propagation, never for directional forecasting.** Loadings are published with bootstrap confidence intervals, and out-of-sample R² is reported honestly — which usually means "close to zero." That is the truthful answer, and hiding it would defeat the point of the project.

### Scenario engine

You cannot forecast a ceasefire. You can enumerate outcomes, weight them, and propagate them — which is what risk desks actually do.

Each axis (geopolitics, CPI, payrolls) carries branches with a shock vector over the factors. Branch probabilities are pulled from **prediction markets and rate futures** — Polymarket, Kalshi, Metaculus, CME FedWatch — and every probability records its `prior_source` in the output artifact. Where no market exists, the historical base rate of surprise direction is used and labelled as such.

The final forecast is a mixture: sample a branch by its probability, apply the shock through the factor loadings, then draw the FHS residual.

**Design constraint:** the scenario engine is built to *widen* the distribution, not to shift its centre. Any directional tilt must be traceable to a market-implied probability. A confident tilt with no market behind it is treated as a bug.

### Trade layer

```
cost      = grams × spot × (1 + premium) × (1 + gst)
proceeds  = grams × spot_T × (1 − buyback)
breakeven = (1 + premium)(1 + gst) / (1 − buyback)
```

At typical Indian retail parameters — 3% dealer premium, 3% GST, 3% buyback discount — the break-even move is **+9.4%**, before gold has done anything at all. Against a two-week one-standard-deviation move of roughly ±4.4%, that is a two-sigma requirement in one specific direction.

This is why the project exists. The friction is deterministic and knowable; the price move is not. Most retail tools model the uncertain part and ignore the certain one.

Friction profiles for gold ETFs and sovereign gold bonds are included for comparison.

## Calibration

A forecast that is never scored is marketing. Aurex grades itself on:

| Metric | Question it answers |
|---|---|
| PIT histogram | Are the predicted distributions the right *shape*? Uniform = calibrated |
| CRPS skill score | Is the full distribution better than a driftless random walk? |
| Kupiec / Christoffersen | Do 95% and 99% VaR breaches occur at the right rate, and independently? |
| Brier score + reliability | When it says 20%, does it happen 20% of the time? |

Walk-forward, expanding window, no lookahead, 2015→present.

## Benchmarks

Every model must beat a **driftless random walk** on out-of-sample CRPS. Models that fail ship anyway, labelled as failing.

> _Results pending first full backtest run. This table will be populated by CI and is not hand-edited._

| Model | CRPS skill vs RW | PIT uniformity (KS p) | Directional accuracy |
|---|---|---|---|
| Random walk (baseline) | — | — | — |
| GJR-GARCH + FHS | _pending_ | _pending_ | _pending_ |
| HAR-RV + FHS | _pending_ | _pending_ | _pending_ |
| AutoARIMA | _pending_ | _pending_ | _pending_ |
| NHITS | _pending_ | _pending_ | _pending_ |
| Chronos (zero-shot) | _pending_ | _pending_ | _pending_ |

**Expected outcome, stated in advance so it cannot be retrofitted:** the GARCH family should win on volatility and distribution shape. No model, including the time-series foundation models, is expected to beat the random walk on 10-day *directional* accuracy. If that is what the data shows, it will be published as the headline result rather than buried — a rigorous public demonstration that a modern foundation model cannot call two-week gold direction is more useful than another repository claiming it can.

## Limitations

- **Horizon.** Everything here targets 5–30 trading days. Longer horizons have different dynamics and are out of scope.
- **Scenario priors are only as good as their source.** Prediction markets are thin on geopolitical questions and can be badly mispriced.
- **Factor loadings are unstable.** Gold's relationship to its drivers shifts with regime. The confidence intervals are wide for a reason — read them.
- **Retail friction varies enormously.** The defaults are representative, not universal. Enter your own dealer's actual quotes.
- **The local premium series depends on published Indian rates**, which differ across sources and carry their own reporting lag.
- **Calibration is not accuracy.** A well-calibrated model that says "55/45" is honest, not useful for timing. That distinction is the whole point.

## Roadmap

- [ ] Regime-switching volatility (Markov-switching GARCH)
- [ ] Options-implied vol surface from COMEX where available
- [ ] Sovereign gold bond and ETF friction comparison in the dashboard
- [ ] Multi-horizon term structure of forecast distributions
- [ ] Public API for the nightly artifact

## Contributing

Issues and PRs welcome. Two rules: no point forecasts, and any new model must be scored against the random-walk baseline before it merges.

## Licence

MIT — see [LICENSE](LICENSE).

---

Aurex is a research and education tool. It produces probability distributions, not advice. Short-horizon price direction is not reliably forecastable, and nothing here changes that.
