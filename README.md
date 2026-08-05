# Aurex

**A calibrated uncertainty engine for gold. Distributions, not predictions.**

[![CI](https://github.com/MadScientistX92/aurex/actions/workflows/ci.yml/badge.svg)](https://github.com/MadScientistX92/aurex/actions/workflows/ci.yml)
[![Nightly](https://github.com/MadScientistX92/aurex/actions/workflows/nightly.yml/badge.svg)](https://github.com/MadScientistX92/aurex/actions/workflows/nightly.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)

> **Build status: steps 1, 1.5, 2, 3a, 3b, 7 and most of 5 are complete.** Built: the data layer, dated tax schedules, import parity, currency lenses, the asset abstraction, the volatility and distribution engine, the scoring layer with a walk-forward backtest from 2015, routes with per-jurisdiction friction and the breakeven hurdle, the nightly job that publishes one dated forecast a night or refuses and says why, and the dashboard's three buildable views. Not built: factor attribution, the scenario engine, the benchmark shootout, and the second asset. The dashboard's *Drivers* and *Scenarios* views wait on step 4's factor loadings and are omitted rather than stubbed. Sections below are marked *(not built yet)* where that is the case — they are commitments, not claims.
>
> **Calibration has now been measured, and the headline is a negative result.** Across 2,876 out-of-sample forecasts from January 2015, the CRPS skill against the random walk is between −0.4% and +0.9% depending on the horizon, and a Diebold-Mariano test rejects at none of them — the smallest p-value in the table is 0.23. The distributions are close to the right *shape*: PIT uniformity survives a KS test at all five horizons, and a chi-square rejects at one of them. Direction carries no information at all. An earlier version of this section reported up to +4.6% CRPS skill; that number was a model carrying drift beating a null that had been denied one, it has been withdrawn, and the simulation no longer carries the drift. It would not have survived a Diebold-Mariano test either — measured, p = 0.27. Everything below is published with the test behind it, in both directions.

---

## The thesis

Search for a gold forecasting tool and you will find hundreds that output a price and an arrow. Almost none of them tell you how often they have been right, because almost none of them keep score.

Aurex takes the opposite position:

> Short-horizon price **direction** is not reliably forecastable. Short-horizon **volatility** partly is. So Aurex never predicts a price — it produces a probability distribution, and then publicly grades how well-calibrated that distribution turned out to be.

Every forecast is timestamped and committed to this repository. Once its horizon elapses, it is scored automatically — and from that moment the file may never be rewritten, which is enforced by code and by CI rather than by good intentions. The git history is the track record. Each run writes a dated copy to `public-data/forecasts/`, because `latest.json` is state and a track record needs a record. A night the engine could not price honestly produces no forecast and a failed build, never a forecast built on last week's prices.

Five rules follow, and they are enforced in code rather than promised in prose:

1. **No point forecasts anywhere.** A number without an interval or a distribution behind it is a bug. `engine/tests/test_no_overclaiming.py` fails the build if point-forecast or marketing vocabulary reaches user-facing text — including this file.
2. **Every probability gets scored.** A forecast that is never scored is marketing. The scoring layer refuses to score without a baseline, and refuses to attach a p-value to overlapping windows.
3. **The null hypothesis is the random walk, and so is the model.** Every model must beat a driftless random walk on out-of-sample CRPS. It must also *be* driftless: a simulation whose median sits above spot is a directional forecast however it got there, so the resampled residual pool is centred by default and a drift-carrying pool is a declared option rather than an accident. Models that lose ship anyway, labelled as losing. Negative results get published. *(The GARCH result is below; the wider shootout is step 6.)*
4. **Every comparison carries its test.** A skill score is a difference of two sample means, and a difference of two sample means is not a result. Each one is published with a Diebold-Mariano statistic, a p-value and a count. This binds in both directions: "+4.6%" and "worth approximately nothing" are claims about the same population, and neither ships without evidence.
5. **No overclaiming.** The credibility is the product.

A sixth rule emerged while building the data layer, and it earned its place:

6. **Never substitute a computed value for a missing observation.** `local_premium_bps` is `observed / parity − 1`. Filling a missing observation with parity would make the residual identically zero and fabricate the very signal it measures. Where there is no observation, there is no number.

## What it does today

- Resolves every input series through a **priority chain** — preferred source, fallback, cache — and records in the artifact which one actually answered
- Prices the Indian import stack from a **dated schedule**, where every duty and GST entry carries its own source URL and confidence level
- Presents one price through **currency lenses**, so the dollar and rupee views come from one engine rather than two code paths
- Fits **GJR-GARCH(1,1,1)** with Student-t innovations, alongside HAR-RV and a rolling-σ baseline, to the price and the exchange rate
- Generates return distributions by **filtered historical simulation** — no assumed parametric shape — and **keeps the paths**
- Reports **first-passage statistics**: what share of paths touch a level, how long they take, and what the survivors are holding
- Couples the two exposures with a **t-copula** so tail co-movement survives into the rupee price
- Models the **real cost of a round trip per route and jurisdiction** — dealer premium, consumption tax, buyback spread, carry — and reports the move required to break even, with every regulatory rate carrying its own source
- Scores **whether that hurdle was cleared** as one more binary event through the same reliability machinery, withholding the diagram where the positive count cannot support one
- **Grades itself**: walks the engine forward over eleven years of history, one refit per week, and scores every distribution it would have published on PIT, CRPS against the random walk, VaR coverage, and reliability — with a Diebold-Mariano test on every skill score, so a win and a shrug are held to the same standard
- **Publishes one dated forecast a night, or refuses and says why** — a declared staleness tolerance per series, a non-zero exit rather than a warning, and an index that lists the dates a forecast is missing from so an outage cannot pass for a quiet run of unscored days

## What it does not do yet

- **Factor attribution** and the crude → CPI → policy → rupee transmission chain *(step 4)*
- **Event scenarios** with probabilities sourced from prediction markets *(steps 4–5)*
- **The dashboard**, the currency toggle, and user-set leverage *(step 5)*
- **The benchmark table** against the random walk and the foundation models *(step 6)*

## What it does not do at all

- It does not tell you where gold is going.
- It does not tell you to buy or sell anything.
- It does not claim to outperform anyone, and the benchmark table will report the cases where it loses.

## Quick start

```bash
git clone https://github.com/MadScientistX92/aurex.git
cd aurex/engine
uv sync

uv run pytest                                    # 644 tests, no network
uv run aurex schedule                            # duty history with provenance
uv run aurex duty 2026-07-29                     # rate in force, and its source
uv run aurex pipeline                            # live run; refuses on stale prices
uv run aurex score --from 2015-01-01             # walk-forward backtest and calibration report
uv run aurex routes --asset gold                 # route x jurisdiction terms, with provenance
uv run aurex index                               # published forecasts, and the dates missing
uv run aurex livelog                             # score the forecasts published in real time

# offline from the committed seed cache, no network at all
AUREX_CACHE_DIR=tests/fixtures/seed-cache uv run aurex pipeline --dry-run
```

The dashboard reads the committed artifacts and nothing else:

```bash
cd ..                     # repository root; the site reads ../public-data from web/
npm --prefix web ci
npm --prefix web run dev   # http://localhost:3000
```

`aurex pipeline` exits non-zero and writes no forecast when the price series does not reach the run date — see [Nightly automation](#nightly-automation). The committed seed cache is always older than today, so exploring from it needs `--allow-stale`, which is opt-in for exactly that reason.

The working cache at `engine/.cache/` is gitignored; a fresh clone either runs once online or uses the committed seed cache shown above.

## Methodology

### Volatility

Gold's volatility clusters — turbulent days follow turbulent days — and responds asymmetrically to shocks. GJR-GARCH captures both:

```
σ²ₜ = ω + (α + γ·I[rₜ₋₁ < 0])·r²ₜ₋₁ + β·σ²ₜ₋₁
```

The `γ` term lets negative and positive shocks move volatility differently. It is **estimated, never assumed**, and reported with its standard error: for some series it is the largest term in the model and for others it is indistinguishable from zero, and which of those is true is a finding rather than an input.

| Model | Role |
|---|---|
| GJR-GARCH(1,1,1) | Student-t innovations, fitted by bounded MLE. The only one of the three where each simulated path carries its own variance trajectory |
| HAR-RV | Corsi's cascade on realised variance measured from OHLC ranges. Needs a true high and low; a close-only series is refused rather than fed squared returns |
| Rolling σ | The parameter-free baseline the others have to beat in step 6 |

Two choices are worth stating plainly:

**The fitted mean is zero unless someone insists otherwise.** A constant fitted over twenty years of returns is a drift, and a drift is a directional forecast wearing a mean's clothes. The random walk is the null, so it is also the default.

**And the simulation is driftless too, which it did not used to be.** Filtered historical simulation resamples the empirical standardised residuals, and in a sample that rose those have a positive mean — +0.0379 over gold's 5,008 residuals here, which is a drift of +0.000409 per session against a realised +0.000362. Nothing fitted a mean, but the median simulated price still sat 2.9% above spot at sixty-three sessions with P(up) = 0.61, and a distribution shaped like that is a directional forecast whatever produced it. §0 says direction is not forecastable, so the pool is now centred before it is resampled: the same fit and the same history give a median 0.2% *below* spot at sixty-three sessions and P(up) = 0.49.

The drift-carrying pool is still there, as a declared option that lands in the artifact rather than as a default nobody chose. Where it is used, the like-for-like null moves to the drift-matched random walk automatically, because a model with a drift and a null without one is not a comparison. The engine records which null is the fair one for the run that produced it.

**Fitting is break-aware.** A policy step is a mechanical jump in the price, not information about volatility. Each known discontinuity is excluded from the likelihood and enters the recursion as no shock at all, so the variance decays across a duty revision instead of spiking at it. Those residuals are also dropped from the resampling pool, because a duty revision is not a shock the bootstrap should be allowed to redraw.

### Return distribution

Aurex does **not** sample from a fitted normal or Student-t. It uses filtered historical simulation, which is standard practice on institutional risk desks:

1. Fit the volatility model, extract standardised residuals `zₜ = rₜ / σₜ`
2. Centre the pool — subtract its mean, so what gets resampled is the shape of the residual distribution and not the sample's direction
3. Block-bootstrap from that pool — contiguous blocks, drawn circularly so every residual is equally likely to appear, including the most recent ones
4. Run the model's own variance recursion forward over those shocks and walk the result into prices, one session at a time

The result inherits gold's actual fat tails and skew rather than a convenient assumption about them. This is the single most defensible technical choice in the project.

Step 2 is the one that is easy to leave out, and leaving it out is not a small error: a pool mean of +0.038 standard deviations is invisible per session and compounds into a 2.9% median displacement at a quarter. The drift policy is therefore carried by the object being resampled rather than by a keyword argument at the call site — the bootstrap refuses a bare array of residuals, and a simulation whose declared policy disagrees with the pool it actually drew from raises rather than publishing the claim.

**The paths are kept.** Every distribution here used to be *terminal*: simulate to the horizon, discard the intermediate days. That discards the event a leveraged, stopped or margined position actually meets, because such a position is closed out on the path and never experiences the terminal distribution at all. Against a barrier it can reach, the engine measures a touch probability around 1.5× the terminal one at a 21-session horizon, and the artifact reports both side by side.

That ratio is below the factor of two a continuous-monitoring argument gives, and the reason is published rather than tuned: paths are monitored at session close, so a level breached and recovered inside one session is not counted. **Every touch probability here is a floor, not an estimate.**

Where a venue caps daily moves, the cap is simulated: the session is truncated and the remainder carries into the next one, so a shock large enough to lock the market takes more than one session to arrive.

Every ensemble records its seed, its model parameters and its block length. A distribution nobody can reproduce cannot be scored.

### Joint rupee exposure

An Indian buyer holds two risks: the dollar gold price and the rupee. Modelling them independently understates joint tail events. Aurex fits a **t-copula** to the standardised residuals of both series and samples jointly, then composes the rupee path.

The family is estimated, not assumed. A Gaussian dependence structure has zero tail dependence *by construction*, however high the correlation; a t-copula does not, and the fitted degrees of freedom decide which describes a given pair. Where that fit runs to its ceiling, the artifact says so rather than implying evidence of fat tails it never found. A synchronised-resampling mode is included as the check on the parametric one — it cannot invent tail co-movement the sample never contained, and comparing the two is how you find out whether the copula is doing real work.

The underlying is simulated **once**, and each lens is a multiplication applied to those same paths, so the two currency views cannot disagree about what the metal did.

It also computes a **local premium** — the residual between the observed Indian retail rate and import parity:

```
parity_ex_gst   = XAUUSD / 31.1034768 × USDINR × (1 + duty)
local_premium_bps = (observed_ibja / parity_ex_gst − 1) × 10⁴
```

Three things make that residual mean what it says.

**The tax stack is a dated schedule, not a constant.** GST did not exist before 1 July 2017 and the import duty has moved ten times, most recently to 15% on 13 May 2026. Every entry in `engine/aurex/data/schedules/duty.yaml` carries its own `source_url` and `source_confidence` (`primary` | `secondary`); no entry inherits a table-level default, and a schema test fails the build if either field is missing.

**The comparison is like-for-like.** IBJA publishes its 999 rate *exclusive* of GST, so the premium is measured against `parity_ex_gst`. Measuring against the GST-inclusive figure would print a spurious ≈ −291bps at all times.

**Parity uses spot, not futures.** `GC=F` is the COMEX front-month *future* and carries a cost-of-carry basis over spot — measured at +2.40% against the London PM fix on 2026-07-29. Sourcing parity from it pushes that basis into the premium, and because the basis moves with rates and time to expiry it would look like a moving domestic-demand signal. Parity uses the London PM fix, the same benchmark IBJA prints in its own daily report. Futures are still loaded, as `xau_futures`, because the realised-volatility estimators want true OHLC.

**A lens must be arithmetic, not policy.** A currency view is valid where the buyer's-currency price is a *mechanical* function of the quote — an exchange rate, a unit conversion, a statutory rate, a published settlement formula — and invalid where that price is administered. An exchange-settled contract quoted in rupees converts by formula and is a legitimate lens; a policy-set retail price is not, because presenting it as a view on the world price asserts a passthrough nobody has measured. Lenses declare which they are, so the rule is enforced rather than remembered.

### Structural breaks are recorded, not smoothed

`policy_breaks.yaml` lists every known discontinuity — each duty revision, the 2017 GST rollout, the 2013–2014 80:20 import-linkage rule, demonetisation — and the pipeline emits them in the artifact, so a policy step is never mistaken for market noise.

Whether retail prices actually follow a duty change is a **measurement**, not an assumption. The pipeline reports a passthrough diagnostic either side of each break and takes no position on what it should be. This distinction is load-bearing in the test suite: the wiring test asserts the mechanical step in *parity*, because under complete passthrough the *premium* does not move at all — an assertion there would fail on correct code.

### Factor attribution *(not built yet — step 4)*

An elastic-net regression of weekly returns on real yields, the dollar index, crude, VIX, ETF flows, and lagged momentum, on a rolling three-year window.

**This is for attribution and scenario propagation, never for directional forecasting.** Loadings will be published with bootstrap confidence intervals, and out-of-sample R² reported honestly — which usually means "close to zero". That is the truthful answer, and hiding it would defeat the point of the project.

Step 4 also carries the India transmission chain: crude → import bill → CPI and the current account → policy response → the rupee → the rupee gold price. Every arrow is estimated by local projections, lagged, and uncertain; uncertainty compounds multiplicatively across the links, so the compounded band is what gets displayed. If it spans zero at every horizon, that is the finding and it will be stated as one.

### Scenario engine *(not built yet — steps 4–5)*

You cannot forecast a ceasefire. You can enumerate outcomes, weight them, and propagate them — which is what risk desks actually do.

Each axis (geopolitics, CPI, payrolls) carries branches with a shock vector over the factors. Branch probabilities come from **prediction markets and rate futures** — Polymarket, Kalshi, Metaculus, CME FedWatch — and every probability records its `prior_source` in the output artifact. Where no market exists, the historical base rate of surprise direction is used and labelled as such.

**Design constraint:** the scenario engine is built to *widen* the distribution, not to shift its centre. Any directional tilt must be traceable to a market-implied probability. A confident tilt with no market behind it is treated as a bug.

### Trade layer

```
cost      = grams × spot × (1 + premium) × (1 + gst)
proceeds  = grams × spot_T × (1 − buyback)
breakeven = (1 + premium)(1 + gst) / (1 − buyback)
```

**No jurisdiction is the default**, so there is no single breakeven number. The same metal reached by a different route, or held in a different country, faces a different hurdle — which is the whole of §20 and the reason the table below has rows rather than a headline. Leaving the jurisdiction unset is a supported state: it gives the quote-currency benchmark with friction *excluded and labelled*, never one country's tax stack applied silently.

<!-- BEGIN GENERATED breakeven-table -->
| Route | 5 sessions | 21 sessions | 63 sessions | 252 sessions | Accrues |
|---|---|---|---|---|---|
| cfd — United Kingdom | 0.14% | 0.27% | 0.62% | 2.17% | yes |
| physical — Germany | 5.10% | 5.10% | 5.10% | 5.10% | no |
| physical — United Kingdom | 5.10% | 5.10% | 5.10% | 5.10% | no |
| physical — India | 9.37% | 9.37% | 9.37% | 9.37% | no |
| physical — United States | 5.10% | 5.10% | 5.10% | 5.10% | no |
<!-- END GENERATED breakeven-table -->

This table is **generated from `engine/aurex/data/schedules/routes.yaml`** and a test regenerates it and compares, so it cannot drift from the data behind it. Reproduce with `uv run aurex routes --markdown`. Three of the physical rows agree only because their representative dealer spreads happen to be the same and all three jurisdictions exempt investment gold; the tax rates behind them are cited separately and independently.

Friction takes a **horizon**, because the two shapes are structurally different: physical friction is paid at the door, while carry friction accrues — visible above as the only row whose numbers move across the columns. The `Accrues` column exists so that is readable rather than inferred.

This is why the project exists. The friction is deterministic and knowable; the price move is not. Most retail tools model the uncertain part and ignore the certain one. At retail physical parameters the hurdle is +9.4%, against a two-week one-standard-deviation move of roughly ±3.2% — close to a **three**-sigma requirement in one specific direction. What that does to a probability forecast is measured below rather than asserted, and the sigma behind that sentence is this sample's rather than a remembered one: gold's daily standard deviation over 2015–2026 is 1.00%, which annualises to 15.9%.

## Asset abstraction

Everything asset-specific lives in `engine/aurex/assets/`. Nothing in `vol/`, `dist/`, `factors/`, `scenarios/`, `trade/` or `score/` may name an asset, and neither may `pipeline.py` or `forecast.py`, which compose assets with the model layer and must know what a lens *is* without knowing which one they are holding.

Two tests enforce it: a synthetic asset runs the entire pipeline end to end, and a static guard fails the build on an asset literal in a guarded module. The behavioural test catches leaks that change something; the static one catches leaks that have not broken anything yet.

**Return transforms are an internal representation.** Crude settled negative in April 2020, so log returns need a shift — and a shifted-log transform silently rescales anything quoted as a percentage from transform space. On the WTI series Aurex already caches, the annualised standard deviation reads 55.2% at a shift of 50 and 24.6% at a shift of 100. The number more than halves on a constant that is arbitrary by construction. So every reported quantity is mapped back to price space first, and the transforms and volatility modules deliberately expose no volatility, quantile or annualisation helper: there is nothing there to accidentally report. A round-trip test cannot catch this — a badly-scaled transform round-trips perfectly.

## Data sources

No API key is required to run Aurex. `FRED_API_KEY` is used if set and ignored otherwise.

| Series | Source | Notes |
|---|---|---|
| XAU/USD (spot) | LBMA London PM fix | Daily to 1968. Close-only; the parity basis |
| XAU futures | `yfinance` `GC=F` | OHLC, for volatility work. Never for parity |
| USD/INR | `yfinance` `INR=X` → FRED `DEXINUS` | |
| VIX | `yfinance` `^VIX` → FRED `VIXCLS` | |
| 10y TIPS real yield | FRED `DFII10` | |
| Dollar index | FRED `DTWEXBGS` | |
| WTI crude | FRED `DCOILWTICO` | |
| India 24K + ETF flow | IBJA daily bullion report (PDF) | 999 AM/PM rates, SPDR tonnes, London fix |

Each series resolves through a priority chain, and the artifact records which source actually answered, so provenance is never implied. Yahoo rate-limits aggressively, which is why nothing depends on it alone.

IBJA's daily PDF replaced two dead ends: their homepage rate block is rendered client-side, and SPDR's published `.csv` endpoint now serves a PDF. The report also supplies SPDR tonnes, a better ETF-flow proxy than shares outstanding.

## Calibration

A forecast that is never scored is marketing. Aurex grades itself on:

| Metric | Question it answers | Step |
|---|---|---|
| PIT histogram | Are the predicted distributions the right *shape*? Uniform = calibrated | 3a |
| CRPS skill score | Is the full distribution better than a driftless random walk? | 3a |
| Diebold-Mariano | Is that skill score distinguishable from zero, or is it noise? | 3a |
| Kupiec / Christoffersen | Do 95% and 99% VaR breaches occur at the right rate, and independently? | 3a |
| PIT chi-square | Same question as KS, but sensitive to mass piled in one bin rather than to a shift | 3a |
| Brier score + reliability | When it says 20%, does it happen 20% of the time? | 3a |
| …on clearing the breakeven hurdle | Same question, asked of the one event friction defines | 3b |

**These score the price distribution, not a position.** None of them needs a venue, a route or a jurisdiction: the binary events are direction and touching a barrier, and both are properties of the metal. The single probability that *is* defined by friction — whether a move clears breakeven — is not a different metric, it is one more event fed to the same reliability machinery, so it waits for the routes that decide which friction applies.

### Results

Walk-forward, expanding window, no lookahead. One refit every five sessions, 4,000 paths per forecast, GJR-GARCH(1,1,1) with Student-t innovations and filtered historical simulation over a centred residual pool. 2,876 scored forecasts, no skipped dates.

**The sample is bounded at both ends, and the command below is the one that produced every number in this section:**

```bash
uv run aurex score --asset gold --from 2015-01-01 --to 2026-07-29 --step 5 --horizons 5,10,21,42,63 --paths 4000
```

`--to` matters more than it looks. Without it the run ends at whatever the price series had reached on the day it was typed, so the sample depends on the clock and a reader re-running the command gets a different one — a table nobody else can reproduce is not a published result. The bound truncates the price series rather than filtering the forecast dates, so it holds back the *outcomes* too: a window that would run past the stated end is dropped rather than scored against data the declared sample says is not there. It defaults to the series' own last observation, never to today, and `calibration-gold.json` records the window it resolved to alongside the command that reproduces it.

The 2,876 forecasts here span **2 January 2015 to 21 July 2026** as-of dates, over a price series running to 29 July 2026 — the gap at the end is the longest horizon needing somewhere to land.

The null is the driftless random walk §0 names: iid empirical increments with the sample mean removed. It is also the like-for-like null, because the model is now driftless too — which is the whole reason this table can be read as a measurement of the volatility layer.

| Horizon | Forecasts | Independent | PIT KS p | PIT χ² p | CRPS skill | DM p (HAC / thinned) | 95% VaR breaches |
|---|---|---|---|---|---|---|---|
| 5 | 580 | 580 | 0.15 | **0.011** | +0.6% | 0.42 / 0.42 | 19 vs 29.0 (p = 0.043) |
| 10 | 579 | 290 | 0.07 | 0.21 | −0.4% | 0.60 / 0.41 | 9 vs 14.5 (p = 0.11) |
| 21 | 577 | 116 | 0.29 | 0.24 | −0.1% | 0.90 / 0.59 | 2 vs 5.8 (p = 0.062) |
| 42 | 572 | 64 | 0.08 | 0.58 | +0.3% | 0.75 / 0.70 | 0 vs 3.2 (p = 0.078) |
| 63 | 568 | 44 | 0.14 | 0.06 | +0.9% | 0.38 / 0.23 | 0 vs 2.2 (p = 0.17) |

**Conditioning on volatility is not measurably worth anything at these horizons, and that sentence now has a test behind it.** The skill score wanders between −0.4% and +0.9% with no sign pattern, and Diebold-Mariano rejects at none of the five horizons on either the overlapping sample or the thinned one; the smallest p-value anywhere in the table is 0.23. This is not a demonstration that the volatility layer is worth zero. It is a demonstration that eleven years of weekly forecasts cannot tell the difference between it and a random walk, which is a weaker claim and the only one the data supports.

**The shape does not decay either, and theory said it should.** Conditional variance mean-reverts to unconditional, so whatever the model knows about next week it should know less about next quarter, and skill should fall with the horizon. Regressing skill on log horizon and bootstrapping the slope over the shared as-of dates in blocks — the five horizons come from the same forecasts on the same days, so they are not five independent observations of anything — gives a slope of **+0.0015, p = 0.73, R² = 0.08**, with a 95% interval of [−0.0071, +0.0100]. No decay, no trend, and the interval is wide enough to contain a decline of 2.5 percentage points of skill across the horizon range. So the honest reading is not "helps at a week and washes out beyond": it is that nothing is distinguishable from zero anywhere, and the test has no power to rule out a decay of the size theory predicts.

**The distributions are close to the right shape, with one rejection.** KS does not reject at any horizon. The chi-square rejects at five sessions (p = 0.011) and is marginal at sixty-three (p = 0.06). The two are run together because they fail on different things — KS reads the largest gap in the cumulative distribution and is weak against mass piled in one bin — and here the chi-square is reading the displacement described below. Worth noting *where* it rejects: the displacement is smallest at five sessions and largest at sixty-three, so the test rejects where the effect is weakest and the sample is 580 rather than 44. That is a statement about power, not about which horizon is worst calibrated.

**The 95% breach count is now short at every horizon**, and short enough to reject at five sessions (19 against 29.0 expected, p = 0.043). It is the same displacement: a distribution centred at spot, scored over a sample that rose, has its lower tail pulled away from the observations. Displacement accounts for about half of it and no more — see below.

**The direction forecast carries no information, as predicted in advance.** "Ends higher" now comes back at 0.509 to 0.512 at every horizon, against realised rates from 0.533 at a week to 0.644 at a quarter. Its resolution is 0.002 or less everywhere: it cannot tell one week from another, which is the point. Its *level* used to look better than that. The drift in the residual pool was pushing the long-horizon forecast up toward the rate the sample actually realised — 0.5915 against a realised 0.6444 at a quarter — and the Brier score collected the benefit. Removing the drift removed that, and the numbers move accordingly:

| At 63 sessions | Drift-carrying pool | Centred pool |
|---|---|---|
| Forecast "ends higher" | 0.5915 | 0.5090 |
| Brier score | 0.23412 | 0.24851 |
| Reliability term (lower is better) | 0.00464 | 0.02093 |
| Resolution | 0.00093 | 0.00200 |

The Brier score is now worse than a constant forecast of the sample's own base rate at every horizon, where before it was better at 42 and 63 sessions. That is the right trade and it is worth being blunt about why: the old number came from a directional bias nobody chose, which happened to point the way the sample went for eleven years. Resolution — the part that would represent actual skill — was nil then and is nil now.

### The test that decides whether a skill score is a finding

Every skill score above is a difference between two sample means, taken over windows that overlap by construction, and it needs a test before it means anything in either direction. The one used here is Diebold-Mariano on the CRPS loss differential `d_t = crps_model − crps_null`:

- **HAC/Newey-West variance with Bartlett weights.** Bartlett rather than the rectangular weights of the original paper, because rectangular weights can return a negative variance estimate — a number no p-value can be built from, arriving exactly when the data are least cooperative.
- **The truncation lag is the overlap measured in records.** The textbook rule for an *h*-step forecast is `h − 1`, and that rule assumes a forecast made every session. These are made every five, so an *h*-session window overlaps the previous `⌈h/5⌉ − 1` records — twelve at a quarter, not sixty-two. The lag is derived from the sampling scheme rather than defaulted, and at a step of one it reduces to `h − 1` exactly.
- **Harvey-Leybourne-Newbold, always.** The uncorrected statistic is asymptotically normal and over-rejects badly at 44 independent windows. There is no uncorrected variant exposed anywhere, because there is no circumstance here where it is the number to read.
- **Run twice.** Once on every record with the HAC variance, once on the thinned non-overlapping subsample, consistent with the discipline every other p-value in the repository follows. On the thinned series the correction reduces *algebraically* to a paired t-test, which is asserted in the test suite rather than checked approximately. Agreement between the two is evidence the lag was long enough; disagreement would be a finding about the dependence rather than a result to choose between.

Diebold-Mariano is the one test in the scoring layer allowed to see overlapping windows. It earns that by modelling the dependence instead of ignoring it: it takes the same declared sampling every other test takes and turns it into a truncation lag, rather than raising.

**What the overlap correction is worth, measured on this run.** Handing the same 568 quarterly differentials to a test that assumed they were independent doubles the statistic, from −0.87 to −1.76, and takes the p-value from 0.38 to 0.079. That is the difference between "no evidence" and a number a reader would be tempted to call marginal, produced entirely by counting 44 independent windows as 568. The correction does nothing at five sessions, where the windows genuinely do not overlap, and grows monotonically with the horizon — which is what it should do, and is the reason the lag is derived from the sampling rather than set by hand.

**The test would have caught the withdrawn number on its own.** Running the old drift-carrying configuration through this machinery — the option is still there, so this is a measurement rather than a reconstruction — reproduces +4.6% CRPS skill at sixty-three sessions and attaches **p = 0.27** to it. So that result was never significant either, and a Diebold-Mariano column would have stopped it being published as a headline regardless of whether anyone had noticed where the drift came from. Two independent guards would each have been enough on their own; the repository had neither.

### The second null, and what it costs to be driftless

The drift-matched random walk still runs alongside, and with a driftless model it has changed meaning. It is now a **hindsight benchmark**: a walk handed the sample mean it could not have known in advance. Losing to it is therefore not a finding about the model, and beating it would be a strong one.

| Horizon | Skill vs drift-matched | DM p |
|---|---|---|
| 5 | +0.1% | 0.88 |
| 10 | −1.0% | 0.38 |
| 21 | −1.4% | 0.48 |
| 42 | −2.2% | 0.50 |
| 63 | −4.0% | 0.38 |

The gap widens with the horizon and none of it is distinguishable from zero either. Read it as the size of the drift rather than as a verdict: it is the price, in CRPS, of refusing to forecast a direction over a sample that went up. The previous version of this README was collecting that price as skill.

### One mechanism, named

Three things looked like separate failures — a heavy top PIT bin, a direction forecast whose level slips with horizon, and breaches that go missing. The first two are one effect, and it has a shape that could have failed to appear.

A model centred at `mu_model` scored over a sample that drifted at `mu_sample` is displaced in standardised units by `(mu_sample − mu_model) / sigma × sqrt(h)`: linear in the horizon over a spread that grows as its square root. Regressing mean PIT on `sqrt(h)` across the five horizons, with the intercept pinned at 0.5 because a sample with no displacement must give one half everywhere, gives **R² = 0.995** on one parameter, implying a standardised drift of 0.0412 per session. The law held, and it fits better now than it did before the pool was centred — which is what should happen, because `mu_model` is now exactly zero instead of approximately the sample's own.

The two symptoms are the same displacement read from different reference points. Mean PIT (0.5209 → 0.5980) measures it against the forecast's own centre. The direction gap measures it against *spot*, which is why it grows faster.

**Breaches are half of this and half something else.** The fitted displacement predicts breach counts of 23.9, 11.0, 3.9, 1.8 and 1.1 against nominal 29.0, 14.5, 5.8, 3.2 and 2.2. Observed were 19, 9, 2, 0 and 0. So displacement accounts for roughly half the deficit at every horizon and the remainder is a lower tail that is too wide — one finding, not two, and the second is smaller than the first but does not go away.

**This is the cost of the settlement, stated plainly.** Centring the pool made the displacement larger, not smaller: the model used to carry very nearly the sample's own drift, so its centre tracked the sample and the PIT sat near one half. Now the centre is fixed at spot and the sample walked away from it, so the displacement is the full drift and every symptom of it is bigger — a chi-square rejection at five sessions, a breach deficit that rejects at five sessions, a long-horizon Brier score worse than the base rate. All of that was previously hidden by a directional bias nobody chose, and a well-calibrated-looking histogram bought that way is not worth having.

**None of this is an argument for fitting a drift.** A fitted mean over a rising sample is a directional forecast wearing a mean's clothes. The displacement is measured and published; it is not tuned away.

### Pre-registered: what the hurdle event was expected to do

Step 3b adds one event to the machinery above — did the realised move clear the round-trip breakeven a route and jurisdiction define. This section was written **before** that event existed, because its failure mode was predictable and a limitation discovered after the fact reads like an excuse. It is left exactly as written; the measured outcome follows it.

**The arithmetic.** At Indian retail parameters the hurdle is around +9.4%. A one-standard-deviation move at ten sessions is roughly ±4.4%, so the hurdle is about 2.1 sigma in one specific direction. That puts the base rate somewhere near 2–3%, and over roughly 580 walk-forward windows that is on the order of **15 positive events**.

**What that means for the score.** Fifteen positives cannot support a ten-bin reliability diagram. The Brier score will be dominated by its uncertainty term — a base rate of 0.025 has an uncertainty of 0.024, and any forecaster that says "probably not" every time will score close to that. Resolution will be unmeasurable at short horizons, and a reliability curve drawn on those counts would be a picture of sampling noise with an axis on it.

**So the following are fixed now, not chosen after seeing the numbers:**

- Every hurdle Brier score is reported with its **base rate and its positive-event count** beside it. A Brier score of 0.02 that looks excellent next to direction's 0.25 is measuring a rare event, not a better forecast, and the count is what makes that visible.
- The reliability curve is **withheld** where the positive count cannot support one. The threshold is **10 positive events**, chosen to match the `MIN_EXPECTED_BREACHES` rule of thumb the coverage tests already use rather than tuned to what this event happens to produce. Below it the score, base rate and count are still published; the diagram is not, and the artifact says why.
- Resolution and reliability terms are still computed and published where the curve is withheld, because they are functions of the score rather than of the diagram — but a withheld curve is the signal that neither should be read as a measurement.

**The comparison is the point, and it is also pre-registered.** Different jurisdictions face different hurdles on the same metal and the same distribution. If a low-friction route clears its hurdle often enough to score and a high-friction one does not, that difference is not an inconvenience in the reporting — it *is* the finding, and it is the project's thesis stated in event counts rather than in prose. The expectation stated in advance: the hurdle rises with friction, the base rate falls with it, and at the top of the friction range the event becomes unmeasurable on eleven years of data. If that is what happens, the honest output is a table of base rates and counts with most of the reliability column empty, and that table is the result rather than a failure to produce one.

### The hurdle result, against that prediction

Measured over the same 2,876 forecasts, with the hurdle for each route and jurisdiction generated from the routes table. Counts are positive events — times the realised move cleared the round-trip breakeven — out of 580, 579, 577, 572 and 568 windows.

| Route (hurdle) | 5 | 10 | 21 | 42 | 63 |
|---|---|---|---|---|---|
| CFD, United Kingdom (0.14% → 0.62%, accrues) | 293 | 303 | 307 | 312 | 353 |
| Physical, United States / United Kingdom / Germany (5.10%) | **11** | 37 | 97 | 177 | 213 |
| Physical, India (9.37%) | **0** | **5** | 22 | 70 | 113 |

**The prediction was right in direction and optimistic in magnitude.** It said the base rate would be near 2–3% and that 580 windows would yield about fifteen positive events at ten sessions. The measured base rate at ten sessions is **0.86%** and the count is **five** — about a third of what was predicted, so the event is rarer and less measurable than the pre-registration allowed for. At five sessions the retail hurdle was cleared **zero** times in eleven years. The withholding rule fired exactly where it was designed to: the curve is withheld at five and ten sessions and drawn from twenty-one onwards, and no diagram was ever drawn on counts that could not support one.

**The prediction missed because the volatility input was wrong, and that is worth naming precisely.** The pre-registered arithmetic used ±4.4% for a ten-session one-standard-deviation move, which came from 22% annualised — a crisis-period figure, not this sample's. Gold's 2015–2026 daily standard deviation is **1.00%**, so ten-session sigma is **3.16%**, not 4.4%. The 9.37% hurdle is therefore about **2.97 sigma** in simple-return terms rather than the 2.1 that was written down, and the pre-registration's "2–3%, so about fifteen events" followed from the wrong denominator rather than from a wrong method.

Being consistent about *which* sigma matters here, because the difference is the whole point of the next paragraph. The engine simulates log returns, so the like-for-like comparison puts the hurdle in log space too: `ln(1.0937) / 0.0316` = **2.83 sigma**, and a driftless Gaussian at 2.83 sigma gives **0.23%**. The simple-return reading gives 2.97 sigma and 0.15%. Both are far below the measured 0.86% — the realised rate is roughly four times the Gaussian one on the like-for-like figure and nearly six times on the simple-return one — and the log-space number is the one to read against this engine, because it is the only one measured on the same scale the model works in.

**The residual gap is the fat tail, which is exactly what filtered historical simulation exists to preserve.** Gold's daily returns over this sample carry an excess kurtosis of 4.9 and a skew of −0.38. A Gaussian assumption discards both by construction; resampling the empirical standardised residuals keeps them. So the question worth asking is not whether the engine is non-Gaussian — it is by construction — but whether its tail is the *right* shape. That has an answer, and it is the model's own forecast probability against the realised rate:

| Hurdle | Horizon | Driftless Gaussian | **Aurex forecast** | Realised |
|---|---|---|---|---|
| 5.10% | 5 | 0.0131 | **0.0174** | 0.0190 |
| 5.10% | 10 | 0.0579 | **0.0528** | 0.0639 |
| 5.10% | 21 | 0.1388 | **0.1212** | 0.1681 |
| 5.10% | 63 | 0.2654 | **0.2500** | 0.3750 |
| 9.37% | 10 | 0.0023 | **0.0079** | 0.0086 |
| 9.37% | 21 | 0.0253 | **0.0290** | 0.0381 |
| 9.37% | 63 | 0.1296 | **0.1177** | 0.1989 |

**Where the tail is what decides the answer, it is close to the right shape.** The sharpest case is the 9.37% hurdle at ten sessions — 2.83 sigma, the deepest event in this table with a non-zero count. A driftless Gaussian forecasts **0.23%**. Aurex forecasts **0.79%**. It happened at **0.86%**. The empirical tail closes roughly **nine tenths** of the Gaussian's shortfall on the one event where tail shape is nearly the whole answer.

The pattern repeats wherever the tail dominates. At five sessions against the 5.10% hurdle, at 2.2 sigma: Gaussian 1.31%, Aurex 1.74%, realised 1.90%. At twenty-one sessions against 9.37%, at 1.95 sigma: Gaussian 2.53%, Aurex 2.90%, realised 3.81%. In every case Aurex sits between the Gaussian and the truth, and on the correct side of the Gaussian.

This is the clearest evidence in the repository that filtered historical simulation is doing real work, and it was invisible until step 5 — the mean forecast was being withheld along with the reliability diagram, so the two rows that carry the sharpest result were the two rows showing nothing.

**Where the horizon is long, the engine falls *below* the Gaussian, and that is the drift displacement rather than the tail.** At sixty-three sessions the hurdle is barely a tail event at all — 0.63 sigma at 5.10% — so tail shape stops mattering and location takes over. A model centred at spot, scored over a sample that rose, under-forecasts every upside event, and the gap widens with the horizon exactly as [the displacement law above](#one-mechanism-named) predicts. That is the same finding appearing in a third place, not a new one, and it is the price of refusing to forecast a direction rather than a defect in the distribution's shape.

**Only the zero-count row cannot be read.** At five sessions the India event has no positives at all, so there is nothing to compare a forecast against — Aurex says 0.17%, which over 580 windows is about one expected event, and observing none is entirely consistent with that. Everything else in the table is a live comparison, including the two rows that used to be blank: the reliability *curve* is still withheld below ten positives, but `mean_forecast` is a scalar over every forecast rather than a ten-bin diagram, and withholding it with the curve was a reporting bug rather than a finding.

**The comparison the pre-registration named is the finding, and it is not the one the counts first suggest.** On the same metal, the same distribution and the same 580 days, the CFD route clears its hurdle 293 times at five sessions and the retail Indian route clears it zero times. It is tempting to read 293 as a score. It is not one. At five sessions the CFD hurdle is 0.14%, its base rate is **0.5052**, and the event has collapsed into "did gold end higher" — which the same run shows has a resolution of 0.002 or less at every horizon. So 293 clears are **293 coin flips landing heads**, not 293 wins.

The correct reading is about the hurdle, not about the route: **at low friction the hurdle is irrelevant, and at high friction it is decisive.** At 0.14% the breakeven test tells you nothing you did not already know from the sign of the move; at 9.37% it is the entire question, and eleven years of five-session windows never answered it once. Nobody succeeded at anything in this table. What changed between the two rows is not skill and not the forecast — it is what the holder had to pay to get in and out, which was knowable in advance and is the only part of this that was.

**Resolution is nil at every friction level, which is the same negative result as direction.** The largest resolution term anywhere in the table is 0.002. Where the event is near-even the Brier score sits at its uncertainty term; where it is rare the Brier score is small because the event is rare. The model cannot tell a window in which the hurdle will be cleared from one in which it will not, at any hurdle, at any horizon. This is what makes the friction comparison above a statement about arithmetic rather than about forecasting.

**Read the Brier scores down a column, never across one.** The India row at five sessions has a Brier score of 0.00004 and the CFD row has 0.25034. The first is not six thousand times better: it is a forecast of an event with a base rate of zero. Every hurdle score is published with its base rate and its positive count for exactly this reason, and the artifact says so in the same block.

### Two conventions fixed in advance, because both would otherwise look like results

**A realised touch is measured at session close**, the same convention the forecast uses. Scoring simulated closes against intraday extremes would charge the model for a floor it already declares. The enforcement is structural: the object carrying a realised outcome holds closes and nothing else, so there is no high or low available to score against.

**Overlapping windows are not independent observations.** Sampling a 21-session horizon weekly makes consecutive scores share three quarters of their path, which is fine for a PIT histogram or a mean CRPS and fatal for a breach-independence test. Every p-value is therefore computed on the thinned subsample, and the function that computes one *raises* on an overlapping series rather than quietly returning a number — 568 forecasts at a 63-session horizon are 44 observations, and a test told otherwise reports a confidence it has not earned. Diebold-Mariano is the single exception, and it is granted the exception because it models the overlap in its variance estimator instead of assuming it away; it is still reported on the thinned subsample too.

**At zero breaches the chi-square approximation is not usable.** Kupiec's statistic is asymptotically chi-square, and zero breaches puts the unrestricted estimate on the boundary of the parameter space, where that asymptotics does not hold. Both p-values are always computed and the exact binomial is the one reported at a boundary. It matters: 0 breaches in 44 windows against a 5% quantile gives a chi-square p of 0.034 and an exact p of 0.17. An earlier version of the table above reported the first and read it as the run's only rejection. It was not one.

## The dashboard

Next.js 15, TypeScript, statically prerendered on Vercel. It reads the committed JSON in `public-data/` and nothing else: no model on the server, no runtime fetch, no API key in the browser. The consequence is the point — **there is no code path that could compute a fresher number, so there is no code path that could invent one.** A stale deploy is visibly stale rather than quietly wrong, and the nightly commit is what triggers the next build.

### A dashboard is where honest uncertainty goes to die

Every interface convention pushes toward one number, one arrow, one confident headline. This engine's entire finding is that it cannot produce one. So the resistance is structural rather than tasteful — the things that would let a page overclaim do not exist to reach for:

- **The hero is the distribution.** There is no hero-number class in the stylesheet, no trend arrow, no success colour. The largest element on the page is a fan chart; the figures beside it sit in a definition list at body size, so none can become the headline by typography alone.
- **The hurdle is a required prop, not an option.** `FanChart` and `ExceedanceChart` will not compile without one, so a forecast cannot be rendered without the move it has to beat drawn across it and labelled with the number. On the exceedance chart the hurdle is a vertical rule, and where it crosses the curve *is* the probability of clearing it — a geometric fact rather than a figure quoted beside a picture.
- **Below even odds, the headline is the loss.** `P(profit) = 0.089` and `P(loss) = 0.911` are the same measurement; the first reads as an opportunity and the second reads as what it is. The rule already existed in the engine; the interface is where it bites.
- **No jurisdiction is the default.** The control opens unset, and unset is a real state: the quote-currency benchmark with friction *excluded and labelled*, never one country's tax stack applied because it sorted first.
- **The negative result is on the front page**, above the distribution it qualifies — not filed under methodology where a reader will not go.

### What is there, and what is deliberately not

| View | Status |
|---|---|
| **Today** | The distribution across horizons, the exceedance curve against breakeven, and every route × jurisdiction hurdle with its odds |
| **Track record** | Sample window and the command that reproduces it, CRPS skill with its Diebold-Mariano p-values, PIT histograms, reliability diagrams, and the live log |
| **Calculator** | Grams, route, jurisdiction → the specific breakeven, the gross gain needed, and the published odds of clearing it |
| ~~Drivers~~ | **Omitted.** Needs step 4's factor loadings |
| ~~Scenarios~~ | **Omitted.** Needs step 4 |

The last two are absent rather than stubbed. An empty tab implies the work is done and merely unpopulated; naming them as missing costs a line and says something true.

**The empty state is the content.** The live log has `n = 0` today and will be in single digits for months. It renders the count and *"no test is possible yet"* rather than hiding until it looks like something — a visitor who sees `n = 3` beside that label learns more about how this repository works than one who sees a section that quietly does not exist.

### Rules the dashboard inherits

- **The leak guard extends to `web/`.** No asset literal, no jurisdiction code, in `.ts`, `.tsx` or `.css`. The site iterates over whatever the artifacts declare; a view that special-cases one asset stops being a view of the engine and becomes a second implementation of it, with its own copy of the tax stack drifting from the schedule that has the citations. It caught a driver named in a caption during the build.
- **Nothing is recomputed.** Probabilities come from a committed exceedance grid by lookup — 101 points at half-percent steps — and breakevens come from the routes artifact. A round-trip hurdle lands wherever a dealer spread and a tax rate put it, almost never on a published quantile, and interpolating a five-point grid in a browser would put invented precision in front of the one number a reader would act on.
- **Every chart carries a table.** Not a nicety: the `Figure` component cannot render without one. Charts are unreadable to a screen reader, in forced-colors mode, and to anyone the light-mode aqua fails contrast for.
- **Monospace tabular figures on every numeral**, because these tables are meant to be read down a column.
- **Accessibility: 100/100 on Lighthouse** for all three views at a 412×823 mobile viewport, with no failing audits. Visible keyboard focus, one orchestrated entrance that `prefers-reduced-motion` removes, and colour never the sole encoding — the hurdle is dashed *and* labelled, the legend is always present, and both themes are stepped and validated rather than flipped.

## Nightly automation

One forecast a night, committed to this repository, or no forecast and a loud failure. There is no third outcome, and the reason is the whole of this section.

### The failure mode this is built around

A nightly runner is a fresh machine with no cache. Yahoo rate-limits, IBJA does not publish on Indian holidays, and the source chain is deliberately built to fall back rather than fail — which is right for a human at a terminal and dangerous for an unattended job. Combine the three and you get a run that quietly resolves week-old prices, simulates from them, and commits a forecast dated today.

That is fabrication with a timestamp on it, and it is the exact class of error this repository exists to catch. It is also invisible afterwards: it has the same shape, the same fields and the same date as a real forecast. **A missing night is a visible hole in the track record. A fabricated night is not visible at all**, and every claim in §0 rests on the dated log being what the engine actually said on the day it is dated.

So the job **refuses to publish**, exits non-zero, and writes nothing but a record of the refusal:

| Series | Blocking | Tolerance | Calendar |
|---|---|---|---|
| `xauusd` (LBMA PM fix) | **yes** | 4 days | London business days |
| `usdinr` | **yes** | 4 days | FX trading days |
| `xau_futures` | no | 4 days | COMEX trading days |
| `ibja_gold` | no | 6 days | Indian business days |
| `real_yield_10y`, `vix` | no | 5 days | US business days |
| `wti` | no | 7 days | US business days, in arrears |
| `dxy` | no | 10 days | US business days, in arrears |

Every tolerance is declared beside the loaders it describes, carries the publication calendar it was derived from and a written rationale, and is cited in the artifact of any forecast that passed it. Four days is the binding number and it is set by the worst *ordinary* case rather than the average one: a 02:00 UTC run on the Tuesday after a Monday holiday reads Friday's fix, which is four days behind and entirely healthy. A guard that fired there would be switched off within a week.

Three properties are worth stating because each was a decision:

- **Only the price series and a published lens's exchange rate block.** Those are the inputs a fabricated price would come from. A stale factor moves a fitted loading, which is a smaller and different problem — and `dxy` runs about a week behind by FRED's own schedule in the ordinary case, so blocking on it would mean refusing most healthy nights. Non-blocking series are still measured and reported.
- **An undeclared tolerance blocks too.** A series added without a policy would otherwise inherit "anything goes" and the guard would degrade one series at a time with nothing failing. A test asserts every registered series declares one.
- **Freshness is measured on the price column, not the index.** A frame whose dates reach today over a NaN close is what a partial fetch leaves behind, and judging it on the index alone would pass exactly the case the guard exists for.

A long market closure exceeds the tolerance and is refused. That is intended: there is no new price, so there is nothing new to publish, and the gap is recorded rather than papered over.

### Track record integrity

**A forecast whose horizon has elapsed may never be rewritten.** Before it elapses, a rerun is a correction to a live forecast and git carries both versions. After it elapses the outcome exists, and rewriting the forecast that preceded it is editing the past. The rule is enforced in code rather than by discipline, in two places: the writer refuses, and CI refuses a diff that modifies a settled file — the second being the stronger of the two, because it catches an edit made by any means including by hand.

Two details decide whether that rule means anything:

- **The clock starts at the anchor, not at the run time.** A Monday run prices from Friday's fix, so the horizon started on Friday. Dating the freeze from `generated_at` would start the clock late and leave a forecast rewritable after its horizon had actually run.
- **The shortest horizon freezes the file, not the longest.** Once any horizon has an outcome, rewriting the file revises a claim whose result is known — even though the quarterly horizon beside it is still live. The ability to correct the live part is not worth the ability to quietly revise the settled one.

**A gap has to be detectable in the data, not merely absent.** A three-week outage that leaves no trace is indistinguishable from three weeks of forecasts nobody scored. So a refusal writes a skip record — the refusal is itself data — and `public-data/forecasts/index.json` lists every date that should carry a forecast and does not, split by whether anything explains it:

```json
"counts": { "published": 41, "gaps": 3, "gaps_explained": 2, "gaps_unexplained": 1 }
```

An *explained* gap is an outage the engine noticed and declined to paper over. An *unexplained* one is a night nothing survives from — the job died, never started, or was never scheduled — and it is the materially weaker position. Publishing both under one heading, distinguished by whether an explanation exists, is what stops a silence from reading like a run of unscored forecasts.

**The nightly writes `public-data/` and nothing else.** It never touches this file. A job that edits prose can trip the §0 no-overclaiming guard at 02:00 UTC with nobody watching, so the workflow stages `public-data/` explicitly and fails if anything else is staged, and a test asserts the command itself writes nowhere else.

### The live log is not the backtest

The walk-forward above is a *simulation of what the engine would have said*. Every one of its 2,876 forecasts was scored against an outcome that already existed when the code ran. That is a legitimate measurement and a weaker claim than it is usually read as.

The nightly log is what the engine **did** say, committed to a public repository before the outcome existed. It cannot be re-run, tuned, or accidentally given lookahead. It is the stronger claim by some distance and it will have `n` in single digits for months.

**The two are reported separately, each with its own count, and they are never pooled.** Pooling would make the live sample look testable years earlier by diluting it with observations carrying a weaker guarantee. The threshold for reporting any p-value on the live log — **30 independent windows**, thinned to non-overlapping, the same discipline every other p-value here follows — is fixed in advance rather than chosen once the numbers exist. At a five-session horizon a nightly job accrues roughly one a week, so that is on the order of half a year. Publishing `n = 4` with *no test is possible* is the honest output for that whole period, not a placeholder.

Two things the live log deliberately does not report. **CRPS skill**, because it needs the null's distribution for the same date and the published artifact does not carry one. And an **uncensored PIT**: the artifact carries quantiles rather than paths, so the live PIT is interpolated on a five-point grid and a realised value outside that grid is recorded as censored rather than clamped to zero or one. A clamped PIT is a number that looks measured and is not.

### Reproducibility, since the log claims it

- **`uv.lock` is committed and CI installs with `uv sync --frozen`.** A recorded seed reproduces nothing if SciPy's optimiser moved underneath it, and `--frozen` fails rather than silently re-resolving.
- **Every artifact records the git SHA** beside `engine_version`, which is a static `0.1.0` carried by every forecast this project has ever published — including the ones produced by the drift-carrying simulation that was later withdrawn. It identifies nothing; the commit does. A dirty working tree is published as `dirty: true` rather than implying a reproducibility the SHA does not cover.

### CI

`ruff`, `mypy --strict`, the full suite behind an 80% coverage gate on `engine/aurex/`, an offline end-to-end run from the committed seed cache, and the §0 guards. The `bench` extra — torch, Chronos, neuralforecast, about 2.5GB — stays out: it exists for the step 6 shootout alone and nothing in the default suite imports it.

One job runs with **no secrets configured at all**, asserting the environment really is bare before it starts. "No API key is required to run Aurex" is a stated feature, which makes it a claim, which means it needs a guard — and it is the kind that will not break loudly. It breaks the day someone adds a source that reads a key and works fine on their machine because their machine has one. That job catches the code-level regression; the nightly catches the source-level one, because it runs keyless against live endpoints and fails visibly when one starts demanding authentication.

## Benchmarks *(not built yet — step 6)*

Every model must beat a **driftless random walk** on out-of-sample CRPS. Models that fail ship anyway, labelled as failing.

> _Results pending the first full backtest run. This table will be populated by CI and is not hand-edited._

| Model | CRPS skill vs RW | PIT uniformity (KS p) | Directional accuracy |
|---|---|---|---|
| Random walk (baseline) | — | — | — |
| GJR-GARCH + FHS | _pending_ | _pending_ | _pending_ |
| HAR-RV + FHS | _pending_ | _pending_ | _pending_ |
| AutoARIMA | _pending_ | _pending_ | _pending_ |
| NHITS | _pending_ | _pending_ | _pending_ |
| Chronos (zero-shot) | _pending_ | _pending_ | _pending_ |

**One row of this is already in.** Against the driftless random walk, GJR-GARCH + FHS scores between −0.4% and +0.9% CRPS skill on gold over 2015–2026, with Diebold-Mariano rejecting at no horizon. The prediction below said the GARCH family should win on volatility; on this asset and sample it did not win and did not measurably lose, and the sentence stays as written rather than being edited after the fact. Every row added here will carry the same test — a shootout that ranks six models by an untested difference of means is a leaderboard, not a measurement.

**Expected outcome, stated in advance so it cannot be retrofitted:** the GARCH family should win on volatility and distribution shape. No model, including the time-series foundation models, is expected to beat the random walk on 10-day *directional* accuracy. If that is what the data shows, it will be published as the headline result rather than buried — a rigorous public demonstration that a modern foundation model cannot call two-week gold direction is more useful than another repository claiming it can.

**Two methodological requirements recorded now, so step 6 cannot be built without them.** Both follow from step 3a and are cheaper to design in than to retrofit:

- **Six models against one null is a multiple-comparisons problem, and six Diebold-Mariano tests at α = 0.05 is the wrong instrument for it.** Run enough pairwise tests and one rejects by construction; a shootout scored that way finds a winner whether or not there is one. Step 6 uses Hansen's Superior Predictive Ability test, or the Model Confidence Set where the question is which models cannot be excluded rather than whether any beats the benchmark. The per-model DM statistic is still reported, as a description rather than as a decision.
- **The minimum detectable effect gets computed per horizon, and published whether or not anything rejects.** Given the observed loss-differential variance and the independent window count, there is a smallest CRPS skill the test could have detected at 80% power. *"We could have detected anything above X% and did not"* is a result. *"We found nothing"* is not — it is a sentence that reads identically whether the effect is absent or the sample is too small, and step 3a has already shown which of those this repository is usually in.

## Limitations

- **Calibration is measured on one asset, one model, one sample.** The results above are gold, GJR-GARCH, and 2015–2026. A sample containing one regime is not evidence about another, and the wider shootout against AutoARIMA, NHITS and Chronos is still step 6.
- **The volatility layer has not been shown to pay for itself, and has not been shown not to.** Its CRPS skill against the random walk is within noise of zero at every horizon, and the Diebold-Mariano tests reject nowhere. With 44 independent windows at a quarter the test would miss a real effect of ordinary size, so this is an absence of evidence and it is reported as one. The distributions it produces are close to well calibrated, which is a different and weaker claim than being better than the null.
- **The tests behind the negative result are underpowered, and that is the finding's main limitation.** The skill-decay interval spans [−0.007, +0.010] per log-unit of horizon, wide enough to contain the decay theory predicts as well as no decay at all. Nothing here rules out a volatility layer worth a percent or two; it rules out one worth enough to see in eleven years of weekly forecasts.
- **The breach deficit is half explained.** Displacement accounts for roughly half of it at every horizon; the rest is a lower tail that is too wide. With 44 to 580 independent windows depending on horizon this cannot be resolved further on the present sample, and it is recorded rather than diagnosed.
- **The PIT chi-square rejects at five sessions.** p = 0.011, against a KS test that does not reject anywhere. The histogram is reading the drift displacement, and the rejection lands at the horizon with the most observations rather than the largest effect. Both tests are published at every horizon so this cannot be read selectively.
- **The backtest scores the dollar view, not the rupee one.** A currency lens composes the base paths with an exchange rate through a copula, so scoring it means walking that joint simulation forward too. Fitting the converted price directly would be easier and would grade something the engine never publishes, so it is not done.
- **Barrier probabilities are monitored at session close.** A level breached and recovered inside one session is not counted, so every touch probability is a floor.
- **Simulated policy is fixed policy.** Duty and GST enter a simulation at the rates in force on the last observed day and stay there. A revision inside the horizon is not modelled, because forecasting one would be a political prediction.
- **The simulation is driftless by choice, and a rising sample will punish that.** Centring the residual pool is §0's position, not a neutral one: over a sample that trended it guarantees a displaced PIT, a short breach count and a direction forecast worse than the base rate. Those are the symptoms of refusing to extrapolate, and they are published rather than tuned away. The drift-carrying pool is available for anyone who wants to measure the other choice.
- **The HAR cascade has no variance-of-variance.** Its forecast is iterated deterministically, because a simulated path has no intraday high and low to measure a range from, so every path in that ensemble shares one variance trajectory. Where path dependence is the question, the recursive model is the one to use.
- **The observed premium series is short.** Each IBJA report carries about four days of history, so a fresh clone starts with days, not years, and the nightly job extends it. The premium is not backfilled.
- **Pre-2017 parity is indicative only.** Before GST, the regime was state-varying VAT plus excise with no single national rate. Those rows are tagged `confidence: low`.
- **Pre-2012 parity does not exist.** Duty was a specific levy (₹300/10g), not ad valorem. Rather than invent a percentage, those dates are dropped.
- **The current duty entry is `secondary`.** The CBIC primary document is not machine-retrievable: legacy PDF paths 404 for 2026, the portal exposes PDFs only via non-guessable numeric IDs, and its search API returns HTTP 401. Three independent secondary sources agree, and the level is corroborated observationally — against the IBJA 999 print of ₹142,224/10g on 2026-07-29, a 15% duty implies a −43bps premium while 6% implies +803bps.
- **Spot is close-only.** The London fix has no intraday range, so OHLC-based volatility estimators must use `xau_futures` and accept the basis.
- **Horizon.** Everything here targets 5–30 trading days. Longer horizons have different dynamics and are out of scope.
- **Factor loadings will be unstable.** Gold's relationship to its drivers shifts with regime. The confidence intervals are wide for a reason — read them.
- **Retail friction varies enormously.** The defaults are representative, not universal. Enter your own dealer's actual quotes. Dealer premiums and buyback discounts carry a `spread_basis` saying they are representative and user-editable; they are structurally prevented from sharing the citation that covers the tax rate on the same entry, because one of those two numbers has a regulator behind it and the other does not.
- **The routes table is four jurisdictions and two routes.** It is not a survey. Where a route is not listed as available somewhere, that is an absence of data in Aurex and never a statement about what a reader may hold — availability is informational and the lookup says so when it fails. Leverage caps are recorded only where the national regulator's own instrument was read.
- **The hurdle event is unmeasurable at short horizons and high friction.** Zero positive events at five sessions and five at ten, against a pre-registered expectation of about fifteen. That expectation was wrong because its sigma was wrong — 22% annualised where the sample carries 15.9% — and the residual gap between a Gaussian rate and the measured one is the fat tail. The reliability curve is withheld there rather than drawn on noise, and the base rate, count and mean forecast are published beside every score so a low Brier on a rare event cannot be read as a good forecast.
- **The dashboard is deployed from a snapshot, not a live feed.** Pages are prerendered at build time from the committed artifacts, so what a visitor sees is what the engine had published when the site last built — and the nightly commit is what rebuilds it. A deploy that has not run shows yesterday's forecast with yesterday's date on it, which is the failure mode this design prefers to a page that recomputes.
- **The exceedance grid is half-percent steps, and the calculator interpolates between them.** That is a real approximation, bounded by half a percent of move — far tighter than the sampling error on a 20,000-path ensemble, which is why it is acceptable here and would not be on a five-point quantile grid. A hurdle outside the published ±25% range gets no probability rather than an extrapolated one.
- **The live track record is empty, and will be small for months.** The nightly job has published nothing yet. Until it accrues 30 independent windows at a horizon, no test is possible on it and none will be reported — the count and the distributions will be, labelled as untestable. It is never pooled with the walk-forward, so nothing here will make the live sample look larger than it is.
- **The staleness tolerances are judgements, not vendor guarantees.** They were set from observed publication lag on a handful of runs, and each carries the calendar and the reasoning behind it so a reader can disagree with a specific number rather than with the idea. A source that changes its schedule will read as a fault until the tolerance is revisited, which is the failure direction to prefer.
- **The safe-haven channel is not yet estimated.** The factor set declares a geopolitical-risk regressor, but no source is wired to it, so it currently reports as unavailable. This matters more than it looks: without it, a scenario chain like *escalation → crude up → inflation up → Fed hawkish → gold down* runs entirely through the real-yield and dollar channels, and would very likely produce the wrong sign with honestly-estimated loadings and a clean causal story attached — gold historically rallies on escalation. Omitted-variable bias is more dangerous here than a hand-typed view, because it survives the check against hand-typed views.
- **Calibration is not accuracy.** A well-calibrated model that says "55/45" is honest, not useful for timing. That distinction is the whole point.

## Roadmap

| Step | Scope |
|---|---|
| ~~3a~~ | ~~Scoring the distributions the engine already produces: PIT, CRPS skill against the random walk, Kupiec, Christoffersen, reliability. Walk-forward, expanding window, 2015→present~~ — **done**, results above |
| ~~3b~~ | ~~Routes and jurisdictions; friction per route; the breakeven hurdle and the calibration of clearing it~~ — **done**, results above |
| 4 | Elastic-net factor attribution; market-sourced scenario priors; the crude → CPI → policy → rupee transmission chain by local projections |
| 5 | ~~Dashboard: Today, Track record, Calculator~~ — **done**, above. Uncertainty decomposition and user-set leverage with a live liquidation probability remain, and the Drivers and Scenarios views wait on step 4 |
| 6 | Benchmark shootout vs random walk, AutoARIMA, NHITS, Chronos — **including the rows where Aurex loses** |
| 7 | ~~Nightly automation: CI, the staleness refusal, track-record integrity, the live log~~ — **done**, above. Deploy waits on step 5 |
| 8 | A second asset, as a traded instrument: exchange-listed rupee-quoted futures, shifted-log returns, roll friction at each expiry, futures friction including transaction tax |
| 9 | Cross-asset scenario view — one geopolitical tree, two conditional distributions |

**3a came before everything else because it was the gate.** It needed nothing that did not already exist, and it decided whether the rest was worth building: a badly non-uniform PIT histogram would have meant the volatility layer needed work before more surface area went on top of it. KS does not reject at any horizon and the chi-square rejects at one, in a direction the drift displacement explains, so 3b proceeds — but the same run found the volatility layer earning no CRPS skill distinguishable from zero against a fair null, at any horizon and with no decay across them. The honest reading is that the distributions are close to trustworthy while the model behind them has not been shown to beat the null. That belongs to step 6, not to 3b.

Later, once the above stands up: regime-switching volatility, an options-implied vol surface from COMEX where available, a multi-horizon term structure of forecast distributions, and a public API for the nightly artifact.

## Contributing

Issues and PRs welcome. Three rules: no point forecasts; any new model must be scored against the random-walk baseline before it merges; and the skill score must arrive with its Diebold-Mariano p-value and observation count, whichever way it points.

## Licence

MIT — see [LICENSE](LICENSE).

---

Aurex is a research and education tool. It produces probability distributions, not advice. Short-horizon price direction is not reliably forecastable, and nothing here changes that.
