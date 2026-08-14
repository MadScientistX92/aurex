# Aurex

**A calibrated uncertainty engine for gold. Distributions, not predictions.**

[![CI](https://github.com/MadScientistX92/aurex/actions/workflows/ci.yml/badge.svg)](https://github.com/MadScientistX92/aurex/actions/workflows/ci.yml)
[![Nightly](https://github.com/MadScientistX92/aurex/actions/workflows/nightly.yml/badge.svg)](https://github.com/MadScientistX92/aurex/actions/workflows/nightly.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)

> **Build status: steps 1, 1.5, 2, 3a, 3b, 4, 6, 7 and most of 5 are complete.** Built: the data layer, dated tax schedules, import parity, currency lenses, the asset abstraction, the volatility and distribution engine, the scoring layer with a walk-forward backtest from 2015, routes with per-jurisdiction friction and the breakeven hurdle, the benchmark shootout and the directional grading beside it, factor attribution with the transmission chain, the nightly job that publishes one dated forecast a night or refuses and says why, and the dashboard's three buildable views. Not built: the scenario engine and the second asset. The dashboard's *Drivers* and *Scenarios* views are still omitted rather than stubbed — step 4's loadings now exist, but the views wait on the scenario engine that reads them. Sections below are marked *(not built yet)* where that is the case — they are commitments, not claims.
>
> **Calibration has now been measured, and the headline is a negative result.** Across 2,876 out-of-sample forecasts from January 2015, the CRPS skill against the random walk is between −0.4% and +0.9% depending on the horizon, and a Diebold-Mariano test rejects at none of them — the smallest p-value in the table is 0.23. The distributions are close to the right *shape*: PIT uniformity survives a KS test at all five horizons, and a chi-square rejects at one of them. Direction carries no information at all — and that is now measured for every model in the benchmark set rather than for this one alone: resolution is indistinguishable from zero at four of the five horizons on both binnings, with the fifth published beside the reasons it reads as noise. An earlier version of this section reported up to +4.6% CRPS skill; that number was a model carrying drift beating a null that had been denied one, it has been withdrawn, and the simulation no longer carries the drift. It would not have survived a Diebold-Mariano test either — measured, p = 0.27. Everything below is published with the test behind it, in both directions.

---

## The thesis

Search for a gold forecasting tool and you will find hundreds that output a price and an arrow. Almost none of them tell you how often they have been right, because almost none of them keep score.

Aurex takes the opposite position:

> Short-horizon price **direction** is not reliably forecastable. Short-horizon **volatility** partly is. So Aurex never predicts a price — it produces a probability distribution, and then publicly grades how well-calibrated that distribution turned out to be.

Every forecast is timestamped and committed to this repository. Once its horizon elapses, it is scored automatically — and from that moment the file may never be rewritten, which is enforced by code and by CI rather than by good intentions. The git history is the track record. Each run writes a dated copy to `public-data/forecasts/`, because `latest.json` is state and a track record needs a record. A night the engine could not price honestly produces no forecast and a failed build, never a forecast built on last week's prices.

Five rules follow, and they are enforced in code rather than promised in prose:

1. **No point forecasts anywhere.** A number without an interval or a distribution behind it is a bug. `engine/tests/test_no_overclaiming.py` fails the build if point-forecast or marketing vocabulary reaches user-facing text — including this file.
2. **Every probability gets scored.** A forecast that is never scored is marketing. The scoring layer refuses to score without a baseline, and refuses to attach a p-value to overlapping windows.
3. **The null hypothesis is the random walk, and so is the model.** Every model must beat a driftless random walk on out-of-sample CRPS. It must also *be* driftless: a simulation whose median sits above spot is a directional forecast however it got there, so the resampled residual pool is centred by default and a drift-carrying pool is a declared option rather than an accident. Models that lose ship anyway, labelled as losing. Negative results get published. *(The GARCH result and the five-model shootout are both below. Nothing beat the null.)*
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

The result inherits gold's actual fat tails and skew rather than a convenient assumption about them. It is the cheapest defensible choice in the project — it assumes nothing about the shape of the residual distribution and costs one bootstrap. What it is *worth* is a separate question and a smaller number than this paragraph used to imply: against a Gaussian handed the engine's own forecast variance, the empirical shape moves the deepest measurable tail probability by about a tenth of the distance to the realised rate, and on most horizons it moves it the other way, because a negative skew thins the upside tail. [The hurdle result below](#the-hurdle-result-against-that-prediction) has the decomposition and the count test behind it.

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

### Factor attribution

An elastic-net regression of weekly returns on real yields, the dollar index, crude, VIX, geopolitical risk, ETF flows, and lagged momentum, on a rolling three-year window.

**This is for attribution and scenario propagation, never for directional forecasting.** Loadings are published with bootstrap confidence intervals, and out-of-sample R² reported honestly — which usually means "close to zero". That is the truthful answer, and hiding it would defeat the point of the project.

Step 4 also carries the India transmission chain: crude → import bill → CPI and the current account → policy response → the rupee → the rupee gold price. Every arrow is estimated by local projections, lagged, and uncertain; uncertainty compounds multiplicatively across the links, so the compounded band is what gets displayed. It spans zero at every horizon, which is the finding, and it is stated as one below.

### The one bias this project named in advance

The driver set has declared a geopolitical-risk regressor since it was written, with no source behind it. That gap was not neutral, and this section is the reason step 4 could not start until it was closed.

Every other regressor reaches an escalation through a channel that pushes the metal **down**. Crude rises, expected inflation rises, policy turns hawkish, the real yield rises, and a loading fitted on real yields alone will duly report that escalation is bearish for gold. The safe-haven bid — the thing that actually moves the price on the day of an escalation — had no regressor to load on, so it would have been absorbed into the residual. The output would have been a wrong sign carrying honestly estimated coefficients and a clean causal story, and it would have passed the guard against hand-typed views **because nothing was hand-typed**. An omitted variable is more dangerous here than a fabricated one, because a fabricated view is visible in a diff.

So the Caldara-Iacoviello daily index is wired first, from the authors' own file, under the same provenance rule as every other external fact. Two decisions about it were made before a single loading was fitted, and are recorded here so neither can be read as a choice made after seeing a coefficient:

- **It enters as a change, not a level**, though the factor set originally declared `level`. Every other regressor here is news; a persistent level regressed on near-unpredictable returns is a well-known way to manufacture a significant-looking coefficient. And the channel being measured is the bid that arrives when risk *rises* — risk that has been elevated for a month is already in the price.
- **It is a required factor, not an optional one.** Optional factors drop out with a recorded reason when their source fails. This one cannot, because the reason would be recorded in a field nobody reads while the loadings underneath told a confident, inverted story about escalation. If the index is unavailable, attribution refuses to publish rather than degrading to the factor set that has this bug.

### Pre-registered: what step 4 was expected to find

Written **before** the estimator was run, and left exactly as written whatever the numbers turned out to be. Two of this repository's four graded pre-registrations have missed; both were worth more than the number they predicted, which is the reason to keep making them.

**Out-of-sample R² will be low, and plausibly indistinguishable from zero.** Stating it now, before it exists. Attribution that explains little is the expected result for weekly returns on macro factors, and it is not a failure of the estimator — it is the measurement the project exists to publish honestly.

That prediction is only falsifiable if it says *which* R², because the factor set admits two and they answer different questions:

- **Predictive** — factors lagged one week, so the regression is asked to forecast. §5 forbids using this for direction, and it is reported precisely so the ban has a number behind it. **Prediction: indistinguishable from zero, and this one is held with confidence.**
- **Contemporaneous** — factors from the same week as the return, coefficients fitted on windows that end before it. This is what attribution actually means: how much of last week's move do these drivers account for, using loadings that did not see it. **Prediction: positive and material, on the order of 0.2 to 0.45, and carried almost entirely by the dollar and the real yield.** If this one is near zero too, the decomposition is not worth publishing and the honest output is to say so.

**Loading stability across the rolling window will be poor.** Made falsifiable: at least one factor changes sign between windows, and the spread of a factor's rolling loading is wide relative to its full-sample confidence interval. If the rolling loadings are stable, the three-year window was unnecessary and a single full-sample fit would have been the honest estimator.

**The compounded band on crude → CPI → policy → rupee will span zero at every horizon.** Four noisy monthly links multiplied together, on a sample of a couple of hundred months, with a fuel-tax buffer sitting in the middle of the first link. The band is what gets published; a point estimate from this chain would be a story with a number attached.

**The two routes from crude to the rupee gold price will disagree.** Estimating it link-by-link and estimating it directly measure overlapping things through different amounts of noise. That disagreement is published as a finding rather than reconciled, and neither number is tuned toward the other.

**Nothing here is ever summed.** Crude is already in the factor set as an inflation proxy, so the direct loading and the chain measure the same driver twice through different paths. They are published as alternative decompositions, side by side, with the overlap stated — never added together into a total that would double-count it.

### The loadings, against that prediction

`aurex factors` writes `public-data/factors.json`; every figure below is read out of it by `tests/test_readme_factors.py`, which fails the build if the README and the artifact disagree in either direction. **1074 weeks, 2006-01-13 to 2026-08-07.** The sample starts in 2006 because the dollar index does, and the artifact names it as the binding regressor rather than leaving a reader to infer it from a start date.

| Driver | Loading | 95% interval (OLS) | OLS p | Selected | Sign flips | R² without it |
|---|---|---|---|---|---|---|
| `d_dxy` | -0.010532 | [-0.013383, -0.009714] | 0.0000 | 1.000 | 0 | 0.09717 |
| `d_vix` | +0.003129 | [0.001924, 0.007207] | 0.0007 | 0.997 | 2 | 0.22795 |
| `d_real_yield` | -0.002009 | [-0.005256, 0.000157] | 0.0648 | 0.941 | 1 | 0.16769 |
| `d_oil` | +0.001545 | [0.000573, 0.004324] | 0.0105 | 0.942 | 4 | 0.18548 |
| `momentum` | -0.000199 | [-0.002963, 0.000765] | 0.2479 | 0.567 | 1 | 0.19207 |
| `d_geopolitical_risk` | +0.000192 | [-0.000339, 0.002204] | 0.1505 | 0.615 | 4 | 0.18998 |

The loading is the change in the weekly log return per one standard deviation of the driver, which is the only scale on which two of these are comparable. *Selected* is the share of 2000 moving-block bootstrap replicates in which the penalised fit kept the driver at all — reported because a percentile bootstrap around a penalised estimator is not valid at exactly zero, so an interval that excludes zero there is weaker evidence than it looks, and the OLS interval beside it is the one without that caveat. *R² without it* is the whole set refitted with that driver removed. `etf_flow` is not in the table: its source accumulates a few days of history per fetch, and admitting it would have cut the sample from 1074 weeks to 2.

**The out-of-sample R² was pre-registered in two forms and both were graded.**

- **Predictive: -0.00218** over 917 weeks, Diebold-Mariano p **0.8346** against the training-window mean. Pre-registered as indistinguishable from zero and held with confidence. **Hit.** It is also slightly negative, which is what a factor model doing worse out of sample than "the average of the last three years" looks like. §5's ban on using this for direction now has a number under it rather than an assurance.
- **Contemporaneous: 0.18699** over 918 weeks, DM p **0.0000**. Pre-registered as *positive and material, on the order of 0.2 to 0.45*. **Missed, narrowly and on the low side.** The direction and the significance were right; the range was not, and 0.187 sits just below the bottom of it. In-sample R² is 0.23315, so about a fifth of the fit does not survive being asked to work on weeks it did not see.

**The half of that prediction that missed badly was the mechanism, not the magnitude.** The pre-registration said the contemporaneous fit would be *carried almost entirely by the dollar and the real yield*. The dollar half is right and then some: remove it and out-of-sample R² falls from 0.18699 to **0.09717**, roughly half the explanatory power in one regressor. The real yield half is wrong. Its loading is the third largest, its OLS interval covers zero (p = 0.0648), and removing it costs about two points of R². The factor set's own description calls it "the dominant carry channel" and this sample does not support that; equity volatility has a larger loading and a smaller p-value.

**Two drivers make the fit worse out of sample.** Removing `d_vix` raises R² from 0.18699 to **0.22795**, and removing `momentum` raises it to **0.19207**. Both are kept: this is an attribution set, the decomposition is the product, and dropping regressors because they hurt an out-of-sample score would be selecting a model on the thing being reported. But it is stated here, because a reader entitled to the loadings is entitled to know that two of them are net negative for the only forecast-shaped number in the section.

**Stability was pre-registered as poor, and it split.** The prediction was made falsifiable in two parts. *At least one driver changes sign between windows* — **hit**, and comfortably: five of six do, with `d_oil` and `d_geopolitical_risk` flipping four times each across 230 rolling three-year windows. *The rolling spread is wide relative to the full-sample interval* — **missed**. The ratio of a driver's rolling interquartile range to the width of its own interval runs from 0.369 to 0.993, and never exceeds one. So the loadings do wander, and the wandering is contained by intervals that were already wide enough to cover it. The two halves of "unstable" turn out to measure different things, and only the first one holds. The dollar is the exception on both counts: 230 windows, 230 selections, zero sign changes.

### What the index that had to be wired first actually did

Nothing measurable, and that is the result.

`d_geopolitical_risk` loads **positive** — rising geopolitical risk goes with a higher gold return, which is the sign the omitted-variable argument said would otherwise be inverted. But it is not distinguishable from zero: OLS p = 0.1505, kept in 61.5% of bootstrap replicates, and it changes sign in four of the 230 rolling windows. And removing it entirely moves the largest surviving loading by **0.00001566** — twenty-five times less than any other driver in the set — while out-of-sample R² goes *up*, from 0.18699 to 0.18998.

So the honest reading is this. The bias this project identified in advance did not materialise in the loadings on this sample. No other driver flips sign without the index, and none moves appreciably. That is not an argument that wiring it was unnecessary: "we checked and the bias was negligible" and "we did not check" produce the same coefficients and are not the same claim, and the second one is the one that gets published with a confident causal story attached. The trap was always in the *propagation* rather than in the fit — the scenario engine is step 5, and it will now reach for an estimated safe-haven loading that says **small, positive, and not distinguishable from zero**, which is a far weaker input than the clean story a real-yield-only chain would have handed it. Weakening that input was the point.

### The chain, and the two answers it gives

Local projections per §19, monthly, one regression per horizon with its own band and no identifying assumptions imposed. The HAC truncation is at least the horizon, because the cumulative response makes the residual a moving average of order *h* by construction. Sample: 175 months, 2012-02 to 2026-08.

| Horizon (months) | Compounded | 95% band | Direct | Orthogonalised |
|---|---|---|---|---|
| 0 | -0.000002 | [-0.000264, +0.000222] | -0.02770 | -0.02310 |
| 1 | +0.000107 | [-0.000183, +0.000618] | -0.03493 | -0.03343 |
| 3 | +0.001261 | [-0.003317, +0.003031] | -0.06998 | -0.06696 |
| 6 | +0.002690 | [-0.005972, +0.005897] | -0.13401 | -0.13181 |
| 12 | -0.000590 | [-0.006663, +0.008290] | -0.04265 | -0.03851 |

**The compounded band spans zero at every horizon** — pre-registered, and **hit**. The band is not the product of the links' own bands — multiplying three intervals would treat estimates from overlapping windows of one economy as independent. It comes from a bootstrap that resamples months once per replicate and re-estimates every link on the same months, so whatever dependence the sample has survives into the product.

The middle link is why. Inflation into the policy response spans zero at all five horizons on its own, with a point estimate that changes sign between the impact month and one month later. Any product running through it inherits that, and no amount of precision in the other two links can rescue it.

**The two routes disagree — pre-registered, and hit — and this is the finding rather than a discrepancy to tune away.** Chain-implied at six months: +0.0027. Direct at six months: **-0.13401**, with a band of [-0.233278, -0.034744] that is the one cell in the whole section that excludes zero. Opposite signs and two orders of magnitude apart. Neither number was moved toward the other. The direct estimate is one horizon of five in one of two specifications, so it is a single rejection in a section with plenty of cells; it is published at its face value with that noted, and not promoted into a mechanism.

**Orthogonalised** is the same regression with the quote-currency fix held constant, so its coefficient is the part of the path running through the local economy — the tax stack, the exchange rate, the domestic premium — with the global price channel the weekly loadings already carry taken out. It barely moves the estimate, which says the two channels are close to separable here. **The chain and the direct loading are never summed.** A test asserts that no field anywhere in the emitted block is named like a total.

**The last arrow is not estimated.** The currency lens computes the rupee price from the fix and the rate by arithmetic, so its elasticity is one by construction. Measured as a check rather than fitted, it comes back at 1.0695 on the rate and 0.9588 on the fix with R² 0.94804 — away from one because the duty and GST steps moved inside the sample, which is a fact about the schedule and not about transmission.

**The fuel-tax buffer, and the hole in it.** Indian central excise on petrol and diesel is adjusted by hand, in the opposite direction, precisely when crude moves. On 2026-03-27, with the benchmark up about 75% in four weeks, excise was cut by ₹10/litre on both fuels and the pump price did not change at all — a passthrough of zero produced by a fiscal decision, which an uncontrolled regression would score as a property of the pricing mechanism. The control is a schedule with per-entry citations, mostly PIB releases, and it covers **74 months** with moves flagged at 2017-11, 2022-06, 2025-04 and 2026-04. It does **not** cover October 2018 to November 2021, which contains at least three changes with no retrievable primary document behind them; that window is declared a gap, the resolver returns nothing inside it, and those months drop out of the controlled estimate. Holding the 2018 level across the largest fuel-tax increase in the sample would have put a fabricated constant into the one control that exists to absorb discretionary tax moves.

### Scenario engine *(not built yet — step 5)*

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
| Geopolitical risk | Caldara-Iacoviello daily GPR | The authors' own workbook. Daily, weekends included, 1985– |
| India CPI | FRED `INDCPIALLMINMEI` | Monthly. **Discontinued upstream; last observation 2025-03** |
| India money-market rate | FRED `IRSTCI01INM156N` | Monthly. The policy corridor's transmission, not the repo rate itself |
| India 24K + ETF flow | IBJA daily bullion report (PDF) | 999 AM/PM rates, SPDR tonnes, London fix |

Each series resolves through a priority chain, and the artifact records which source actually answered, so provenance is never implied. Yahoo rate-limits aggressively, which is why nothing depends on it alone.

**Every series now carries a `source_confidence`, on the same rule the duty and GST schedules have always followed.** `primary` means the publisher's own file — the LBMA's fix, IBJA's report, the GPR authors' workbook. `secondary` means a redistributor: FRED serves observations computed by the Treasury, the EIA and the OECD, and Yahoo serves exchange data it did not compute. Neither label is a quality judgement; it records how many hands the number passed through, which is the one thing a URL cannot tell you. It is a member of the loader protocol rather than a field read back with `getattr`, so a source that declares no citation is rejected at the boundary instead of quietly serving numbers with no confidence recorded against them.

The geopolitical-risk index is the one series with a stated citation form, and the authors ask for it: Caldara, Dario and Matteo Iacoviello (2022), "Measuring Geopolitical Risk," *American Economic Review*, April, 112(4), pp. 1194–1225, with data downloaded from <https://www.matteoiacoviello.com/gpr.htm>. It is published under CC BY. `cite_as` is filled for that series and empty for the others, because a citation nobody supplied is not a field to fill with something plausible.

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

**The residual gap was attributed to the fat tail, and most of it is not.** Gold's daily returns over this sample carry an excess kurtosis of 4.9 and a skew of −0.38. A Gaussian assumption discards both by construction; resampling the empirical standardised residuals keeps them. So the question worth asking is not whether the engine is non-Gaussian — it is by construction — but whether its tail is the *right* shape, and whether the comparison that used to appear here was measuring shape at all. It was not, mostly. The table now carries a second Gaussian:

| Hurdle | Horizon | Gaussian at the sample's σ | Gaussian at the engine's own σ | **Aurex forecast** | Realised | Events |
|---|---|---|---|---|---|---|
| 5.10% | 5 | 0.0130 | 0.0198 | **0.0174** | 0.0190 | 11 |
| 5.10% | 10 | 0.0576 | 0.0600 | **0.0528** | 0.0639 | 37 |
| 5.10% | 21 | 0.1385 | 0.1343 | **0.1212** | 0.1681 | 97 |
| 5.10% | 42 | 0.2211 | 0.2168 | **0.2024** | 0.3094 | 177 |
| 5.10% | 63 | 0.2652 | 0.2638 | **0.2500** | 0.3750 | 213 |
| 9.37% | 10 | 0.0023 | 0.0074 | **0.0079** | 0.0086 | 5 |
| 9.37% | 21 | 0.0252 | 0.0315 | **0.0291** | 0.0381 | 22 |
| 9.37% | 42 | 0.0833 | 0.0857 | **0.0764** | 0.1224 | 70 |
| 9.37% | 63 | 0.1293 | 0.1319 | **0.1177** | 0.1989 | 113 |

**The two Gaussian columns differ only in what volatility they were given, and that is the whole correction.** The first uses the scored window's own realised daily σ of 1.00% — the number an earlier version of this section quoted, and a number nobody had on the first as-of date. The second gives the Gaussian the standard deviation the *engine itself* forecast for that same window, so the two differ in kurtosis and skew and in nothing else. Holding the second moment fixed is the only way to ask whether the tail is the right shape without the answer being a disagreement about how volatile the fortnight was.

On the sharpest row — the 9.37% hurdle at ten sessions, the deepest event here with a non-zero count — the arithmetic comes apart cleanly. A Gaussian at the sample's σ forecasts **0.23%**. A Gaussian at the engine's own σ forecasts **0.74%**. Aurex forecasts **0.79%**. It happened at **0.86%**. So of the 0.56 percentage points between the published Gaussian and Aurex, **0.51 is volatility level and 0.05 is tail shape** — 91% and 9%. The previous version of this section said the empirical tail closed "roughly nine tenths" of the gap. It closes roughly one tenth, and the sentence had the decomposition inverted.

**Where the extra volatility comes from is not a defect, but it is not skill either.** The engine's forecasts over these windows imply a daily σ of **1.16%** against the 1.00% the scored window realised — about 16% hot. It fits on every session available at each as-of date, which reaches back to 2006 and includes both the 2008 crisis and the 2011–2013 decline; the unconditional level it reverts to is therefore higher than what 2015–2026 delivered. A driftless Gaussian handed the *same* expanding-window history forecasts **0.74%** on that row too, indistinguishable from the variance-matched figure. Which is the uncomfortable reading: on this event, a plain unconditional Gaussian with no GARCH and no filtered historical simulation would have produced the same number the engine did.

**And across the rest of the table the model is mostly *thinner*-tailed than a variance-matched Gaussian, not fatter.** Seven of the nine rows put Aurex below the engine-σ Gaussian. That is what a skew of −0.38 does to an *upside* hurdle: negative skew thins the right tail, and every hurdle here is a move up. Only the two deepest rows — 9.37% at ten sessions and 5.10% at five — come out fatter, which is consistent with kurtosis winning over skew far enough into the tail and with nothing else in this table.

**Now the count test, which is what was missing.** Every row above was a comparison of two rates with no test behind it, which is the same error the CRPS skill scores are held to account for three sections up. Each hurdle event is now graded by an exact Poisson-binomial on the window-by-window forecast probabilities — the count is a sum of Bernoulli draws with *different* parameters, so a plain binomial would test a rate the forecaster never stated — reported one-sided in the direction a fat tail predicts, on the full sample and on the thinned non-overlapping subsample alike:

| Reference | Windows | Expected | Observed | p (one-sided) |
|---|---|---|---|---|
| Gaussian at the sample's σ | 579 | 1.33 | 5 | **0.011** |
| Gaussian at the engine's own σ | 579 | 4.29 | 5 | 0.428 |
| Aurex | 579 | 4.58 | 5 | 0.485 |
| Gaussian at the sample's σ, thinned | 290 | 0.67 | 2 | 0.143 |
| Aurex, thinned | 290 | 2.31 | 2 | 0.677 |

**The full sample rejects the published Gaussian and the thinned sample does not, and that gap is the finding.** Five events against 1.33 expected is p = 0.011 over 579 overlapping windows; the same events over 290 independent ones give p = 0.143. Nothing here is strong enough to survive being counted honestly. The engine's own rate is comfortable on both — it forecast 4.58 events and five happened — but "the model is consistent with what occurred" is a much weaker statement than the one this section used to make, and it is the only one the sample supports. Eleven years of gold contains five of these events. That is not a sample that can settle a question about tail shape, and the right conclusion is that the tail is *not inconsistent* with the data rather than that it has been shown to be correctly shaped.

**The one place the engine is emphatically rejected is the long horizon, and there the rejection survives thinning.** At sixty-three sessions the model forecast 66.9 clears of the 9.37% hurdle and 113 happened (p < 0.0001 on every window, p = 0.012 on the forty-four independent ones); at forty-two it forecast 43.7 against 70 (p = 0.048 thinned). This is not tail shape. It is the drift displacement appearing for a fourth time: a distribution centred at spot, scored over a sample that rose, under-forecasts every upside event, and the gap widens with the horizon exactly as [the displacement law above](#one-mechanism-named) predicts. It is the price of refusing to forecast a direction — but until now it was asserted, and it is one of the very few rejections in this repository that does not evaporate when the overlapping windows are thinned away.

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
| ~~Drivers~~ | **Omitted.** Step 4's loadings exist; the view waits on the scenario engine that propagates them |
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

**Four skip records state the wrong cause, and they have not been rewritten.** The records for **2026-08-06, 2026-08-08, 2026-08-10 and 2026-08-11** each give their reason as *"price series did not reach the run date within its declared tolerance"*. That is not what happened on any of the four. `xauusd` did not resolve at all — every one of those records carries `"verdict": "unavailable"` with `lag_days: null` and no last observation, because the sole source for the London fix returned a 2xx response whose body was not JSON and there was no cache behind it. The reason line was a fixed string written for every refusal regardless of which verdict caused it; the per-series `verdict` field beside it was correct throughout, which is the only reason this was recoverable at all.

This is corrected in code rather than in the files: the reason is now derived from the verdicts that actually blocked, and unavailability, staleness, emptiness and an undeclared tolerance each read differently. The four records stand as written. Editing them would be editing the past under the same rule that governs an elapsed forecast, and their detail blocks were never wrong — only the sentence summarising them was.

It is named here rather than only in the commit for one reason beyond the record being wrong. A stale series invites a look at its tolerance, and the tolerance is the one thing that must not move here: widening it would have published nothing truer on any of those four nights, and would have retired the signal that the anchor series has a single source which failed five times in eight days. A published cause that points at the wrong repair is worse than a published cause that is merely vague, because a reader can act on it.

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

`ruff`, `mypy --strict`, the full suite behind an 80% coverage gate on `engine/aurex/`, an offline end-to-end run from the committed seed cache, and the §0 guards. The `bench` extra — torch, Chronos, neuralforecast, about 2.5GB — stays out: it exists for the step 6 shootout alone, which runs in [its own manually-triggered workflow](.github/workflows/shootout.yml). That "nothing in the default suite imports it" is a test rather than a convention, because the failure is silent: a module-level import added to the shootout code would keep every job green while quietly making all of them install torch.

The split is deliberate about *which* half is guarded on every push. The heavy adapters run in the shootout workflow; the statistics behind it — Hansen's SPA, the Model Confidence Set, the minimum detectable effect — live in `aurex/score/shootout.py`, depend on nothing beyond SciPy, and are tested on every commit. A multiple-comparisons correction exercised only when somebody manually starts a two-hour job is a correction nobody is checking, and its own tests are Monte Carlo: the size test runs the whole procedure 120 times against data with no winner in it and asserts it finds one less than 15% of the time, and the MDE is validated by simulating an effect of exactly the size it claims to detect and counting how often the test fires.

One job runs with **no secrets configured at all**, asserting the environment really is bare before it starts. "No API key is required to run Aurex" is a stated feature, which makes it a claim, which means it needs a guard — and it is the kind that will not break loudly. It breaks the day someone adds a source that reads a key and works fine on their machine because their machine has one. That job catches the code-level regression; the nightly catches the source-level one, because it runs keyless against live endpoints and fails visibly when one starts demanding authentication.

## Benchmarks

Every model must beat a **driftless random walk** on out-of-sample CRPS. Models that fail ship anyway, labelled as failing. Five of them did.

**The headline is that nothing beat the null.** Hansen's Superior Predictive Ability test does not reject at any horizon — the smallest p-value across the five is **0.19** — so on eleven years of gold, no model in this set has been shown to have a smaller expected CRPS than a random walk with no drift and no conditioning. The Model Confidence Set never excludes the random walk either. That is the result, it is negative, and it is the one the pre-registration below asked for in advance.

Generated by [`aurex bench`](engine/aurex/cli.py) into [`public-data/benchmarks.json`](public-data/benchmarks.json), 2,876 scored forecasts, first as-of 2015-01-02 and the sample bounded at the 2026-07-29 close, every model refitted at every as-of date. The last as-of date is earlier at each horizon — 2026-07-21 at a week, 2026-04-24 at a quarter — because a forecast whose horizon would run past the bound is dropped rather than scored against data the stated window says is not there. Not hand-edited, and reproducible with the command the artifact carries.

**CRPS skill against the driftless random walk**, by horizon in sessions. Positive is better than the null:

| Model | 5 | 10 | 21 | 42 | 63 |
|---|---|---|---|---|---|
| Random walk (baseline) | — | — | — | — | — |
| GJR-GARCH + FHS | **+0.60%** | −0.44% | −0.12% | **+0.31%** | **+0.94%** |
| AutoARIMA | +0.13% | −0.02% | +0.09% | +0.00% | +0.05% |
| HAR-RV + FHS | −1.12% | −2.00% | −2.30% | −5.59% | −7.69% |
| NHITS | −5.03% | −7.32% | −8.30% | −6.33% | −4.32% |
| Chronos-t5-small (zero-shot) | −26.02% | −26.13% | −24.85% | −25.61% | −25.81% |

**And the tests that decide what that table means**, because a shootout that ranks six models by an untested difference of means is a leaderboard rather than a measurement:

| Horizon | Windows | Independent | Hansen SPA *p* | Model Confidence Set at 90% | MDE range |
|---|---|---|---|---|---|
| 5 | 580 | 580 | 0.192 | AutoARIMA, GJR-GARCH, random walk | 0.26% – 5.80% |
| 10 | 579 | 290 | 0.718 | AutoARIMA, GJR-GARCH, random walk | 0.23% – 6.44% |
| 21 | 577 | 116 | 0.385 | AutoARIMA, GJR-GARCH, random walk | 0.25% – 9.76% |
| 42 | 572 | 64 | 0.626 | + HAR-RV, + NHITS | 0.26% – 13.77% |
| 63 | 568 | 44 | 0.453 | + NHITS | 0.20% – 14.74% |

**The minimum detectable effect is per model, not per horizon, and the spread is the point.** Given the loss-differential variance each pairing actually has, the smallest true CRPS skill a Diebold-Mariano test would reject at four times in five ranges from 0.20% to 14.74%. So GJR-GARCH's +0.94% at a quarter sits well inside an interval where the test could only have found 3.02%, and Chronos's −25.8% is far outside its own 14.74% — one of those is an absence of evidence and the other is evidence of absence, and the single number "no model beat the random walk" does not distinguish them. Quoting one MDE per horizon would have hidden a 70-fold spread between the models being compared.

**AutoARIMA is the random walk, and its own MDE is what says so.** Its skill is between −0.02% and +0.13% at every horizon, and its detectable effect is 0.20–0.26% — an order of magnitude tighter than anything else here, because a model whose loss differential against the null has almost no variance is a model that *is* the null. Order selection on demeaned gold returns lands on essentially no structure, which is the correct answer to the question ARIMA is being asked and not an interesting one. It is in the confidence set at every horizon for the same reason the random walk is.

**This is exactly where six separate tests would have found a winner.** Across five models, five horizons and both samples there are fifty Diebold-Mariano p-values in the artifact, and at 5% roughly two or three should reject by chance alone. They do. The cleanest example is AutoARIMA at sixty-three sessions: p = 0.029 on the thinned subsample, which read alone is a rejection at 5%, attached to a skill score of **+0.05%** — a fifth of a percent of nothing. Its overlapping-sample p-value is 0.512. That is a coin landing on its edge, and a shootout scored by "which model has a small p-value" would have published it. SPA does not, because the multiplicity is inside its null distribution rather than corrected for afterwards.

**Chronos loses by a quarter of its CRPS at every horizon, and that is the strongest signal in the table.** −24.9% to −26.1%, consistent across horizons, rejecting on both the overlapping and the thinned sample at every horizon (the largest of those ten p-values is 0.0004), and eliminated from the Model Confidence Set at every horizon — the only model that is. It is the one place in this repository where a test rejects emphatically in either direction. Two caveats belong beside it rather than underneath it: the model is shown 512 sessions of context and asked for the next h, so it is being used exactly as a zero-shot forecaster is meant to be used, but the checkpoint's *training* corpus is not auditable from here — if it contained gold this is not a clean out-of-sample test and nothing in this repository could detect that. The artifact pins the resolved commit (`a971ba21…`) so at least the weights are identified. And its ensemble is 200 sampled paths against 4,000 for the simulation models; the CRPS estimator is the fair one so that is unbiased rather than penalised, but it is noisier, and noise widens a standard error — which cuts *against* significance, not toward it. A −26% loss is not an artifact of ensemble size.

**Expected outcome, stated in advance so it cannot be retrofitted:** the GARCH family should win on volatility and distribution shape. No model, including the time-series foundation models, is expected to beat the random walk on 10-day *directional* accuracy. If that is what the data shows, it will be published as the headline result rather than buried — a rigorous public demonstration that a modern foundation model cannot call two-week gold direction is more useful than another repository claiming it can.

**Against that prediction, kept as written.** The first half is half right and unproven: GJR-GARCH has the best skill of the five at three of the five horizons and is in the confidence set at all of them, which is the direction the prediction named — but SPA does not reject anywhere, so it did not *win*, and "the GARCH family" as a whole did not, because HAR-RV loses at every horizon and loses more as the horizon lengthens. The second half is not graded by *this* run at all, and that is worth being exact about rather than quietly claiming: **directional accuracy per model was not measured here.** The harness scored binary events for the subject only, and every model in this set is centred to driftless by the shared drift policy, so all six forecast P(up) ≈ 0.5 by construction. Measuring direction across them would have graded the policy rather than the models. It is now graded by [its own run](#the-directional-result-against-that-prediction) below, which is uncentred and carries the drift-matched walk as its null — and which was pre-registered before it ran, because a claim graded on the artifact that was supposed to test it is not graded at all.

**Two methodological requirements were recorded before this was built, and both are load-bearing in it.** Both followed from step 3a:

- **Six models against one null is a multiple-comparisons problem, and six Diebold-Mariano tests at α = 0.05 is the wrong instrument for it.** Run enough pairwise tests and one rejects by construction; a shootout scored that way finds a winner whether or not there is one. Step 6 uses Hansen's Superior Predictive Ability test, or the Model Confidence Set where the question is which models cannot be excluded rather than whether any beats the benchmark. The per-model DM statistic is still reported, as a description rather than as a decision. *Built: `aurex/score/shootout.py`, with the recentring that keeps a few hopeless entrants from destroying the power to detect a real one, and both the recentred and uncentred p-values published so the gap between them is visible.*
- **The minimum detectable effect gets computed per horizon, and published whether or not anything rejects.** Given the observed loss-differential variance and the independent window count, there is a smallest CRPS skill the test could have detected at 80% power. *"We could have detected anything above X% and did not"* is a result. *"We found nothing"* is not — it is a sentence that reads identically whether the effect is absent or the sample is too small, and step 3a has already shown which of those this repository is usually in. *Built, and computed per model as well as per horizon for the reason above.*

**Fairness is structural rather than promised.** The shootout is a single walk-forward call carrying every model, so each one sees the identical price slice on the identical date and is graded against the identical realised outcome; a date any model cannot fit is a date none of them is scored on, because a challenger allowed to sit out its hardest windows would post a better mean CRPS for having declined to forecast. Every model is driftless: the engine's models and the null centre the pool they resample, and each challenger's simulated returns are centred before they are walked to prices. That last one is not housekeeping — three of these models fit a conditional mean by construction and one infers a trend from its context, and a model carrying drift scored against a null denied one is worth several percent of CRPS skill that belongs to neither. It is the error this project already withdrew once, and a six-model shootout is where it would come back six times over.

**What this run does not contain.** Per-model PIT uniformity — the harness computes distributional diagnostics for the subject and CRPS for the rest. Per-model directional accuracy was the other gap and is closed by [the run below](#the-directional-result-against-that-prediction), which needed the harness to score binary events for every forecaster rather than the subject alone. One asset, one sample, one set of hyperparameters per challenger: NHITS at 200 training steps and Chronos at its small checkpoint are defensible defaults and not tuned ones, and a tuned NHITS is a different experiment that this result does not speak to.

### Pre-registered: the directional claim, and why it needs its own run

The sentence above — *"no model, including the time-series foundation models, is expected to beat the random walk on 10-day directional accuracy"* — is the one people will quote, and it was the only part of §0's benchmark promise still ungraded. This section states what is expected of it before the numbers exist, as with [the hurdle event](#pre-registered-what-the-hurdle-event-was-expected-to-do), where the prediction missed and the miss was explainable, which is what made it worth having.

**It cannot be graded on the shootout above, and re-reading that artifact would not do it.** Every model in the CRPS run is centred, so all six forecast P(up) ≈ 0.5 by construction. Scoring direction there grades the drift policy and returns "no model can call direction" whatever the models did — an answer that cannot come out any other way is not a measurement. The fix is not to abandon centring, which is still right for the distributional comparison. It is to grade direction on the **uncentred** forecasts with the **drift-matched random walk** as the null, so every competitor including the benchmark carries whatever drift it infers. That is the like-for-like comparison the forecaster already knows how to pick: `ModelForecaster.like_for_like_null` derives it from the drift policy rather than taking it as configuration.

**Resolution is the primary metric, not Brier.** Brier confounds level with discrimination — a model that knows the base rate and nothing else beats a model with real signal and a biased level, which is the opposite of what calling direction means. Resolution is level-invariant: it measures whether a model can tell one window from another. A model with the base rate right and no discrimination scores resolution zero; a model with the wrong level and real discrimination scores positive. Reliability and the level are reported separately so the two are never read as one number.

**Expected outcome, stated in advance so it cannot be retrofitted:** **resolution at or near zero for every model at every horizon, including Chronos.** The foundation model has the most room to surprise here — it infers a trend from its context window rather than fitting a mean, so it is the one entrant that could in principle discriminate. If it does, that is a finding and it gets published as one. If it does not, the sentence the pre-registration already committed to is earned rather than assumed.

**What the run is bound to.** Six models — the five challengers plus the drift-matched walk, which is graded as a competitor and not only as a null, because a walk carrying drift makes a real directional claim and a null that cannot lose the comparison it defines is not a competitor. Per model per horizon: the Brier score, its full Murphy decomposition, the base rate, the positive count, and the mean forecast beside the realised rate. The sample is bounded with `--to` and the artifact carries the command that reproduces it.

**And the same multiple-comparisons discipline**, because six resolution figures compared informally is the leaderboard problem again. The decision is one test over the whole set — the largest *studentised* resolution against a null built by resampling the outcomes against fixed forecasts — which is Hansen's SPA's shape for the same reason. Two details are load-bearing and were fixed before the run:

- **Resolution is a sum of squares, so it is positive under the null.** "Close to zero" is not a reading on its own. The reference distribution says what zero is worth on this sample and every figure is published beside its own `resolution_under_null`.
- **On overlapping windows the resampling is a circular shift, not a permutation.** A shift preserves the outcome series' autocorrelation exactly; a permutation destroys it, makes the null's bin rates less variable than they really are, and turns ordinary persistence into a rejection. `engine/tests/test_shootout.py` asserts that the permutation over-rejects on the same dependent data, rather than describing it.

**One threat to the metric was found while building it and is handled rather than discovered afterwards.** A direction forecast lives near one half, so a model's entire range across every window it ever saw can fall inside a single equal-width bin — and its resolution is then zero *by construction*, whatever it knew. Publishing that as "no discrimination" would make this pre-registration self-fulfilling. So the bin count travels with every score, and the whole test is run a second time on bins of equal *count*, which adapt to whatever range a model actually used and ask whether its ordering of windows carries information. Where the two binnings disagree, the equal-count one is the one with power and the disagreement is itself the finding.

### The directional result, against that prediction

Generated by [`aurex direction`](engine/aurex/cli.py) into [`public-data/direction.json`](public-data/direction.json), the same 2,876 forecasts on the same windows as the shootout above, bounded at the same 2026-07-29 close, uncentred, against the drift-matched random walk. No date was skipped.

**The prediction was right at four horizons out of five, and it missed at the fifth.** That is the headline and it is stated in that order deliberately.

**Resolution, equal-width bins — the published Murphy decomposition.** Read every figure against the `null` column beside it, which is what that model's own binning produces with no skill at all, and against the uncertainty term of 0.229–0.249:

| Model | 5 | 10 | 21 | 42 | 63 |
|---|---|---|---|---|---|
| Random walk (drift-matched) | 0.00000 | 0.00000 | 0.00000 | 0.00023 | 0.00116 |
| GJR-GARCH + FHS | 0.00000 | 0.00000 | 0.00000 | 0.00234 | 0.00402 |
| HAR-RV + FHS | 0.00076 | 0.00000 | 0.00137 | 0.00083 | 0.00018 |
| AutoARIMA | 0.00042 | 0.00073 | 0.00374 | 0.00467 | 0.00581 |
| NHITS | 0.00192 | 0.00634 | 0.00626 | 0.00312 | 0.00473 |
| Chronos-t5-small | 0.00313 | 0.00441 | 0.00170 | 0.00934 | 0.00612 |

**Half of those zeros are the axis, not the models, and the pre-registration said to check.** GJR-GARCH, HAR-RV and the drift-matched walk put every forecast they ever made into **one** equal-width bin at the short horizons — their P(up) ranges over a few points either side of 0.55 and never leaves the `[0.5, 0.6)` decile. A resolution of exactly 0.00000 from a forecaster confined to one bin is not a measurement, and publishing it as the cleanest possible confirmation would have been assuming the answer. Binned by rank instead:

| Model | 5 | 10 | 21 | 42 | 63 |
|---|---|---|---|---|---|
| Random walk (drift-matched) | 0.00538 | 0.00375 | 0.00439 | 0.00665 | 0.00404 |
| GJR-GARCH + FHS | 0.00443 | 0.00587 | 0.00412 | 0.00621 | 0.01087 |
| HAR-RV + FHS | 0.00627 | **0.01097** | 0.00221 | 0.00801 | 0.00710 |
| AutoARIMA | 0.00324 | 0.00370 | 0.00453 | 0.00264 | 0.00520 |
| NHITS | 0.00704 | 0.00528 | 0.00553 | 0.00138 | 0.00307 |
| Chronos-t5-small | 0.00437 | 0.00295 | 0.00488 | 0.01157 | 0.01448 |

**And the tests that decide what those tables mean.** Each cell is one test over all six models at once; a decision needs both runs of a binning to reject:

| Horizon | Windows | Independent | Equal-width, full | Equal-width, thinned | Equal-count, full | Equal-count, thinned | Rejects |
|---|---|---|---|---|---|---|---|
| 5 | 580 | 580 | 0.743 | 0.743 | 0.313 | 0.313 | no |
| 10 | 579 | 290 | 0.329 | 0.431 | **0.036** | **0.025** | **equal-count** |
| 21 | 577 | 116 | 0.149 | 1.000 | 0.907 | 0.849 | no |
| 42 | 572 | 64 | 0.719 | 0.730 | 0.844 | 0.627 | no |
| 63 | 568 | 44 | 0.465 | 0.518 | 0.796 | 0.569 | no |

**Chronos did not surprise, and the way it failed is sharper than the prediction was.** It was named in advance as the one entrant that could in principle discriminate, and it is indeed the only model that genuinely *varies* its directional call: it occupies all ten equal-width bins at every horizon, where the GARCH family and the walk sit in one or two. So it is not declining to discriminate — it is discriminating, and the discrimination is noise. Its observed resolution sits *below* its own null mean at three of five horizons (0.00170 against 0.00581 at twenty-one sessions), and its marginal p-value never falls below 0.31. It pays for all that movement with the worst Brier score in the table by a factor of two — 0.361 to 0.476 against roughly 0.24 for everything else — and almost all of the difference is the reliability term, which reaches **0.250** at sixty-three sessions. A model can move its probability around confidently, be wrong about which way every time, and end up exactly where a model that never moved would be. That is what the resolution column is for.

**The miss: at ten sessions the equal-count screen rejects on both runs, and this is published as a finding because the pre-registration said it would be.** *p* = 0.036 on the full sample and *p* = 0.025 thinned. HAR-RV rejects marginally on both (*p* = 0.0069 and 0.0070) with a resolution of 0.011 against a null of 0.004; AutoARIMA rejects on the thinned run only. `both_screens_reject_equal_count` is `true` at that horizon and nowhere else. That field was first called `distinguishable_from_zero_equal_count` and has been renamed: the original name is factually correct and reads as a verdict, which the multiplicity and concentration caveats below contradict.

**Two of the four reasons first published here were wrong, and controls run afterwards are what showed it.** This section originally called the ten-session cell a disagreement between the two runs, and left standing an implicit worry that equal-count bins might be measuring something other than the forecast. Neither survives contact with a test. The replaced reasoning is kept visible rather than quietly overwritten, because the correction is worth more to a reader than the paragraph reading smoothly — the same move the tail section above makes.

- **It is a near-tie, not a disagreement.** On the thinned run AutoARIMA studentises to +3.494 (*p* = 0.0046) and HAR-RV to +3.244 (*p* = 0.0070); on the full run HAR-RV leads at +3.493 with the next model at +0.993. HAR-RV is top-two in both runs and is displaced by neither. "The two runs name different models" was true of the maximum and false of the evidence.
- **The bins are measuring the forecast.** `_equal_count_assignment` ranks with a stable sort, so ties break by position, which on this series is date order — and five of six models resolve to fewer than a third as many distinct probabilities as there are windows. Breaking those ties at random instead leaves the set *p* at a median of 0.025 across ten seeds, rejecting on both runs ten times out of ten; the published 0.036 sits at the *weak* end of that distribution. HAR-RV's bin index tracks the as-of date at ρ = +0.60, but so does its raw probability, to within 0.002 — the ordering is the model's, not the sort's.

**And the calendar trend that ρ = +0.60 suggested is not what is being scored, which took two controls to establish.** The realised up-rate does drift across the sample (Spearman +0.075 against date rank, *p* = 0.073), so a forecast that was only a slow function of the calendar could in principle post resolution while knowing nothing.

- **A calendar placebo does not score.** A "forecaster" whose probability is the as-of date rank and nothing else posts resolution 0.00525 against a null mean of **0.00904** — below chance — for a marginal *p* of 0.964, and 0.00694 against 0.00770 on the thinned run. Added as a seventh entrant it ranks last of seven, fifth of seven thinned, and leaves the set *p* at 0.036 unchanged. The level of its null is the part worth reading: a calendar binning has a null mean twice the real models' 0.004, because under a circular shift the outcome series keeps its clustering and contiguous-date bins pick that up in nearly any alignment. **The shift over-represents trend alignment on this sample rather than under-representing it** — the opposite of the concern that motivated the check — and the studentisation has been pricing it in all along.
- **A null that keeps the trend rejects harder, not softer.** Permuting outcomes only *within* calendar year preserves the year-level up-rate structure — 2015 at 0.39 through 2025 at 0.76 — and destroys only within-year alignment. The set *p* goes to **0.0138** (0.0238 thinned) and HAR-RV's marginal to 0.0030. A within-year *cyclic shift*, which also preserves within-year persistence and so cannot be accused of over-rejecting the way an iid permutation can, gives 0.0132 and 0.0018. The control is not toothless — it discriminates correctly: under the same null the calendar placebo's *p* is 0.895, its null mean still well above its score, while HAR-RV's null mean does not move at all (0.00403 to 0.00397). HAR-RV's bins drift with the calendar without *being* calendar blocks: 8.6 of ten bins occupied within an average year, against the placebo's 1.8. Chronos settles it from the other side — more calendar-aligned than HAR-RV on every measure, and dead last in the set with resolution below its own null.

**Both controls are in the code and both are in the artifact, which they were not when this section first published them.** They were run once in a scratch file, quoted here, and the scratch file was deleted — leaving the strongest claims in this section tied to nothing, in a repository whose whole discipline is that a published number carries the run that produced it. `resolution_screen` now takes the null as a parameter and ships three (the circular shift, within-year permutation, within-year cyclic shift); each screen records which one it used, because *which null* is load-bearing here rather than incidental. The calendar placebo is a named control in `aurex.score.shootout`, not a fixture. Leave-one-year-out is a function that refits the screen twelve times. All of it lands in `public-data/direction.json` under `discrimination.controls`, and a test parses the two tables and every figure in the prose above and fails if any of them stops matching the artifact — in either direction, which is what makes a re-run that moves a number visible instead of quiet.

**Four reasons it still does not amount to "HAR-RV can call ten-day gold direction" — all of them reasoning done *after* seeing the number, which is exactly the move that has to be labelled rather than performed smoothly:**

- **Multiplicity across horizons and binnings is not corrected anywhere in this table, and it now carries the conclusion.** The screen handles the six models; nothing handles five horizons times two binnings. Ten cells with one at 0.036 is about what chance produces — roughly 0.3 if it were corrected. Step 6 already published this exact pattern — AutoARIMA at *p* = 0.029 attached to a skill score of +0.05%, "a coin landing on its edge". Adding a correction now, having seen which cell it would erase, would be inventing a standard to dispose of an inconvenient number, so it is named as the limitation and the number is left standing.
- **Taken at face value it is small.** 0.011 of resolution against an uncertainty of 0.248 is about 4% of the event's variance, 7.7% on the thinned run — from the model that lost the CRPS shootout by the widest margin of the five at that horizon.
- **It reads as a property of one period rather than a persisting ability.** Dropping each calendar year in turn and refitting the whole screen, HAR-RV's marginal *p* runs from 0.0019 to 0.6528. Removing **2017** alone — 8.6% of the sample — takes it from **0.0069 to 0.6528** and puts the resolution *below its own null*, 0.00354 against 0.00482. Every one of 2015 through 2019 is individually load-bearing: drop any single one of those five and the rejection is gone, at *p* between 0.085 and 0.65. No year from 2020 to 2025 is — each of those can be dropped and the cell still rejects, and 2025, the 0.76 up-rate year that was the candidate anticipated in advance, turns out not to be load-bearing at all. Only the partial 2026 block moves it above the line, to 0.067. Six of twelve refits still reject marginally, two of twelve at set level. An ability to rank ten-session windows should not be carried by the first third of the sample.
- **It appears at one horizon out of five and at neither neighbour**: HAR-RV's equal-count *p* is 0.10 at five sessions and 0.92 at twenty-one. An ability to rank ten-session windows by direction that vanishes at five and at twenty-one has no mechanism behind it.

**The leave-one-year-out table behind that third reason.** Each row drops one calendar year, re-ranks the survivors into equal-count bins, rebuilds the shift null for the shorter series, and refits the whole six-model screen:

| Dropped | n | That year's up-rate | HAR-RV resolution | Its null | Marginal *p* | Set *p* | Best |
|---|---|---|---|---|---|---|---|
| — | 579 | — | 0.01097 | 0.00403 | **0.0069** | **0.0364** | HAR-RV |
| 2015 | 528 | 0.392 | 0.00745 | 0.00458 | 0.1008 | 0.3973 | NHITS |
| 2016 | 529 | 0.520 | 0.00643 | 0.00451 | 0.1765 | 0.6679 | HAR-RV |
| 2017 | 529 | 0.640 | 0.00354 | 0.00482 | **0.6528** | 0.7097 | NHITS |
| 2018 | 529 | 0.440 | 0.00522 | 0.00466 | 0.3454 | 0.5484 | walk |
| 2019 | 529 | 0.560 | 0.00821 | 0.00473 | 0.0854 | 0.2676 | walk |
| 2020 | 528 | 0.647 | 0.01249 | 0.00461 | 0.0076 | 0.0266 | HAR-RV |
| 2021 | 529 | 0.560 | 0.01385 | 0.00453 | 0.0019 | 0.0095 | HAR-RV |
| 2022 | 530 | 0.429 | 0.00970 | 0.00457 | 0.0341 | 0.1591 | HAR-RV |
| 2023 | 529 | 0.480 | 0.01154 | 0.00455 | 0.0095 | 0.0550 | HAR-RV |
| 2024 | 528 | 0.647 | 0.00928 | 0.00451 | 0.0361 | 0.1711 | HAR-RV |
| 2025 | 529 | 0.760 | 0.01132 | 0.00453 | 0.0152 | 0.0778 | HAR-RV |
| 2026 | 552 | 0.407 | 0.00781 | 0.00413 | 0.0673 | 0.2745 | GJR-GARCH |

**What would settle it** is a horizon grid this run does not have and a second asset, neither of which is a reason to withhold the rejection now. The claim this repository is entitled to make is the one at the top of this section: resolution is indistinguishable from zero at four of five horizons on both binnings, and at the fifth, one binning rejects with a small effect that survives every control aimed at it and does not survive leaving out a single year.

**One consequence for the sentence people quote.** *"No model, including the time-series foundation models, is expected to beat the random walk on 10-day directional accuracy"* is now graded rather than assumed — and ten sessions is precisely the horizon where the grading is least clean. The foundation model is not the reason: Chronos is nowhere near rejecting at any horizon, and the drift-matched walk it was supposed to beat is itself in the confidence-free zone with everything else. The sentence stands, with the ten-session caveat attached to it rather than filed away.

## Limitations

- **Calibration is measured on one asset, one sample.** The results above are gold and 2015–2026. A sample containing one regime is not evidence about another, and the shootout adds five more models on the *same* sample rather than a second sample — so it widens the comparison and does nothing at all for this limitation.
- **Nothing in the shootout beat the null, and at most horizons the test could not have seen a small effect if there was one.** Hansen's SPA rejects at no horizon. The minimum detectable effect runs from 0.20% to 14.7% of CRPS depending on the pairing, so the negative result is strong for Chronos, which loses far outside its own interval, and weak for GJR-GARCH, whose +0.94% at a quarter sits inside an interval where only 3.02% was detectable. One sentence covers both cases and should not be read as covering them equally.
- **The volatility layer has not been shown to pay for itself, and has not been shown not to.** Its CRPS skill against the random walk is within noise of zero at every horizon, and the Diebold-Mariano tests reject nowhere. With 44 independent windows at a quarter the test would miss a real effect of ordinary size, so this is an absence of evidence and it is reported as one. The distributions it produces are close to well calibrated, which is a different and weaker claim than being better than the null.
- **The tests behind the negative result are underpowered, and that is the finding's main limitation.** The skill-decay interval spans [−0.007, +0.010] per log-unit of horizon, wide enough to contain the decay theory predicts as well as no decay at all. Nothing here rules out a volatility layer worth a percent or two; it rules out one worth enough to see in eleven years of weekly forecasts.
- **Open question: the ten-session directional cell may be a volatility regime, and it is left for a reader rather than tested.** HAR-RV fits no mean. Its directional probability moves only because a variance forecast rescales a residual pool that is skewed and carries a non-zero drift, so its call is a function of forecast volatility and of nothing else. The leave-one-year-out table in step 6b puts the whole effect in 2015–2019 and none of it in 2020–2025 — and that split is also, plausibly, a low-volatility block against a high-volatility one, which would make the concentration an artifact of regime rather than a period of genuine discrimination. Three mechanisms have now been proposed against this cell — tie ordering, calendar-trend alignment, and a null that could not carry the trend — and the data killed all three. A fourth control chosen after the third failed would be motivated skepticism rather than a test, so the observation is recorded and the cell is left where the multiplicity and concentration caveats already put it.
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
- **The hurdle event is unmeasurable at short horizons and high friction.** Zero positive events at five sessions and five at ten, against a pre-registered expectation of about fifteen. That expectation was wrong because its sigma was wrong — 22% annualised where the sample carries 15.9% — and the residual gap between a Gaussian rate and the measured one is mostly a difference in *volatility level* rather than in tail shape, which is a correction to what this line used to say. The reliability curve is withheld there rather than drawn on noise, and the base rate, count and mean forecast are published beside every score so a low Brier on a rare event cannot be read as a good forecast.
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
| 4 | ~~Elastic-net factor attribution; the crude → CPI → policy → rupee transmission chain by local projections~~ — **done**, above. Market-sourced scenario priors remain and belong to the scenario engine |
| 5 | ~~Dashboard: Today, Track record, Calculator~~ — **done**, above. Uncertainty decomposition and user-set leverage with a live liquidation probability remain, and the Drivers and Scenarios views wait on step 4 |
| ~~6~~ | ~~Benchmark shootout vs random walk, AutoARIMA, NHITS, Chronos — **including the rows where Aurex loses**~~ — **done**, results above. Nothing beat the null; Chronos lost by a quarter of its CRPS. Per-model **direction** is now graded too, in its own uncentred run: resolution is indistinguishable from zero at four of five horizons, and the fifth is published with the reasons it reads as noise. Per-model PIT uniformity remains unmeasured |
| 7 | ~~Nightly automation: CI, the staleness refusal, track-record integrity, the live log~~ — **done**, above. Deploy waits on step 5 |
| 8 | A second asset, as a traded instrument: exchange-listed rupee-quoted futures, shifted-log returns, roll friction at each expiry, futures friction including transaction tax |
| 9 | Cross-asset scenario view — one geopolitical tree, two conditional distributions |

**3a came before everything else because it was the gate.** It needed nothing that did not already exist, and it decided whether the rest was worth building: a badly non-uniform PIT histogram would have meant the volatility layer needed work before more surface area went on top of it. KS does not reject at any horizon and the chi-square rejects at one, in a direction the drift displacement explains, so 3b proceeds — but the same run found the volatility layer earning no CRPS skill distinguishable from zero against a fair null, at any horizon and with no decay across them. The honest reading is that the distributions are close to trustworthy while the model behind them has not been shown to beat the null. That belonged to step 6, which has since run: five challengers on the same windows, and Hansen's SPA rejects at no horizon either.

Later, once the above stands up: regime-switching volatility, an options-implied vol surface from COMEX where available, a multi-horizon term structure of forecast distributions, and a public API for the nightly artifact.

## Contributing

Issues and PRs welcome. Three rules: no point forecasts; any new model must be scored against the random-walk baseline before it merges; and the skill score must arrive with its Diebold-Mariano p-value and observation count, whichever way it points.

## Licence

MIT — see [LICENSE](LICENSE).

---

Aurex is a research and education tool. It produces probability distributions, not advice. Short-horizon price direction is not reliably forecastable, and nothing here changes that.
