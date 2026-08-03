# Aurex

**A calibrated uncertainty engine for gold. Distributions, not predictions.**

[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)

> **Build status: steps 1, 1.5, 2 and 3a of 9 complete.** Built: the data layer, dated tax schedules, import parity, currency lenses, the asset abstraction, the volatility and distribution engine, and the scoring layer with a walk-forward backtest from 2015. Not built: routes and per-jurisdiction friction, factor attribution, the scenario engine, the dashboard, the benchmark shootout, and the second asset. Sections below are marked *(not built yet)* where that is the case — they are commitments, not claims.
>
> **Calibration has now been measured, and the headline is a negative result.** Across 2,876 out-of-sample forecasts from January 2015, the distributions are the right *shape* — PIT uniformity survives both a KS and a chi-square test at all five horizons. But against a random walk allowed the same drift the model itself carries, the CRPS skill is zero or slightly negative at every horizon. An earlier version of this section reported up to +4.6%; that number was the model's drift beating a null that had been denied one, and it has been withdrawn. Direction, separately, carries no information at all. Both negative results are below with the counts behind them.

---

## The thesis

Search for a gold forecasting tool and you will find hundreds that output a price and an arrow. Almost none of them tell you how often they have been right, because almost none of them keep score.

Aurex takes the opposite position:

> Short-horizon price **direction** is not reliably forecastable. Short-horizon **volatility** partly is. So Aurex never predicts a price — it produces a probability distribution, and then publicly grades how well-calibrated that distribution turned out to be.

Every forecast is timestamped and committed to this repository. Once its horizon elapses, it is scored automatically. The git history is the track record, and it cannot be quietly edited after the fact. Each run writes a dated copy to `public-data/forecasts/`, because `latest.json` is state and a track record needs a record.

Four rules follow, and they are enforced in code rather than promised in prose:

1. **No point forecasts anywhere.** A number without an interval or a distribution behind it is a bug. `engine/tests/test_no_overclaiming.py` fails the build if point-forecast or marketing vocabulary reaches user-facing text — including this file.
2. **Every probability gets scored.** A forecast that is never scored is marketing. The scoring layer refuses to score without a baseline, and refuses to attach a p-value to overlapping windows.
3. **The null hypothesis is the random walk.** Every model must beat a driftless random walk on out-of-sample CRPS. Models that lose ship anyway, labelled as losing. Negative results get published. *(The GARCH result is below; the wider shootout is step 6.)*
4. **No overclaiming.** The credibility is the product.

A fifth rule emerged while building the data layer, and it earned its place:

5. **Never substitute a computed value for a missing observation.** `local_premium_bps` is `observed / parity − 1`. Filling a missing observation with parity would make the residual identically zero and fabricate the very signal it measures. Where there is no observation, there is no number.

## What it does today

- Resolves every input series through a **priority chain** — preferred source, fallback, cache — and records in the artifact which one actually answered
- Prices the Indian import stack from a **dated schedule**, where every duty and GST entry carries its own source URL and confidence level
- Presents one price through **currency lenses**, so the dollar and rupee views come from one engine rather than two code paths
- Fits **GJR-GARCH(1,1,1)** with Student-t innovations, alongside HAR-RV and a rolling-σ baseline, to the price and the exchange rate
- Generates return distributions by **filtered historical simulation** — no assumed parametric shape — and **keeps the paths**
- Reports **first-passage statistics**: what share of paths touch a level, how long they take, and what the survivors are holding
- Couples the two exposures with a **t-copula** so tail co-movement survives into the rupee price
- Models the **real cost of physical gold in India** — dealer premium, GST, buyback spread — and reports the move required to break even
- **Grades itself**: walks the engine forward over eleven years of history, one refit per week, and scores every distribution it would have published on PIT, CRPS against the random walk, VaR coverage, and reliability

## What it does not do yet

- **Routes and jurisdictions**: friction per route, the breakeven hurdle, and whether clearing it is a calibrated probability *(step 3b)*
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

uv run pytest                                    # 461 tests, no network
uv run aurex schedule                            # duty history with provenance
uv run aurex duty 2026-07-29                     # rate in force, and its source
uv run aurex pipeline                            # live run, writes public-data/latest.json
uv run aurex score --from 2015-01-01             # walk-forward backtest and calibration report

# offline from the committed seed cache, no network at all
AUREX_CACHE_DIR=tests/fixtures/seed-cache uv run aurex pipeline --dry-run
```

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

**But the simulation is not driftless, and it is worth being exact about why.** Filtered historical simulation resamples the empirical standardised residuals, and in a sample that rose those have a positive mean — +0.039 over gold's history here, which is a drift of +0.000418 per session against a realised +0.000423. So the median simulated price sits above spot (3.2% at sixty-three sessions) even though nothing fitted a mean. That is a defensible consequence of resampling what happened rather than a bug, but it is a drift, it is empirical rather than estimated, and every scoring comparison has to account for it — which is why the backtest runs a drift-matched null alongside the driftless one.

**Fitting is break-aware.** A policy step is a mechanical jump in the price, not information about volatility. Each known discontinuity is excluded from the likelihood and enters the recursion as no shock at all, so the variance decays across a duty revision instead of spiking at it. Those residuals are also dropped from the resampling pool, because a duty revision is not a shock the bootstrap should be allowed to redraw.

### Return distribution

Aurex does **not** sample from a fitted normal or Student-t. It uses filtered historical simulation, which is standard practice on institutional risk desks:

1. Fit the volatility model, extract standardised residuals `zₜ = rₜ / σₜ`
2. Block-bootstrap from the empirical `z` pool — contiguous blocks, drawn circularly so every residual is equally likely to appear, including the most recent ones
3. Run the model's own variance recursion forward over those shocks and walk the result into prices, one session at a time

The result inherits gold's actual fat tails and skew rather than a convenient assumption about them. This is the single most defensible technical choice in the project.

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

At typical Indian retail parameters — 3% dealer premium, 3% GST, 3% buyback discount — the break-even move is **+9.4%**, before gold has done anything at all. Against a two-week one-standard-deviation move of roughly ±4.4%, that is a two-sigma requirement in one specific direction.

This is why the project exists. The friction is deterministic and knowable; the price move is not. Most retail tools model the uncertain part and ignore the certain one.

Friction takes a **horizon**, because the two shapes are structurally different: physical friction is paid at the door and is horizon-independent, while a futures roll drag compounds. Profiles for gold ETFs and sovereign gold bonds are included for comparison. Attaching this to simulated P&L is step 3b, along with the routes that decide which friction applies and the jurisdiction that sets its rate.

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
| Kupiec / Christoffersen | Do 95% and 99% VaR breaches occur at the right rate, and independently? | 3a |
| PIT chi-square | Same question as KS, but sensitive to mass piled in one bin rather than to a shift | 3a |
| Brier score + reliability | When it says 20%, does it happen 20% of the time? | 3a |
| …on clearing the breakeven hurdle | Same question, asked of the one event friction defines | 3b |

**These score the price distribution, not a position.** None of them needs a venue, a route or a jurisdiction: the binary events are direction and touching a barrier, and both are properties of the metal. The single probability that *is* defined by friction — whether a move clears breakeven — is not a different metric, it is one more event fed to the same reliability machinery, so it waits for the routes that decide which friction applies.

### Results

Walk-forward, expanding window, no lookahead. One refit every five sessions from 2 January 2015 to 21 July 2026, 4,000 paths per forecast, GJR-GARCH(1,1,1) with Student-t innovations and filtered historical simulation. 2,876 scored forecasts, no skipped dates. Reproduce with `uv run aurex score --asset gold --from 2015-01-01 --horizons 5,10,21,42,63`.

Two nulls, because they answer different questions. **Driftless** is the random walk §0 names: iid empirical increments with the sample mean removed. **Drift-matched** is the same walk with the drift left in. The second exists because the model is not driftless either — see below — so only that column isolates what the volatility machinery is actually worth.

| Horizon | Forecasts | Independent | PIT KS p | PIT χ² p | Skill vs driftless | Skill vs drift-matched | 95% VaR breaches |
|---|---|---|---|---|---|---|---|
| 5 | 580 | 580 | 0.80 | 0.14 | +0.7% | **+0.2%** | 23 vs 29.0 (p = 0.24) |
| 10 | 579 | 290 | 0.54 | 0.33 | −0.2% | **−0.8%** | 13 vs 14.5 (p = 0.68) |
| 21 | 577 | 116 | 0.74 | 0.25 | +0.1% | **−1.2%** | 4 vs 5.8 (p = 0.42) |
| 42 | 572 | 64 | 0.46 | 0.71 | +1.7% | **−0.7%** | 1 vs 3.2 (p = 0.14) |
| 63 | 568 | 44 | 0.55 | 0.13 | +4.6% | **−0.1%** | 0 vs 2.2 (p = 0.17) |

**The distributions are the right shape.** Uniformity survives both tests at every horizon. They are run together because they fail on different things: KS reads the largest gap in the cumulative distribution and is weak against mass piled in one bin, which is exactly the pattern here. Neither rejects. Breach counts are within tolerance everywhere and independent everywhere the independence test is defined.

**Conditioning on volatility is worth approximately nothing at these horizons.** Against the drift-matched null the skill score is +0.2% at a week and negative at every horizon beyond it. The earlier +4.6% at 63 sessions was an artefact of scoring a model that carries drift against a null that had been stripped of it; the honest figure is −0.1%. This is a losing result for the GARCH layer on this asset and sample, and §0 says losing models ship labelled as losing.

**The direction forecast carries no information, as predicted in advance.** At five sessions the Brier score on "ends higher" is 0.24873 against an uncertainty term of 0.24893 — a resolution of 0.00000 to five decimal places. Its *level* is close to right, because the resampled residuals carry the drift, but its resolution is nil: it cannot tell one week from another. At 42 and 63 sessions the level slips too (0.5915 forecast against 0.6444 realised), so there the Brier score is worse than a constant forecast of the sample's own base rate — itself a hindsight benchmark.

### One mechanism, named

Three things looked like separate failures — a heavy top PIT bin, a direction forecast that decays with horizon, and breaches that go missing at long horizons. The first two are one effect, and it has a shape that could have failed to appear.

A model centred at `mu_model` scored over a sample that drifted at `mu_sample` is displaced in standardised units by `(mu_sample − mu_model) / sigma × sqrt(h)`: linear in the horizon over a spread that grows as its square root. Regressing mean PIT on `sqrt(h)` across the five horizons, with the intercept pinned at 0.5 because a sample with no displacement must give one half everywhere, gives **R² = 0.95** on one parameter. The law held.

The two symptoms are the same displacement read from different reference points. Mean PIT (0.5038 → 0.5422) measures it against the forecast's own centre. The direction gap measures it against *spot*, which is why it grows faster. Breaches are **not** part of this: the fitted displacement predicts a 63-session breach rate near 3.8%, and the observed is 1.8%, so most of that deficit is a differently-shaped lower tail and calling it drift would hide a second finding behind the first.

**The engine is not driftless, and this README previously said it was.** The GJR fit sets the conditional mean to zero, but filtered historical simulation resamples the *empirical* standardised residuals, whose mean is +0.039 over this history. That is a drift of +0.000418 per session against a realised +0.000423 — very nearly all of it. The median simulated price sits 0.4% above spot at five sessions and 3.2% above at sixty-three. The drift is empirical rather than fitted, which is defensible, but it was undocumented and it was inflating a published skill score.

**None of this is an argument for fitting a drift.** A fitted mean over a rising sample is a directional forecast wearing a mean's clothes. The displacement is measured and published; it is not tuned away.

### Two conventions fixed in advance, because both would otherwise look like results

**A realised touch is measured at session close**, the same convention the forecast uses. Scoring simulated closes against intraday extremes would charge the model for a floor it already declares. The enforcement is structural: the object carrying a realised outcome holds closes and nothing else, so there is no high or low available to score against.

**Overlapping windows are not independent observations.** Sampling a 21-session horizon weekly makes consecutive scores share three quarters of their path, which is fine for a PIT histogram or a mean CRPS and fatal for a breach-independence test. Every p-value is therefore computed on the thinned subsample, and the function that computes one *raises* on an overlapping series rather than quietly returning a number — 568 forecasts at a 63-session horizon are 44 observations, and a test told otherwise reports a confidence it has not earned.

**At zero breaches the chi-square approximation is not usable.** Kupiec's statistic is asymptotically chi-square, and zero breaches puts the unrestricted estimate on the boundary of the parameter space, where that asymptotics does not hold. Both p-values are always computed and the exact binomial is the one reported at a boundary. It matters: 0 breaches in 44 windows against a 5% quantile gives a chi-square p of 0.034 and an exact p of 0.17. An earlier version of the table above reported the first and read it as the run's only rejection. It was not one.

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

**One row of this is already in.** Against a drift-matched random walk, GJR-GARCH + FHS scores +0.2% CRPS skill at five sessions and negative beyond it, on gold over 2015–2026. The prediction below said the GARCH family should win on volatility; on this asset and sample it did not, and the sentence stays as written rather than being edited after the fact.

**Expected outcome, stated in advance so it cannot be retrofitted:** the GARCH family should win on volatility and distribution shape. No model, including the time-series foundation models, is expected to beat the random walk on 10-day *directional* accuracy. If that is what the data shows, it will be published as the headline result rather than buried — a rigorous public demonstration that a modern foundation model cannot call two-week gold direction is more useful than another repository claiming it can.

## Limitations

- **Calibration is measured on one asset, one model, one sample.** The results above are gold, GJR-GARCH, and 2015–2026. A sample containing one regime is not evidence about another, and the wider shootout against AutoARIMA, NHITS and Chronos is still step 6.
- **The volatility layer does not currently pay for itself.** Against a drift-matched random walk its CRPS skill is within noise of zero at a week and negative beyond. The distributions it produces are well calibrated, which is a different and weaker claim than being better than the null.
- **The breach deficit at long horizons is unexplained.** Displacement accounts for perhaps a third of it; the rest is a lower tail that is too wide at 42 and 63 sessions. With 44 to 64 independent windows this cannot be resolved on the present sample, and it is recorded rather than diagnosed.
- **The backtest scores the dollar view, not the rupee one.** A currency lens composes the base paths with an exchange rate through a copula, so scoring it means walking that joint simulation forward too. Fitting the converted price directly would be easier and would grade something the engine never publishes, so it is not done.
- **Barrier probabilities are monitored at session close.** A level breached and recovered inside one session is not counted, so every touch probability is a floor.
- **Simulated policy is fixed policy.** Duty and GST enter a simulation at the rates in force on the last observed day and stay there. A revision inside the horizon is not modelled, because forecasting one would be a political prediction.
- **The HAR cascade has no variance-of-variance.** Its forecast is iterated deterministically, because a simulated path has no intraday high and low to measure a range from, so every path in that ensemble shares one variance trajectory. Where path dependence is the question, the recursive model is the one to use.
- **The observed premium series is short.** Each IBJA report carries about four days of history, so a fresh clone starts with days, not years, and the nightly job extends it. The premium is not backfilled.
- **Pre-2017 parity is indicative only.** Before GST, the regime was state-varying VAT plus excise with no single national rate. Those rows are tagged `confidence: low`.
- **Pre-2012 parity does not exist.** Duty was a specific levy (₹300/10g), not ad valorem. Rather than invent a percentage, those dates are dropped.
- **The current duty entry is `secondary`.** The CBIC primary document is not machine-retrievable: legacy PDF paths 404 for 2026, the portal exposes PDFs only via non-guessable numeric IDs, and its search API returns HTTP 401. Three independent secondary sources agree, and the level is corroborated observationally — against the IBJA 999 print of ₹142,224/10g on 2026-07-29, a 15% duty implies a −43bps premium while 6% implies +803bps.
- **Spot is close-only.** The London fix has no intraday range, so OHLC-based volatility estimators must use `xau_futures` and accept the basis.
- **Horizon.** Everything here targets 5–30 trading days. Longer horizons have different dynamics and are out of scope.
- **Factor loadings will be unstable.** Gold's relationship to its drivers shifts with regime. The confidence intervals are wide for a reason — read them.
- **Retail friction varies enormously.** The defaults are representative, not universal. Enter your own dealer's actual quotes.
- **The safe-haven channel is not yet estimated.** The factor set declares a geopolitical-risk regressor, but no source is wired to it, so it currently reports as unavailable. This matters more than it looks: without it, a scenario chain like *escalation → crude up → inflation up → Fed hawkish → gold down* runs entirely through the real-yield and dollar channels, and would very likely produce the wrong sign with honestly-estimated loadings and a clean causal story attached — gold historically rallies on escalation. Omitted-variable bias is more dangerous here than a hand-typed view, because it survives the check against hand-typed views.
- **Calibration is not accuracy.** A well-calibrated model that says "55/45" is honest, not useful for timing. That distinction is the whole point.

## Roadmap

| Step | Scope |
|---|---|
| ~~3a~~ | ~~Scoring the distributions the engine already produces: PIT, CRPS skill against the random walk, Kupiec, Christoffersen, reliability. Walk-forward, expanding window, 2015→present~~ — **done**, results above |
| 3b | Routes and jurisdictions; friction per route; the breakeven hurdle and the calibration of clearing it |
| 4 | Elastic-net factor attribution; market-sourced scenario priors; the crude → CPI → policy → rupee transmission chain by local projections |
| 5 | Dashboard, currency toggle, uncertainty decomposition, user-set leverage with the liquidation probability recomputed live |
| 6 | Benchmark shootout vs random walk, AutoARIMA, NHITS, Chronos — **including the rows where Aurex loses** |
| 7 | Nightly automation and deploy |
| 8 | A second asset, as a traded instrument: exchange-listed rupee-quoted futures, shifted-log returns, roll friction at each expiry, futures friction including transaction tax |
| 9 | Cross-asset scenario view — one geopolitical tree, two conditional distributions |

**3a came before everything else because it was the gate.** It needed nothing that did not already exist, and it decided whether the rest was worth building: a non-uniform PIT histogram would have meant the volatility layer needed work before more surface area went on top of it. The histogram came back uniform at all five horizons, so 3b proceeds — but the same run found the volatility layer earning no CRPS skill against a fair null, and the honest reading is that the distributions are trustworthy while the model behind them is not yet better than the null. That belongs to step 6, not to 3b.

Later, once the above stands up: regime-switching volatility, an options-implied vol surface from COMEX where available, a multi-horizon term structure of forecast distributions, and a public API for the nightly artifact.

## Contributing

Issues and PRs welcome. Two rules: no point forecasts, and any new model must be scored against the random-walk baseline before it merges.

## Licence

MIT — see [LICENSE](LICENSE).

---

Aurex is a research and education tool. It produces probability distributions, not advice. Short-horizon price direction is not reliably forecastable, and nothing here changes that.
