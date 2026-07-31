# Aurex

**A calibrated uncertainty engine for gold. Distributions, not predictions.**

[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)

> **Build status: steps 1, 1.5 and 2 of 9 complete.** Built: the data layer, dated tax schedules, import parity, currency lenses, the asset abstraction, and the volatility and distribution engine. Not built: scoring and calibration, factor attribution, the scenario engine, the dashboard, benchmark results, and the second asset. Sections below are marked *(not built yet)* where that is the case — they are commitments, not claims.
>
> **Nothing here has been shown to be calibrated.** The engine now produces distributions; step 3 is what will score them. Until then there is no PIT histogram and no CRPS skill score, and this README will not pretend otherwise.

---

## The thesis

Search for a gold forecasting tool and you will find hundreds that output a price and an arrow. Almost none of them tell you how often they have been right, because almost none of them keep score.

Aurex takes the opposite position:

> Short-horizon price **direction** is not reliably forecastable. Short-horizon **volatility** partly is. So Aurex never predicts a price — it produces a probability distribution, and then publicly grades how well-calibrated that distribution turned out to be.

Every forecast is timestamped and committed to this repository. Once its horizon elapses, it is scored automatically. The git history is the track record, and it cannot be quietly edited after the fact. *(The scoring half of that promise is step 3.)*

Four rules follow, and they are enforced in code rather than promised in prose:

1. **No point forecasts anywhere.** A number without an interval or a distribution behind it is a bug. `engine/tests/test_no_overclaiming.py` fails the build if point-forecast or marketing vocabulary reaches user-facing text — including this file.
2. **Every probability gets scored.** A forecast that is never scored is marketing. *(not built yet — step 3.)*
3. **The null hypothesis is the random walk.** Every model must beat a driftless random walk on out-of-sample CRPS. Models that lose ship anyway, labelled as losing. Negative results get published. *(not built yet — step 6.)*
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

## What it does not do yet

- **Scoring**: PIT histograms, CRPS, Kupiec/Christoffersen, reliability diagrams *(step 3)*
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

uv run pytest                                    # 384 tests, no network
uv run aurex schedule                            # duty history with provenance
uv run aurex duty 2026-07-29                     # rate in force, and its source
uv run aurex pipeline                            # live run, writes public-data/latest.json

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

**The mean is zero unless someone insists otherwise.** A constant fitted over twenty years of returns is a drift, and a drift is a directional forecast wearing a mean's clothes — it would put the median terminal price above spot and every path statistic downstream would inherit it. The random walk is the null, so it is also the default.

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

Friction takes a **horizon**, because the two shapes are structurally different: physical friction is paid at the door and is horizon-independent, while a futures roll drag compounds. Profiles for gold ETFs and sovereign gold bonds are included for comparison. Attaching this to simulated P&L is step 3.

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

## Calibration *(not built yet — step 3)*

A forecast that is never scored is marketing. Aurex will grade itself on:

| Metric | Question it answers |
|---|---|
| PIT histogram | Are the predicted distributions the right *shape*? Uniform = calibrated |
| CRPS skill score | Is the full distribution better than a driftless random walk? |
| Kupiec / Christoffersen | Do 95% and 99% VaR breaches occur at the right rate, and independently? |
| Brier score + reliability | When it says 20%, does it happen 20% of the time? |

Walk-forward, expanding window, no lookahead, 2015→present.

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

**Expected outcome, stated in advance so it cannot be retrofitted:** the GARCH family should win on volatility and distribution shape. No model, including the time-series foundation models, is expected to beat the random walk on 10-day *directional* accuracy. If that is what the data shows, it will be published as the headline result rather than buried — a rigorous public demonstration that a modern foundation model cannot call two-week gold direction is more useful than another repository claiming it can.

## Limitations

- **No calibration evidence exists yet.** Until step 3 lands there is no PIT histogram and no CRPS skill score. The engine produces distributions; whether they are any good is unmeasured.
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
| 3 | Friction and P&L; PIT, CRPS, Kupiec, Christoffersen, Brier; walk-forward from 2015 |
| 4 | Elastic-net factor attribution; market-sourced scenario priors; the crude → CPI → policy → rupee transmission chain by local projections |
| 5 | Dashboard, currency toggle, uncertainty decomposition, user-set leverage with the liquidation probability recomputed live |
| 6 | Benchmark shootout vs random walk, AutoARIMA, NHITS, Chronos — **including the rows where Aurex loses** |
| 7 | Nightly automation and deploy |
| 8 | A second asset, as a traded instrument: exchange-listed rupee-quoted futures, shifted-log returns, roll friction at each expiry, futures friction including transaction tax |
| 9 | Cross-asset scenario view — one geopolitical tree, two conditional distributions |

Later, once the above stands up: regime-switching volatility, an options-implied vol surface from COMEX where available, a multi-horizon term structure of forecast distributions, and a public API for the nightly artifact.

## Contributing

Issues and PRs welcome. Two rules: no point forecasts, and any new model must be scored against the random-walk baseline before it merges.

## Licence

MIT — see [LICENSE](LICENSE).

---

Aurex is a research and education tool. It produces probability distributions, not advice. Short-horizon price direction is not reliably forecastable, and nothing here changes that.
