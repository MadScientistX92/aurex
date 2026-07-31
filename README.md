# Aurex

A calibrated uncertainty engine for gold, priced in INR for an Indian retail buyer.

Aurex does not tell you where the gold price is going. It produces a probability
distribution, states what you have to beat to break even, and then scores its own
distributions in public.

> **Build status: steps 1, 1.5 and 2 of 9 complete** — data layer, tax schedules,
> import parity, currency lenses, the asset abstraction, and the volatility and
> distribution engine. The scoring layer, scenario tree, dashboard, benchmark results
> and the oil module are not built yet. Sections marked *(not built yet)* are
> commitments, not claims. **Nothing here has been shown to be calibrated**: the
> engine now produces distributions, and step 3 is what will score them.

---

## Philosophy

Most retail gold tools output a number and a direction. They are wrong most of the
time and have no way of knowing it. Aurex takes the opposite position:

> **Direction is not forecastable at short horizons. Volatility partly is. Therefore
> Aurex never predicts a price — it produces a calibrated probability distribution,
> and then proves the calibration.**

Four rules follow, and they are enforced in code rather than promised in prose:

1. **No point forecasts anywhere.** A number without an interval or a distribution
   behind it is a bug. `engine/tests/test_no_overclaiming.py` fails the build if
   point-forecast or marketing vocabulary reaches user-facing text.
2. **Every probability gets scored.** A forecast that is never scored is marketing.
   *(not built yet — step 3.)*
3. **The null hypothesis is the random walk.** Every model must beat a driftless
   random walk on out-of-sample CRPS. Models that lose ship anyway, labelled as
   losing. Negative results get published. *(not built yet — step 6.)*
4. **No overclaiming.** The credibility is the product.

A fifth rule emerged while building the data layer, and it earned its place:

5. **Never substitute a computed value for a missing observation.** `local_premium_bps`
   is `observed / parity - 1`. Filling a missing observation with parity would make
   the residual identically zero and fabricate the very signal it measures. Where
   there is no observation, there is no number.

---

## What is built: parity and the domestic premium

The Indian retail gold price is the world price plus the rupee plus a tax stack:

```
parity_ex_gst   = XAUUSD / 31.1034768 * USDINR * (1 + duty)
parity_incl_gst = parity_ex_gst * (1 + gst_metal)

local_premium_bps = (observed_ibja / parity_ex_gst - 1) * 10_000
```

`local_premium_bps` is the residual — the part that import friction and domestic
demand pressure actually move. Three things make it mean what it says.

**The tax stack is a dated schedule, not a constant.** GST did not exist before
1 July 2017 and the import duty has moved ten times, most recently to 15% on
13 May 2026. Every entry in `engine/aurex/data/schedules/duty.yaml` carries its own
`source_url` and `source_confidence` (`primary` | `secondary`); no entry inherits a
table-level default, and a schema test fails the build if either field is missing.

**The comparison is like-for-like.** IBJA publishes its 999 rate *exclusive* of GST,
so the premium is measured against `parity_ex_gst`. Measuring against the
GST-inclusive figure would print a spurious ≈ −291bps at all times.

**Parity uses spot, not futures.** The obvious gold ticker, `GC=F`, is the COMEX
front-month *future*, which carries a cost-of-carry basis over spot — measured at
+2.40% against the London PM fix on 2026-07-29. Sourcing parity from it pushes that
basis into the premium, and because the basis moves with rates and time to expiry it
would look like a moving domestic-demand signal. Parity uses the London PM fix, the
same benchmark IBJA prints in its own daily report. Futures are still loaded, as
`xau_futures`, because step 2's realised-volatility estimators want true OHLC.

## Two lenses, one engine

The same price, seen from two places. A lens is a toggle rather than a second page,
because two code paths computing the same thing will drift.

| | INR view | USD view |
|---|---|---|
| FX exposure | XAU/USD × USD/INR | none — single exposure |
| Import duty | 15% (dated schedule) | none |
| Consumption tax | GST 3% | state sales tax, defaults to 0% |
| Breakeven hurdle | 9.37% | 5.10% (coin) |
| Local premium | measured vs import parity | not applicable |

The USD path is the cleaner case, and that is the point: the INR price is the USD
price with currency and policy layered on top. Making the decomposition visible
teaches something real about why the Indian buyer's distribution is wider.

It is displayed as **measured, never asserted**. XAU/USD and USD/INR are dependent —
fitting a copula over exactly that dependence is the point of the distribution layer
— so their variances do not add, and the FX leg can offset rather than compound. The
INR distribution is usually wider; the interface must still be able to render the
case where it is not.

**A lens must be arithmetic, not policy.** A currency view is valid where the
buyer's-currency price is a *mechanical* function of the quote — an exchange rate, a
unit conversion, a statutory rate, a published settlement formula — and invalid where
that price is administered. An exchange-settled contract quoted in rupees converts by
formula and is a legitimate lens; a policy-set retail price is not, because
presenting it as a view on the world price asserts a passthrough nobody has measured.
Lenses declare which they are, so the rule is enforced rather than remembered.

A note, not a calculation: US long-term gains on physical gold are taxed as a
collectible at a higher maximum rate than equities. Aurex links the IRS guidance and
computes nothing.

## Asset abstraction

Everything asset-specific lives in `engine/aurex/assets/`. Nothing in `vol/`,
`dist/`, `factors/`, `scenarios/`, `trade/` or `score/` may name an asset. Two tests
enforce it: a synthetic asset runs the entire pipeline end to end, and a static guard
fails the build on an asset literal in a downstream package. The behavioural test
catches leaks that change something; the static one catches leaks that have not
broken anything yet.

The guard also covers `pipeline.py` and `forecast.py` — the two modules that compose
assets with the model layer. They necessarily know what a lens *is*; they must never
know which one they are holding. Extending the guard to them immediately found a
stale asset name in a pipeline docstring, which is the argument for the guard in one
sentence.

The interface carries two decisions worth knowing about:

**Friction takes a horizon.** Physical friction is paid at the door and is
horizon-independent; a futures roll drag compounds. A horizon-free interface would
have forced the second into the shape of the first.

**Return transforms are an internal representation.** Crude settled negative in April
2020, so log returns need a shift — and a shifted-log transform silently rescales
anything quoted as a percentage from transform space. On the WTI series Aurex already
caches, the annualised standard deviation reads 55.2% at a shift of 50 and 24.6% at a
shift of 100. The number more than halves on a constant that is arbitrary by
construction. So every reported quantity is mapped back to price space first, and the
transforms module deliberately exposes no volatility or quantile helper. A round-trip
test cannot catch this — a badly-scaled transform round-trips perfectly.

## Distributions, and why the paths are kept

The engine fits a conditional variance model, resamples its standardised residuals in
blocks, and walks them forward into price paths. Three choices are worth stating.

**Terminal distributions are not enough.** Every path is retained rather than
collapsed to its endpoint, because a position that can be closed out early — anything
leveraged, stopped, or margined — never experiences the terminal distribution. Against
a barrier it can reach, the chance of *touching* a level is materially higher than the
chance of *ending* beyond it: on a 21-session horizon the engine currently measures a
ratio of about 1.5. The artifact reports both numbers side by side at each horizon,
never the terminal one alone.

That ratio is below the factor of two a continuous-monitoring argument gives, and the
reason is stated rather than tuned: paths are monitored at session close, so a barrier
breached and recovered inside one session is not counted. The published touch
probability is a floor.

**The mean is zero unless someone insists.** A constant fitted on twenty years of
returns is a drift, and a drift is a directional forecast wearing a mean's clothes —
it would put the median terminal price above spot and every path statistic downstream
would inherit it. The random walk is the null, so it is also the default.

**One simulation, two views.** The underlying is simulated once and each lens is a
multiplication applied to those same paths, so the dollar and rupee views cannot
disagree about what the metal did. The rupee view adds an exchange rate with its own
fitted variance, linked to the price by a **bivariate t-copula** — a Gaussian
dependence has zero tail dependence by construction however high the correlation, and
whether that describes a given pair is an estimated question. Where the fitted degrees
of freedom run to their ceiling, the artifact says so instead of implying evidence of
fat tails that the fit never found.

| Model | Role |
|---|---|
| GJR-GARCH(1,1,1) | Asymmetric conditional variance, Student-t innovations. Each path carries its own variance trajectory |
| HAR-RV | Cascade on realised variance measured from OHLC ranges. Needs a true high and low; a close-only series is refused rather than fed squared returns |
| Rolling σ | The parameter-free baseline the others have to beat in step 6 |

Fitting is break-aware: a policy step is a mechanical jump, so its date is excluded
from the likelihood and enters the recursion as no shock at all. Those residuals are
also dropped from the resampling pool, because a duty revision is not a shock the
bootstrap should be allowed to redraw.

Daily price limits are simulated where a venue has them: the session is truncated to
the cap and the remainder carries into the next one, so a shock big enough to lock the
market takes more than one session to arrive.

Every ensemble records its seed, its model parameters and its block length. A
distribution nobody can reproduce cannot be scored, and step 3 has to be able to
regenerate exactly what was published.

### Structural breaks are recorded, not smoothed

`policy_breaks.yaml` lists every known discontinuity — each duty revision, the 2017
GST rollout, the 2013–2014 80:20 import-linkage rule, demonetisation — and the
pipeline emits them in the artifact. Downstream layers consume them so a policy step
is never mistaken for market noise: parity moves *mechanically* at a duty change, and
the premium is only comparable within a regime.

Whether retail prices actually follow a duty change is a **measurement**, not an
assumption. The pipeline reports a passthrough diagnostic either side of each break
and takes no position on what it should be. This distinction is load-bearing in the
test suite: the wiring test asserts the mechanical step in *parity*, because under
complete passthrough the *premium* does not move at all — an assertion there would
fail on correct code.

---

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

Each series resolves through a **priority chain**: preferred source → fallback →
cache. The artifact records which source actually answered, so provenance is never
implied. Yahoo rate-limits aggressively, which is why nothing depends on it alone.

IBJA's daily PDF replaced two dead ends: their homepage rate block is rendered
client-side, and SPDR's published `.csv` endpoint now serves a PDF. The report also
supplies SPDR tonnes, a better ETF-flow proxy than shares outstanding.

---

## Not built yet

| Step | Scope |
|---|---|
| 3 | Friction and P&L; PIT, CRPS, Kupiec, Christoffersen, Brier; walk-forward from 2015 |
| 4 | Elastic-net factor attribution; market-sourced scenario priors; the crude → CPI → policy → rupee → gold transmission chain, estimated by local projections and published with its bands even where they straddle zero |
| 5 | Next.js dashboard, currency toggle, uncertainty decomposition, user-set leverage with the liquidation probability recomputed live |
| 6 | Benchmark shootout vs random walk, AutoARIMA, NHITS, Chronos — **including the rows where Aurex loses** |
| 7 | Nightly automation and deploy |
| 8 | Oil as a traded instrument: exchange-listed rupee-quoted crude futures, shifted-log returns, roll friction at each expiry, futures friction including CTT |
| 9 | Cross-asset scenario view — one geopolitical tree, two conditional distributions |

---

## Limitations

- **The observed premium series is short.** Each IBJA report carries about four days
  of history, so a fresh clone starts with days, not years, and the nightly job
  extends it. The premium is not backfilled.
- **Pre-2017 parity is indicative only.** Before GST, the regime was state-varying
  VAT plus excise with no single national rate. Those rows are tagged
  `confidence: low`.
- **Pre-2012 parity does not exist.** Duty was a specific levy (₹300/10g), not ad
  valorem. Rather than invent a percentage, those dates are dropped.
- **The current duty entry is `secondary`.** The CBIC primary document is not
  machine-retrievable: legacy PDF paths 404 for 2026, the portal exposes PDFs only
  via non-guessable numeric IDs, and its search API returns HTTP 401. Three
  independent secondary sources agree, and the level is corroborated observationally
  — against the IBJA 999 print of ₹142,224/10g on 2026-07-29, a 15% duty implies a
  −43bps premium while 6% implies +803bps.
- **Spot is close-only.** The London fix has no intraday range, so OHLC-based
  volatility estimators must use `xau_futures` and accept the basis.
- **No calibration evidence exists yet.** Until step 3 lands there is no PIT
  histogram and no CRPS skill score, so nothing here has been shown to be calibrated.
  The engine produces distributions; whether they are any good is unmeasured.
- **Barrier probabilities are monitored at session close.** A level breached and
  recovered inside one session is not counted, so every touch probability is a floor
  rather than an estimate.
- **Simulated policy is fixed policy.** Duty and GST enter a simulation at the rates
  in force on the last observed day and stay there. A revision inside the horizon is
  not modelled, because forecasting one would be a political prediction.
- **The HAR cascade has no variance-of-variance.** Its forecast is iterated
  deterministically, because a simulated path has no intraday high and low to measure
  a range from, so every path in that ensemble shares one variance trajectory. Where
  path dependence is the question, the recursive model is the one to use.
- **The safe-haven channel is not yet estimated.** The factor set declares a
  geopolitical-risk regressor, but no source is wired to it, so it currently reports
  as unavailable. This matters more than it looks: without it, a scenario chain like
  *escalation → oil up → inflation up → Fed hawkish → gold down* runs entirely
  through the real-yield and dollar channels, and would very likely produce the wrong
  sign with honestly-estimated loadings and a clean causal story attached — gold
  historically rallies on escalation. Omitted-variable bias is more dangerous here
  than a hand-typed view, because it survives the check against hand-typed views. The
  cross-asset sign will be validated against an event study of historical escalation
  episodes before it is published, and a disagreement will be published as a finding
  rather than tuned away.
- **Oil will need an optional API key.** EIA's v2 API requires free registration.
  `EIA_API_KEY` will be optional like `FRED_API_KEY`; without it the inventory
  factors drop out with the reason recorded. The engine will never require a key.

---

## Quick start

```bash
cd engine
uv sync

uv run pytest                                    # 370 tests, no network
uv run aurex schedule                            # duty history with provenance
uv run aurex duty 2026-07-29                     # rate in force, and its source
uv run aurex pipeline                            # live run, writes public-data/latest.json

# offline from the committed seed cache, no network at all
AUREX_CACHE_DIR=tests/fixtures/seed-cache uv run aurex pipeline --dry-run
```

The working cache at `engine/.cache/` is gitignored; a fresh clone either runs once
online or uses the committed seed cache shown above.

---

## Licence

MIT. See [LICENSE](LICENSE).

---

Aurex is a research and education tool. It produces probability distributions, not advice. Short-horizon price direction is not reliably forecastable, and nothing here changes that.
