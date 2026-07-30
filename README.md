# Aurex

A calibrated uncertainty engine for gold, priced in INR for an Indian retail buyer.

Aurex does not tell you where the gold price is going. It produces a probability
distribution, states what you have to beat to break even, and then scores its own
distributions in public.

> **Build status: step 1 of 7 complete** — data layer, tax schedules, and import
> parity. The volatility models, distribution engine, scenario tree, dashboard and
> benchmark results are not built yet. Sections marked *(not built yet)* are
> commitments, not claims.

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
| 2 | GJR-GARCH / HAR-RV / rolling vol; filtered historical simulation; t-copula on XAU × USDINR |
| 3 | Friction and P&L (retail, ETF, SGB); PIT, CRPS, Kupiec, Christoffersen, Brier; walk-forward from 2015 |
| 4 | Elastic-net factor attribution; market-sourced scenario priors |
| 5 | Next.js dashboard |
| 6 | Benchmark shootout vs random walk, AutoARIMA, NHITS, Chronos — **including the rows where Aurex loses** |
| 7 | Nightly automation and deploy |

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

---

## Quick start

```bash
cd engine
uv sync

uv run pytest                                    # 134 tests, no network
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
