# Aurex — handoff

**Read this before touching anything.** It is the state of the project as of 2026-08-14, written so a new session can pick up without re-deriving context. The rules in §1 are not stylistic preferences; they are the reason the project has any credibility, and every one of them was written after something went wrong.

- **Repo:** https://github.com/MadScientistX92/aurex (public)
- **Dashboard:** https://aurex-five.vercel.app
- **Local:** `~/aurex` on a Mac mini. Engine in `engine/`, dashboard in `web/`, artifacts in `public-data/`.
- **Built with:** Claude Code, manual-approve mode. Agent memory lives in the Claude Code project memory directory.

---

## 1. The working method — diagnose, never assume

**This is the most important section. If you read nothing else, read this.**

When something fails, when a number looks wrong, or when you are about to explain a result: **investigate before asserting.** Read the log. Run the query. Check the artifact. Say "I don't know yet" and go find out. Never reason from what a failure *probably* was, never fill a gap with a plausible-sounding mechanism, and never propose a fix built on an unverified diagnosis.

This is not caution for its own sake. It is empirical. Across the build, the human collaborator advanced roughly **thirteen** confident hypotheses and **the data killed essentially all of them**:

| Claimed | Actually |
|---|---|
| India gold import duty is 6% | 15% (BCD 10 + AIDC 5), from May 2026 |
| CBIC Notification 27/2026 | 15–18/2026, dated 12 May 2026 |
| HS headings 7107/7109/7110/7111/7112/7118 | Two different lists — and gold bullion is **7108**, in neither |
| Touch-probability ≈ 2× terminal | 1.35× at 10 sessions (continuous vs discrete monitoring) |
| "The FHS mean is zero by construction" | Drift leaks back through undemeaned residuals |
| DM truncation lag = h−1 | Must derive from the sampling step; h−1 over-specifies at step>1 |
| 10-day σ ≈ 4.4% (pre-registration) | ≈3.2%; 4.4% came from crisis-period vol |
| Tail closes 9/10 of the Gaussian shortfall | Inverted: 91% volatility level, 9% tail shape |
| Vercel Root Directory = repo root | Must be `web/` with include-outside-files enabled |
| 2025 is the load-bearing LOYO year | **2017** is; dropping it takes p from 0.0069 → 0.6528 |
| Tie-ordering inflates the h=10 result | Backwards — position ordering made it look *weaker* |
| Cyclic shift under-represents trend alignment | It **over**-represents it on this sample |
| CI failure was the GPR/DesignError path | Price-series path; GPR resolved fine and never entered |

Not one of those reached the codebase as a wrong fix, because each went to a diagnostic prompt rather than an implementation prompt. **That is the entire method.** Preserve it.

The agent side has the same discipline and the same track record. Bugs caught by *reading output* rather than by a green test suite: `df_at_bound` at 59.9994 reporting `False`; a double FX division putting the rupee anchor at ₹330 instead of ₹17,784; `applies_at` scoring a hurdle at horizons it was never priced for; a `?? 1` fallback publishing a p-value of 1.00; the root `.gitignore` silently swallowing `web/lib/`; a README figure of 0.010 against an artifact value of 0.0095. **A green suite is not evidence. Look at the numbers.**

### Corollaries

- **A missing value that reads as a stronger claim is the signature failure mode here.** `?? 1` → "definitively no effect". A NaN → SPA p = 0.000, maximally significant. A `False` from a boundary check → "found fat tails". Hunt absences that read as confidence, not just crashes.
- **Report what you measured, not what the spec predicted.** When a brief says "expect roughly double" and you measure 1.35, report 1.35 and explain the gap. Never tune toward the brief.
- **A change is not made until it is pushed.** Reporting a change as done while it sits uncommitted in the working tree is the same class of error as a published number nobody can reproduce. This has happened; it cost two failed deploys.
- **Say when you are unsure.** "This is the leading hypothesis, not a proven mechanism, and it stays unproven until someone captures the body" is exactly the right register.

---

## 2. What Aurex is (§0, the philosophy)

> Short-horizon price **direction** is not reliably forecastable. Short-horizon **volatility** partly is. So Aurex never predicts a price — it produces a probability distribution, and then publicly grades how well-calibrated that distribution turned out to be.

Concretely, and enforced by tests:

- **No point forecasts anywhere.** Not in the API, not in the UI, not in the README. A number without an interval or a distribution behind it is a bug.
- **Every probability is scored.** A forecast that is never scored is marketing.
- **The null is the driftless random walk.** Any model that does not beat it ships anyway, labelled as not beating it. Negative results are published, not buried.
- **No overclaiming.** A CI guard greps user-facing text for a curated list of banned phrases: marketing vocabulary claiming to outperform a benchmark or to carry professional credentials, promises of certain gain or of no downside, and point-forecast language naming a level with no distribution behind it. The list is in `engine/tests/test_no_overclaiming.py`, which is deliberately the only file exempt from its own scan — the phrases are not repeated anywhere else, this line included, because a document that quotes them is a document the guard would have to be widened around rather than widened onto.
- **Every published figure is mechanically tied to the artifact that produced it.** `tests/test_readme_direction.py` and `tests/test_readme_factors.py` parse the README cell by cell against the JSON and fail the build on disagreement. This mechanism has already caught a real error.

---

## 3. Build state

**Done:** steps 1, 1.5, 2, 3a, 3b, 4, 6, 7, and most of 5.

| Step | What |
|---|---|
| 1 | Data layer, source chains with parquet cache, dated duty/GST schedules, import parity, `local_premium_bps` |
| 1.5 | Currency lenses (USD native, INR taxed-import), asset abstraction (`Asset` protocol) |
| 2 | GJR-GARCH / HAR-RV / rolling-std; filtered historical simulation with **paths preserved**; t-copula; first-passage statistics |
| 3a | PIT, CRPS, Kupiec (exact binomial at boundaries), Christoffersen, Diebold-Mariano with HAC |
| 3b | Routes × jurisdictions, friction profiles, breakeven hurdle, hurdle-clearing Brier |
| 4 | Elastic-net factor loadings, GPR index wired, crude→CPI→policy→rupee chain by local projections, fuel-excise control |
| 5 | Dashboard: **Today**, **Track record**, **Calculator**. Drivers and Scenarios **omitted, not stubbed** |
| 6 | Benchmark shootout (RW, GJR-GARCH, HAR-RV, AutoARIMA, NHITS, Chronos) + direction grading with SPA/MCS and minimum detectable effect |
| 7 | CI, nightly job with staleness refusal, track-record integrity, live log |

**Not built:** the scenario engine (remainder of step 5), the Drivers and Scenarios dashboard views (blocked on scenario work), and oil as a second asset (steps 8–9, specified in addenda §17/§18).

---

## 4. Results as they stand

**Calibration — the headline is a negative result.** Across 2,876 out-of-sample forecasts from January 2015, CRPS skill against the random walk runs between −0.4% and +0.9% depending on horizon, and Diebold-Mariano rejects at **none** of them; the smallest p in the table is 0.23. PIT uniformity survives KS at all five horizons; chi-square rejects at one.

**A withdrawn number.** An earlier version reported up to +4.6% CRPS skill. That was a model carrying sample drift beating a null denied one. It has been withdrawn, the residual pool is now centred by default, and — measured afterwards — it was never significant anyway (p = 0.27). Centring made *every* published number worse and it was taken anyway.

**Direction carries no information.** Resolution at or near zero for every model at every horizon, including Chronos zero-shot. One cell (HAR-RV, h=10, equal-count screen) rejects at p = 0.036 / 0.025 thinned. It is published as a finding and read as noise, for three stated reasons: multiplicity across ten uncorrected cells; effect size of 0.011 resolution against 0.248 uncertainty; and leave-one-year-out showing **2017 is load-bearing** — dropping that year alone takes the marginal p from 0.0069 to 0.6528 and puts the resolution below its own null. Two earlier published reasons were withdrawn after controls contradicted them, and the correction is visible in the README rather than smoothed over.

**Tail shape.** The claim that filtered historical simulation closed "nine tenths of the Gaussian shortfall" was **inverted** on proper baselining. Held to the engine's own forecast variance, the split is 91% volatility level and 9% tail shape, and on seven of nine rows the engine is *thinner*-tailed than a variance-matched Gaussian (skew −0.38). Corrected in all three places that inherited it.

**Step 4 against its pre-registration.** Predictive out-of-sample R² = −0.00218, DM p = 0.8346 — **hit**. Contemporaneous R² = 0.18699 — **missed**, just below the registered 0.2–0.45. Stability **split**: five of six drivers change sign across rolling windows, but the rolling spread never exceeds the full-sample interval. Chain band spans zero at every horizon — **hit**. The two routes disagree (+0.0027 chain-implied vs −0.13401 direct at six months) — **hit**, published as a finding and not reconciled.

**The GPR result is the most instructive.** The geopolitical-risk index was wired as a hard prerequisite specifically to prevent an inverted safe-haven sign. Removing it moves the largest surviving loading by **0.0000157** — twenty-five times less than any other driver — with no sign flips anywhere, and out-of-sample R² goes *up*. The bias identified in advance **did not materialise in the loadings on this sample**. That is published as the result. "We checked and it was negligible" and "we did not check" produce identical coefficients and are not the same claim.

**Friction is the thesis.** Same metal, same distribution, same day: P(profit) is 0.483 through an oil-style CFD route and 0.089 through Indian retail. Breakeven hurdle 9.37% (India retail) vs 0.14% (CFD). The hurdle is irrelevant at low friction and decisive at high friction — that is the whole point, and the CFD route's 293 "clears" at five sessions are 293 coin flips landing heads, not 293 wins.

---

## 5. The open problem — start here

`xauusd`, the anchor price series that every published price and every simulated path is built from, **has a source chain of length one**: `LbmaGoldLoader`. `usdinr` looks better — Yahoo plus a FRED fallback — but that redundancy has since been measured and is nominal; see below. The one blocking series with no second source is the one everything depends on, and the one that appears to have a second source has one that cannot arrive in time.

From GitHub runners it fails roughly **56% of the time** (4 resolved, 5 failed, 2026-08-05 to 08-13) with a 2xx response whose body is not JSON. From a residential IP it answered 12/12. On **2026-08-13 a runner resolved it at 04:05 UTC and another failed on it at 19:12** — same endpoint, same day — so this is per-run, not an outage anyone could wait out.

**Mechanism, settled.** Probe run 31906804255: 20/20 attempts returned HTTP 200 carrying a complete ~12KB Cloudflare interstitial ("One moment, please…", `cf-edge-cache: no-cache`), 20 distinct bodies, none converting. Not a truncated body. Its only remedy is a JavaScript reload after five seconds, and `http.get` calls the module-level `requests.get`, which builds and discards a `Session` per call — so no state survives to be sent back and **a courtesy-delayed retry is mechanically dead**. Probe run 31974956815 then answered 5/5 clean from a different runner IP, which fixes the shape of the problem: per-egress, not per-day. The repair is a second source on a different host.

**The candidate landscape is thin, and that is a finding rather than a gap in the search.** FRED's LBMA fix was removed with all ICE Benchmark Administration data on 2022-01-31. ICE, which administers the benchmark, publishes no unauthenticated endpoint at all — delivery is ICE Connect / Data API / Data Files / Global Network, under licence, contact `iba-licensing@ice.com` — and LBMA moved its own historic tables to the members' portal in the week commencing 2025-11-24, keeping only the latest daily auction price and chart data public. The Bundesbank's directory holds exactly one gold price series, `BBEX3.D.XAU.DEM.EA.AC.C01`, the Frankfurt fixing in D-marks per kilo, last observation **1998-12-30**. stooq.com and fsapi.gold.org publish `User-agent: * / Disallow: /`. fxratesapi serves a runner but is rejected on the data: weekend carry-forward that would satisfy the freshness guard with a repeated Friday quote, ≈ −0.34% against the PM fix, wrong fixing time, and a 366-day history wall. **The status of the LBMA JSON feed under the portal move is itself unresolved, and an enquiry to LBMA is drafted at `docs/lbma-enquiry.md`.**

### The second single-source fragility: `usdinr`'s fallback is nominal

Measured 2026-08-17, and the reason it matters is that `usdinr` is *blocking* — the rupee lens converts through it, so a nightly that cannot resolve it publishes nothing.

The fallback is FRED `DEXINUS`, which carries daily observations but is **published weekly**. Evidence, not inference: 76 consecutive ALFRED vintages (2026-06-02 → 2026-08-16, no errors) show the newest available observation changing on exactly 11 dates, every one a Monday except 2026-06-02, which followed a Monday holiday — and each release reaches only the **preceding Friday**. The Federal Reserve's own H.10 schedule says the same: released "On Mondays at 4:15 p.m.", next business day if Monday is a holiday, covering "the previous business week".

`usdinr` declares `max_lag_days=4` (`engine/aurex/assets/gold.py`), and the engine refuses when `(run_date − last_observation).days > 4` (`engine/aurex/data/freshness.py`). The nightly runs at 02:00 UTC, *before* the 20:15 UTC Monday release. So the lag a nightly would actually see is:

| Night (02:00 UTC) | On-time week (release Mon 20:15 UTC) | Lag | Holiday week (release slips to Tue) | Lag |
|---|---|---|---|---|
| Mon | Friday of the week before last | 10 | Friday of the week before last | 10 |
| Tue | previous Friday | **4 — clears** | still Friday of the week before last | 11 |
| Wed | previous Friday | 5 | previous Friday | 5 |
| Thu | previous Friday | 6 | previous Friday | 6 |
| Fri | previous Friday | 7 | previous Friday | 7 |
| Sat | previous Friday | 8 | previous Friday | 8 |
| Sun | previous Friday | 9 | previous Friday | 9 |

**≈12% of nights, not one in seven.** An on-time week clears on exactly one night, Tuesday, at the boundary. A week whose release slips to Tuesday clears on **no night at all**: the Tuesday run happens before that day's release and sees a lag of 11, and by Wednesday the newly released data is already 5 days old. Nine of the eleven release weeks measured were on time, so the rate is **9 clearing nights in 77 ≈ 12%**. On every other night, if Yahoo `INR=X` fails, the chain resolves `DEXINUS` successfully and the freshness guard then correctly refuses it, so the run skips exactly as if there were no fallback at all. The redundancy is real in the source chain and absent in the freshness budget: the tolerance was derived for a daily-published FX series, which the primary is and the fallback is not.

**Do not fix this by moving the tolerance.** The number is right for what it guards. The honest repairs are a fallback that publishes daily, or a declared acceptance that `usdinr` is single-sourced in practice.

### Current branch: `fix-skip-reason-and-probe-lbma` (pushed, **not merged**)

Two commits:
- `a2907e4` — skip reasons derived from verdicts instead of a hardcoded string, plus `http.get_json` carrying status, headers (`content-type`, `server`, `cf-ray`, `cf-mitigated`, `cf-cache-status`), byte count and the first 200 bytes `repr`'d.
- `ab29bfb` — `.github/workflows/probe-lbma.yml`, a **throwaway** 20-attempt probe that records what a runner actually receives. `permissions: contents: read`; writes nothing to the repo. Delete once the mechanism is known.

**The probe cannot be dispatched from a branch** — GitHub only offers `workflow_dispatch` for workflows present on the default branch. It must land on `main` first.

### Immediate sequence

1. Merge `fix-skip-reason-and-probe-lbma` to `main`, push.
2. Dispatch **Probe LBMA** (`gh workflow run probe-lbma.yml`, or Actions tab → Probe LBMA → Run workflow).
3. Download the workflow artifact and **read it before proposing anything.**
4. Then decide the repair: second source (different host; **GC=F is excluded** — its cost-of-carry basis, +2.40% on 2026-07-29, would land in the domestic premium and read as a demand signal) versus courtesy-delayed retry.
5. Delete the probe workflow.
6. Separately: seed cache for `factors.yml` only — see §6.

### Also outstanding

- **`direction.yml` executed for the first time on 2026-08-17 (run 31983796753) and was killed by its own ceiling at 94%.** What it settled: `xauusd` resolved live from LBMA on the first attempt, so the resolution gate costs about a minute, not a retry loop. What killed it: `timeout-minutes: 360`. The NHITS seed advances by 5,000 per window, which is how the log can be read at all — **547 windows in 5h58m, ~39.3s each, against the ~582 the committed artifact implies.** It needed roughly 23 more minutes. The "~2h wall time" figure in this section was a local-machine number; on a 4-vCPU runner it is ≈6.4 hours. Raise the ceiling, or cut the work (`--step 10` roughly halves the windows), but do not read 360 as adequate.
  - **Its step ordering is now half-tested.** Steps 7, 8 and 10 — guard, README test, commit — were **skipped**, `git add` never ran, and `origin/main` carries no direction commit. So a failure at or after the re-run stages and commits nothing, as designed. The guard and README steps themselves have still never run against a fresh artifact.
  - **The `if: always()` upload published a decoy, and this is the part to fix first.** `aurex direction` writes `public-data/direction.json` once, at the end. A cancelled run therefore leaves the *checked-out* copy in the workspace, and the upload ships that: the artifact from this run is byte-identical to the one already committed on `main` (sha256 `f1a34f34…`). `if-no-files-found: error` protects nothing, because the file exists. An artifact named `direction` from a run that computed nothing is indistinguishable from one that computed everything unless you hash it — the §1 signature failure exactly. Cheapest honest repair: delete `public-data/direction.json` before the run so the guard has something real to fail on.
  - **Six hours produced no progress output.** Between 01:04:52 and the cancellation the only stdout was `lightning_fabric` seed lines. There is no way to tell working from hung, and no ETA.
- Four skip records (2026-08-06, 08-08, 08-10, 08-11) state the wrong cause. **They have not been rewritten and must not be** — their `verdict` fields were correct throughout; only the summarising sentence was wrong. The README names them in *Track record integrity* and explains why the misstatement was the dangerous kind: it pointed the reader at the freshness tolerance, which is the one knob that must not move.
- **Every Yahoo series is fetched against a blanket `Disallow`.** `query1.finance.yahoo.com` and `query2.finance.yahoo.com` — where `yfinance` fetches — serve exactly `User-agent: * / Disallow: /` (26 bytes, verified 2026-08-17 with Aurex's own client). `YahooLoader` delegates transport to `yfinance`, which never consults `robots.txt`, so `usdinr` (blocking), `xau_futures` and `vix` have resolved that way for the whole build. This is recorded here because the probe workflow that found it is throwaway and the finding must outlive it. Two honest options and no third: stop using Yahoo, or decide deliberately and in writing that this one is fetched anyway. `finance.yahoo.com` itself does **not** disallow the quote pages; only the API hosts do.
- Five pre-existing `mypy --strict` errors in `aurex/score/shootout.py` and `bench/`. Present on `main` before this work.
- `bench/` sits at 0% coverage, deliberately. **Do not add an omit** — a visible zero on 192 statements is information. The gate passes at 89%.

---

## 6. Standing constraints — do not violate

Each of these exists because violating it was tempting at some specific moment.

- **Never widen a freshness tolerance to make a run pass.** On all four skipped nights it would have published nothing truer and would have retired the signal that the anchor series has a single point of failure.
- **Never make `d_geopolitical_risk` optional again.** It is `required=True` so attribution refuses rather than degrading. A dropped-out reason recorded in a field nobody reads is not a defence against an inverted sign.
- **Never sum a chain response with a factor loading.** Crude is in both; they are alternative decompositions of overlapping paths. A test asserts no field in the emitted block is named like a total.
- **Never rewrite an elapsed forecast or a skip record.** Correct in code and note the correction in the README. Same rule that governs the withdrawn +4.6%.
- **Seed cache is correct for `factors.yml`, wrong for `nightly.yml`.** `aurex factors` applies no freshness guard and makes no claim about today's price, so a cached series ending weeks ago runs correctly and dissolves nothing. The nightly is the opposite: a cache lets a stale copy resolve, which the guard then correctly refuses. Seeding it buys nothing honest. **If `factors.yml` runs from a cache, the artifact must name the cache's end date** — a run reproducible only from a cache nobody can see is the `--to` problem wearing a different hat.
- **Bound every published run with `--to`.** Default to the series' own last observation, never to today. The artifact records requested *and* resolved bounds plus the exact reproducing command.
- **Pre-register before running.** Every graded pre-registration so far has been more useful than the number it predicted, including the two that missed. Leave the prediction as written when it misses; explain the miss.
- **Leak guard:** no asset name and no jurisdiction code (uppercase alpha-3, case-sensitive) as a literal outside `assets/` and the routes table — including in `web/` (`.ts`, `.tsx`, `.css`). It has already caught a driver named in a UI caption.
- **Provenance:** every external fact carries `source_url` and `source_confidence`. `primary` = the publisher's own file; `secondary` = a redistributor (FRED, Yahoo). `cite_as` is filled only where the publisher states one. Never mark something primary without reading the primary document. Where a schedule does not know a value, declare a **gap** rather than carrying the last value forward — the fuel-excise schedule does exactly this for Oct 2018–Nov 2021.
- **HAR-RV must be unavailable for any leveraged route.** Deterministic variance iteration means every path shares one trajectory, so barrier and liquidation statistics from it silently assume constant volatility.
- **Multiple comparisons.** Six models against one null at α = 0.05 finds a winner whether or not one exists. Use Hansen's SPA or the Model Confidence Set; per-model DM statistics are description, not decision.

---

## 7. If you need more detail

The README is the source of truth and is long (~750 lines) and heavily annotated — methodology, results, limitations, and the corrections made along the way. The agent's memory files in the Claude Code project memory directory encode the same standards as short rules and have been audited for drift.

Four spec addenda exist as separate documents from the design conversation: §15–16 (USD lens, asset abstraction — both built), §17 (oil module — not built), §18–19 (oil as a traded instrument, India transmission chain — chain built, oil not), §20 (routes and jurisdictions — built).

---

**Last note.** The most credible thing in this repository is not any positive result. It is the collection of negative ones that survived contact with the data, published with the arithmetic that proves them, alongside a visible record of the claims that were withdrawn when they did not. Protect that. It is worth more than any finding this project could produce.
