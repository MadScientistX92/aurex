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
- **No overclaiming.** A CI guard greps for banned vocabulary ("beats the market", "institutional-grade", etc.).
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

`xauusd`, the anchor price series that every published price and every simulated path is built from, **has a source chain of length one**: `LbmaGoldLoader`. Compare `usdinr`, which carries Yahoo plus a FRED fallback. The one blocking series with no second source is the one everything depends on.

From GitHub runners it fails roughly **56% of the time** (4 resolved, 5 failed, 2026-08-05 to 08-13) with a 2xx response whose body is not JSON. From a residential IP it answered 12/12. On **2026-08-13 a runner resolved it at 04:05 UTC and another failed on it at 19:12** — same endpoint, same day — so this is per-run, not an outage anyone could wait out.

Two candidate mechanisms needing **opposite** repairs:
- **Cloudflare challenge page** → the host will not serve this client from that network; a second source on a **different host** is required.
- **Empty or truncated body** → transient; a courtesy-delayed retry inside the loader may be the whole fix, which is far smaller.

The evidence to distinguish them was being discarded — `http.get` raised for non-2xx *before* decoding, so `JSONDecodeError: Expecting value: line 1 column 1` was emitted identically by both. That is now fixed.

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

- **`direction.yml` has never executed.** It is committed and dispatch-only. ~2h wall time, zero attention. Its step *ordering* — re-run, then guard, then commit — has never actually happened.
- Four skip records (2026-08-06, 08-08, 08-10, 08-11) state the wrong cause. **They have not been rewritten and must not be** — their `verdict` fields were correct throughout; only the summarising sentence was wrong. The README names them in *Track record integrity* and explains why the misstatement was the dangerous kind: it pointed the reader at the freshness tolerance, which is the one knob that must not move.
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
