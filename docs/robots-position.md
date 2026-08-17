# Does `robots.txt` bind Aurex?

**Status: decision brief. The decision is not made here and must not be inferred from the
ordering below.** Written 2026-08-17. Every fact in it was already recorded somewhere in
this repository before this page existed; nothing here was fetched to write it, and the two
hosts that publish a blanket `Disallow` were not contacted.

The question is forced rather than academic. `xauusd` — the anchor every published price
and every simulated path is built from — is fetched from a host whose `robots.txt` answers
401, and `usdinr`, which is *blocking*, is fetched from a host that publishes a blanket
refusal. Whatever answer is given here, something changes: either the nightly stops, or
this project states in writing that it fetches against a stated `Disallow`.

---

## 1. What is measured, and where it is written down

| Host | What its `robots.txt` says | Measured | Recorded in |
|---|---|---|---|
| `query1.finance.yahoo.com`, `query2.finance.yahoo.com` | exactly `User-agent: *` / `Disallow: /`, 26 bytes | 2026-08-17, with Aurex's own client | HANDOFF §5; `probe-lbma.yml` round-4 header |
| `finance.yahoo.com` (the HTML quote pages) | does **not** disallow them | 2026-08-17 | HANDOFF §5 |
| `stooq.com` | `User-agent: *` / `Disallow: /` to Aurex's client, **404 to `urllib`'s** | 2026-08-17 | `data/sources/http.py` `_read_robots`; `tests/test_sources.py::TestRobotsIsReadAsAurex` |
| `fsapi.gold.org` (World Gold Council) | `User-agent: *` / `Disallow: /` | round 2 | HANDOFF §5; `probe-lbma.yml` round-2 header |
| `prices.lbma.org.uk` | **HTTP 401.** RFC 9309 §2.3.1.3 and Aurex's own corrected checker both read that as a complete disallow | 2026-08-17 | `docs/lbma-enquiry.md` §0; `http.py` 401/403 branch |
| `fred.stlouisfed.org` | **never measured** | — | — |

And the part that decides the whole question, which is not about any host:

| Loader | Series it resolves | Does Aurex's robots guard run? |
|---|---|---|
| `IbjaReportLoader` | `ibja_gold` | **Yes** — plain `http.get`, guard consulted |
| `GprDailyLoader` | `gpr` | **Yes** — plain `http.get`, guard consulted |
| `LbmaGoldLoader` | `xauusd` | No — `check_robots=False` (`lbma.py:54`), documented, against the 401 |
| `FredLoader` | `real_yield_10y`, `dxy`, `wti`, `local_cpi`, `local_policy_rate`; and the `usdinr` and `vix` fallbacks | No — `check_robots=False` (`fred.py:53`), **no recorded reason anywhere** |
| `YahooLoader` | `usdinr`, `xau_futures`, `vix` primaries | **Structurally cannot** — `yahoo.py` never imports `http`; it calls `yf.download`, and `yfinance` does not consult `robots.txt` |

So of the eleven series this project loads, the guard governs **two**. Six are fetched
through a flag that turns it off — five of those with no reason recorded at any call site —
and three through a dependency that has never asked, two of which fall back onto the flag
when Yahoo fails. That is the state a position has to be taken against: not a clean rule
with one exception, but a rule that reaches only the two series that are neither the anchor
nor blocking.

---

## 2. The two positions

**Position A — `robots.txt` binds Aurex generally.** It is a statement of the operator's
wishes about automated access to their host. Aurex is automated access. The document does
not distinguish a crawler from a targeted read, so neither may we, and a `Disallow: /`
means every path including the one we want.

**Position B — `robots.txt` binds crawling, and Aurex is not crawling.** The standard grew
up around discovery: following links, enumerating a site, building an index. Aurex fetches
one named document per series, on a schedule, with a courtesy delay and an honest
`User-Agent` pointing at a contact route. Under this reading a blanket `Disallow: /` bars
indexing the site and does not bar retrieving a specific published file.

Both are arguable. Neither is chosen here.

---

## 3. Under Position A

**Breaks — `xauusd`, and therefore everything.** The 401 at `prices.lbma.org.uk` reads as a
total refusal, `check_robots=False` becomes indefensible rather than merely undocumented,
and the anchor series has no second source (that is §5's open problem). Every published
price, every simulated path, the nightly, the dashboard's *Today* view: all of it stops
until a compliant source for the anchor exists. **This is the consequence most likely to be
overlooked, because the discussion started with Yahoo.** Position A is not a Yahoo
question.

**Breaks — `usdinr`, which is blocking.** The `INR=X` primary goes. The `DEXINUS` fallback
survives only if FRED's `robots.txt` permits it, which nobody has measured, and even then
it is nominal: it is published weekly, and the freshness guard correctly refuses it on
about 88% of nights (§5 — 9 clearing nights in 77 measured). So the rupee lens has no
working source on almost every night, and the honest reading is that `usdinr` becomes
single-sourced-and-refused rather than merely degraded.

**Breaks — `xau_futures`, and with it HAR-RV.** `GC=F` is a chain of length one and it is
the asset's `ohlc_series_id`. Without measured OHLC there is no Parkinson realised
variance, and HAR-RV is *omitted rather than fed squared returns* (`bench/runner.py`), so
the six-model shootout and the direction run become five-model runs. Note this is a
different exclusion from the existing one: `GC=F` is already barred as a *price* source on
cost-of-carry grounds (+2.40% on 2026-07-29, which would land in the domestic premium and
read as a demand signal). Position A removes its remaining, legitimate use.

**Degrades — `vix`.** The `^VIX` primary goes; `VIXCLS` at FRED survives, subject to the
same unmeasured FRED question.

**Unknown, and it is the widest exposure — everything reached through FRED.**
`real_yield_10y`, `dxy`, `wti`, `local_cpi` and `local_policy_rate` are FRED-only, and both
surviving fallbacks above are FRED. All seven are fetched with the guard switched off and
nobody has read `fred.stlouisfed.org/robots.txt`. If it permits us, Position A costs the
prices and keeps the factor set; if it does not, Position A also takes out step 4's
attribution and the transmission chain, leaving `gpr` and `ibja_gold` — the two series the
guard already clears — as the only things this project could still load. **The FRED question
has to be measured before Position A can be costed at all**, and measuring it is one request
Position A itself permits.

**Reopens — nothing.** No candidate becomes available under the stricter rule. Position A
strictly shrinks the source landscape, and §5's finding that the landscape is already thin
is what makes that fatal rather than inconvenient.

---

## 4. Under Position B

**Breaks — nothing mechanical.** Every series resolves exactly as it does today. What
changes is what this project can say about itself, and the LBMA enquiry is where that bill
arrives first: `docs/lbma-enquiry.md` currently discloses the 401 and asks whether it is
policy or a Cloudflare artifact. Under Position B the same letter has to say that we read
it as a refusal and fetched anyway. That is still disclosable — it is a stated position
rather than an oversight — but it is a harder paragraph to write, and it must be written
before the letter goes.

**Reopens — `stooq.com`, and this is the substantive prize.** It was excluded from the
candidate list *solely* on the `Disallow`, and if that ground is scoped away the exclusion
has no remaining basis. What the record actually says about it, and nothing more:

- Endpoint: `https://stooq.com/q/d/l/?s=xauusd&i=d`, recorded in the round-2 probe under
  `robots_only` with the note *"free daily XAUUSD with full history; the obvious
  candidate"*.
- A `usdinr` symbol exists on the same endpoint shape — `?s=usdinr&i=d` was allow-listed in
  this repository's local tooling settings on an earlier round. **So a single reopened host
  would be a candidate for the anchor series *and* for the blocking FX series**, and unlike
  `DEXINUS` a daily file would sit inside the 4-day freshness budget rather than outside
  it. That makes stooq the only recorded candidate that addresses both single-source
  fragilities in §5 at once.
- It is on a different host from LBMA, which is the requirement §5 states for a second
  source (the Cloudflare failure is per-egress, not per-day, so a retry is mechanically
  dead and only a different host helps).

**What is not known about stooq, and it is most of what matters.** The probe never fetched
it, deliberately: *"a probe that ignores a Disallow to find out what is behind it has
answered a question nobody may act on."* So every test that disqualified fxratesapi is
unrun here — weekend carry-forward (the mechanism by which a repeated Friday quote
satisfies a freshness guard), agreement with the PM fix (fxratesapi missed by ≈ −0.34%),
fixing time, history depth, and whether the host serves a GitHub runner at all. **stooq
reopening makes it a candidate to be measured, not a repair.** Anyone reading this page as
"we can fix the anchor with stooq" has read it wrong; the honest claim is "we would be
permitted to go and find out".

**Reopens — `fsapi.gold.org`, uselessly.** The World Gold Council feed is recorded as
downsampled to ~491 points over the requested window, so it fails on the data
independently of robots. It reopens and stays rejected.

---

## 5. The asymmetry, plainly

**We rejected stooq on grounds we depend on at Yahoo.**

`stooq.com` and `query1/query2.finance.yahoo.com` serve the *same* file — `User-agent: *`,
`Disallow: /`. stooq was struck from the candidate list for it, in writing, with the
principle stated explicitly. Yahoo has been fetched against it nightly for the entire build
— `usdinr` (blocking), `xau_futures` and `vix` — and nobody noticed until 2026-08-17.

The difference between the two was never a principle. It was transport. A stooq loader
would have gone through `http.get`, where the guard runs; `YahooLoader` delegates to
`yfinance`, which never asks. The rule was applied to the candidate we had not written yet
and not to the three series we already depended on, and the deciding factor was which side
of a function boundary the request happened to fall on.

Two further asymmetries in the same shape, both smaller and both real:

- The guard was **fixed** on 2026-08-17 so it reads `robots.txt` as Aurex rather than as
  `urllib`. Before that fix stooq's file was invisible to us (404 to `urllib`, which the
  parser maps to allow-all). The stricter check arrived after the exclusion it would have
  justified, and it has never been applied to Yahoo, because it cannot reach it.
- `FredLoader` passes `check_robots=False` with **no recorded reason**, and it is the widest
  bypass in the codebase — five series' primary route and two more on fallback. The LBMA
  bypass at least has a documented one. Whatever position is taken, that flag needs either a
  reason or removal.

Naming this is not a reason to prefer Position B. A position taken because it retroactively
excuses what the code already does is not a position, and Position A's cost — the anchor
series, and with it everything — is the actual price of consistency in the other direction.
The asymmetry is a reason not to leave the question open, because *unresolved* is the one
state that lets both rules operate at once, each where it happens to be convenient.

---

## 6. What a §6 constraint would say

Under **Position A**:

> **Never fetch a path a host's `robots.txt` disallows, and never pass
> `check_robots=False`.** A 401 or 403 on `robots.txt` is a refusal, not silence. A
> blocking series with no compliant source is a series Aurex does not publish — the
> nightly refuses and says why, exactly as it does for a stale price. Removing the flag
> from `LbmaGoldLoader` stops the engine; that is the constraint working, not a bug in it.

Under **Position B**:

> **`robots.txt` governs crawling; Aurex retrieves named documents and does not crawl.**
> No discovery, no enumeration, no following links, one known URL per series, the courtesy
> delay and an honest `User-Agent` with a live contact route. Where a host publishes a
> blanket `Disallow`, the decision to fetch it anyway is **recorded per host** with the
> date, the body observed and the reason — a `Disallow` fetched silently is the failure,
> not a `Disallow` fetched deliberately. `check_robots=False` carries the recorded reason
> at the call site or comes out.

Either way, one more constraint is needed that belongs to neither position:

> **A source's robots status is measured before it is excluded, and re-measured with
> Aurex's own client.** The candidate list was shortened by a check that was reading
> somebody else's `robots.txt`.

---

## 7. Open questions either position has to answer

1. **What does `fred.stlouisfed.org/robots.txt` say?** Two fallbacks depend on it and
   nobody has looked. Position A's survivors are unknown until this is measured.
2. **Is LBMA's 401 policy or a Cloudflare artifact?** `docs/lbma-enquiry.md` asks. Under
   Position A the answer decides whether the engine can run at all; under Position B it
   decides how the disclosure paragraph is written.
3. **Does the stooq exclusion get withdrawn, and by whom?** It is recorded in a throwaway
   workflow. If Position B is taken, the withdrawal has to land somewhere that outlives
   the probe.
4. **Do the four fxratesapi tests get run against stooq before anything depends on it?**
   Weekend carry-forward, PM-fix agreement, fixing time, history depth, runner
   reachability. A source that clears robots and fails those is not a second source.
