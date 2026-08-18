# Does `robots.txt` bind Aurex?

**Status: decided 2026-08-19. The decision is §0; everything else in this document is the
costing that produced it and is left exactly as it was written, including the sentences the
decision overrules.** Written 2026-08-17 as a decision brief, revised the same day. Every fact in it was already
recorded in this repository, with **one exception made deliberately**: `fred.stlouisfed.org`
was fetched once, because Position A could not be costed without it and reading a
`robots.txt` is the request every position permits (§1a). The two hosts that publish a
blanket `Disallow` were not contacted, and nothing behind a `Disallow` was fetched.

The question is forced rather than academic. `xauusd` — the anchor every published price
and every simulated path is built from — is fetched from a host whose `robots.txt` answers
401, and `usdinr`, which is *blocking*, is fetched from a host that publishes a blanket
refusal. Whatever answer is given here, something changes: either the nightly stops, or
this project states in writing that it fetches against a stated `Disallow`.

**Three options are costed, not two.** §2–§4 are the two readings of `robots.txt`. **§4a**
is a third and separate one — workflow-level retry — which answers the *other* refusal, the
Cloudflare interstitial, and is costed here because it raises the same question about
proceeding past a machine-readable "no". It is **not implemented**, and this document does
not recommend it.

---

## 0. The decision, 2026-08-19

**An explicit `Disallow` is honoured. An ambiguous refusal is unknown, and unknown is not
permission.**

Neither position as §2 frames them. The costing was built around crawling versus
retrieval, and that turned out not to be the line that decides anything here: what
separates the two hosts this project actually depends on is not what we do with the file
but **how clearly its operator said anything at all.**

**Where a host serves a `robots.txt` that names a path Aurex fetches, that binds Aurex.**
It binds whether or not the fetch is a crawl, whether or not one document per night is
modest, and whether or not the data is unobtainable anywhere else. The three things that
follow are not softened:

- `stooq.com` and `fsapi.gold.org` **stay excluded**, on the `Disallow` and on nothing
  else. §4 calls stooq "the substantive prize" and names it the only recorded candidate
  that could answer both single-source fragilities at once. That prize is not claimed. The
  four measurements §4 lists as unrun against it stay unrun, because a probe that ignores a
  `Disallow` to find out what is behind it has answered a question nobody may act on.
- `query1.finance.yahoo.com` and `query2.finance.yahoo.com` serve the same 26-byte file, so
  `usdinr` (**blocking**), `xau_futures` and `vix` are **in breach of this decision from the
  day it is written**, and will stay in breach until they are re-sourced or dropped. That is
  a deadline, not a footnote, and it is recorded in HANDOFF §6 as one. The asymmetry §5 of
  this document names — stooq struck for the file Yahoo serves us nightly — is resolved in
  the strict direction, which is the only direction that resolves it: both are out.
- The mechanism that let Yahoo through, `yfinance` never consulting `robots.txt`, is not a
  defence. A rule that reaches only the requests that happen to go through `http.get` is a
  rule about a function boundary.

**Where a host answers 401 or 403 on `robots.txt` itself, nothing has been stated to us.**
That is a different fact from a `Disallow`, and collapsing the two in either direction is
the error. Read as refusal, it stops every published price on an inference nobody has
verified — the file behind a CDN that also serves this host's interstitial is at least as
likely an edge artifact as a policy. Read as permission, it is the convenient reading of a
silence, which is what this document exists to refuse. So it is read as **unresolved**, and
an unresolved question is handled by asking it:

1. **Disclose** it wherever the source is described — the loader's call site, this
   document, and the enquiry itself.
2. **Ask the administrator directly.** `docs/lbma-enquiry.md` is that letter.
3. **Continue the single nightly fetch while the enquiry is open**, unchanged in rate,
   pattern or identification. Continuing is not a finding in our favour and must never be
   described as one.
4. **Stop the day they say stop** — and stop on an explicit `Disallow` appearing at that
   host, without waiting for a reply.

`prices.lbma.org.uk` is the only host in this state. `LbmaGoldLoader`'s `check_robots=False`
therefore names one open question rather than granting a general bypass, and it must not be
copied to a host that publishes an explicit `Disallow`.

**No re-rolling of egress addresses, under any position.** §4a stays costed and **unbuilt**.
Disclosure cannot cure it, because the thing disclosed would be that we re-run a job until
we draw an address the host's edge does not challenge — which is looking for a door that is
not being watched. It is the same convenient reading of a machine-readable "no" that the
paragraph above refuses, and a project cannot hold one standard for the refusal that is
expensive to honour and another for the one that is cheap.

**What this costs, stated once and not softened.** No second source for `xauusd` reopens.
`usdinr`'s only compliant fallback is FRED's weekly-published `DEXINUS`, which the freshness
guard correctly refuses on about 88% of nights, so the rupee lens is single-sourced in
practice and will stay that way until a compliant daily FX source is found. `xau_futures`
going takes measured OHLC with it, and HAR-RV is omitted rather than fed squared returns, so
the shootout and the direction run become five-model runs. Those are the terms. They were
costed in §3 before the decision and none of them is a reason to reopen it.

**What changed in the code on the day of the decision.** `FredLoader`'s `check_robots=False`
came out — measured unnecessary in §1a, and a bypass that is unnecessary is
indistinguishable from the outside from one that is load-bearing. `LbmaGoldLoader`'s stayed
and was re-documented as the one open question above. Nothing else. In particular no loader
was added, removed or re-pointed: the Yahoo breach is named, not yet repaired, because the
replacement candidates are still being measured and a series dropped before its replacement
is measured is a nightly that publishes nothing while we find out.

---

## 1. What is measured, and where it is written down

| Host | What its `robots.txt` says | Measured | Recorded in |
|---|---|---|---|
| `query1.finance.yahoo.com`, `query2.finance.yahoo.com` | exactly `User-agent: *` / `Disallow: /`, 26 bytes | 2026-08-17, with Aurex's own client | HANDOFF §5; `probe-lbma.yml` round-4 header |
| `finance.yahoo.com` (the HTML quote pages) | does **not** disallow them | 2026-08-17 | HANDOFF §5 |
| `stooq.com` | `User-agent: *` / `Disallow: /` to Aurex's client, **404 to `urllib`'s** | 2026-08-17 | `data/sources/http.py` `_read_robots`; `tests/test_sources.py::TestRobotsIsReadAsAurex` |
| `fsapi.gold.org` (World Gold Council) | `User-agent: *` / `Disallow: /` | round 2 | HANDOFF §5; `probe-lbma.yml` round-2 header |
| `prices.lbma.org.uk` | **HTTP 401.** RFC 9309 §2.3.1.3 and Aurex's own corrected checker both read that as a complete disallow | 2026-08-17 | `docs/lbma-enquiry.md` §0; `http.py` 401/403 branch |
| `www.rbi.org.in` | **HTTP 418** with a WAF block page ("Unauthorised Access", 626 bytes, a support ID). Aurex's guard returns **True** for it — only 401 and 403 are read as withheld, so an ambiguous status reads as permission | 2026-08-19, with Aurex's own client | HANDOFF §6, second bullet; probe round 5 |
| `data.rbi.org.in` | 404, no file. Nothing to apply | 2026-08-19 | probe round 5 |
| `www.fbil.org.in` | **Not measured — read timeout at 20s, repeatedly**, while the data endpoints on the same host answered in 144ms. The guard fails open on an unreachable file, so it would fetch; under §0 that is an unmeasured host, not a permitted one | 2026-08-19 | probe round 5 |
| `www.ecb.europa.eu` | Served, 1,141 bytes, and **specific**: `Disallow`s for translated `_content.*.html` pages, some video and asset directories, none matching `/stats/eurofxref/`. Also **`Crawl-delay: 5`**, which nothing in this codebase parses and `HTTP_COURTESY_DELAY` (1.0s) does not meet | 2026-08-19, with Aurex's own client | probe round 5; HANDOFF §5 |
| `fred.stlouisfed.org` | **Permits everything Aurex fetches.** `User-agent: *` gets `Crawl-delay: 1` and six specific `Disallow`s — `/graph/graph-landing.php`, `/graph/image.php`, `/graph/fredgraph.png`, `/searchresults`, `/fred-glance-widget.php`, `/seriesBeta`. None matches `/graph/fredgraph.csv`. | 2026-08-17, with Aurex's own client, one request | §1a below |

### 1a. FRED, measured

The one gap that stopped Position A being costable is closed. `https://fred.stlouisfed.org/robots.txt`, fetched once on 2026-08-17 with Aurex's own client and `User-Agent`: **HTTP 200, `text/plain`, 960 bytes, `Server: Apache`.** Four groups — `GPTBot`, `ChatGPT-User` and `Google-Extended` are each held to `Crawl-delay: 2592000` (one crawl a month), and then:

```
# Allows all robots to visit all files.
User-agent: *
Crawl-delay: 1
Disallow: /graph/graph-landing.php
Disallow: /graph/image.php
Disallow: /graph/fredgraph.png
Disallow: /searchresults
Disallow: /fred-glance-widget.php
Disallow: /seriesBeta
```

Aurex matches `*`. `robots_allows()` returns **True** for every path involved: `/graph/fredgraph.csv?id=DEXINUS` (what `FredLoader` actually fetches), the same for `VIXCLS`, the bare CSV endpoint, `/series/<code>` (cited as `source_url`, never fetched) and the site root.

**The line FRED drew is deliberate, not incidental.** `/graph/fredgraph.png` is disallowed and `/graph/fredgraph.csv` is not — same endpoint family, one extension apart. The rendered image is barred; the underlying data is not. This is a publisher who thought about it and permitted exactly the thing Aurex reads, which is a stronger fact than a blanket `Allow` would have been.

**One honest caveat about the delay.** `Crawl-delay: 1` is what FRED asks of `*`, and `HTTP_COURTESY_DELAY` is 1.0 second per host, so Aurex complies — **by coincidence**. Nothing in the codebase parses `Crawl-delay`; a host asking for more would not be honoured. That is a small, separate gap, and it is now the only unmeasured thing in this table.

And the part that decides the whole question, which is not about any host:

| Loader | Series it resolves | Does Aurex's robots guard run? |
|---|---|---|
| `IbjaReportLoader` | `ibja_gold` | **Yes** — plain `http.get`, guard consulted |
| `GprDailyLoader` | `gpr` | **Yes** — plain `http.get`, guard consulted |
| `LbmaGoldLoader` | `xauusd` | No — `check_robots=False` (`lbma.py:54`), documented, against the 401 |
| `FredLoader` | `real_yield_10y`, `dxy`, `wti`, `local_cpi`, `local_policy_rate`; and the `usdinr` and `vix` fallbacks | No — `check_robots=False` (`fred.py:53`), **no recorded reason anywhere**. Now known to be *unnecessary*: the guard would pass (§1a), so the flag can come out with no change in behaviour |
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

**Survives, and this is now measured rather than assumed — everything reached through FRED
is permitted.** `real_yield_10y`, `dxy`, `wti`, `local_cpi` and `local_policy_rate` are
FRED-only, and both surviving fallbacks above are FRED. That was the widest exposure in this
document and the reason Position A could not be costed; §1a closes it. **Position A costs
the prices and keeps the entire factor set.** Step 4's attribution, the transmission chain,
the `DEXINUS` and `VIXCLS` fallbacks and the geopolitical-risk control all survive it
untouched, and `FredLoader`'s `check_robots=False` turns out to be hiding nothing — the
guard would pass, so the flag can be removed with no change in behaviour whatsoever.

So the cost of Position A is now bounded and specific rather than open-ended: **`xauusd`,
`xau_futures` and the `usdinr`/`vix` primaries.** Everything else this project loads is
already compliant. That is a much smaller bill than this document could state a day ago —
and it is still a fatal one, because the first item on it is the anchor.

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

## 4a. A third option: workflow-level retry — costed here, **not implemented**

The two positions above are about `robots.txt`. This one is about the *other* refusal —
Cloudflare's interstitial at `prices.lbma.org.uk` — and it belongs in this document because
it is the same question wearing different clothes: an operator has said something in a
machine-readable way, and we are deciding whether to proceed anyway.

**The mechanism, and why only this kind of retry could work.** An in-process retry is
already known to be dead: the interstitial's only remedy is a JavaScript reload after five
seconds, and `http.get` calls the module-level `requests.get`, which builds and discards a
`Session` per call, so no state survives to be sent back. What the record does show is that
the failure follows the **egress address**, and that the address is drawn per job and fixed
within one. Read off the four probe artifacts rather than the run summaries:

| Probe run | Date | Egress address | LBMA arm | Distinct addresses *within* the job |
|---|---|---|---|---|
| 31906804255 | 2026-08-15 | `52.188.198.99` | **0 of 20** | 1 |
| 31974956815 | 2026-08-16 | `135.232.208.115` | 5 of 5 | 1 |
| 31981226755 | 2026-08-17 | `68.154.54.35` | 5 of 5 | 1 |
| 31984657960 | 2026-08-17 | `172.172.87.80` | 5 of 5 | 1 |

Four sittings, four addresses, one dead and three clean. Every attempt record in every job
carries the same address as its siblings, so "fixed within a job" is measured and not
assumed — and the two sittings on 2026-08-17 drew **different** addresses, which is what
makes it per-job rather than per-day. So the only retry that could change anything is one
that gets a **new runner**, i.e. a job-level or workflow-level retry.

**It would not be a guaranteed fix, either.** Failure rates from the record: 5 of 9 nightlies
(2026-08-05 to 08-13) and 1 of 4 probe sittings — 46% pooled, and the two disagree enough
that neither should be quoted alone. At *p* = 0.56 a three-job retry still fails 18% of the
time; at *p* = 0.25, 1.6%. It converts a coin-flip into a long shot, not into a source.

**What it is, said plainly. Re-rolling egress addresses until one is not challenged is
evasion.** It does not solve the challenge — nothing here executes the JavaScript — it
looks for a door that is not being watched. The data behind that door is public and
unauthenticated and we are not forging anything, and that is exactly the argument that would
be available for Yahoo too. **This belongs on the same ledger as the Yahoo question**: in
both cases a host has stated a boundary in a way software can read, and in both cases the
proposal is to proceed because proceeding is technically easy and the data is otherwise out
of reach. A project whose most credible asset is its negative results does not get to hold
two standards, one for the refusal that is expensive to honour and one for the refusal that
is cheap.

**If it is adopted, it must be disclosed in the LBMA letter before that letter is sent.**
`docs/lbma-enquiry.md` currently discloses the `robots.txt` 401 and asks whether it is
policy or an artifact. A retry policy of this kind makes that paragraph materially harder —
it would have to say that we re-run the job until we draw an address their edge does not
challenge — and a letter that asks about one boundary while silently working around a second
is worse than no letter. The disclosure is the price of the option, not an optional extra.

**And if it is adopted, two things follow that are worth wanting.**

1. **`xauusd` no longer needs a second source.** §5's open problem — the anchor series with a
   source chain of length one — closes by making the one source reliable enough rather than
   by finding another. Note what it does *not* fix: `usdinr`'s fallback is still nominal
   (that is a Yahoo/`DEXINUS` problem and untouched by any of this), and the robots question
   is still open, because this option answers a different refusal.
2. **The second-source search becomes a publishable negative result rather than an
   unfinished task.** The landscape is already mapped and the map is the finding: FRED
   dropped the LBMA fix with all IBA data on 2022-01-31; ICE publishes no unauthenticated
   endpoint and licenses delivery; LBMA moved its historic tables to the members' portal in
   the week of 2025-11-24; the Bundesbank's only gold series ends 1998-12-30; stooq and
   `fsapi.gold.org` serve `Disallow: /`; fxratesapi fails on the data, not the transport;
   `GC=F` is excluded on cost-of-carry. **"The world's benchmark gold price has no second
   public source a cookie-less client can read"** is a real result about public data
   availability, and this project's whole method is that a negative result published with
   the arithmetic behind it beats an open TODO.

**Not implemented, and nothing in this repository has been changed to enable it.** This
section exists so the option is costed beside the other two, with its price named.

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
  bypass at least has a documented one. Now that §1a has measured the host, this one is the
  easiest thing in this document to settle: FRED permits every path we fetch, so the flag
  buys nothing and removing it changes no behaviour. **A bypass that was never needed is
  still worth removing** — it is indistinguishable, from the outside, from one that is
  load-bearing, and it is the reason this document had to spend a day not knowing whether
  Position A cost five series or none.

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

And if **§4a's retry** is adopted, on top of whichever position:

> **Retrying for a different egress address is evasion, and is disclosed wherever the
> source is described.** Not in a comment — in the artifact's provenance, in the README's
> data-sources section, and in the enquiry to the administrator before it is sent. A retry
> policy adopted and not disclosed is the failure; adopted and disclosed is a position.
> And the retry is bounded and counted: a job that needed four attempts records four, so
> the rate is visible rather than smoothed away by the loop that hides it.

Either way, one more constraint is needed that belongs to neither position:

> **A source's robots status is measured before it is excluded, and re-measured with
> Aurex's own client.** The candidate list was shortened by a check that was reading
> somebody else's `robots.txt`.

---

## 7. Open questions either position has to answer

1. ~~**What does `fred.stlouisfed.org/robots.txt` say?**~~ **Answered 2026-08-17 — see §1a.**
   It permits every path `FredLoader` fetches, so all seven FRED-reached series survive
   Position A and `check_robots=False` there is unnecessary rather than load-bearing. The
   residue is smaller and separate: nothing in the codebase parses `Crawl-delay`, and Aurex
   complies with FRED's by coincidence.
2. **Is LBMA's 401 policy or a Cloudflare artifact?** **Still open, and it is now the only
   question in this document whose answer changes what Aurex does.** `docs/lbma-enquiry.md`
   asks it. §0 says how it is handled while unanswered — disclose, ask, keep the single
   nightly fetch, stop when told — so the engine runs meanwhile and the reader is told on
   what footing.
3. ~~**Does the stooq exclusion get withdrawn, and by whom?**~~ **Answered by §0: it is not
   withdrawn.** It now rests on a stated position recorded here rather than on a line in a
   throwaway workflow, which is what the question was actually asking for.
4. ~~**Do the four fxratesapi tests get run against stooq?**~~ **Moot under §0** — the tests
   are not run because the source is not a candidate. The question survives in a different
   form for every *new* candidate: weekend carry-forward, PM-fix agreement, fixing time,
   history depth and runner reachability are what separate a transport that answers from a
   source, and a candidate that clears robots and fails those is not a second source.
5. **What replaces the three Yahoo series, and by when?** New, and it is the open item §0
   creates. `usdinr` is blocking, so the deadline it carries is the sharpest one in this
   repository.
