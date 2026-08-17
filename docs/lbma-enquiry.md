# Draft enquiry to LBMA — access route for the daily gold price JSON

**Status: DRAFT. Not sent.** Written 2026-08-17. Nothing here should be sent without the
maintainer reading it first and filling in the two placeholders.

## Why this is being sent at all

Two things changed the picture:

1. LBMA moved its historic benchmark price tables to the members' portal in the week
   commencing **2025-11-24**, keeping "the most recent daily auction price … and chart
   data" public
   ([announcement](https://www.lbma.org.uk/articles/lbma-benchmark-prices-data-tables-are-moving)).
   Aurex reads `https://prices.lbma.org.uk/json/gold_pm.json`, which still serves the
   full history back to 1968. Whether that feed is inside or outside the move is not
   something this project can determine by reading it, and guessing in our own favour is
   not acceptable.
2. The `User-Agent` Aurex presents now points at a repository that exists. Until
   2026-08-17 it named `github.com/aurex-engine/aurex`, which is a 404 — so any operator
   who tried to follow it got nowhere. Asking a question while presenting a dead contact
   route would have been worse than not asking.

## Where to send it

- **LBMA**, via <https://www.lbma.org.uk/contact>. That page lists separate addresses for
  general enquiries, membership and press; they are rendered client-side, so pick the
  general enquiries address off the page rather than copying one from here.
- **Copy to IBA licensing, `iba-licensing@ice.com`.** ICE Benchmark Administration
  administers the LBMA Gold Price and is the party that licenses its use and
  redistribution ([ICE](https://www.ice.com/iba/lbma-gold-silver-price)). If the answer is
  "you need a licence", it is theirs to give.

## Draft

> **Subject:** Access route for the daily gold price JSON — open-source research project
>
> Hello,
>
> I maintain Aurex, an open-source, non-commercial research project that publishes
> probabilistic forecasts of gold and then publicly grades how well calibrated they turn
> out to be. It is a personal project with no revenue, no clients and no product; the
> code and every published number are at https://github.com/MadScientistX92/aurex, and the
> results it publishes are mostly negative ones.
>
> It reads the London PM fix from `https://prices.lbma.org.uk/json/gold_pm.json`. I want
> to check that we are doing this by the route you would prefer, for two reasons.
>
> First, the access itself. The job runs **once per night** and makes **one** request for
> that single file — no polling, no crawling, no other endpoint on the host. Every request
> identifies itself honestly as
> `aurex-research/0.1 (+https://github.com/MadScientistX92/aurex; open-source gold research)`,
> with a per-host courtesy delay and a 20-second timeout. If a different route — a
> different endpoint, a lower frequency, a registered client, or an outright "please stop"
> — would suit you better, I would rather be told than guess.
>
> Second, the terms. I saw that the historic benchmark price tables moved to the Members'
> Portal from the week commencing 24 November 2025, with the latest daily auction price
> and chart data remaining public. I am not clear whether the JSON file above falls under
> that change, and I would rather ask than assume the reading that happens to suit me. If
> using it — or holding a local copy of the history so a nightly job does not have to
> refetch it — needs a licence from ICE Benchmark Administration, I would be grateful to
> be pointed at the right process; I have copied IBA licensing on this note.
>
> One thing I should be straight about: from GitHub's hosted runners, requests to that
> endpoint intermittently receive a Cloudflare interstitial page rather than the JSON,
> roughly half the time and varying by source address. I am not writing to ask you to
> change that — it may well be deliberate, and if it is, that is an answer in itself and I
> will stop reaching the endpoint from those addresses. I mention it only so that you know
> what the traffic you may be seeing from us looks like.
>
> Happy to provide anything useful about the project, and happy to remove or change the
> integration on request.
>
> Thank you for your time,
>
> [name]
> [email]
> https://github.com/MadScientistX92/aurex

## Things this draft deliberately does not do

- It does not claim a licence, an exemption, or "fair use". It asks.
- It does not ask them to whitelist an IP range or to relax a control they may have put
  there on purpose.
- It does not describe Aurex as anything other than what it is: a personal, non-commercial
  project whose headline result is that the model does not beat a random walk.
