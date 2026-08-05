# Deploying the dashboard

You run these; I do not have your Vercel account. Everything below assumes `main` is
pushed and green.

## What the project needs to be told

The site lives in `web/`, but it reads `public-data/` at the **repository root**. So the
Vercel *Root Directory* must be the repository root, not `web/` — pointing it at `web/`
is the one misconfiguration that will build cleanly and then fail at data-read time.
`vercel.json` at the root already carries the rest:

| Setting | Value |
|---|---|
| Root Directory | `.` (repository root — **not** `web/`) |
| Framework Preset | Next.js |
| Install Command | `npm --prefix web ci` |
| Build Command | `npm --prefix web run build` |
| Output Directory | `web/.next` |
| Node version | 22.x |
| Environment variables | **none** — the site needs no secret of any kind |

`vercel.json` sets all four commands plus a strict CSP, HSTS, `X-Frame-Options: DENY`
and a `Permissions-Policy` that switches off camera, microphone, geolocation, payment
and USB. Vercel reads them from the file, so the dashboard fields should be left alone.

## Option A — the dashboard (no CLI)

1. Go to <https://vercel.com/new> and import `MadScientistX92/aurex`.
2. On the configure screen, leave **Root Directory** as `./`. Do not set it to `web`.
3. Framework Preset should auto-detect as **Next.js**. The install, build and output
   settings come from `vercel.json`; leave the overrides switched off.
4. Add **no** environment variables.
5. Deploy.

## Option B — the CLI

```bash
npm i -g vercel
cd /Users/mmks/aurex     # the repository root, not web/
vercel login
vercel link              # create or link the project; accept "./" as the root
vercel --prod
```

## After the first deploy

Confirm three things before treating the URL as live:

```bash
URL=https://<your-deployment>.vercel.app

# 1. All three views render.
for path in "" track-record calculator; do
  printf '%s %s\n' "$(curl -s -o /dev/null -w '%{http_code}' "$URL/$path")" "/$path"
done

# 2. The page shows the forecast the repository actually published.
curl -s "$URL" | grep -o 'Anchored to the close of[^<]*<[^>]*>[^<]*'

# 3. Accessibility is still at the floor on the deployed URL.
npx lighthouse "$URL" --only-categories=accessibility --view
```

The third is the one worth actually running: local runs score 100/100 on all three
views, and a regression would most likely come from a header or a font substitution
that only exists in production.

## Wiring the nightly to the deploy

Vercel's GitHub integration rebuilds on every push to `main`, and the nightly job pushes
`public-data/` to `main` — so a new forecast redeploys the site with no further wiring.
`vercel.json` sets `github.silent`, so those deploys will not comment on commits.

Two consequences worth knowing:

- **A night the engine refuses still triggers a build.** The skip record is a change
  under `public-data/`, so the site rebuilds and the forecast index gains a gap. That is
  correct: the hole should be visible on the site the same night it happens.
- **The deployed page is a snapshot.** It shows what the engine had published when the
  build ran. If Vercel is disconnected or a build fails, the site keeps serving the last
  good forecast with its own date on it — visibly stale rather than silently wrong.

## Only after the URL is live

Add the deploy badge and the link to the README. They are deliberately absent until then
— a badge pointing at a deployment that does not exist is the same class of error as a
CI badge pointing at a workflow that does not exist, which is why those were removed
once before.

```markdown
[![Deploy](https://img.shields.io/badge/dashboard-live-black)](https://<your-deployment>.vercel.app)
```

## The one thing I could not verify

`npm audit` reports two high-severity advisories in `postcss` and `sharp`, both
transitive dependencies of Next 15 and both fixable only by moving to Next 16, which is
a major version and outside what was asked for. Neither is reachable here: the postcss
advisories concern processing attacker-controlled CSS, and this site compiles one
stylesheet written by hand at build time; the `sharp` advisories reach through Next's
image optimiser, and there are no images and no `next/image` anywhere in `web/`. Pinning
Next `15.5.22` cleared all twenty-nine advisories that applied to Next itself. Worth
revisiting when Next 16 is the intended target.
