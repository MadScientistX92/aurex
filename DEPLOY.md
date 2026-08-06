# Deploying the dashboard

You run these; I do not have your Vercel account. Everything below assumes `main` is
pushed and green.

## What the project needs to be told

The site lives in `web/` and reads `public-data/` at the **repository root**. Those two
facts pull in opposite directions, and the settings below are what reconciles them.

| Setting | Value |
|---|---|
| Root Directory | `web` |
| Include source files outside of the Root Directory | **on** |
| Framework Preset | Next.js (auto-detected from `web/package.json`) |
| Install Command | default (`npm ci`) — override **off** |
| Build Command | default (`next build`) — override **off** |
| Output Directory | default (`.next`) — override **off** |
| Ignored Build Step | `exit 1` |
| Node version | 22.x |
| Environment variables | **none** — the site needs no secret of any kind |

### Why Root Directory is `web` and not the repository root

An earlier version of this file said the Root Directory had to be the repository root.
That was wrong, and it is worth being precise about how it fails, because the error does
not look like a configuration error.

Vercel resolves the framework preset by reading `package.json` in the Root Directory.
There is no `package.json` at the repository root — the only one is `web/package.json` —
so with the Root Directory set to `.`, framework detection has nothing to read. What you
see in the build log is an install step that completes normally (a `npm --prefix web ci`
override does find the manifest) followed by a failure to determine the Next.js version.
It reads like a broken or missing dependency. It is not: the dependency tree is fine and
the manifest is fine, they are simply not where Vercel is looking.

### Why the "outside the Root Directory" toggle is not optional

With the Root Directory at `web`, the build only receives `web/` unless you tell Vercel
otherwise. `web/lib/data.ts` reads the committed artifacts from `public-data/`, one level
up, at build time. Turn the toggle **on** — under Settings → Build & Deployment, beside
the Root Directory field — and the repository root is included in the build context, so
the relative read resolves.

Leave it off and you get the other failure: the install and the framework detection both
succeed, and the build then dies inside static generation with `Could not find
public-data/`, listing the paths it tried. That message is `data.ts` telling you exactly
this, and it is the only remaining way to misconfigure this project that gets that far.

### Why the commands are defaults now

With the Root Directory at `web`, Vercel runs every command with the working directory
already inside `web/`. `npm ci`, `next build` and `.next` are then all correct, and the
`--prefix web` forms are actively wrong — they would resolve to `web/web/`. `vercel.json`
has moved to `web/vercel.json` (Vercel reads it from the Root Directory) and no longer
carries `installCommand`, `buildCommand` or `outputDirectory`. What it still carries is
the framework field, `github.silent`, and a strict CSP, HSTS, `X-Frame-Options: DENY` and
a `Permissions-Policy` that switches off camera, microphone, geolocation, payment and
USB. Those come from the file, so leave the corresponding dashboard fields alone.

### Why the Ignored Build Step must be `exit 1`

Vercel's default behaviour for a project with a Root Directory is to skip the build when
a push changed nothing inside that directory. That default is wrong here, and it fails
silently in the worst possible way.

The nightly job commits `public-data/` **and nothing else** — it verifies as much before
pushing. `public-data/` sits outside `web/`. So under the default, every single nightly
forecast would push, change no file under the Root Directory, and be skipped. The site
would keep serving whatever forecast was current the last time somebody touched `web/`,
with no failed build, no red check, and no signal anywhere that the dashboard had
stopped tracking the engine. A visitor would read a stale number as a current one.

That is the freshness guard's failure mode moved up one layer, and it gets the same
answer: fail loudly or do not publish. Set **Ignored Build Step** (Settings → Git) to the
literal command:

```
exit 1
```

A non-zero exit means "do not skip". Every push to `main` rebuilds, including the nights
the engine refuses and writes a skip record. A build that costs nothing is worth far more
than a silent freeze.

## Option A — the dashboard (no CLI)

1. Go to <https://vercel.com/new> and import `MadScientistX92/aurex`.
2. On the configure screen, set **Root Directory** to `web`, and enable **Include source
   files outside of the Root Directory in the Build Step**.
3. Framework Preset should now auto-detect as **Next.js**. Leave the Install, Build and
   Output overrides switched off — the defaults are correct and `web/vercel.json` carries
   the rest.
4. Add **no** environment variables.
5. Deploy.
6. After the first deploy, go to Settings → Git and set **Ignored Build Step** to
   `exit 1`. This is not optional; see above.

## Option B — the CLI

Set the Root Directory, the outside-files toggle and the Ignored Build Step in the
dashboard even if you deploy from the CLI — they are project settings, not flags.

```bash
npm i -g vercel
cd /Users/mmks/aurex     # the repository root — the upload must contain public-data/
vercel login
vercel link              # create or link the project
vercel --prod
```

Run `vercel` from the repository root, not from `web/`. The CLI uploads the directory you
invoke it in, and Vercel then applies the project's Root Directory to what it received. A
deploy launched from inside `web/` uploads a tree with no `public-data/` in it, and fails
the same way the missing toggle does.

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

Check 2 also earns its place the first night after deploy: run it again and confirm the
date has moved. If it has not, the Ignored Build Step is not set.

## Wiring the nightly to the deploy

Vercel's GitHub integration rebuilds on every push to `main`, and the nightly job pushes
`public-data/` to `main` — so a new forecast redeploys the site with no further wiring,
**provided the Ignored Build Step is `exit 1`**. `vercel.json` sets `github.silent`, so
those deploys will not comment on commits.

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

`npm audit` reports high-severity advisories in two transitive dependencies of Next 15,
`postcss` and `sharp`, both fixable only by moving to Next 16, which is a major version
and outside what was asked for. (It prints "3 high severity vulnerabilities"; the third
package is `next` itself, flagged only because it depends on the other two.) Neither is
reachable here: the postcss
advisories concern processing attacker-controlled CSS, and this site compiles one
stylesheet written by hand at build time; the `sharp` advisories reach through Next's
image optimiser, and there are no images and no `next/image` anywhere in `web/`. Pinning
Next `15.5.22` cleared all twenty-nine advisories that applied to Next itself. Worth
revisiting when Next 16 is the intended target.
