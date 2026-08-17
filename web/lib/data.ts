import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

import type {
  AssetBlock,
  CalibrationArtifact,
  EventBlock,
  ForecastIndex,
  LatestArtifact,
  LensBlock,
  LiveLogArtifact,
  RoutesArtifact,
} from "./types";

/**
 * Build-time readers for the committed artifacts.
 *
 * Everything is read from disk during `next build` and baked into static pages. No
 * runtime fetch, no database, no engine on the server. The consequence is worth being
 * explicit about: **this site can only ever show what the engine actually published.**
 * There is no code path that could compute a fresher number, so there is no code path
 * that could invent one, and a stale deploy is visibly stale rather than quietly wrong.
 *
 * Absence is a first-class state. The live log starts empty, the forecast index does
 * not exist until a night has run, and a distribution can be unavailable. Each reader
 * returns `null` for "not published yet" and throws only when a file exists and cannot
 * be parsed — a corrupt artifact is a bug and should fail the build; a missing one is
 * just Tuesday.
 */

const CANDIDATE_ROOTS = ["..", ".", "../.."];

function dataRoot(): string {
  const tried: string[] = [];
  for (const candidate of CANDIDATE_ROOTS) {
    const root = join(process.cwd(), candidate, "public-data");
    tried.push(root);
    try {
      readdirSync(root);
      return root;
    } catch {
      continue;
    }
  }
  throw new Error(
    `Could not find public-data/. Tried:\n  ${tried.join("\n  ")}\n` +
      `This site reads the engine's committed artifacts and has no other data source. ` +
      `On Vercel this is what a correct Root Directory of web/ looks like with ` +
      `"Include source files outside of the Root Directory in the Build Step" left off: ` +
      `the build never received the repository root, so there is nothing one level up ` +
      `to read. Turn the toggle on. See DEPLOY.md.`,
  );
}

function readJson<T>(...segments: string[]): T | null {
  let raw: string;
  try {
    raw = readFileSync(join(dataRoot(), ...segments), "utf8");
  } catch {
    return null;
  }
  // A file that exists but will not parse is a broken artifact, and a build that
  // swallowed it would publish a page quietly missing a section.
  return JSON.parse(raw) as T;
}

export function latest(): LatestArtifact | null {
  return readJson<LatestArtifact>("latest.json");
}

export function routes(): RoutesArtifact | null {
  return readJson<RoutesArtifact>("routes.json");
}

export function liveLog(): LiveLogArtifact | null {
  return readJson<LiveLogArtifact>("live-log.json");
}

export function forecastIndex(): ForecastIndex | null {
  return readJson<ForecastIndex>("forecasts", "index.json");
}

/**
 * Every calibration report present, found by pattern rather than by name.
 *
 * The filename carries the asset id, so naming one here would put an asset literal in
 * `web/` and trip the leak guard — correctly. Globbing keeps this page working the day
 * a second asset lands without anybody editing it.
 */
export function calibrations(): CalibrationArtifact[] {
  let names: string[];
  try {
    names = readdirSync(dataRoot());
  } catch {
    return [];
  }
  return names
    .filter((name) => name.startsWith("calibration-") && name.endsWith(".json"))
    .sort()
    .map((name) => readJson<CalibrationArtifact>(name))
    .filter((report): report is CalibrationArtifact => report !== null);
}

/** The assets the latest artifact carries, in a stable order. */
export function assets(artifact: LatestArtifact): [string, AssetBlock][] {
  return Object.entries(artifact.assets).sort(([a], [b]) => a.localeCompare(b));
}

/**
 * The lens to lead with: the asset's own quote currency.
 *
 * Not a default *jurisdiction* — that is a different question and §20 says there is no
 * default for it. This is the currency the engine models in, and leading with anything
 * else would present a converted view as the primary one.
 */
export function quoteLens(block: AssetBlock): [string, LensBlock] | null {
  const entries = Object.entries(block.lenses);
  const native = entries.find(([code]) => code === block.asset.quote_currency);
  return native ?? entries[0] ?? null;
}

export function horizonKeys(block: LensBlock): number[] {
  const horizons = block.distribution?.horizons;
  if (!horizons) return [];
  return Object.keys(horizons)
    .map((key) => Number.parseInt(key, 10))
    .filter((value) => Number.isFinite(value))
    .sort((a, b) => a - b);
}

/**
 * Events whose published numbers are identical, collapsed into one panel each.
 *
 * The track-record page renders a reliability panel per scored event, and three of them
 * were the same panel. Three routes in different jurisdictions carry breakeven hurdles
 * that coincide — 1.05102, a gross move of 5.10% — so the engine correctly scores three
 * events, and correctly gets three identical answers, and the reader correctly wonders
 * why the same chart is on the page three times. Nothing is wrong with the artifact: the
 * hurdles genuinely coincide on this friction table, and if one jurisdiction's duty moves
 * they will separate again on their own.
 *
 * So this collapses **only what is genuinely the same**, and the rule is deliberately
 * blunt: two events group when every published field except `id` and `label` is equal.
 * Not "same definition" — a definition string rounds the multiple to six figures, so two
 * hurdles differing in the seventh would print identically and score differently, and
 * grouping on the text would hide a real difference behind a display convention. Any
 * disagreement in any number, including one that only appears in a bin count, and they
 * render separately. The failure this avoids is the one §1 warns about: a page that looks
 * tidier because it dropped something.
 *
 * The labels travel with the group so the page can say *which* routes share the hurdle.
 * That is the interesting fact, and it was the thing the three identical panels were
 * accidentally hiding.
 */
export interface EventGroup {
  /** The scored event. Any member of the group would do; they are equal by construction. */
  event: EventBlock;
  /** Every route label that produced this identical scoring, in a stable order. */
  labels: string[];
}

function scoringKey(event: EventBlock): string {
  const entries = Object.entries(event as unknown as Record<string, unknown>)
    .filter(([key]) => key !== "id" && key !== "label")
    .sort(([a], [b]) => a.localeCompare(b));
  return JSON.stringify(entries);
}

export function groupIdenticalEvents(events: EventBlock[]): EventGroup[] {
  const groups = new Map<string, EventGroup>();
  for (const event of events) {
    const key = scoringKey(event);
    const existing = groups.get(key);
    const label = event.label;
    if (existing) {
      if (label) existing.labels.push(label);
      continue;
    }
    groups.set(key, { event, labels: label ? [label] : [] });
  }
  return [...groups.values()].map((group) => ({
    ...group,
    labels: [...group.labels].sort((a, b) => a.localeCompare(b)),
  }));
}
