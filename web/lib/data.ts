import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

import type {
  AssetBlock,
  CalibrationArtifact,
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
      `This site reads the engine's committed artifacts and has no other data source; ` +
      `check the Vercel root directory is the repository root rather than web/.`,
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
