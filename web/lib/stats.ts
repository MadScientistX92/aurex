import type { ExceedancePoint, HorizonBlock } from "./types";

/**
 * Reading the published distribution. Nothing here models anything.
 *
 * Every function is a lookup or an average over numbers the engine already computed
 * and committed. That is a hard boundary, not a convenience: the moment this file
 * starts *deriving* a probability, the site has a second model in it, and the two
 * would disagree the first time either changed.
 */

/**
 * P(the move reaches `target` or better) at this horizon, from the published grid.
 *
 * Linear interpolation between adjacent grid points, which is a real approximation and
 * is labelled as one wherever it surfaces. The grid is half-percent steps, so the
 * interpolation spans at most half a percent of move — far tighter than the sampling
 * error on a 20,000-path ensemble, which is why this is acceptable where interpolating
 * a five-point quantile grid would not be.
 *
 * Returns `null` outside the published range rather than clamping to 0 or 1. A clamped
 * probability looks measured and is not.
 */
export function exceedance(
  points: ExceedancePoint[],
  target: number,
  reading: "terminal" | "touch" = "terminal",
): number | null {
  if (points.length === 0) return null;
  const key = reading === "terminal" ? "terminal_probability" : "touch_probability";
  const sorted = [...points].sort((a, b) => a.move - b.move);

  const first = sorted[0];
  const last = sorted[sorted.length - 1];
  if (!first || !last) return null;
  if (target < first.move || target > last.move) return null;

  for (let i = 0; i < sorted.length - 1; i += 1) {
    const lo = sorted[i];
    const hi = sorted[i + 1];
    if (!lo || !hi) continue;
    if (target >= lo.move && target <= hi.move) {
      if (hi.move === lo.move) return lo[key];
      const weight = (target - lo.move) / (hi.move - lo.move);
      return lo[key] + weight * (hi[key] - lo[key]);
    }
  }
  return null;
}

/** The move at which the exceedance curve crosses 0.5 — the median, read off the grid. */
export function medianMove(block: HorizonBlock): number | null {
  const sorted = [...block.exceedance].sort((a, b) => a.move - b.move);
  for (let i = 0; i < sorted.length - 1; i += 1) {
    const lo = sorted[i];
    const hi = sorted[i + 1];
    if (!lo || !hi) continue;
    if (lo.terminal_probability >= 0.5 && hi.terminal_probability <= 0.5) {
      const span = lo.terminal_probability - hi.terminal_probability;
      if (span === 0) return lo.move;
      return lo.move + ((lo.terminal_probability - 0.5) / span) * (hi.move - lo.move);
    }
  }
  return null;
}

export interface Outcome {
  /** Probability the round trip is above water at the horizon. */
  probabilityOfProfit: number | null;
  /** Probability it was ever above water on the way, at session close. A floor. */
  probabilityOfTouch: number | null;
  /** The move required before the round trip returns the outlay. */
  hurdle: number;
  /**
   * True where the honest headline is the loss rather than the win.
   *
   * The engine already refuses to lead with a win probability below one half. The rule
   * is repeated here because the interface is where it actually bites: every dashboard
   * convention pushes toward the encouraging half of a pair, and "38% chance of profit"
   * reads as an opportunity while "62% chance of loss" reads as what it is. They are
   * the same number.
   */
  leadWithLoss: boolean;
}

export function outcome(block: HorizonBlock, hurdle: number): Outcome {
  const terminal = exceedance(block.exceedance, hurdle, "terminal");
  const touch = exceedance(block.exceedance, hurdle, "touch");
  return {
    probabilityOfProfit: terminal,
    probabilityOfTouch: touch,
    hurdle,
    leadWithLoss: terminal !== null && terminal < 0.5,
  };
}

/** Quantile levels present on a horizon block, parsed from `q05`-style keys. */
export function quantileLevels(block: HorizonBlock): { level: number; value: number }[] {
  return Object.entries(block.quantiles)
    .map(([key, value]) => ({ level: Number.parseInt(key.slice(1), 10) / 100, value }))
    .filter((entry) => Number.isFinite(entry.level))
    .sort((a, b) => a.level - b.level);
}

/** Nearest published band to a target coverage, e.g. 0.9 from q05 and q95. */
export function band(
  block: HorizonBlock,
  lower: number,
  upper: number,
): { lo: number; hi: number } | null {
  const levels = quantileLevels(block);
  const lo = levels.find((entry) => Math.abs(entry.level - lower) < 1e-9);
  const hi = levels.find((entry) => Math.abs(entry.level - upper) < 1e-9);
  return lo && hi ? { lo: lo.value, hi: hi.value } : null;
}
