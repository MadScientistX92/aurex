/**
 * Number rendering.
 *
 * Every numeral on this site is monospace with tabular figures, set in CSS on the
 * `.num` class. That is not a stylistic preference: these pages put numbers in
 * columns and ask readers to compare them down a column, and proportional figures
 * make a column of digits ragged enough that the comparison stops being visual.
 */

const FIXED = (digits: number) =>
  new Intl.NumberFormat("en-GB", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });

export function num(value: number, digits = 2): string {
  return FIXED(digits).format(value);
}

export function pct(value: number, digits = 1): string {
  return `${FIXED(digits).format(value * 100)}%`;
}

/** A probability, always with a leading zero and always to the same width. */
export function prob(value: number | null, digits = 3): string {
  return value === null ? "—" : FIXED(digits).format(value);
}

export function signedPct(value: number, digits = 1): string {
  const formatted = FIXED(digits).format(Math.abs(value) * 100);
  const sign = value > 0 ? "+" : value < 0 ? "−" : "±";
  return `${sign}${formatted}%`;
}

/** A p-value, with the convention that very small ones are reported as a bound. */
export function pValue(value: number | null): string {
  if (value === null) return "n/a";
  if (value < 0.001) return "< 0.001";
  return FIXED(3).format(value);
}

export function money(value: number, currency: string, digits = 2): string {
  try {
    return new Intl.NumberFormat("en-GB", {
      style: "currency",
      currency,
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    }).format(value);
  } catch {
    // An unknown or synthetic currency code is not an error worth failing a build for.
    return `${FIXED(digits).format(value)} ${currency}`;
  }
}

export function isoDate(value: string): string {
  return value.slice(0, 10);
}

/** "5 sessions", "1 session" — horizons are always in sessions, never in days. */
export function sessions(count: number): string {
  return `${count} ${count === 1 ? "session" : "sessions"}`;
}

/**
 * An identifier as published, made readable.
 *
 * The engine stores `troy_ounce` and `over_the_counter` because a stable identifier is
 * what downstream code should key on. A reader should not have to see the underscore —
 * and the identifier is still what the site keys on, so this is presentation only.
 */
export function unitLabel(value: string): string {
  return value.replace(/_/g, " ");
}
