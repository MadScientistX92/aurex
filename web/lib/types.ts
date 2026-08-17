/**
 * Shapes of the committed JSON this site reads.
 *
 * Deliberately structural and never nominal about which asset or which jurisdiction
 * it is holding. Nothing in `web/` may name one — the same static guard that keeps
 * `vol/` and `score/` asset-agnostic scans these files, and it is the right rule
 * here for the same reason: a page that special-cases one asset stops being a view
 * of the engine and becomes a second implementation of it.
 *
 * Fields are optional where the engine may legitimately omit them. A distribution can
 * be unavailable, a lens can carry no local premium, and the live log can be empty —
 * all three are states the interface has to render rather than crash on.
 */

export interface CodeProvenance {
  engine_version: string;
  commit: string | null;
  ref: string | null;
  dirty: boolean | null;
  source: string;
  note: string;
}

export interface SeriesAge {
  series_id: string;
  blocking: boolean;
  verdict: string;
  last_observation: string | null;
  lag_days: number | null;
  tolerance: { max_lag_days: number; calendar: string; rationale: string } | null;
  detail: string;
}

export interface Freshness {
  run_date: string;
  publishable: boolean;
  policy: string;
  series: SeriesAge[];
}

export interface ExceedancePoint {
  move: number;
  level: number;
  terminal_probability: number;
  touch_probability: number;
}

export interface AdverseMove {
  move: number;
  barrier: number;
  touch_probability: number;
  terminal_probability: number;
  path_dependence_ratio: number | null;
  median_sessions_to_touch: number | null;
  monitoring: string;
}

export interface HorizonBlock {
  quantiles: Record<string, number>;
  adverse_moves: AdverseMove[];
  exceedance: ExceedancePoint[];
  n_paths: number;
}

export interface Distribution {
  available: boolean;
  reason?: string;
  anchor?: number;
  horizons?: Record<string, HorizonBlock>;
  simulation?: Record<string, unknown>;
  vol_model?: Record<string, unknown> | null;
  note?: string;
  held_constant?: string;
}

export interface LensLatest {
  as_of: string;
  price: number;
  price_ex_consumption_tax: number;
  fx_rate: number;
  duty: number;
  consumption_tax: number;
  confidence: string;
  unit: string;
  currency: string;
}

export interface LensBlock {
  lens: { code: string; unit_label?: string; [key: string]: unknown };
  latest: LensLatest | null;
  distribution?: Distribution;
  local_premium: { observations: number; latest_bps?: number; note: string } | null;
}

export interface AssetBlock {
  asset: {
    id: string;
    label: string;
    quote_currency: string;
    base_unit: string;
    [key: string]: unknown;
  };
  lenses: Record<string, LensBlock>;
  factors: { id: string; available: boolean; required: boolean; reason?: string }[];
}

export interface LatestArtifact {
  schema_version: number;
  generated_at: string;
  engine_version: string;
  code: CodeProvenance;
  freshness: Freshness;
  mode: string;
  assets: Record<string, AssetBlock>;
  sources: Record<string, { source_name: string; source_url: string; end: string | null }>;
  disclaimer: string;
}

export interface DieboldMariano {
  p_value: number | null;
  statistic: number | null;
  observations: number;
  hac_truncation_lag: number;
  mean_loss_differential: number;
  null_compared: string;
  undefined_reason: string | null;
  reading: string;
}

/**
 * Both runs of the test, under the engine's own key names.
 *
 * `overlapping_windows_hac` models the dependence in its variance estimator;
 * `non_overlapping_subsample` thins to independent windows and reduces algebraically to
 * a paired t-test. Agreement between them is evidence the truncation lag was long
 * enough. Neither is "the" p-value, so neither is renamed to look like one here.
 */
export interface Significance {
  distinguishable_from_zero: boolean;
  overlapping_windows_hac: DieboldMariano;
  non_overlapping_subsample: DieboldMariano;
  reading: string;
}

export interface EventBlock {
  id: string;
  definition: string;
  /**
   * Which route priced this hurdle, for the events that come from one. Absent on the
   * events that belong to no route — direction and the touch levels. Carried so the
   * dashboard can name the routes that share an identical hurdle rather than drawing
   * the same reliability panel once per route; see `groupIdenticalEvents`.
   */
  label?: string;
  observations: number;
  positive_events: number;
  base_rate: number;
  mean_forecast: number;
  forecast_bias: number;
  brier: number;
  decomposition: {
    reliability: number;
    resolution: number;
    uncertainty: number;
    implied_brier: number;
    binning_error: number;
  };
  bins: {
    lower: number;
    upper: number;
    count: number;
    forecast_mean: number | null;
    observed_rate: number | null;
  }[];
  curve_withheld: string | null;
  reading: string;
}

export interface HorizonCalibration {
  horizon_sessions: number;
  observations: number;
  independent_observations: number;
  first_as_of: string;
  last_as_of: string;
  pit: {
    mean: number;
    bins: number[];
    uniformity: { p_value: number; statistic: number } | null;
    goodness_of_fit: { p_value: number; statistic: number } | null;
    uniformity_unavailable: string | null;
    note?: string;
  };
  crps: {
    model: number;
    random_walk: number;
    skill_score: number;
    significance: Significance;
    alternative_nulls?: Record<string, { crps: number; significance: Significance }>;
    reading?: string;
  };
  coverage: {
    level: number;
    breaches: number;
    kupiec: { p_value: number | null; boundary_note: string | null; [key: string]: unknown };
    [key: string]: unknown;
  }[];
  events: EventBlock[];
  direction?: { forecast_mean: number; observed_rate: number; gap: number; note: string };
  sampling?: { horizon_sessions: number; step_sessions: number; overlapping: boolean };
}

export interface SampleWindow {
  series_id: string;
  requested: { from: string | null; to: string | null };
  resolved: { from: string; to: string };
  observations: number;
  series_available_to: string;
  truncated: boolean;
  note: string;
}

export interface CalibrationArtifact {
  sample: SampleWindow | null;
  asset: { id: string; label: string; quote_currency: string; base_unit: string };
  calibration: {
    horizons: HorizonCalibration[];
    skill_decay?: {
      slope: number;
      p_value: number;
      r_squared: number;
      interval?: [number, number];
    } | null;
    drift_displacement?: { slope: number; r_squared: number; [key: string]: unknown } | null;
    baseline: Record<string, unknown>;
    conventions: Record<string, string>;
  };
  like_for_like_null: { null: string; why: string };
  generated_at: string;
  engine_version: string;
  code?: CodeProvenance;
  reproduce?: string;
  scope: string;
}

export interface RouteCell {
  route_id: string;
  jurisdiction: string;
  max_leverage: number | null;
  source_url: string;
  source_confidence: string;
  notes: string[];
  route: {
    id: string;
    asset_id: string;
    venue: string;
    instrument: string;
    quote_currency: string;
    available_in: string[];
    source_url: string;
    source_confidence: string;
    notes: string[];
  };
  friction: { id: string; label: string; [key: string]: unknown };
  breakeven: Record<
    string,
    {
      required_move: number;
      required_move_pct: number;
      multiple: number;
      horizon_dependent: boolean;
      components: Record<string, number>;
    }
  >;
}

export interface RoutesArtifact {
  generated_at: string;
  horizons: number[];
  cells: RouteCell[];
  jurisdictions: { code: string; label: string; [key: string]: unknown }[];
  conventions: Record<string, string | null>;
  reading: string;
}

export interface LiveLogHorizon {
  horizon_sessions: number;
  observations: number;
  independent_observations: number;
  censored: number;
  mean_pit: number | null;
  test_possible: boolean;
  test_note: string;
}

export interface LiveLogArtifact {
  generated_at: string;
  scope: string;
  total_observations: number;
  horizons: LiveLogHorizon[];
  conventions: Record<string, string>;
  observations?: unknown[];
}

export interface ForecastIndex {
  generated_at: string;
  as_of_date: string;
  counts: {
    published: number;
    gaps: number;
    gaps_explained: number;
    gaps_unexplained: number;
  };
  forecasts: { date: string; as_of: string | null; frozen: boolean; commit: string | null }[];
  gaps: { date: string; reason: string | null; explained: boolean }[];
  conventions: Record<string, string>;
}
