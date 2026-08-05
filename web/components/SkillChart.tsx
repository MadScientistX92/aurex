import { pValue, sessions, signedPct } from "@/lib/format";
import type { HorizonCalibration } from "@/lib/types";

import { Figure, LegendLine } from "./Figure";

const W = 760;
const H = 260;
const PAD = { top: 22, right: 24, bottom: 48, left: 62 };

/**
 * CRPS skill against the driftless random walk, per horizon, with zero drawn.
 *
 * Diverging about zero, because the reader's question is polarity: is this above or
 * below the null. The zero line is the emphasis, not the bars — the finding is that
 * every bar is indistinguishable from it, and a chart that made the bars vivid would
 * be arguing the opposite of what the test says.
 *
 * No confidence band is drawn, deliberately. The Diebold-Mariano p-value is printed
 * against every bar instead: a band would invite eyeballing overlap with zero, which
 * is not how the test works on overlapping windows.
 */
export function SkillChart({ horizons }: { horizons: HorizonCalibration[] }) {
  const values = horizons.map((h) => h.crps.skill_score);
  const bound = Math.max(0.02, ...values.map((v) => Math.abs(v) * 1.5));

  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const slot = innerW / horizons.length;
  const y = (v: number) => PAD.top + (1 - (v + bound) / (2 * bound)) * innerH;
  const zero = y(0);

  return (
    <Figure
      title="CRPS skill against a driftless random walk, by horizon"
      chart={
        <svg
          viewBox={`0 0 ${W} ${H}`}
          width={W}
          role="img"
          aria-label={
            `Bar chart of CRPS skill against a driftless random walk at ${horizons.length} horizons, ` +
            `ranging from ${signedPct(Math.min(...values), 1)} to ${signedPct(Math.max(...values), 1)}. ` +
            `Every bar is small relative to the zero line, and a Diebold-Mariano test rejects at none ` +
            `of them. Exact figures and p-values are in the table below.`
          }
        >
          <title>CRPS skill against the null, with zero as the reference</title>

          {[-bound, -bound / 2, 0, bound / 2, bound].map((v) => (
            <g key={v}>
              <line
                x1={PAD.left}
                x2={W - PAD.right}
                y1={y(v)}
                y2={y(v)}
                stroke="var(--grid)"
              />
              <text
                x={PAD.left - 10}
                y={y(v) + 4}
                textAnchor="end"
                fontSize={11.5}
                fill="var(--text-muted)"
                fontFamily="var(--mono)"
              >
                {signedPct(v, 1)}
              </text>
            </g>
          ))}

          {horizons.map((h, i) => {
            const v = h.crps.skill_score;
            const top = Math.min(y(v), zero);
            const height = Math.max(2, Math.abs(y(v) - zero));
            const cx = PAD.left + slot * i + slot / 2;
            return (
              <g key={h.horizon_sessions}>
                <rect
                  x={cx - slot * 0.22}
                  y={top}
                  width={slot * 0.44}
                  height={height}
                  rx={3}
                  fill="var(--band-inner)"
                />
                <text
                  x={cx}
                  y={v >= 0 ? top - 8 : top + height + 16}
                  textAnchor="middle"
                  fontSize={12}
                  fontFamily="var(--mono)"
                  fill="var(--text-primary)"
                >
                  {signedPct(v, 1)}
                </text>
                <text
                  x={cx}
                  y={H - PAD.bottom + 18}
                  textAnchor="middle"
                  fontSize={12}
                  fontFamily="var(--mono)"
                  fill="var(--text-muted)"
                >
                  {h.horizon_sessions}
                </text>
                <text
                  x={cx}
                  y={H - PAD.bottom + 33}
                  textAnchor="middle"
                  fontSize={10.5}
                  fill="var(--text-muted)"
                >
                  p ={" "}
                  {pValue(
                    h.crps.significance.non_overlapping_subsample.p_value,
                  )}
                </text>
              </g>
            );
          })}

          {/* Zero: the null. The emphasis of this chart. */}
          <line
            x1={PAD.left}
            x2={W - PAD.right}
            y1={zero}
            y2={zero}
            stroke="var(--hurdle)"
            strokeWidth={2}
          />
          <text
            x={PAD.left + innerW / 2}
            y={H - 6}
            textAnchor="middle"
            fontSize={12}
            fill="var(--text-secondary)"
          >
            horizon, in sessions
          </text>
        </svg>
      }
      legend={<LegendLine color="var(--hurdle)" label="the random walk — zero skill" />}
      caption={
        <>
          Above the line is better than the null and below it is worse. Neither is what
          happened: every bar is within noise of zero, and the Diebold-Mariano test
          printed beneath each rejects at none of them. This is an absence of evidence,
          not evidence of absence — with as few as 44 independent windows at the longest
          horizon, a real effect of ordinary size would be missed.
        </>
      }
      table={
        <table>
          <caption>CRPS skill against the driftless random walk, with its test.</caption>
          <thead>
            <tr>
              <th scope="col">Horizon</th>
              <th scope="col">Forecasts</th>
              <th scope="col">Independent</th>
              <th scope="col">Skill</th>
              <th scope="col">DM p (HAC)</th>
              <th scope="col">DM p (thinned)</th>
            </tr>
          </thead>
          <tbody>
            {horizons.map((h) => (
              <tr key={h.horizon_sessions}>
                <th scope="row">{sessions(h.horizon_sessions)}</th>
                <td>{h.observations}</td>
                <td>{h.independent_observations}</td>
                <td>{signedPct(h.crps.skill_score, 2)}</td>
                <td>{pValue(h.crps.significance.overlapping_windows_hac.p_value)}</td>
                <td>{pValue(h.crps.significance.non_overlapping_subsample.p_value)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
      tableLabel="Show the skill scores and their p-values"
    />
  );
}
