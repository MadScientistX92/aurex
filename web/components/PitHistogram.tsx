import { num, pValue, sessions } from "@/lib/format";
import type { HorizonCalibration } from "@/lib/types";

import { Figure } from "./Figure";

const W = 760;
const H = 220;
const PAD = { top: 18, right: 20, bottom: 40, left: 52 };

/**
 * Are the published distributions the right shape? Uniform means calibrated.
 *
 * A single series, so no legend and no categorical hue — the bars are the sequential
 * ramp and the reference line is the count a perfectly uniform histogram would give.
 * Deviation from that line is the entire content, which is why it is drawn rather than
 * left for the reader to hold in their head.
 */
export function PitHistogram({ horizon }: { horizon: HorizonCalibration }) {
  const bins = horizon.pit.bins;
  const total = bins.reduce((sum, count) => sum + count, 0);
  const expected = total / (bins.length || 1);
  const yMax = Math.max(...bins, expected) * 1.15;

  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const barW = innerW / bins.length;
  const y = (v: number) => PAD.top + (1 - v / yMax) * innerH;

  return (
    <Figure
      title={`Probability integral transform at ${sessions(horizon.horizon_sessions)}`}
      chart={
        <svg
          viewBox={`0 0 ${W} ${H}`}
          width={W}
          role="img"
          aria-label={
            `Histogram of ${total} probability-integral-transform values in ${bins.length} bins at ` +
            `${sessions(horizon.horizon_sessions)}. A flat histogram means the distributions are ` +
            `the right shape. A horizontal reference line marks the ${num(expected, 1)} counts per ` +
            `bin that perfect uniformity would give. Mean PIT is ${num(horizon.pit.mean, 4)} against ` +
            `0.5 for a centred model. Counts are in the table below.`
          }
        >
          <title>PIT histogram against perfect uniformity</title>

          {bins.map((count, i) => {
            const barH = Math.max(0, innerH - (y(count) - PAD.top));
            return (
              <rect
                key={i}
                x={PAD.left + i * barW + 1}
                y={y(count)}
                width={Math.max(1, barW - 2)}
                height={barH}
                rx={3}
                fill="var(--band-inner)"
              />
            );
          })}

          <line
            x1={PAD.left}
            x2={W - PAD.right}
            y1={y(expected)}
            y2={y(expected)}
            stroke="var(--hurdle)"
            strokeWidth={2}
            strokeDasharray="7 4"
          />
          <text
            x={W - PAD.right}
            y={y(expected) - 6}
            textAnchor="end"
            fontSize={11.5}
            fill="var(--text-secondary)"
          >
            uniform = {num(expected, 1)} per bin
          </text>

          <line
            x1={PAD.left}
            x2={W - PAD.right}
            y1={PAD.top + innerH}
            y2={PAD.top + innerH}
            stroke="var(--border-strong)"
          />
          {[0, 0.5, 1].map((v) => (
            <text
              key={v}
              x={PAD.left + v * innerW}
              y={H - PAD.bottom + 20}
              textAnchor="middle"
              fontSize={12}
              fill="var(--text-muted)"
              fontFamily="var(--mono)"
            >
              {v.toFixed(1)}
            </text>
          ))}
          <text
            x={PAD.left + innerW / 2}
            y={H - 6}
            textAnchor="middle"
            fontSize={12}
            fill="var(--text-secondary)"
          >
            PIT value — where the outcome fell in the forecast distribution
          </text>
        </svg>
      }
      caption={
        <>
          KS p = <span className="num">{pValue(horizon.pit.uniformity?.p_value ?? null)}</span>,
          chi-square p ={" "}
          <span className="num">{pValue(horizon.pit.goodness_of_fit?.p_value ?? null)}</span>. Both are
          published because they fail on different things: KS reads the largest gap in the
          cumulative distribution and is weak against mass piled in one bin, which is
          what a location error looks like here.
        </>
      }
      table={
        <table>
          <caption>
            PIT bin counts at {sessions(horizon.horizon_sessions)}, {total} forecasts.
            Uniformity would give {num(expected, 1)} per bin.
          </caption>
          <thead>
            <tr>
              <th scope="col">Bin</th>
              <th scope="col">Count</th>
              <th scope="col">Against uniform</th>
            </tr>
          </thead>
          <tbody>
            {bins.map((count, i) => (
              <tr key={i}>
                <th scope="row">
                  {(i / bins.length).toFixed(1)}–{((i + 1) / bins.length).toFixed(1)}
                </th>
                <td>{count}</td>
                <td>{count > expected ? "+" : ""}{num(count - expected, 1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
      tableLabel="Show the PIT bin counts"
    />
  );
}
