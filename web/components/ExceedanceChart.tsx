import { prob, sessions, signedPct } from "@/lib/format";
import { exceedance } from "@/lib/stats";
import type { ExceedancePoint } from "@/lib/types";

import { Figure, LegendLine } from "./Figure";

const W = 760;
const H = 340;
const PAD = { top: 22, right: 24, bottom: 52, left: 62 };

/**
 * P(move at least x) against x, with the hurdle drawn as a vertical rule.
 *
 * This is the chart where the hurdle stops being an annotation and becomes the
 * question: the rule crosses the curve at exactly the probability of clearing it, so
 * the answer is a geometric fact rather than a number quoted beside a picture. It is
 * also the honest way to show a fat tail — the curve's shape in the right-hand quarter
 * is the whole difference between this engine and a Gaussian one.
 *
 * Both readings are drawn because §18 is about the gap between them: `terminal` is
 * being above water *at* the horizon, `touch` is having been above water at any
 * session close on the way. Touch is a floor, since a level crossed and given back
 * inside one session is not counted.
 */
export function ExceedanceChart({
  points,
  hurdleMove,
  hurdleLabel,
  horizon,
}: {
  points: ExceedancePoint[];
  hurdleMove: number;
  hurdleLabel: string;
  horizon: number;
}) {
  const sorted = [...points].sort((a, b) => a.move - b.move);
  const xMin = Math.min(...sorted.map((p) => p.move), hurdleMove);
  const xMax = Math.max(...sorted.map((p) => p.move), hurdleMove);

  const x = (v: number) => PAD.left + ((v - xMin) / (xMax - xMin)) * (W - PAD.left - PAD.right);
  const y = (v: number) => PAD.top + (1 - v) * (H - PAD.top - PAD.bottom);

  const line = (key: "terminal_probability" | "touch_probability") =>
    sorted.map((p, i) => `${i === 0 ? "M" : "L"} ${x(p.move)},${y(p[key])}`).join(" ");

  const atHurdle = exceedance(sorted, hurdleMove, "terminal");
  const touchAtHurdle = exceedance(sorted, hurdleMove, "touch");

  const xTicks = [-0.2, -0.1, 0, 0.1, 0.2].filter((v) => v >= xMin && v <= xMax);
  const yTicks = [0, 0.25, 0.5, 0.75, 1];

  // Sampled for the table: 101 rows would bury the reader, and the grid is dense
  // enough that every second-and-a-half percent tells the same story.
  const tableRows = sorted.filter((p) => Math.abs(Math.round(p.move * 40) - p.move * 40) < 1e-9);

  return (
    <Figure
      title={`Probability of clearing a given move at ${sessions(horizon)}`}
      chart={
        <svg
          viewBox={`0 0 ${W} ${H}`}
          width={W}
          role="img"
          aria-label={
            `Two falling curves showing the probability of a move reaching at least a given size ` +
            `over ${sessions(horizon)}. A vertical dashed rule marks the round-trip breakeven at ` +
            `${signedPct(hurdleMove)}, where the probability of finishing above water is ` +
            `${prob(atHurdle)} and the probability of having been above water at some session ` +
            `close is ${prob(touchAtHurdle)}. Full numbers in the table below.`
          }
        >
          <title>Probability of clearing a move, against the breakeven that defines it</title>

          {yTicks.map((v) => (
            <g key={v}>
              <line
                x1={PAD.left}
                x2={W - PAD.right}
                y1={y(v)}
                y2={y(v)}
                stroke="var(--grid)"
                strokeWidth={1}
              />
              <text
                x={PAD.left - 10}
                y={y(v) + 4}
                textAnchor="end"
                fontSize={12}
                fill="var(--text-muted)"
                fontFamily="var(--mono)"
              >
                {v.toFixed(2)}
              </text>
            </g>
          ))}

          {xTicks.map((v) => (
            <text
              key={v}
              x={x(v)}
              y={H - PAD.bottom + 20}
              textAnchor="middle"
              fontSize={12}
              fill="var(--text-muted)"
              fontFamily="var(--mono)"
            >
              {signedPct(v, 0)}
            </text>
          ))}

          <path d={line("touch_probability")} fill="none" stroke="var(--series-3)" strokeWidth={2} />
          <path
            d={line("terminal_probability")}
            fill="none"
            stroke="var(--series-1)"
            strokeWidth={2}
          />

          {/* The hurdle, vertical. Where it crosses each curve is the answer. */}
          <line
            x1={x(hurdleMove)}
            x2={x(hurdleMove)}
            y1={PAD.top}
            y2={H - PAD.bottom}
            stroke="var(--hurdle)"
            strokeWidth={2}
            strokeDasharray="7 4"
          />
          {atHurdle !== null ? (
            <circle
              cx={x(hurdleMove)}
              cy={y(atHurdle)}
              r={5}
              fill="var(--series-1)"
              stroke="var(--surface-1)"
              strokeWidth={2}
            />
          ) : null}
          {touchAtHurdle !== null ? (
            <circle
              cx={x(hurdleMove)}
              cy={y(touchAtHurdle)}
              r={5}
              fill="var(--series-3)"
              stroke="var(--surface-1)"
              strokeWidth={2}
            />
          ) : null}
          <text
            x={x(hurdleMove)}
            y={PAD.top - 6}
            textAnchor="middle"
            fontSize={12.5}
            fontWeight={650}
            fill="var(--text-primary)"
            fontFamily="var(--mono)"
          >
            {signedPct(hurdleMove, 2)}
          </text>

          <text
            x={PAD.left + (W - PAD.left - PAD.right) / 2}
            y={H - 10}
            textAnchor="middle"
            fontSize={12}
            fill="var(--text-secondary)"
          >
            move in the underlying price
          </text>
          <text
            x={14}
            y={PAD.top + (H - PAD.top - PAD.bottom) / 2}
            fontSize={12}
            fill="var(--text-secondary)"
            transform={`rotate(-90 14 ${PAD.top + (H - PAD.top - PAD.bottom) / 2})`}
            textAnchor="middle"
          >
            probability of at least that move
          </text>
        </svg>
      }
      legend={
        <>
          <LegendLine color="var(--series-1)" label="above water at the horizon (terminal)" />
          <LegendLine color="var(--series-3)" label="above water at some session close (touch)" />
          <LegendLine color="var(--hurdle)" label={hurdleLabel} dashed />
        </>
      }
      caption={
        <>
          The dashed rule is the round-trip breakeven; where it crosses each curve is the
          probability of clearing it. Touch is monitored at session close, so a level
          crossed and given back within one session does not count — every touch
          probability here is a floor, not an estimate.
        </>
      }
      table={
        <table>
          <caption>
            Exceedance probabilities at {sessions(horizon)}, sampled every 2.5% of move.
            Breakeven is {signedPct(hurdleMove, 2)}.
          </caption>
          <thead>
            <tr>
              <th scope="col">Move</th>
              <th scope="col">Terminal</th>
              <th scope="col">Touch</th>
            </tr>
          </thead>
          <tbody>
            {tableRows.map((p) => (
              <tr key={p.move}>
                <th scope="row">{signedPct(p.move, 1)}</th>
                <td>{prob(p.terminal_probability)}</td>
                <td>{prob(p.touch_probability)}</td>
              </tr>
            ))}
            <tr>
              <th scope="row">{signedPct(hurdleMove, 2)} (breakeven)</th>
              <td>{prob(atHurdle)}</td>
              <td>{prob(touchAtHurdle)}</td>
            </tr>
          </tbody>
        </table>
      }
      tableLabel={`Show the exceedance numbers at ${sessions(horizon)}`}
    />
  );
}
