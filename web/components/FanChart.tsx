import { num, pct, sessions, signedPct, unitLabel } from "@/lib/format";
import type { HorizonBlock } from "@/lib/types";

import { Figure, LegendLine, LegendSwatch } from "./Figure";

export interface Hurdle {
  /** Required move as a fraction, e.g. 0.0937. */
  move: number;
  label: string;
  detail: string;
}

interface Column {
  horizon: number;
  q05: number;
  q25: number;
  q50: number;
  q75: number;
  q95: number;
}

const W = 760;
const H = 380;
const PAD = { top: 24, right: 128, bottom: 46, left: 68 };

function columns(anchor: number, horizons: [number, HorizonBlock][]): Column[] {
  const start: Column = {
    horizon: 0,
    q05: anchor,
    q25: anchor,
    q50: anchor,
    q75: anchor,
    q95: anchor,
  };
  const rest = horizons.map(([horizon, block]) => ({
    horizon,
    q05: block.quantiles.q05 ?? anchor,
    q25: block.quantiles.q25 ?? anchor,
    q50: block.quantiles.q50 ?? anchor,
    q75: block.quantiles.q75 ?? anchor,
    q95: block.quantiles.q95 ?? anchor,
  }));
  return [start, ...rest];
}

/**
 * The distribution, as the largest thing on the page.
 *
 * A fan rather than a line with error bars, because the reader's eye should land on
 * the *width* first. The median is drawn thin and unemphasised on purpose — it is the
 * least informative line here and the one a reader is most tempted to treat as a
 * forecast, so it gets no more weight than the band edges.
 *
 * The hurdle is a required prop. There is no way to render this chart without drawing
 * what the move has to beat, which is the point: a distribution shown without its
 * hurdle invites the reader to admire the upside half and forget the cost of reaching
 * it. Drawn as ink, dashed, always labelled — never a status colour, because nothing
 * here is entitled to say "good" or "bad".
 */
export function FanChart({
  anchor,
  horizons,
  hurdle,
  currency,
  unit,
}: {
  anchor: number;
  horizons: [number, HorizonBlock][];
  hurdle: Hurdle;
  currency: string;
  unit: string;
}) {
  const cols = columns(anchor, horizons);
  const hurdleLevel = anchor * (1 + hurdle.move);

  const lows = cols.map((c) => c.q05);
  const highs = cols.map((c) => c.q95);
  const yMin = Math.min(...lows, hurdleLevel, anchor) * 0.985;
  const yMax = Math.max(...highs, hurdleLevel, anchor) * 1.015;
  const xMax = Math.max(...cols.map((c) => c.horizon));

  const x = (h: number) => PAD.left + (h / xMax) * (W - PAD.left - PAD.right);
  const y = (v: number) =>
    PAD.top + (1 - (v - yMin) / (yMax - yMin)) * (H - PAD.top - PAD.bottom);

  const areaPath = (lo: (c: Column) => number, hi: (c: Column) => number) => {
    const up = cols.map((c) => `${x(c.horizon)},${y(hi(c))}`).join(" L ");
    const down = [...cols].reverse().map((c) => `${x(c.horizon)},${y(lo(c))}`).join(" L ");
    return `M ${up} L ${down} Z`;
  };

  const medianPath = cols.map((c, i) => `${i === 0 ? "M" : "L"} ${x(c.horizon)},${y(c.q50)}`).join(" ");

  const ticks = 4;
  const yTicks = Array.from({ length: ticks + 1 }, (_, i) => yMin + ((yMax - yMin) * i) / ticks);

  return (
    <Figure
      title={`Simulated price distribution to ${sessions(xMax)} ahead, against a hurdle of ${signedPct(hurdle.move)}`}
      chart={
        <svg
          viewBox={`0 0 ${W} ${H}`}
          width={W}
          role="img"
          aria-label={
            `Fan chart of the simulated price distribution from ${num(anchor)} ${currency} per ${unitLabel(unit)}, ` +
            `widening across ${cols.length - 1} horizons up to ${sessions(xMax)}. ` +
            `A dashed horizontal rule marks the round-trip breakeven at ${num(hurdleLevel)}, ` +
            `a move of ${signedPct(hurdle.move)}. The full numbers are in the table below.`
          }
        >
          <title>Simulated distribution against the round-trip breakeven</title>

          {yTicks.map((value) => (
            <g key={value}>
              <line
                x1={PAD.left}
                x2={W - PAD.right}
                y1={y(value)}
                y2={y(value)}
                stroke="var(--grid)"
                strokeWidth={1}
              />
              <text
                x={PAD.left - 10}
                y={y(value) + 4}
                textAnchor="end"
                fontSize={12}
                fill="var(--text-muted)"
                fontFamily="var(--mono)"
              >
                {num(value, 0)}
              </text>
            </g>
          ))}

          <path d={areaPath((c) => c.q05, (c) => c.q95)} fill="var(--band-outer)" />
          <path d={areaPath((c) => c.q25, (c) => c.q75)} fill="var(--band-inner)" />
          <path
            d={medianPath}
            fill="none"
            stroke="var(--band-median)"
            strokeWidth={2}
            strokeLinejoin="round"
          />

          {/* Today's price: where the reader is standing. */}
          <line
            x1={PAD.left}
            x2={W - PAD.right}
            y1={y(anchor)}
            y2={y(anchor)}
            stroke="var(--text-muted)"
            strokeWidth={1}
            strokeDasharray="1 3"
          />

          {/* The hurdle. Always drawn, always labelled with the move it demands. */}
          <line
            x1={PAD.left}
            x2={W - PAD.right + 8}
            y1={y(hurdleLevel)}
            y2={y(hurdleLevel)}
            stroke="var(--hurdle)"
            strokeWidth={2}
            strokeDasharray="7 4"
          />
          <text
            x={W - PAD.right + 14}
            y={y(hurdleLevel) - 4}
            fontSize={12.5}
            fill="var(--text-primary)"
            fontFamily="var(--mono)"
            fontWeight={650}
          >
            {signedPct(hurdle.move)}
          </text>
          <text
            x={W - PAD.right + 14}
            y={y(hurdleLevel) + 12}
            fontSize={11}
            fill="var(--text-secondary)"
          >
            to break even
          </text>

          {cols.slice(1).map((c) => (
            <text
              key={c.horizon}
              x={x(c.horizon)}
              y={H - PAD.bottom + 20}
              textAnchor="middle"
              fontSize={12}
              fill="var(--text-muted)"
              fontFamily="var(--mono)"
            >
              {c.horizon}
            </text>
          ))}
          <text
            x={PAD.left + (W - PAD.left - PAD.right) / 2}
            y={H - 8}
            textAnchor="middle"
            fontSize={12}
            fill="var(--text-secondary)"
          >
            sessions ahead
          </text>

          <line
            x1={PAD.left}
            x2={PAD.left}
            y1={PAD.top}
            y2={H - PAD.bottom}
            stroke="var(--border-strong)"
            strokeWidth={1}
          />
        </svg>
      }
      legend={
        <>
          <LegendSwatch color="var(--band-outer)" label="90% of simulated paths (q05–q95)" />
          <LegendSwatch color="var(--band-inner)" label="50% of paths (q25–q75)" />
          <LegendLine color="var(--band-median)" label="median path" />
          <LegendLine color="var(--hurdle)" label={`breakeven, ${signedPct(hurdle.move)}`} dashed />
        </>
      }
      caption={
        <>
          {hurdle.label} — {hurdle.detail} The band is what the engine claims; the dashed
          rule is what a round trip has to clear before any of it is worth anything.
          Quantiles are simulated, not predicted, and the median is not a forecast.
        </>
      }
      table={
        <table>
          <caption>
            Simulated quantiles per horizon, in {currency} per {unitLabel(unit)}. Breakeven sits at{" "}
            {num(hurdleLevel)}.
          </caption>
          <thead>
            <tr>
              <th scope="col">Horizon</th>
              <th scope="col">q05</th>
              <th scope="col">q25</th>
              <th scope="col">median</th>
              <th scope="col">q75</th>
              <th scope="col">q95</th>
              <th scope="col">Move to breakeven</th>
            </tr>
          </thead>
          <tbody>
            {cols.slice(1).map((c) => (
              <tr key={c.horizon}>
                <th scope="row">{sessions(c.horizon)}</th>
                <td>{num(c.q05)}</td>
                <td>{num(c.q25)}</td>
                <td>{num(c.q50)}</td>
                <td>{num(c.q75)}</td>
                <td>{num(c.q95)}</td>
                <td>{pct(hurdle.move, 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    />
  );
}
