import { num, prob } from "@/lib/format";
import type { EventBlock } from "@/lib/types";

import { Figure, LegendLine } from "./Figure";
import { Notice } from "./Notice";

const W = 380;
const H = 380;
const PAD = { top: 18, right: 18, bottom: 46, left: 52 };

/**
 * When it says 20%, does it happen 20% of the time?
 *
 * Two series — the model's curve and the diagonal it should lie on — so a legend is
 * present and both are direct-labelled. The diagonal is the reference, drawn as ink
 * rather than as a third colour.
 *
 * Where the engine withheld the curve, this renders the reason instead of the diagram.
 * That is the whole point of the withholding rule: a ten-bin curve drawn on five
 * positive events is a picture of sampling noise with an axis on it, and drawing it
 * anyway because the component expects a chart would defeat a decision the engine made
 * deliberately. The score, base rate, count and mean forecast are still shown, because
 * those survive at any sample size.
 */
export function ReliabilityChart({ event }: { event: EventBlock }) {
  const drawn = event.bins.filter((bin) => bin.count > 0 && bin.forecast_mean !== null);

  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const x = (v: number) => PAD.left + v * innerW;
  const y = (v: number) => PAD.top + (1 - v) * innerH;

  const summary = (
    <dl className="readout" style={{ marginTop: "0.75rem" }}>
      <div>
        <dt>Base rate</dt>
        <dd>{prob(event.base_rate, 4)}</dd>
      </div>
      <div>
        <dt>Mean forecast</dt>
        <dd>{prob(event.mean_forecast, 4)}</dd>
      </div>
      <div>
        <dt>Positive events</dt>
        <dd>{event.positive_events} of {event.observations}</dd>
      </div>
      <div>
        <dt>Brier score</dt>
        <dd>
          {num(event.brier, 5)}
          <small>
            Read against the uncertainty term, {num(event.decomposition.uncertainty, 5)},
            never against another event&rsquo;s score.
          </small>
        </dd>
      </div>
      <div>
        <dt>Resolution</dt>
        <dd>
          {num(event.decomposition.resolution, 5)}
          <small>How far forecasts move from the base rate. Higher is better.</small>
        </dd>
      </div>
    </dl>
  );

  if (event.curve_withheld) {
    return (
      <div>
        <Notice title="The reliability curve is withheld here, on purpose.">
          <p>{event.curve_withheld}</p>
        </Notice>
        {summary}
      </div>
    );
  }

  return (
    <div>
      <Figure
        title={`Reliability of "${event.definition}"`}
        chart={
          <svg
            viewBox={`0 0 ${W} ${H}`}
            width={W}
            role="img"
            aria-label={
              `Reliability diagram for the event "${event.definition}". Forecast probability on the ` +
              `horizontal axis against observed frequency on the vertical, with a diagonal marking ` +
              `perfect calibration. Base rate ${prob(event.base_rate, 4)}, mean forecast ` +
              `${prob(event.mean_forecast, 4)}, over ${event.observations} forecasts. Bin figures ` +
              `are in the table below.`
            }
          >
            <title>Forecast probability against observed frequency</title>

            {[0, 0.25, 0.5, 0.75, 1].map((v) => (
              <g key={v}>
                <line x1={x(0)} x2={x(1)} y1={y(v)} y2={y(v)} stroke="var(--grid)" />
                <line x1={x(v)} x2={x(v)} y1={y(0)} y2={y(1)} stroke="var(--grid)" />
                <text
                  x={PAD.left - 8}
                  y={y(v) + 4}
                  textAnchor="end"
                  fontSize={11}
                  fill="var(--text-muted)"
                  fontFamily="var(--mono)"
                >
                  {v.toFixed(2)}
                </text>
                <text
                  x={x(v)}
                  y={H - PAD.bottom + 17}
                  textAnchor="middle"
                  fontSize={11}
                  fill="var(--text-muted)"
                  fontFamily="var(--mono)"
                >
                  {v.toFixed(2)}
                </text>
              </g>
            ))}

            <line
              x1={x(0)}
              x2={x(1)}
              y1={y(0)}
              y2={y(1)}
              stroke="var(--hurdle)"
              strokeWidth={2}
              strokeDasharray="6 4"
            />

            <path
              d={drawn
                .map(
                  (bin, i) =>
                    `${i === 0 ? "M" : "L"} ${x(bin.forecast_mean ?? 0)},${y(bin.observed_rate ?? 0)}`,
                )
                .join(" ")}
              fill="none"
              stroke="var(--series-1)"
              strokeWidth={2}
            />
            {drawn.map((bin, i) => (
              <circle
                key={i}
                cx={x(bin.forecast_mean ?? 0)}
                cy={y(bin.observed_rate ?? 0)}
                r={Math.max(4, Math.min(9, Math.sqrt(bin.count) * 0.7))}
                fill="var(--series-1)"
                stroke="var(--surface-1)"
                strokeWidth={2}
              />
            ))}

            <text
              x={PAD.left + innerW / 2}
              y={H - 6}
              textAnchor="middle"
              fontSize={12}
              fill="var(--text-secondary)"
            >
              forecast probability
            </text>
          </svg>
        }
        legend={
          <>
            <LegendLine color="var(--series-1)" label="observed frequency per bin" />
            <LegendLine color="var(--hurdle)" label="perfect calibration" dashed />
          </>
        }
        caption={
          <>
            Marker area is proportional to how many forecasts landed in that bin. A model
            can sit exactly on the diagonal and still be useless — that is reliability
            without resolution, and resolution here is{" "}
            <span className="num">{num(event.decomposition.resolution, 5)}</span>.
          </>
        }
        table={
          <table>
            <caption>Reliability bins for &ldquo;{event.definition}&rdquo;.</caption>
            <thead>
              <tr>
                <th scope="col">Bin</th>
                <th scope="col">Forecasts</th>
                <th scope="col">Mean forecast</th>
                <th scope="col">Observed</th>
              </tr>
            </thead>
            <tbody>
              {event.bins.map((bin, i) => (
                <tr key={i}>
                  <th scope="row">
                    {bin.lower.toFixed(1)}–{bin.upper.toFixed(1)}
                  </th>
                  <td>{bin.count}</td>
                  <td>{bin.forecast_mean === null ? "—" : prob(bin.forecast_mean)}</td>
                  <td>{bin.observed_rate === null ? "—" : prob(bin.observed_rate)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        }
        tableLabel="Show the reliability bins"
      />
      {summary}
    </div>
  );
}
