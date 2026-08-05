"use client";

import { useMemo, useState } from "react";

import { money, num, prob, sessions, signedPct, unitLabel } from "@/lib/format";
import { outcome } from "@/lib/stats";
import type { HorizonBlock, RouteCell } from "@/lib/types";

import { ExceedanceChart } from "./ExceedanceChart";
import { Notice } from "./Notice";

export interface CalculatorData {
  assetLabel: string;
  currency: string;
  unit: string;
  /** Units of the quoted instrument per gram, so a mass converts to an outlay. */
  unitsPerGram: number;
  anchor: number;
  horizons: [number, HorizonBlock][];
  cells: RouteCell[];
  jurisdictionLabels: Record<string, string>;
}

const NO_JURISDICTION = "";

/**
 * Size, route, jurisdiction — and the specific hurdle and odds that follow.
 *
 * Three rules are enforced here rather than left to whoever edits the copy:
 *
 * 1. **No jurisdiction is the default.** The control opens unset, and unset is a real
 *    state showing the quote-currency benchmark with friction excluded and labelled —
 *    not one country's tax stack applied because it happened to be first in the list.
 * 2. **Where the odds of profit are below one half, the headline is the loss.** The
 *    same number either way; "38% chance of profit" reads as an opportunity and "62%
 *    chance of loss" reads as what it is, and every dashboard convention pushes toward
 *    the first.
 * 3. **Nothing is computed that the engine did not publish.** The probability comes
 *    from the committed exceedance grid by lookup; the breakeven comes from the routes
 *    artifact. There is no model in this file and no arithmetic that could disagree
 *    with the engine's.
 */
export function Calculator({ data }: { data: CalculatorData }) {
  const [grams, setGrams] = useState(10);
  const [jurisdiction, setJurisdiction] = useState<string>(NO_JURISDICTION);
  const [routeId, setRouteId] = useState<string>("");
  const [horizon, setHorizon] = useState<number>(data.horizons[1]?.[0] ?? data.horizons[0]?.[0] ?? 21);

  const jurisdictions = useMemo(
    () => [...new Set(data.cells.map((c) => c.jurisdiction))].sort(),
    [data.cells],
  );

  const availableRoutes = useMemo(
    () =>
      jurisdiction === NO_JURISDICTION
        ? []
        : data.cells.filter((c) => c.jurisdiction === jurisdiction),
    [data.cells, jurisdiction],
  );

  const cell = useMemo(
    () => availableRoutes.find((c) => c.route_id === routeId) ?? availableRoutes[0] ?? null,
    [availableRoutes, routeId],
  );

  const block = useMemo(
    () => data.horizons.find(([h]) => h === horizon)?.[1] ?? null,
    [data.horizons, horizon],
  );

  const quote = cell?.breakeven[String(horizon)] ?? null;
  const hurdleMove = quote?.required_move ?? 0;
  const result = block ? outcome(block, hurdleMove) : null;

  const outlay = (grams / data.unitsPerGram) * data.anchor;
  const grossToBreakeven = outlay * hurdleMove;

  return (
    <>
      <div className="controls">
        <div className="field">
          <label htmlFor="grams">Size, in grams</label>
          <input
            id="grams"
            type="number"
            min={0.1}
            step={0.1}
            value={grams}
            onChange={(e) => setGrams(Math.max(0, Number(e.target.value) || 0))}
          />
        </div>

        <div className="field">
          <label htmlFor="jurisdiction">Jurisdiction</label>
          <select
            id="jurisdiction"
            value={jurisdiction}
            onChange={(e) => {
              setJurisdiction(e.target.value);
              setRouteId("");
            }}
          >
            <option value={NO_JURISDICTION}>Not set — benchmark, friction excluded</option>
            {jurisdictions.map((code) => (
              <option key={code} value={code}>
                {data.jurisdictionLabels[code] ?? code}
              </option>
            ))}
          </select>
        </div>

        <div className="field">
          <label htmlFor="route">Route</label>
          <select
            id="route"
            value={cell?.route_id ?? ""}
            onChange={(e) => setRouteId(e.target.value)}
            disabled={availableRoutes.length === 0}
          >
            {availableRoutes.length === 0 ? (
              <option value="">Choose a jurisdiction first</option>
            ) : (
              availableRoutes.map((c) => (
                <option key={c.route_id} value={c.route_id}>
                  {unitLabel(c.route.instrument)} — {unitLabel(c.route.venue)}
                </option>
              ))
            )}
          </select>
        </div>

        <div className="field">
          <label htmlFor="horizon">Holding period</label>
          <select
            id="horizon"
            value={horizon}
            onChange={(e) => setHorizon(Number(e.target.value))}
          >
            {data.horizons.map(([h]) => (
              <option key={h} value={h}>
                {sessions(h)}
              </option>
            ))}
          </select>
        </div>
      </div>

      {jurisdiction === NO_JURISDICTION ? (
        <Notice title="No jurisdiction is set, and that is a supported state rather than a gap." tone="key">
          <p>
            With none chosen, what follows is the quote-currency benchmark with friction{" "}
            <strong>excluded and labelled</strong> — never one country&rsquo;s tax stack
            applied silently because it happened to be first in the list. A real round
            trip pays a dealer spread, a consumption tax and a buyback discount, and the
            same metal reached by a different route in a different country faces a
            different hurdle. Choose one above to see which.
          </p>
        </Notice>
      ) : null}

      <div className="panel">
        <h2 style={{ marginTop: 0 }}>
          {grams > 0 ? `${num(grams, 1)} g` : "Nothing"} of {data.assetLabel}
          {cell ? ` via ${unitLabel(cell.route.instrument)} in ${data.jurisdictionLabels[cell.jurisdiction] ?? cell.jurisdiction}` : ""}
        </h2>

        <dl className="readout">
          <div>
            <dt>Outlay at the last close</dt>
            <dd>
              {money(outlay, data.currency)}
              <small>
                {num(grams, 1)} g at {num(data.anchor)} per {unitLabel(data.unit)}. Excludes the
                dealer spread, which is inside the hurdle below.
              </small>
            </dd>
          </div>
          <div>
            <dt>Move required to break even</dt>
            <dd>
              {signedPct(hurdleMove, 2)}
              <small>
                {cell
                  ? `Round trip: in, hold ${sessions(horizon)}, out. ${quote?.horizon_dependent ? "This route accrues, so the hurdle grows with the holding period." : "Paid at the door, so the hurdle does not move with the holding period."}`
                  : "Friction excluded — this is the benchmark, not a cost anyone actually faces."}
              </small>
            </dd>
          </div>
          <div>
            <dt>Gross gain needed</dt>
            <dd>
              {money(grossToBreakeven, data.currency)}
              <small>
                What the position has to make before it has made anything.
              </small>
            </dd>
          </div>
        </dl>

        {result ? <Odds result={result} horizon={horizon} /> : null}
      </div>

      {block ? (
        <div className="panel">
          <h3 style={{ marginTop: 0 }}>The distribution, against this hurdle</h3>
          <ExceedanceChart
            points={block.exceedance}
            hurdleMove={hurdleMove}
            hurdleLabel={
              cell
                ? `breakeven, ${signedPct(hurdleMove, 2)}`
                : "benchmark, friction excluded"
            }
            horizon={horizon}
          />
        </div>
      ) : null}

      {cell ? <Provenance cell={cell} horizon={horizon} /> : null}
    </>
  );
}

function Odds({
  result,
  horizon,
}: {
  result: ReturnType<typeof outcome>;
  horizon: number;
}) {
  const p = result.probabilityOfProfit;
  if (p === null) {
    return (
      <Notice tone="loud">
        <p>
          The breakeven for this combination falls outside the published exceedance grid,
          so no probability is shown. An extrapolated one would be invented rather than
          measured.
        </p>
      </Notice>
    );
  }

  return (
    <Notice
      title={
        result.leadWithLoss
          ? `More likely than not to lose money over ${sessions(horizon)}`
          : `Above water at ${sessions(horizon)} in ${prob(p)} of simulated paths`
      }
      tone={result.leadWithLoss ? "loud" : "key"}
    >
      <p>
        {result.leadWithLoss ? (
          <>
            <span className="num">{prob(1 - p)}</span> of simulated paths finish this
            holding period below the round-trip breakeven — a loss, after costs. The
            complement, <span className="num">{prob(p)}</span>, is the chance of finishing
            above it. Both are the same number; the loss is stated first because it is the
            more likely outcome and the one a reader is more likely to discount.
          </>
        ) : (
          <>
            <span className="num">{prob(p)}</span> of simulated paths finish above the
            round-trip breakeven and <span className="num">{prob(1 - p)}</span> finish
            below it.
          </>
        )}{" "}
        {result.probabilityOfTouch !== null ? (
          <>
            A holder able to exit mid-period had{" "}
            <span className="num">{prob(result.probabilityOfTouch)}</span> of being above
            water at <em>some</em> session close — a floor, because a level crossed and
            given back inside one session is not counted.
          </>
        ) : null}
      </p>
      <p>
        These are simulated frequencies, not a prediction and not advice. The engine has
        not been shown to forecast direction, and its resolution on exactly this event is
        nil — so read this as what the round-trip cost does to a distribution, not as a
        view about where the price is going.
      </p>
    </Notice>
  );
}

function Provenance({ cell, horizon }: { cell: RouteCell; horizon: number }) {
  const quote = cell.breakeven[String(horizon)];
  return (
    <div className="panel">
      <h3 style={{ marginTop: 0 }}>Where this hurdle comes from</h3>
      <div className="table-wrap">
        <table>
          <caption>
            Components of the round-trip breakeven at {sessions(horizon)}, each carrying
            its own source.
          </caption>
          <thead>
            <tr>
              <th scope="col">Component</th>
              <th scope="col">Rate</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(quote?.components ?? {}).map(([key, value]) => (
              <tr key={key}>
                <th scope="row">{key.replace(/_/g, " ")}</th>
                <td>{signedPct(value, 2)}</td>
              </tr>
            ))}
            <tr>
              <th scope="row">Required move</th>
              <td>{signedPct(quote?.required_move ?? 0, 2)}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="hint">
        Confidence: <strong>{cell.source_confidence}</strong>.{" "}
        <a href={cell.source_url} rel="noreferrer">
          Source for the regulated rates
        </a>
        . Dealer premiums and buyback discounts are representative and user-editable —
        they have no regulator behind them, and they are structurally prevented from
        sharing the citation that covers the tax rate on the same entry.
        {cell.max_leverage !== null ? (
          <>
            {" "}
            Leverage is capped at <span className="num">{cell.max_leverage}:1</span> here,
            recorded only because the national regulator publishes it.
          </>
        ) : null}
      </p>
      {cell.notes.length > 0 ? (
        <ul className="hint">
          {cell.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
