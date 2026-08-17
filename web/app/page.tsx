import { ExceedanceChart } from "@/components/ExceedanceChart";
import { FanChart } from "@/components/FanChart";
import { Footnote } from "@/components/Footnote";
import { Nav } from "@/components/Nav";
import { Notice } from "@/components/Notice";
import { assets, calibrations, horizonKeys, latest, quoteLens, routes } from "@/lib/data";
import { isoDate, num, prob, sessions, signedPct, unitLabel } from "@/lib/format";
import { outcome } from "@/lib/stats";
import type { HorizonBlock, RouteCell } from "@/lib/types";

/**
 * Today: the distribution, and what it has to beat.
 *
 * The one page where the temptation to produce a headline number is strongest, so the
 * ordering is deliberate. The chart comes first and is the largest element. The
 * numbers beside it are a definition list at body size, none of them typographically
 * privileged. The benchmark hurdle is drawn even when no jurisdiction is chosen —
 * because §20 says no jurisdiction is the default, and because a distribution shown
 * without a cost line invites the reader to read the upper band as an expectation.
 */
export default function TodayPage() {
  const artifact = latest();
  const routeBook = routes();
  const reports = calibrations();

  if (!artifact) {
    return (
      <>
        <Nav current="/" />
        <main className="shell" id="main">
          <h1>No forecast is published</h1>
          <Notice title="This is a real state, not an error page." tone="loud">
            <p>
              The nightly job publishes one dated forecast a night or it publishes
              nothing and fails loudly. Nothing has been published here yet, and the
              site will not invent a number to fill the space.
            </p>
          </Notice>
          <Footnote />
        </main>
      </>
    );
  }

  const entries = assets(artifact);

  return (
    <>
      <Nav current="/" />
      <main className="shell" id="main">
        <h1 className="enter enter-1">Today&rsquo;s distribution</h1>
        <p className="enter enter-1">
          Aurex does not forecast a price. It simulates a distribution of them, publishes
          it before the outcome exists, and grades itself afterwards. Everything below is
          a simulated quantile, and the dashed rule on every chart is the round-trip cost
          a move has to clear before any of it is worth anything.
        </p>

        <HeadlineFinding reports={reports} />

        {entries.map(([assetKey, block], index) => {
          const lensEntry = quoteLens(block);
          if (!lensEntry) return null;
          const [lensCode, lens] = lensEntry;
          const dist = lens.distribution;

          if (!dist?.available || !dist.horizons || dist.anchor === undefined) {
            return (
              <section key={assetKey} className="panel">
                <h2>{block.asset.label}</h2>
                <Notice title="No distribution was produced for this run." tone="loud">
                  <p>{dist?.reason ?? "The engine recorded no reason, which is itself a bug."}</p>
                </Notice>
              </section>
            );
          }

          const anchor = dist.anchor;
          const horizons = horizonKeys(lens);
          const blocks: [number, HorizonBlock][] = horizons.flatMap((h) => {
            const hb = dist.horizons?.[String(h)];
            return hb ? [[h, hb] as [number, HorizonBlock]] : [];
          });

          const cells = (routeBook?.cells ?? []).filter(
            (cell) => cell.route.asset_id === block.asset.id,
          );
          const benchmark = benchmarkHurdle();
          const focusHorizon = horizons[1] ?? horizons[0];
          const focusBlock = focusHorizon ? dist.horizons[String(focusHorizon)] : undefined;

          return (
            <section key={assetKey} className={`enter enter-${Math.min(index + 2, 3)}`}>
              <div className="panel-head">
                <h2>
                  {block.asset.label}{" "}
                  <span className="hint">
                    {lensCode} per {unitLabel(lens.latest?.unit ?? block.asset.base_unit)}
                  </span>
                </h2>
                <p className="hint" style={{ margin: 0 }}>
                  Anchored to the close of <span className="num">{isoDate(lens.latest?.as_of ?? "")}</span>
                </p>
              </div>

              <div className="panel">
                <FanChart
                  anchor={anchor}
                  horizons={blocks}
                  hurdle={benchmark}
                  currency={lensCode}
                  unit={lens.latest?.unit ?? block.asset.base_unit}
                />
              </div>

              <dl className="readout">
                <div>
                  <dt>Last observed close</dt>
                  <dd>
                    {num(anchor)}
                    <small>
                      The price the simulation starts from. Not a forecast, and not
                      today&rsquo;s price if the market has moved since the fix.
                    </small>
                  </dd>
                </div>
                {blocks.map(([horizon, hb]) => {
                  const q05 = hb.quantiles.q05;
                  const q95 = hb.quantiles.q95;
                  return (
                    <div key={horizon}>
                      <dt>{sessions(horizon)}, central 90%</dt>
                      <dd>
                        {q05 !== undefined && q95 !== undefined
                          ? `${num(q05)} – ${num(q95)}`
                          : "—"}
                        <small>
                          Nine simulated paths in ten finish inside this range. One in
                          twenty finishes below it.
                        </small>
                      </dd>
                    </div>
                  );
                })}
              </dl>

              {focusBlock && focusHorizon ? (
                <div className="panel">
                  <h3 style={{ marginTop: 0 }}>
                    What it would take to break even, at {sessions(focusHorizon)}
                  </h3>
                  <ExceedanceChart
                    points={focusBlock.exceedance}
                    hurdleMove={benchmark.move}
                    hurdleLabel="benchmark, friction excluded"
                    horizon={focusHorizon}
                  />
                </div>
              ) : null}

              {focusBlock && focusHorizon ? (
                <HurdleTable
                  cells={cells}
                  block={focusBlock}
                  horizon={focusHorizon}
                  label={block.asset.label}
                />
              ) : null}
            </section>
          );
        })}

        <NotBuilt />
        <Footnote
          code={artifact.code}
          generatedAt={artifact.generated_at}
          disclaimer={artifact.disclaimer}
        />
      </main>
    </>
  );
}

/**
 * With no jurisdiction chosen there is still a hurdle, and it is zero.
 *
 * §20: no jurisdiction is the default, and the correct unset view is the quote-currency
 * benchmark with friction *excluded and labelled* — never one country's tax stack
 * applied quietly. Zero is the honest number here and the label is what stops it being
 * mistaken for "costless".
 */
function benchmarkHurdle() {
  return {
    move: 0,
    label: "No jurisdiction is set, so this is the quote-currency benchmark",
    detail:
      "friction is excluded, not zero — a real round trip pays a dealer spread, a consumption tax and a buyback discount, and the Calculator applies whichever set you choose.",
  };
}

/**
 * The negative result, on the front page, above the fold.
 *
 * §0's rule is that this is surfaced rather than filed under methodology. A reader who
 * sees the distribution without seeing this learns the opposite of what the project
 * measured.
 */
function HeadlineFinding({ reports }: { reports: ReturnType<typeof calibrations> }) {
  const report = reports[0];
  if (!report) return null;

  const skills = report.calibration.horizons.map((h) => h.crps.skill_score);
  if (skills.length === 0) return null;
  const lo = Math.min(...skills);
  const hi = Math.max(...skills);
  const worstP = Math.min(
    ...report.calibration.horizons.map(
      (h) => h.crps.significance.non_overlapping_subsample.p_value ?? 1,
    ),
  );

  return (
    <Notice title="This model has not been shown to beat a random walk." tone="key">
      <p>
        Across {report.calibration.horizons[0]?.observations ?? 0}+ out-of-sample
        forecasts, its CRPS skill against a driftless random walk runs between{" "}
        <span className="num">{signedPct(lo, 1)}</span> and{" "}
        <span className="num">{signedPct(hi, 1)}</span> depending on horizon, and a
        Diebold-Mariano test rejects at none of them — the smallest p-value anywhere is{" "}
        <span className="num">{worstP.toFixed(2)}</span>. Conditioning on volatility is
        not measurably worth anything at these horizons. The distributions are close to
        the right <em>shape</em>; that is a weaker and different claim from being better
        than the null. <a href="/track-record">The full scoring is here</a>.
      </p>
    </Notice>
  );
}

function HurdleTable({
  cells,
  block,
  horizon,
  label,
}: {
  cells: RouteCell[];
  block: HorizonBlock;
  horizon: number;
  label: string;
}) {
  if (cells.length === 0) return null;

  const rows = cells
    .map((cell) => {
      const quote = cell.breakeven[String(horizon)];
      if (!quote) return null;
      const result = outcome(block, quote.required_move);
      return { cell, quote, result };
    })
    .filter((row): row is NonNullable<typeof row> => row !== null)
    .sort((a, b) => a.quote.required_move - b.quote.required_move);

  return (
    <div className="panel">
      <h3 style={{ marginTop: 0 }}>Every route and jurisdiction, at {sessions(horizon)}</h3>
      <p className="hint">
        The same metal, the same distribution, the same day. What changes down this table
        is only what a holder pays to get in and out — and it is the part that was
        knowable in advance.
      </p>
      <div className="table-wrap">
        <table>
          <caption>
            Round-trip breakeven per route and jurisdiction for {label}, with the
            probability of clearing it over {sessions(horizon)}.
          </caption>
          <thead>
            <tr>
              <th scope="col" className="wrap">
                Route
              </th>
              <th scope="col">Jurisdiction</th>
              <th scope="col">Breakeven</th>
              <th scope="col">P(above water at horizon)</th>
              <th scope="col">P(loss)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ cell, quote, result }) => (
              <tr key={`${cell.route_id}-${cell.jurisdiction}`}>
                <th scope="row" className="wrap">
                  {unitLabel(cell.route.instrument)}
                </th>
                <td>{cell.jurisdiction}</td>
                <td>{signedPct(quote.required_move, 2)}</td>
                <td>{prob(result.probabilityOfProfit)}</td>
                <td>
                  {result.probabilityOfProfit === null
                    ? "—"
                    : prob(1 - result.probabilityOfProfit)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Notice>
        <p>
          Read the loss column, not the profit column. Every route here has a probability
          of loss above one half at this horizon, which is what a round-trip cost does to
          a distribution centred on today&rsquo;s price. Jurisdiction codes are ISO
          alpha-3 and are listed only where a regulator publishes the route as available;
          an absence is an absence of data in Aurex, never a statement about what anyone
          may hold.
        </p>
      </Notice>
    </div>
  );
}

function NotBuilt() {
  return (
    <section>
      <h2>Not built yet</h2>
      <Notice title="Drivers and Scenarios wait on the scenario engine, and are omitted rather than stubbed.">
        <p>
          Factor attribution is built. The elastic-net loadings on the macro drivers each
          asset declares are estimated, carry bootstrap intervals, and are published as an
          artifact with the leave-one-driver-out check that the central claim rests on —
          including the result that removing the safe-haven control moves the largest
          surviving loading by less than a fifth of a basis point, which is the kind of
          negative finding this project exists to publish.
        </p>
        <p>
          What is missing is the engine that propagates a shock through those loadings and
          returns a distribution rather than a number. Both views need it: Drivers without
          it is a table of coefficients with no way to ask what they imply, and Scenarios
          is nothing else. A tab that showed loadings and could not answer the question the
          tab&rsquo;s name promises would be worse than no tab, so there is no tab. When
          the scenario work lands, both appear together.
        </p>
      </Notice>
    </section>
  );
}
