import { Footnote } from "@/components/Footnote";
import { LiveLogPanel } from "@/components/LiveLogPanel";
import { Nav } from "@/components/Nav";
import { Notice } from "@/components/Notice";
import { PitHistogram } from "@/components/PitHistogram";
import { ReliabilityChart } from "@/components/ReliabilityChart";
import { SkillChart } from "@/components/SkillChart";
import { calibrations, forecastIndex, groupIdenticalEvents, liveLog } from "@/lib/data";
import { num, pValue, sessions, signedPct } from "@/lib/format";

export const metadata = {
  title: "Track record — Aurex",
  description:
    "How well-calibrated the published distributions turned out to be, including where the model loses.",
};

/**
 * The scoring, including every place it comes out negative.
 *
 * §0's second rule is that a forecast which is never scored is marketing, and its fifth
 * is that the credibility is the product. So this page leads with the result that is
 * least flattering to the engine, and the reliability diagrams that were withheld for
 * lack of positive events say so rather than being quietly absent.
 */
export default function TrackRecordPage() {
  const reports = calibrations();
  const log = liveLog();
  const index = forecastIndex();

  if (reports.length === 0) {
    return (
      <>
        <Nav current="/track-record" />
        <main className="shell" id="main">
          <h1>No scoring has been published</h1>
          <Notice tone="loud">
            <p>The calibration report is not in the published data.</p>
          </Notice>
          <Footnote />
        </main>
      </>
    );
  }

  return (
    <>
      <Nav current="/track-record" />
      <main className="shell" id="main">
        <h1 className="enter enter-1">Track record</h1>
        <p className="enter enter-1">
          A forecast that is never scored is marketing. Every distribution the engine
          would have published over eleven years is graded here on shape, on sharpness
          against a null, and on whether its probabilities happen at the rate it claimed.
          Where the answer is unflattering it is the headline rather than a footnote.
        </p>

        {reports.map((report) => {
          const horizons = report.calibration.horizons;
          const skills = horizons.map((h) => h.crps.skill_score);
          const decay = report.calibration.skill_decay;

          return (
            <div key={report.asset.id}>
              <Notice title="The headline is a negative result." tone="key">
                <p>
                  CRPS skill against a driftless random walk runs between{" "}
                  <span className="num">{signedPct(Math.min(...skills), 1)}</span> and{" "}
                  <span className="num">{signedPct(Math.max(...skills), 1)}</span> across
                  horizons, and a Diebold-Mariano test rejects at none of them.
                  Conditioning on volatility is not measurably worth anything at these
                  horizons — which is a weaker claim than &ldquo;worth zero&rdquo;, and
                  the only one the data supports. With as few as{" "}
                  <span className="num">
                    {Math.min(...horizons.map((h) => h.independent_observations))}
                  </span>{" "}
                  independent windows at the longest horizon, a real effect of ordinary
                  size would be missed.
                </p>
              </Notice>

              {report.sample ? (
                <section className="panel">
                  <h2 style={{ marginTop: 0 }}>The sample these numbers came from</h2>
                  <dl className="readout">
                    <div>
                      <dt>Price history used</dt>
                      <dd>
                        {report.sample.resolved.from} → {report.sample.resolved.to}
                        <small>
                          {report.sample.observations} observations. The bound holds back
                          outcomes as well as forecasts, so no window is scored against
                          data outside it.
                        </small>
                      </dd>
                    </div>
                    <div>
                      <dt>Forecast dates</dt>
                      <dd>
                        {horizons[0]?.first_as_of} →{" "}
                        {horizons.map((h) => h.last_as_of).sort().at(-1)}
                        <small>
                          One refit every five sessions, expanding window, no lookahead.
                        </small>
                      </dd>
                    </div>
                    <div>
                      <dt>Scored forecasts</dt>
                      <dd>
                        {horizons.reduce((sum, h) => sum + h.observations, 0)}
                        <small>Across {horizons.length} horizons, no skipped dates.</small>
                      </dd>
                    </div>
                  </dl>
                  {report.reproduce ? (
                    <>
                      <p style={{ marginTop: "1rem", marginBottom: "0.4rem" }}>
                        Every number on this page is reproduced by:
                      </p>
                      <pre>
                        <code>{report.reproduce}</code>
                      </pre>
                      <p className="hint">
                        The <code>--to</code> is what makes that true. Without it the run
                        ends wherever the data had reached on the day it was typed, and a
                        reader following the command gets a different sample.
                      </p>
                    </>
                  ) : null}
                </section>
              ) : null}

              <section className="panel">
                <h2 style={{ marginTop: 0 }}>Is it better than a random walk?</h2>
                <SkillChart horizons={horizons} />
                {decay ? (
                  <Notice title="And the theory said skill should decay with horizon.">
                    <p>
                      Regressing skill on log horizon and bootstrapping the slope over
                      shared as-of dates gives{" "}
                      <span className="num">{num(decay.slope, 5)}</span>, p ={" "}
                      <span className="num">{pValue(decay.p_value)}</span>, R² ={" "}
                      <span className="num">{num(decay.r_squared, 2)}</span>. No decay, no
                      trend — and an interval wide enough to contain the decline theory
                      predicts as well as no decline at all. The honest reading is that
                      nothing is distinguishable from zero anywhere, and the test has no
                      power to rule out a decay of the size theory expects.
                    </p>
                  </Notice>
                ) : null}
              </section>

              <section>
                <h2>Are the distributions the right shape?</h2>
                <p>
                  The probability integral transform asks whether outcomes fall through
                  the forecast distribution evenly. A flat histogram means calibrated. A
                  hump on one side means the distribution is displaced; a hump in the
                  middle means it is too wide.
                </p>
                {horizons.map((horizon) => (
                  <div className="panel" key={horizon.horizon_sessions}>
                    <h3 style={{ marginTop: 0 }}>
                      {sessions(horizon.horizon_sessions)} — mean PIT{" "}
                      <span className="num">{num(horizon.pit.mean, 4)}</span>
                    </h3>
                    <PitHistogram horizon={horizon} />
                  </div>
                ))}
              </section>

              <section>
                <h2>When it says 20%, does it happen 20% of the time?</h2>
                <p>
                  Each binary event the engine publishes is scored on its Brier score and
                  its reliability curve. Read the scores down a column and never across
                  one: a rare event has a low achievable Brier because it is rare, not
                  because the forecast is good, so the base rate and the positive count
                  are published beside every score.
                </p>
                {horizons.slice(0, 1).map((horizon) => (
                  <div key={horizon.horizon_sessions}>
                    {groupIdenticalEvents(horizon.events).map(({ event, labels }) => (
                      <div className="panel" key={event.id}>
                        <h3 style={{ marginTop: 0 }}>{event.definition}</h3>
                        <p className="hint">
                          At {sessions(horizon.horizon_sessions)}, over{" "}
                          {event.observations} forecasts.
                          {labels.length > 1 && (
                            <>
                              {" "}
                              Priced by {labels.length} routes whose breakeven hurdles
                              coincide on this friction table — {labels.join(", ")} — so
                              they are one scored event and are shown once. They separate
                              again the moment any of those frictions moves.
                            </>
                          )}
                        </p>
                        <ReliabilityChart event={event} />
                      </div>
                    ))}
                  </div>
                ))}
              </section>
            </div>
          );
        })}

        <LiveLogPanel log={log} index={index} />

        <Footnote
          code={reports[0]?.code}
          generatedAt={reports[0]?.generated_at}
        />
      </main>
    </>
  );
}
