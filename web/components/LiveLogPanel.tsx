import { num, sessions } from "@/lib/format";
import type { ForecastIndex, LiveLogArtifact } from "@/lib/types";

import { Notice } from "./Notice";

/**
 * The empty state is the content.
 *
 * The live log has n = 0 today and will have single digits for months. Hiding the panel
 * until the number looks like something would be a publication decision made on the
 * basis of the result, which is precisely what this project exists not to do — and a
 * visitor who sees `n = 3` beside "no test is possible yet" learns more about how this
 * repository works than one who sees a section that quietly does not exist.
 *
 * The count and the gap index are shown together on purpose. A live log with three
 * entries and no gaps means three nights ran. Three entries and eleven unexplained gaps
 * means something else entirely, and a reader cannot tell those apart from the count.
 */
export function LiveLogPanel({
  log,
  index,
}: {
  log: LiveLogArtifact | null;
  index: ForecastIndex | null;
}) {
  const total = log?.total_observations ?? 0;
  const published = index?.counts.published ?? 0;

  return (
    <section className="panel">
      <div className="panel-head">
        <h2 style={{ marginTop: 0 }}>The live log</h2>
        <p className="hint" style={{ margin: 0 }}>
          What the engine <em>did</em> say, before the outcome existed
        </p>
      </div>

      <p>
        The scoring above is a walk-forward simulation: honest about lookahead, but every
        one of its forecasts was graded against an outcome that already existed when the
        code ran. This section is different. These are forecasts committed to a public
        repository on a date git records, scored only after their horizon elapsed. It is
        the stronger claim, and it is tiny.
      </p>

      <dl className="readout">
        <div>
          <dt>Forecasts published</dt>
          <dd>
            {published}
            <small>Dated files in the public record, one per night the engine ran.</small>
          </dd>
        </div>
        <div>
          <dt>Horizons elapsed and scored</dt>
          <dd>
            {total}
            <small>
              A forecast enters this count only once its horizon has run. Nothing here is
              scored early.
            </small>
          </dd>
        </div>
        {index ? (
          <div>
            <dt>Nights with no forecast</dt>
            <dd>
              {index.counts.gaps}
              <small>
                {index.counts.gaps_explained} explained by a recorded refusal,{" "}
                {index.counts.gaps_unexplained} with no trace at all — the weaker case,
                and the reason both are counted separately.
              </small>
            </dd>
          </div>
        ) : null}
      </dl>

      {total === 0 ? (
        <Notice title="No test is possible yet, and none is reported." tone="key">
          <p>
            {published === 0
              ? "Nothing has been published yet. This panel will fill one night at a time."
              : `${published} forecast${published === 1 ? " has" : "s have"} been published, but no horizon has elapsed. The shortest is five sessions, so the first scored entry is about a week behind the first published one.`}{" "}
            The threshold for reporting any p-value here is 30 independent
            non-overlapping windows, fixed in advance rather than chosen once the numbers
            exist. At a five-session horizon a nightly job accrues roughly one a week, so
            that is on the order of half a year of saying exactly this.
          </p>
        </Notice>
      ) : (
        <>
          <div className="table-wrap">
            <table>
              <caption>
                Live forecasts scored, by horizon. Independent counts are thinned to
                non-overlapping windows, the same rule the backtest follows.
              </caption>
              <thead>
                <tr>
                  <th scope="col">Horizon</th>
                  <th scope="col">Scored</th>
                  <th scope="col">Independent</th>
                  <th scope="col">Censored</th>
                  <th scope="col">Mean PIT</th>
                  <th scope="col">Test possible?</th>
                </tr>
              </thead>
              <tbody>
                {(log?.horizons ?? []).map((h) => (
                  <tr key={h.horizon_sessions}>
                    <th scope="row">{sessions(h.horizon_sessions)}</th>
                    <td>{h.observations}</td>
                    <td>{h.independent_observations}</td>
                    <td>{h.censored}</td>
                    <td>{h.mean_pit === null ? "—" : num(h.mean_pit, 4)}</td>
                    <td>{h.test_possible ? "yes" : "no"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Notice tone="key">
            <p>{log?.horizons[0]?.test_note}</p>
          </Notice>
        </>
      )}

      <p className="hint">
        This is never pooled with the walk-forward. Pooling would make the live sample
        look testable years earlier by diluting it with observations that carry a weaker
        guarantee, and the counts are reported apart so a reader can see how much of the
        evidence is of which kind. The PIT here is interpolated on the published
        five-point quantile grid rather than computed against the ensemble, so it is
        coarser than the backtest&rsquo;s and not comparable with it point for point.
      </p>
    </section>
  );
}
