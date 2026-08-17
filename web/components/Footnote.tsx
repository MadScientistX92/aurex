import type { CodeProvenance } from "@/lib/types";

/**
 * What produced the page, and the standing disclaimer.
 *
 * The commit is here rather than buried because it is the only thing that identifies
 * the code behind these numbers — `engine_version` is a static string carried by every
 * artifact this project has ever published, including the withdrawn ones.
 *
 * **A page built from two artifacts has to name both.** The calculator renders today's
 * distribution from one file and its hurdle table from another, and for a while it
 * reported only the first: a route table generated on 2026-08-05 from a modified working
 * tree sat under a line saying "generated 2026-08-17" with no such warning. Nobody was
 * misled about the distribution; the numbers a reader would actually act on — the
 * breakevens — were the ones the footnote did not describe. That is the same failure as
 * an artifact uploaded from a run that did not produce it, and it is why `also` exists
 * rather than the page picking whichever provenance looked best.
 */
export function Footnote({
  code,
  generatedAt,
  disclaimer,
  also,
}: {
  code?: CodeProvenance;
  generatedAt?: string;
  disclaimer?: string;
  /** Further artifacts this page is built from, each named so a reader can tell them apart. */
  also?: { label: string; code?: CodeProvenance; generatedAt?: string }[];
}) {
  return (
    <footer className="footnote">
      <p>
        {disclaimer ??
          "Aurex is a research and education tool. It produces probability distributions, not advice. Short-horizon price direction is not reliably forecastable, and nothing here changes that."}
      </p>
      <p>
        {generatedAt ? (
          <>
            Artifact generated <span className="num">{generatedAt.slice(0, 19).replace("T", " ")}</span>{" "}
            UTC.{" "}
          </>
        ) : null}
        {code?.commit ? (
          <>
            Produced by commit <span className="num">{code.commit.slice(0, 12)}</span>
            {code.dirty ? " (working tree modified — not reproducible from that commit alone)" : ""}.{" "}
          </>
        ) : null}
        {also?.map((source) => (
          <span key={source.label}>
            {source.label} comes from a separate artifact
            {source.generatedAt ? (
              <>
                {" "}
                generated{" "}
                <span className="num">
                  {source.generatedAt.slice(0, 19).replace("T", " ")}
                </span>{" "}
                UTC
              </>
            ) : null}
            {source.code?.commit ? (
              <>
                {" "}
                by commit <span className="num">{source.code.commit.slice(0, 12)}</span>
                {source.code.dirty
                  ? " (working tree modified — not reproducible from that commit alone)"
                  : ""}
              </>
            ) : null}
            .{" "}
          </span>
        ))}
        This page is statically built from committed JSON; there is no model running on
        the server and no key of any kind in the browser.{" "}
        <a href="https://github.com/MadScientistX92/aurex">Source and method</a>.
      </p>
    </footer>
  );
}
