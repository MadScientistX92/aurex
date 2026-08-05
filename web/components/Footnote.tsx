import type { CodeProvenance } from "@/lib/types";

/**
 * What produced the page, and the standing disclaimer.
 *
 * The commit is here rather than buried because it is the only thing that identifies
 * the code behind these numbers — `engine_version` is a static string carried by every
 * artifact this project has ever published, including the withdrawn ones.
 */
export function Footnote({
  code,
  generatedAt,
  disclaimer,
}: {
  code?: CodeProvenance;
  generatedAt?: string;
  disclaimer?: string;
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
        This page is statically built from committed JSON; there is no model running on
        the server and no key of any kind in the browser.{" "}
        <a href="https://github.com/MadScientistX92/aurex">Source and method</a>.
      </p>
    </footer>
  );
}
