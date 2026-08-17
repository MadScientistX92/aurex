import { Calculator, type CalculatorData } from "@/components/Calculator";
import { Footnote } from "@/components/Footnote";
import { Nav } from "@/components/Nav";
import { Notice } from "@/components/Notice";
import { assets, horizonKeys, latest, quoteLens, routes } from "@/lib/data";
import type { HorizonBlock } from "@/lib/types";

export const metadata = {
  title: "Calculator — Aurex",
  description:
    "Your size, your route, your jurisdiction: the specific move a round trip has to clear, and the odds of clearing it.",
};

/**
 * The hurdle, made specific.
 *
 * Everything else on this site reports a benchmark. This page asks the reader for the
 * three things that decide what they actually face — how much, by what route, under
 * whose rules — and reports the breakeven that follows and the published odds of
 * clearing it. It computes no probability of its own: the numbers are looked up in the
 * committed exceedance grid, so the page cannot disagree with the engine.
 */
export default function CalculatorPage() {
  const artifact = latest();
  const routeBook = routes();

  if (!artifact || !routeBook) {
    return (
      <>
        <Nav current="/calculator" />
        <main className="shell" id="main">
          <h1>Not enough published data</h1>
          <Notice tone="loud">
            <p>
              The calculator reads the published forecast and the published route table.
              One of them is missing, and nothing here will be estimated in its absence.
            </p>
          </Notice>
          <Footnote />
        </main>
      </>
    );
  }

  const jurisdictionLabels = Object.fromEntries(
    routeBook.jurisdictions.map((j) => [j.code, j.label]),
  );

  const datasets: CalculatorData[] = assets(artifact).flatMap(([, block]) => {
    const lensEntry = quoteLens(block);
    if (!lensEntry) return [];
    const [lensCode, lens] = lensEntry;
    const dist = lens.distribution;
    if (!dist?.available || !dist.horizons || dist.anchor === undefined) return [];

    const gramsPerUnit = (lens.lens as { grams_per_unit?: number | null }).grams_per_unit;
    if (typeof gramsPerUnit !== "number" || gramsPerUnit <= 0) return [];

    const horizons: [number, HorizonBlock][] = horizonKeys(lens).flatMap((h) => {
      const hb = dist.horizons?.[String(h)];
      return hb ? [[h, hb] as [number, HorizonBlock]] : [];
    });
    if (horizons.length === 0) return [];

    return [
      {
        assetLabel: block.asset.label,
        currency: lensCode,
        unit: lens.latest?.unit ?? block.asset.base_unit,
        unitsPerGram: gramsPerUnit,
        anchor: dist.anchor,
        horizons,
        cells: routeBook.cells.filter((cell) => cell.route.asset_id === block.asset.id),
        jurisdictionLabels,
      },
    ];
  });

  return (
    <>
      <Nav current="/calculator" />
      <main className="shell" id="main">
        <h1 className="enter enter-1">What it has to beat</h1>
        <p className="enter enter-1">
          The friction is deterministic and knowable. The price move is not. Most tools
          model the uncertain part and ignore the certain one — this page does the
          opposite first, then shows what the published distribution says about clearing
          it.
        </p>

        {datasets.length === 0 ? (
          <Notice tone="loud">
            <p>
              No published distribution carries a mass basis, so a size in grams cannot be
              converted into an outlay. Rather than guess at a conversion, the calculator
              is unavailable.
            </p>
          </Notice>
        ) : (
          datasets.map((data) => (
            <section key={data.assetLabel} className="enter enter-2">
              <Calculator data={data} />
            </section>
          ))
        )}

        <Notice title="What this is not.">
          <p>
            Not advice, and not a recommendation to hold anything. Availability is
            informational — where a route is not listed for a jurisdiction that is an
            absence of data in Aurex, never a statement about what a reader may hold, and
            leverage caps are recorded only where the national regulator&rsquo;s own
            instrument was read. Dealer premiums and buyback discounts are representative
            defaults, not quotes: enter your own dealer&rsquo;s actual numbers before
            treating any of this as your hurdle.
          </p>
        </Notice>

        <Footnote
          code={artifact.code}
          generatedAt={artifact.generated_at}
          disclaimer={artifact.disclaimer}
          also={[
            {
              label: "The breakeven table",
              code: routeBook.code,
              generatedAt: routeBook.generated_at,
            },
          ]}
        />
      </main>
    </>
  );
}
