import type { ReactNode } from "react";

/**
 * A chart, its caption, and the table that says the same thing in words.
 *
 * The table is not optional and is not a nicety. Every chart on this site encodes a
 * number a reader might act on, and a chart alone is unreadable to a screen reader, in
 * forced-colors mode, and to anyone for whom the light-mode aqua fails contrast. So
 * the component makes the alternative structural: you cannot render a figure here
 * without supplying one.
 */
export function Figure({
  title,
  caption,
  chart,
  table,
  legend,
  tableLabel = "Show the numbers behind this chart",
}: {
  title: string;
  caption?: ReactNode;
  chart: ReactNode;
  table: ReactNode;
  legend?: ReactNode;
  tableLabel?: string;
}) {
  return (
    <figure>
      <div className="figure-scroll">{chart}</div>
      {legend ? <ul className="legend">{legend}</ul> : null}
      {caption ? <figcaption>{caption}</figcaption> : null}
      <details className="table-alt">
        <summary>{tableLabel}</summary>
        <div className="table-wrap">{table}</div>
      </details>
      <span className="visually-hidden">{title}</span>
    </figure>
  );
}

export function LegendSwatch({ color, label }: { color: string; label: string }) {
  return (
    <li>
      <span className="swatch" style={{ background: color }} aria-hidden="true" />
      {label}
    </li>
  );
}

export function LegendLine({
  color,
  label,
  dashed = false,
}: {
  color: string;
  label: string;
  dashed?: boolean;
}) {
  return (
    <li>
      <span
        className="swatch swatch-line"
        style={{ borderTopColor: color, borderTopStyle: dashed ? "dashed" : "solid" }}
        aria-hidden="true"
      />
      {label}
    </li>
  );
}
