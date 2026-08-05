import type { ReactNode } from "react";

/**
 * The block the negative results live in.
 *
 * Deliberately not a "warning" or an "error" — those are states a system is in, and
 * these are findings the project stands behind. It gets a border and a title and sits
 * inline with the content, because §0's rule is that the negative results are surfaced
 * rather than filed under methodology, and a notice styled like an incident banner is
 * one a reader learns to skip.
 */
export function Notice({
  title,
  tone = "plain",
  children,
}: {
  title?: string;
  tone?: "plain" | "key" | "loud";
  children: ReactNode;
}) {
  const cls = tone === "key" ? "notice notice-key" : tone === "loud" ? "notice notice-loud" : "notice";
  return (
    <aside className={cls}>
      {title ? <strong className="notice-title">{title}</strong> : null}
      {children}
    </aside>
  );
}
