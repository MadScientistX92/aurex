import Link from "next/link";

/**
 * Three views, and an explicit note about the two that are not built.
 *
 * Drivers and Scenarios need step 4's factor loadings. Shipping them as empty shells
 * would put two dead tabs in front of a reader and imply the work is done; naming them
 * as absent costs one line and says something true.
 */
const VIEWS = [
  { href: "/", label: "Today" },
  { href: "/track-record", label: "Track record" },
  { href: "/calculator", label: "Calculator" },
] as const;

export function Nav({ current }: { current: string }) {
  return (
    <header className="masthead">
      <div className="masthead-inner">
        <Link href="/" className="wordmark">
          Aurex
          <span>distributions, not predictions</span>
        </Link>
        <nav aria-label="Views">
          <ul>
            {VIEWS.map((view) => (
              <li key={view.href}>
                <Link
                  href={view.href}
                  aria-current={view.href === current ? "page" : undefined}
                >
                  {view.label}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
      </div>
    </header>
  );
}
