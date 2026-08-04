"""Routes and jurisdictions: the same metal, reached different ways, taxed differently.

§20 supersedes §18's India-specific framing. A **route** is a way of holding an asset —
a venue, an instrument, a quote currency. A **jurisdiction** is who is holding it. The
two together decide what a round trip costs and how much leverage a regulator permits,
and neither decides that on its own.

**The split is the whole design, and it was got wrong first.** §20's original dataclass
put ``friction`` and ``max_leverage`` on ``Route``. They do not live there: the same CFD
route carries different caps under different regulators, and the same physical route
pays different consumption tax in different countries. Putting them on the route makes
whichever jurisdiction was written first into the default — which is exactly the bug
§20 exists to fix. So :class:`Route` holds only what is jurisdiction-invariant, and
everything a regulator or a tax authority sets lives in :class:`RouteTerms`, keyed by
``(route, jurisdiction)``.

**No jurisdiction is the default.** Leaving it unset is a supported state, not a missing
input: it yields the quote-currency benchmark with friction *excluded and labelled*, via
:meth:`RouteBook.benchmark`. An engine that silently applied one country's tax stack to
an unset jurisdiction would be making a claim about the reader.

**Availability is informational.** ``available_in`` records where a route is published
as available, with a link. ``restricted_in`` is *derived* from it against the known
jurisdictions rather than hand-maintained, because two lists that can disagree
eventually will. Nothing here gates a calculation or advises anyone; it states what a
regulator publishes and cites it.

**Blocs are not jurisdictions.** The EU is a valid ISO-3166 exceptionally-reserved code
and is still wrong here, for a substantive reason rather than a notational one: VAT
exemption on investment gold is a *Directive*, implemented by each member state in its
own national instrument. So a bloc is a named grouping that carries the Directive, and
every member's terms must cite its own national law — enforced, not requested.

**Leverage caps cite the regulator that sets them.** A cap traced to a pan-regional
body rather than to the national regulator is refused at load, because the national
regulator is who a reader would have to argue with.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from aurex.assets.friction import FrictionProfile, PhysicalFriction, RollFriction
from aurex.data.schedules.provenance import (
    Confidence,
    ScheduleError,
    read_yaml,
    require_provenance,
)

ROUTES_FILE = "routes.yaml"

#: ISO-3166-1 alpha-3, uppercase, matched case-sensitively. Alpha-2 is unusable as a
#: leak-guard token — ``IN`` and ``IT`` are English words and collide with over a
#: hundred lines of ordinary prose in the guarded modules. Alpha-3 matched
#: case-insensitively still collides (``ARE``). Uppercase alpha-3 matched
#: case-sensitively collides with nothing, which is what makes the guard usable.
JURISDICTION_CODE = re.compile(r"^[A-Z]{3}$")

#: Bodies that coordinate rather than set. A leverage cap sourced here is refused: the
#: reader's regulator is the national one, and that is who publishes the binding number.
NON_SETTING_BODIES = ("esma.europa.eu", "iosco.org")


class RouteError(ScheduleError):
    """The routes table is malformed, or violates one of §20's rules."""


@dataclass(frozen=True, slots=True)
class Jurisdiction:
    """Somewhere a route can be held, and who says so."""

    code: str
    label: str
    #: Bloc this jurisdiction belongs to, where one applies. Informational.
    bloc: str | None
    source_url: str
    source_confidence: Confidence

    def describe(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "label": self.label,
            "bloc": self.bloc,
            "source_url": self.source_url,
            "source_confidence": self.source_confidence,
        }


@dataclass(frozen=True, slots=True)
class Bloc:
    """A grouping whose rule is implemented nationally — never a jurisdiction itself."""

    code: str
    label: str
    members: tuple[str, ...]
    #: What the bloc actually publishes. Members implement it; they do not inherit it.
    instrument: str
    source_url: str
    source_confidence: Confidence

    def describe(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "label": self.label,
            "members": list(self.members),
            "instrument": self.instrument,
            "source_url": self.source_url,
            "source_confidence": self.source_confidence,
            "note": (
                "A bloc is not a jurisdiction. Its instrument is implemented in each "
                "member's national law, and every member's terms cite that law rather "
                "than this one."
            ),
        }


@dataclass(frozen=True, slots=True)
class Route:
    """A way of holding an asset. Only what does not vary by jurisdiction.

    If a field here would take a different value for a reader in another country, it
    belongs in :class:`RouteTerms` instead. That is the entire rule, and the reason
    ``friction`` and ``max_leverage`` are conspicuously absent.
    """

    id: str
    asset_id: str
    venue: str
    #: ``physical`` | ``etf`` | ``futures`` | ``cfd`` | ``bond``.
    instrument: str
    quote_currency: str
    #: Where the route is published as available. Informational; see the module
    #: docstring. Never used to gate a calculation.
    available_in: tuple[str, ...]
    source_url: str
    source_confidence: Confidence
    notes: tuple[str, ...] = ()

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "asset_id": self.asset_id,
            "venue": self.venue,
            "instrument": self.instrument,
            "quote_currency": self.quote_currency,
            "available_in": list(self.available_in),
            "source_url": self.source_url,
            "source_confidence": self.source_confidence,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class RouteTerms:
    """What one jurisdiction's rules do to one route. The (route x jurisdiction) cell.

    ``max_leverage`` is ``None`` for an unleveraged route, and that is load-bearing
    rather than cosmetic: it is what :func:`aurex.vol.model_for` reads to decide whether
    a deterministic variance model may be used at all. See
    :meth:`RouteBook.models_barred_for`.
    """

    route_id: str
    jurisdiction: str
    friction: FrictionProfile
    max_leverage: float | None
    source_url: str
    source_confidence: Confidence
    notes: tuple[str, ...] = ()

    @property
    def leveraged(self) -> bool:
        return self.max_leverage is not None

    def describe(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "jurisdiction": self.jurisdiction,
            "max_leverage": self.max_leverage,
            "friction": self.friction.describe(),
            "source_url": self.source_url,
            "source_confidence": self.source_confidence,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkView:
    """A route with no jurisdiction set: the quote-currency view, friction excluded.

    Not a degenerate :class:`RouteTerms` and deliberately not duck-compatible with one.
    It has no ``friction``, so a caller cannot reach for a hurdle that does not exist
    and silently get one country's. Asking it for a hurdle raises and says why.
    """

    route_id: str
    quote_currency: str

    @property
    def friction_excluded(self) -> bool:
        return True

    def describe(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "jurisdiction": None,
            "quote_currency": self.quote_currency,
            "friction": None,
            "friction_excluded": True,
            "reading": (
                "No jurisdiction was set, so this is the quote-currency benchmark with "
                "friction excluded. It is not what any particular buyer would pay: "
                "consumption tax, dealer spread and leverage caps are all set "
                "nationally. Set a jurisdiction to price a round trip."
            ),
        }


def _require_spread_basis(
    name: str, key: str, raw: dict[str, Any], spreads: dict[str, float]
) -> str:
    """A dealer spread has no regulator, and must not borrow the tax entry's citation.

    ``source_url`` on a terms entry documents what an authority *sets* — a consumption
    tax, a leverage cap. A dealer premium and a buyback discount are market conventions:
    representative, user-editable, and nobody's published rule. Letting them sit under
    the same citation would make one verified link stand behind two very different
    kinds of number, which is the exact failure the per-entry provenance rule exists to
    prevent. So any non-zero spread must declare where it came from, in its own words.
    """
    if not any(value > 0.0 for value in spreads.values()):
        return "none: every spread parameter is zero"
    basis = raw.get("spread_basis")
    if not basis:
        moving = sorted(k for k, v in spreads.items() if v > 0.0)
        raise RouteError(
            f"{name}[{key}]: {moving} are non-zero and no spread_basis is declared. "
            f"These are market conventions, not published rules, and they may not "
            f"shelter under the regulatory source_url on this entry. State what they "
            f"are representative of."
        )
    return str(basis)


def _friction_from(name: str, key: str, raw: dict[str, Any]) -> FrictionProfile:
    """Build the declared friction shape. The ``kind`` discriminator is required."""
    kind = raw.get("kind")
    label = str(raw.get("label", key))
    notes = tuple(str(n) for n in raw.get("notes", ()))

    if kind == "physical":
        spreads = {
            "dealer_premium": float(raw["dealer_premium"]),
            "buyback_discount": float(raw["buyback_discount"]),
        }
        basis = _require_spread_basis(name, key, raw, spreads)
        return PhysicalFriction(
            id=key,
            label=label,
            dealer_premium=spreads["dealer_premium"],
            consumption_tax=float(raw["consumption_tax"]),
            buyback_discount=spreads["buyback_discount"],
            notes=(*notes, f"Spread parameters: {basis}"),
        )
    if kind == "roll":
        spreads = {"bid_ask_spread": float(raw.get("bid_ask_spread", 0.0))}
        basis = _require_spread_basis(name, key, raw, spreads)
        return RollFriction(
            id=key,
            label=label,
            annual_expense_ratio=float(raw["annual_expense_ratio"]),
            annual_roll_drag=float(raw["annual_roll_drag"]),
            bid_ask_spread=spreads["bid_ask_spread"],
            pinned_days=int(raw.get("pinned_days", 30)),
            notes=(*notes, f"Spread parameters: {basis}"),
        )
    raise RouteError(
        f"{name}[{key}]: friction kind {kind!r} is not one of 'physical' or 'roll'. "
        f"The shape is declared rather than inferred, because a carry cost forced into "
        f"a one-off spread understates a long hold and a spread spread over a year "
        f"understates a short one."
    )


@dataclass(frozen=True, slots=True)
class RouteBook:
    """Every route, jurisdiction and cell, with the invariants already checked."""

    routes: tuple[Route, ...]
    jurisdictions: tuple[Jurisdiction, ...]
    blocs: tuple[Bloc, ...]
    terms: tuple[RouteTerms, ...]

    def route(self, route_id: str) -> Route:
        for entry in self.routes:
            if entry.id == route_id:
                return entry
        raise KeyError(f"unknown route {route_id!r}; known: {[r.id for r in self.routes]}")

    def jurisdiction(self, code: str) -> Jurisdiction:
        for entry in self.jurisdictions:
            if entry.code == code:
                return entry
        known = [j.code for j in self.jurisdictions]
        raise KeyError(f"unknown jurisdiction {code!r}; known: {known}")

    def terms_for(self, route_id: str, jurisdiction: str) -> RouteTerms:
        """The cell, or a loud failure naming what is recorded.

        Failing here means Aurex has no published terms for that pairing — not that a
        reader may not hold it. The distinction matters enough to be in the message:
        this layer reports what regulators publish and never advises.
        """
        for entry in self.terms:
            if entry.route_id == route_id and entry.jurisdiction == jurisdiction:
                return entry
        recorded = sorted(t.jurisdiction for t in self.terms if t.route_id == route_id)
        raise KeyError(
            f"no terms recorded for route {route_id!r} in {jurisdiction!r}; recorded "
            f"for: {recorded}. This is an absence of data in Aurex, not a statement "
            f"about whether the route may be held there."
        )

    def benchmark(self, route_id: str) -> BenchmarkView:
        """The unset-jurisdiction view: quote currency, friction excluded and labelled."""
        entry = self.route(route_id)
        return BenchmarkView(route_id=entry.id, quote_currency=entry.quote_currency)

    def restricted_in(self, route_id: str) -> tuple[str, ...]:
        """Derived, never stored: every known jurisdiction the route is not available in."""
        available = set(self.route(route_id).available_in)
        return tuple(sorted(j.code for j in self.jurisdictions if j.code not in available))

    def for_asset(self, asset_id: str) -> tuple[Route, ...]:
        return tuple(entry for entry in self.routes if entry.asset_id == asset_id)

    def leveraged_terms(self) -> tuple[RouteTerms, ...]:
        return tuple(entry for entry in self.terms if entry.leveraged)

    def codes(self) -> tuple[str, ...]:
        """Every jurisdiction and bloc code, for the static leak guard."""
        return tuple(sorted({j.code for j in self.jurisdictions} | {b.code for b in self.blocs}))

    def describe(self) -> dict[str, Any]:
        return {
            "routes": [entry.describe() for entry in self.routes],
            "jurisdictions": [entry.describe() for entry in self.jurisdictions],
            "blocs": [entry.describe() for entry in self.blocs],
            "terms": [entry.describe() for entry in self.terms],
            "conventions": {
                "default_jurisdiction": None,
                "availability": (
                    "Informational. Aurex states what a regulator publishes and links "
                    "it; it never gates a calculation or advises."
                ),
                "restricted_in": (
                    "Derived from available_in against the known jurisdictions, never "
                    "stored, so the two cannot disagree."
                ),
                "leverage_and_friction": (
                    "Set per (route, jurisdiction). Route carries only what does not "
                    "vary by jurisdiction."
                ),
            },
        }


def _load_jurisdictions(raw: dict[str, Any]) -> tuple[Jurisdiction, ...]:
    entries: list[Jurisdiction] = []
    for code, item in (raw.get("jurisdictions") or {}).items():
        if not JURISDICTION_CODE.match(str(code)):
            raise RouteError(
                f"{ROUTES_FILE}: jurisdiction code {code!r} must be uppercase ISO-3166 "
                f"alpha-3. Alpha-2 codes are unusable as leak-guard tokens and lowercase "
                f"alpha-3 collides with ordinary English."
            )
        confidence = require_provenance(ROUTES_FILE, f"jurisdictions.{code}", item)
        entries.append(
            Jurisdiction(
                code=str(code),
                label=str(item["label"]),
                bloc=None if item.get("bloc") is None else str(item["bloc"]),
                source_url=str(item["source_url"]),
                source_confidence=confidence,
            )
        )
    if not entries:
        raise RouteError(f"{ROUTES_FILE}: no jurisdictions")
    entries.sort(key=lambda j: j.code)
    return tuple(entries)


def _load_blocs(raw: dict[str, Any], known: set[str]) -> tuple[Bloc, ...]:
    entries: list[Bloc] = []
    for code, item in (raw.get("blocs") or {}).items():
        confidence = require_provenance(ROUTES_FILE, f"blocs.{code}", item)
        members = tuple(str(m) for m in item.get("members", ()))
        if str(code) in known:
            raise RouteError(
                f"{ROUTES_FILE}: {code!r} is registered as both a bloc and a "
                f"jurisdiction. A bloc publishes an instrument that its members "
                f"implement nationally; it is not somewhere a reader lives."
            )
        unknown = sorted(set(members) - known)
        if unknown:
            raise RouteError(f"{ROUTES_FILE}: blocs.{code} names unknown members {unknown}")
        if not members:
            raise RouteError(
                f"{ROUTES_FILE}: blocs.{code} has no members. A bloc with no explicit "
                f"members is a jurisdiction by another name."
            )
        entries.append(
            Bloc(
                code=str(code),
                label=str(item["label"]),
                members=members,
                instrument=str(item["instrument"]),
                source_url=str(item["source_url"]),
                source_confidence=confidence,
            )
        )
    entries.sort(key=lambda b: b.code)
    return tuple(entries)


def _load_routes(raw: dict[str, Any], known: set[str]) -> tuple[Route, ...]:
    entries: list[Route] = []
    for route_id, item in (raw.get("routes") or {}).items():
        if "restricted_in" in item:
            raise RouteError(
                f"{ROUTES_FILE}: routes.{route_id} declares restricted_in. It is derived "
                f"from available_in against the known jurisdictions — two hand-maintained "
                f"lists eventually disagree, and the one that is wrong is the one nobody "
                f"reads."
            )
        for banned in ("friction", "max_leverage"):
            if banned in item:
                raise RouteError(
                    f"{ROUTES_FILE}: routes.{route_id} declares {banned}, which varies by "
                    f"jurisdiction and belongs in terms[]. Leaving it on the route makes "
                    f"whichever jurisdiction was written first into the default."
                )
        confidence = require_provenance(ROUTES_FILE, f"routes.{route_id}", item)
        available = tuple(str(code) for code in item.get("available_in", ()))
        unknown = sorted(set(available) - known)
        if unknown:
            raise RouteError(
                f"{ROUTES_FILE}: routes.{route_id} is available_in unknown jurisdictions {unknown}"
            )
        entries.append(
            Route(
                id=str(route_id),
                asset_id=str(item["asset_id"]),
                venue=str(item["venue"]),
                instrument=str(item["instrument"]),
                quote_currency=str(item["quote_currency"]),
                available_in=available,
                source_url=str(item["source_url"]),
                source_confidence=confidence,
                notes=tuple(str(n) for n in item.get("notes", ())),
            )
        )
    if not entries:
        raise RouteError(f"{ROUTES_FILE}: no routes")
    entries.sort(key=lambda r: r.id)
    return tuple(entries)


def _load_terms(
    raw: dict[str, Any],
    routes: tuple[Route, ...],
    jurisdictions: tuple[Jurisdiction, ...],
    blocs: tuple[Bloc, ...],
) -> tuple[RouteTerms, ...]:
    by_id = {entry.id: entry for entry in routes}
    known = {entry.code for entry in jurisdictions}
    bloc_urls = {entry.source_url for entry in blocs}
    in_bloc = {entry.code for entry in jurisdictions if entry.bloc is not None}

    entries: list[RouteTerms] = []
    seen: set[tuple[str, str]] = set()

    for index, item in enumerate(raw.get("terms") or []):
        confidence = require_provenance(ROUTES_FILE, f"terms[{index}]", item)
        route_id = str(item["route_id"])
        jurisdiction = str(item["jurisdiction"])
        key = (route_id, jurisdiction)

        if route_id not in by_id:
            raise RouteError(f"{ROUTES_FILE}: terms[{index}] names unknown route {route_id!r}")
        if jurisdiction not in known:
            raise RouteError(
                f"{ROUTES_FILE}: terms[{index}] names unknown jurisdiction {jurisdiction!r}"
            )
        if key in seen:
            raise RouteError(f"{ROUTES_FILE}: terms[{index}] duplicates the cell {key}")
        if jurisdiction not in by_id[route_id].available_in:
            raise RouteError(
                f"{ROUTES_FILE}: terms[{index}] sets terms for {key}, but the route is "
                f"not available_in {jurisdiction!r}. Availability and terms are the two "
                f"lists §20 warns can disagree; they are checked against each other "
                f"here rather than trusted."
            )
        seen.add(key)

        source_url = str(item["source_url"])
        max_leverage = item.get("max_leverage")
        if max_leverage is not None:
            max_leverage = float(max_leverage)
            if max_leverage <= 1.0:
                raise RouteError(
                    f"{ROUTES_FILE}: terms[{index}] sets max_leverage={max_leverage}, "
                    f"which is not leverage. Use null for an unleveraged route."
                )
            for body in NON_SETTING_BODIES:
                if body in source_url:
                    raise RouteError(
                        f"{ROUTES_FILE}: terms[{index}] sources a leverage cap to "
                        f"{body}, which coordinates rather than sets. Cite the national "
                        f"regulator that publishes the binding number."
                    )
        if jurisdiction in in_bloc and source_url in bloc_urls:
            raise RouteError(
                f"{ROUTES_FILE}: terms[{index}] cites its bloc's own instrument for "
                f"{jurisdiction!r}. A Directive is implemented in national law; cite "
                f"the national instrument."
            )

        entries.append(
            RouteTerms(
                route_id=route_id,
                jurisdiction=jurisdiction,
                friction=_friction_from(
                    ROUTES_FILE, f"{route_id}.{jurisdiction}", item["friction"]
                ),
                max_leverage=max_leverage,
                source_url=source_url,
                source_confidence=confidence,
                notes=tuple(str(n) for n in item.get("notes", ())),
            )
        )

    missing = sorted(
        (entry.id, code)
        for entry in routes
        for code in entry.available_in
        if (entry.id, code) not in seen
    )
    if missing:
        raise RouteError(
            f"{ROUTES_FILE}: routes claim availability with no terms recorded: {missing}. "
            f"A route available somewhere with no friction is a hurdle of zero, which is "
            f"the one number this project must never publish by omission."
        )

    entries.sort(key=lambda t: (t.route_id, t.jurisdiction))
    return tuple(entries)


def build_routes(raw: dict[str, Any]) -> RouteBook:
    """Validate a routes mapping. Every invariant in this module is checked here.

    Separate from :func:`load_routes` so the schema test can feed it malformed tables
    directly. A validator only reachable through the real file is a validator tested
    exactly once, against the one input already known to pass.
    """
    jurisdictions = _load_jurisdictions(raw)
    known = {entry.code for entry in jurisdictions}
    blocs = _load_blocs(raw, known)

    declared = {entry.bloc for entry in jurisdictions if entry.bloc is not None}
    unknown_blocs = sorted(declared - {entry.code for entry in blocs})
    if unknown_blocs:
        raise RouteError(f"{ROUTES_FILE}: jurisdictions claim unknown blocs {unknown_blocs}")
    for bloc in blocs:
        wrong = sorted(
            entry.code
            for entry in jurisdictions
            if entry.code in bloc.members and entry.bloc != bloc.code
        )
        if wrong:
            raise RouteError(
                f"{ROUTES_FILE}: blocs.{bloc.code} lists {wrong} as members, but they do "
                f"not declare that membership. Membership is stated on both sides so "
                f"neither list can drift."
            )

    routes = _load_routes(raw, known)
    return RouteBook(
        routes=routes,
        jurisdictions=jurisdictions,
        blocs=blocs,
        terms=_load_terms(raw, routes, jurisdictions, blocs),
    )


@lru_cache(maxsize=1)
def load_routes() -> RouteBook:
    """The routes table as shipped."""
    return build_routes(read_yaml(ROUTES_FILE))


__all__ = [
    "JURISDICTION_CODE",
    "NON_SETTING_BODIES",
    "ROUTES_FILE",
    "BenchmarkView",
    "Bloc",
    "Jurisdiction",
    "Route",
    "RouteBook",
    "RouteError",
    "RouteTerms",
    "build_routes",
    "load_routes",
]
