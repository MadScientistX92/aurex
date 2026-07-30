"""Assets. Every asset-specific fact in Aurex lives behind this package.

Adding an asset means adding one module here and registering it below. Nothing in
``vol/``, ``dist/``, ``factors/``, ``scenarios/``, ``trade/``, ``score/`` or ``web/``
may name an asset — enforced by ``tests/test_asset_abstraction.py``.
"""

from aurex.assets.base import Asset, FactorSpec, VolConfig, describe_asset, lens_by_code
from aurex.assets.friction import (
    US_COLLECTIBLES_NOTE,
    FrictionProfile,
    FrictionQuote,
    PhysicalFriction,
    RollFriction,
)
from aurex.assets.gold import GOLD
from aurex.assets.lens import (
    LENS_COLUMNS,
    CurrencyLens,
    LensContext,
    NativeLens,
    TaxedImportLens,
)
from aurex.assets.transforms import (
    Difference,
    LogReturn,
    ReturnTransform,
    ShiftedLogReturn,
    TransformDomainError,
    round_trip,
)

#: Assets the pipeline runs by default. The synthetic asset is deliberately absent —
#: it exists for the leak test and is imported directly there.
REGISTRY: dict[str, Asset] = {GOLD.id: GOLD}


def get(asset_id: str) -> Asset:
    """Look up a registered asset, or fail with the available ids."""
    try:
        return REGISTRY[asset_id]
    except KeyError:
        raise KeyError(f"unknown asset {asset_id!r}; known: {sorted(REGISTRY)}") from None


__all__ = [
    "GOLD",
    "LENS_COLUMNS",
    "REGISTRY",
    "US_COLLECTIBLES_NOTE",
    "Asset",
    "CurrencyLens",
    "Difference",
    "FactorSpec",
    "FrictionProfile",
    "FrictionQuote",
    "LensContext",
    "LogReturn",
    "NativeLens",
    "PhysicalFriction",
    "ReturnTransform",
    "RollFriction",
    "ShiftedLogReturn",
    "TaxedImportLens",
    "TransformDomainError",
    "VolConfig",
    "describe_asset",
    "get",
    "lens_by_code",
    "round_trip",
]
