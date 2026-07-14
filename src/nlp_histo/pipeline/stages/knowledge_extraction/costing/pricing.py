"""
Configurable model price book.

Prices are loaded from a JSON file with this shape::

    {
        "batch_discount_multiplier": 0.5,
        "cached_input_discount_multiplier": 0.25,   // optional
        "models": {
            "gpt-4o-mini": {"input_per_1m": 0.15, "output_per_1m": 0.60},
            ...
        }
    }

If no file path is supplied, the bundled ``config/model_prices.json`` is used.
A missing model produces a warning and a price of ``None`` — token counts are
still recorded; only dollar costs are skipped for that model.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


def default_price_path():
    """The packaged price table, or NLP_HISTO_MODEL_PRICES if set.

    Shipped as package data and read via importlib.resources — an installed wheel has
    no repository tree, so locating it by filesystem depth silently yields an empty
    price book (every cost renders as n/a).
    """
    import os
    from importlib import resources

    override = os.getenv("NLP_HISTO_MODEL_PRICES")
    if override:
        return Path(override).expanduser()
    return resources.files("nlp_histo.resources").joinpath("model_prices.json")


@dataclass(frozen=True)
class ModelPrice:
    """USD price per 1M tokens for one model."""

    input_per_1m: float
    output_per_1m: float
    cached_input_per_1m: float | None = None

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_1m / 1_000_000.0
            + output_tokens * self.output_per_1m / 1_000_000.0
        )


class PriceBook:
    """In-memory price lookup. Centralizes every dollar-cost computation."""

    def __init__(
        self,
        prices: dict[str, ModelPrice],
        batch_discount_multiplier: float = 0.5,
        cached_input_discount_multiplier: float | None = None,
        source_path: Path | None = None,
    ) -> None:
        self._prices = prices
        self.batch_discount_multiplier = float(batch_discount_multiplier)
        self.cached_input_discount_multiplier = cached_input_discount_multiplier
        self.source_path = source_path
        self._missing_warned: set[str] = set()

    @classmethod
    def load(cls, path: Path | str | None = None) -> PriceBook:
        """Load a price book from JSON. Returns an empty book if the file is missing."""
        p = Path(path) if path is not None else default_price_path()
        if not p.is_file():
            logger.warning(
                "PriceBook: %s does not exist — proceeding with no prices "
                "(token usage will be reported, dollar costs will be omitted).",
                p,
            )
            return cls(prices={}, source_path=p)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("PriceBook: failed to parse %s: %s", p, exc)
            return cls(prices={}, source_path=p)

        raw = data.get("models", {})
        prices: dict[str, ModelPrice] = {}
        for name, fields in raw.items():
            try:
                prices[name] = ModelPrice(
                    input_per_1m=float(fields["input_per_1m"]),
                    output_per_1m=float(fields["output_per_1m"]),
                    cached_input_per_1m=(
                        float(fields["cached_input_per_1m"])
                        if "cached_input_per_1m" in fields else None
                    ),
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "PriceBook: skipping malformed entry for %r: %s", name, exc,
                )
        return cls(
            prices=prices,
            batch_discount_multiplier=float(
                data.get("batch_discount_multiplier", 0.5)
            ),
            cached_input_discount_multiplier=data.get(
                "cached_input_discount_multiplier"
            ),
            source_path=p,
        )

    def has(self, model: str) -> bool:
        return self._lookup(model) is not None

    def get(self, model: str) -> ModelPrice | None:
        return self._lookup(model)

    def cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        *,
        batch: bool = False,
        cached_input_tokens: int = 0,
    ) -> float | None:
        """Compute USD cost or return None when no price is configured.

        ``cached_input_tokens`` are billed at the cached_input rate if present
        in the price book; otherwise they are billed at the normal input rate
        (the conservative choice — we do not silently invent a discount).
        """
        price = self._lookup(model)
        if price is None:
            return None
        billable_input = max(0, input_tokens - cached_input_tokens)
        c = (
            billable_input * price.input_per_1m / 1_000_000.0
            + output_tokens * price.output_per_1m / 1_000_000.0
        )
        if cached_input_tokens > 0:
            cached_rate = (
                price.cached_input_per_1m
                if price.cached_input_per_1m is not None
                else price.input_per_1m
            )
            c += cached_input_tokens * cached_rate / 1_000_000.0
        if batch:
            c *= self.batch_discount_multiplier
        return c

    def known_models(self) -> list[str]:
        return sorted(self._prices.keys())

    def _lookup(self, model: str) -> ModelPrice | None:
        if not model:
            return None
        if model in self._prices:
            return self._prices[model]
        # Best-effort fuzzy match: strip version suffixes like "@20251001"
        # and try the bare name. Never invent a price for an unrecognized model.
        stripped = model.split("@", 1)[0]
        if stripped in self._prices:
            return self._prices[stripped]
        if model not in self._missing_warned:
            logger.warning(
                "PriceBook: no price configured for model %r — "
                "tokens will be reported but dollar cost omitted.",
                model,
            )
            self._missing_warned.add(model)
        return None
