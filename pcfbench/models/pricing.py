"""USD pricing per million tokens for the eight PCFBench baseline models.

Used to compute ``cost_usd`` from a ``pydantic_ai.usage.Usage`` capture.

Prices are public list prices from each provider's pricing page as of
2026-05-03; cache-read and cache-write are priced separately for the
Anthropic models. Update this table when providers re-price.

Reasoning / "thinking" tokens are billed as output tokens by every
provider we use, so we don't need a separate column for them — the
Usage object already counts them under output_tokens.
"""

from __future__ import annotations

import dataclasses
from typing import Final

from pydantic_ai.usage import Usage


@dataclasses.dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float = 0.0
    cache_write_per_mtok: float = 0.0


# Per-million-token USD list prices.
_PRICES: Final[dict[str, ModelPrice]] = {
    # Anthropic on Vertex (same prices as direct API).
    "claude-opus-4-6": ModelPrice(15.00, 75.00, 1.50, 18.75),
    "claude-sonnet-4-6": ModelPrice(3.00, 15.00, 0.30, 3.75),
    "claude-haiku-4-5@20251001": ModelPrice(1.00, 5.00, 0.10, 1.25),
    # OpenAI.
    "gpt-5.5": ModelPrice(1.25, 10.00),
    "gpt-5.4-mini": ModelPrice(0.25, 2.00),
    # Google on Vertex.
    "gemini-3.1-pro-preview": ModelPrice(1.25, 10.00),
    "gemini-3-flash-preview": ModelPrice(0.30, 2.50),
    # DeepSeek-via-Vertex (MaaS list price).
    "deepseek-ai/deepseek-v3.2-maas": ModelPrice(0.27, 1.10),
}


def get_price(model_id: str) -> ModelPrice | None:
    return _PRICES.get(model_id)


def cost_usd(model_id: str, usage: Usage) -> float | None:
    """Compute USD cost from a Usage object. Returns None if the model
    isn't in the price table."""
    price = _PRICES.get(model_id)
    if price is None:
        return None
    cost = (
        usage.input_tokens * price.input_per_mtok
        + usage.output_tokens * price.output_per_mtok
        + usage.cache_read_tokens * price.cache_read_per_mtok
        + usage.cache_write_tokens * price.cache_write_per_mtok
    ) / 1_000_000.0
    return cost
