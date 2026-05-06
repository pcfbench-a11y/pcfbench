"""Build a Pydantic AI ``Agent`` for any of the eight PCFBench models.

Reasoning configs are taken from ``pcfbench.models.registry``.
"""

from __future__ import annotations

from typing import Any

import httpx
import tenacity
from anthropic import AsyncAnthropicVertex
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.models.openai import (
    OpenAIChatModel,
    OpenAIChatModelSettings,
    OpenAIResponsesModel,
    OpenAIResponsesModelSettings,
)
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after

from pcfbench.models.registry import (
    AGENTIC_OPENAI_REASONING_EFFORT,
    AGENTIC_THINKING_BUDGETS,
    ANTHROPIC_MODELS,
    DEEPSEEK_VERTEX_MODELS,
    GEMINI_MODELS,
    OPENAI_MODELS,
    SINGLESHOT_OPENAI_REASONING_EFFORT,
    SINGLESHOT_THINKING_BUDGETS,
    has_reasoning_agentic,
    has_reasoning_singleshot,
)
from pcfbench.models.vertex_auth import (
    VertexBearerAuth,
    vertex_maas_base_url,
    vertex_project_id,
)

# Anthropic on Vertex: pin to the "global" region — it's a valid region
# for every Claude model PCFBench uses (Haiku 4.5 is "global"-only).
_ANTHROPIC_VERTEX_REGION = "global"


def _retrying_transport(
    *, wrapped: httpx.AsyncBaseTransport | None = None
) -> AsyncTenacityTransport:
    """Wrap a transport with exponential-backoff 429 / 5xx retries.

    ``validate_response`` raises ``httpx.HTTPStatusError`` on non-2xx, so
    tenacity's ``retry_if_exception_type`` fires for the same conditions
    Anthropic's SDK retry-on-429 already covers. Honors ``Retry-After``
    when the upstream sets it.
    """
    config = RetryConfig(
        stop=tenacity.stop_after_attempt(6),
        wait=wait_retry_after(
            fallback_strategy=tenacity.wait_random_exponential(
                multiplier=2, min=2, max=60
            ),
            max_wait=120,
        ),
        retry=tenacity.retry_if_exception_type(httpx.HTTPStatusError),
        reraise=True,
    )

    def _validate(response: httpx.Response) -> None:
        # Retry on 429 + transient 5xx; 4xx (other than 429) are real errors.
        if response.status_code == 429 or 500 <= response.status_code < 600:
            response.raise_for_status()

    return AsyncTenacityTransport(
        config=config, wrapped=wrapped, validate_response=_validate
    )


def _retrying_http_client(
    *, auth: httpx.Auth | None = None, timeout: float = 120.0
) -> httpx.AsyncClient:
    """Build an ``httpx.AsyncClient`` with 429 + 5xx retries baked in."""
    base = httpx.AsyncHTTPTransport()
    transport = _retrying_transport(wrapped=base)
    kwargs: dict[str, Any] = {"transport": transport, "timeout": timeout}
    if auth is not None:
        kwargs["auth"] = auth
    return httpx.AsyncClient(**kwargs)


def _anthropic_thinking(budget: int | None) -> dict[str, Any] | None:
    if budget is None or budget <= 0:
        return None
    return {"type": "enabled", "budget_tokens": budget}


def _gemini_thinking(budget: int | None) -> dict[str, Any] | None:
    """Return the Gemini thinking config dict, or None to use SDK default.

    ``budget=None``: no override (model auto-decides — Gemini 3 Flash
    interprets this as "dynamic" thinking, which can blow past the
    default ``max_output_tokens`` of 8192 before producing any
    response). ``budget=0``: explicitly disable thinking. ``budget>0``:
    set the explicit thinking budget.
    """
    if budget is None:
        return None
    if budget == 0:
        return {"thinking_budget": 0}
    return {"thinking_budget": budget}


def _build_anthropic(
    model_id: str, *, budget: int | None, has_reasoning: bool
) -> AnthropicModel:
    settings_kwargs: dict[str, Any] = {
        "temperature": 1.0 if has_reasoning else 0.0,
        # max_tokens must exceed thinking.budget_tokens; we set 16k by
        # default which comfortably exceeds the 8192-token reasoning
        # budget we use and leaves headroom for the actual response.
        "max_tokens": 16000,
    }
    thinking = _anthropic_thinking(budget)
    if thinking is not None:
        settings_kwargs["anthropic_thinking"] = thinking
    vertex_client = AsyncAnthropicVertex(
        project_id=vertex_project_id(),
        region=_ANTHROPIC_VERTEX_REGION,
        max_retries=5,
    )
    return AnthropicModel(
        model_id,
        provider=AnthropicProvider(anthropic_client=vertex_client),
        settings=AnthropicModelSettings(**settings_kwargs),
    )


def _build_openai(model_id: str, *, reasoning_effort: str | None):
    """Build an OpenAI model. Reasoning models require the Responses API
    (``/v1/responses``) — function-tool calls with reasoning_effort are not
    supported on ``/v1/chat/completions`` for gpt-5.5 et al."""
    if reasoning_effort is not None:
        return OpenAIResponsesModel(
            model_id,
            provider=OpenAIProvider(http_client=_retrying_http_client()),
            settings=OpenAIResponsesModelSettings(
                openai_reasoning_effort=reasoning_effort
            ),
        )
    return OpenAIChatModel(
        model_id,
        provider=OpenAIProvider(http_client=_retrying_http_client()),
        settings=OpenAIChatModelSettings(),
    )


def _build_gemini(
    model_id: str, *, budget: int | None, has_reasoning: bool
) -> GoogleModel:
    settings_kwargs: dict[str, Any] = {
        "temperature": 1.0 if has_reasoning else 0.0,
    }
    thinking = _gemini_thinking(budget)
    if thinking is not None:
        settings_kwargs["google_thinking_config"] = thinking
        # Gemini's default ``max_output_tokens`` (8192) is the same size
        # as our thinking_budget; if the model spends its full budget on
        # reasoning, no tokens remain for the actual response and the
        # SDK raises ``Model token limit exceeded before any response
        # was generated``. Add headroom for the response on top of
        # reasoning. (Anthropic's _build_anthropic already uses 16000.)
        settings_kwargs["max_tokens"] = 16000
    return GoogleModel(
        model_id,
        provider=GoogleProvider(
            vertexai=True,
            project=vertex_project_id(),
            location="global",
            http_client=_retrying_http_client(),
        ),
        settings=GoogleModelSettings(**settings_kwargs),
    )


def _build_deepseek_vertex(model_id: str) -> OpenAIChatModel:
    base_url = vertex_maas_base_url()
    http_client = _retrying_http_client(auth=VertexBearerAuth())
    provider = OpenAIProvider(
        base_url=base_url,
        api_key="placeholder-overridden-by-bearer-auth",
        http_client=http_client,
    )
    return OpenAIChatModel(model_id, provider=provider)


def build_model(*, model_id: str, agentic: bool):
    """Return a Pydantic AI ``Model`` configured for this model & path."""
    if model_id in ANTHROPIC_MODELS:
        budget = (
            AGENTIC_THINKING_BUDGETS if agentic else SINGLESHOT_THINKING_BUDGETS
        ).get(model_id)
        has_reasoning = (
            has_reasoning_agentic(model_id)
            if agentic
            else has_reasoning_singleshot(model_id)
        )
        return _build_anthropic(model_id, budget=budget, has_reasoning=has_reasoning)
    if model_id in OPENAI_MODELS:
        effort = (
            AGENTIC_OPENAI_REASONING_EFFORT
            if agentic
            else SINGLESHOT_OPENAI_REASONING_EFFORT
        ).get(model_id)
        return _build_openai(model_id, reasoning_effort=effort)
    if model_id in GEMINI_MODELS:
        budget = (
            AGENTIC_THINKING_BUDGETS if agentic else SINGLESHOT_THINKING_BUDGETS
        ).get(model_id)
        has_reasoning = (
            has_reasoning_agentic(model_id)
            if agentic
            else has_reasoning_singleshot(model_id)
        )
        return _build_gemini(model_id, budget=budget, has_reasoning=has_reasoning)
    if model_id in DEEPSEEK_VERTEX_MODELS:
        return _build_deepseek_vertex(model_id)
    raise ValueError(f"Unknown model: {model_id}")


def build_agent(
    *,
    model_id: str,
    agentic: bool,
    output_type,
    system_prompt: str,
    tools=None,
    deps_type=None,
) -> Agent:
    model = build_model(model_id=model_id, agentic=agentic)
    kwargs: dict[str, Any] = {
        "output_type": output_type,
        "system_prompt": system_prompt,
        # Bump retries for both tool-call validation (``retries``) and
        # output-type validation (``output_retries``) from the Pydantic
        # AI default of 1 to 3. The 1-retry default tripped on ~1% of
        # items in the 2026-05-03 sweep when models returned briefly
        # malformed args. Three retries catches those without slowing
        # the happy path.
        "retries": 3,
        "output_retries": 3,
    }
    if tools is not None:
        kwargs["tools"] = tools
    if deps_type is not None:
        kwargs["deps_type"] = deps_type
    return Agent(model, **kwargs)
