"""LLM backends. Each one turns (system, user, JSON schema) into a parsed JSON object."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-5-5"


class Backend(Protocol):
    def complete_json(self, system: str, user: str, schema: dict) -> dict: ...


class BackendError(RuntimeError):
    """The model returned no usable response: a refusal, truncation, or unparseable JSON."""


def parse_json(text: str) -> dict:
    """Parse a JSON object, tolerating code fences or prose around it."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise BackendError(f"response is not valid JSON: {text[:200]!r}")


def _schema_instruction(schema: dict) -> str:
    return (
        "\n\nRespond with only a JSON object matching this JSON schema, with no other text:\n"
        + json.dumps(schema, indent=2)
    )


class AnthropicBackend:
    """Claude through the official Anthropic SDK, with schema-constrained structured output.

    api_key and base_url default to the ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL environment
    variables (or an `ant auth login` profile). Extra keyword arguments are passed to
    messages.create on every request.
    """

    def __init__(
        self,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        effort: str | None = None,
        max_tokens: int = 16000,
        client: Any = None,
        **request_kwargs,
    ):
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.request_kwargs = request_kwargs

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        output_config: dict = {"format": {"type": "json_schema", "schema": schema}}
        if self.effort is not None:
            output_config["effort"] = self.effort
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config=output_config,
            **self.request_kwargs,
        )
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise BackendError(f"model refused ({getattr(details, 'category', None)})")
        if response.stop_reason == "max_tokens":
            raise BackendError("response truncated at max_tokens")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise BackendError("response has no text block")
        return parse_json(text)


class OpenAIBackend:
    """Any OpenAI-compatible chat completions endpoint (OpenAI, OpenRouter, vLLM, Ollama, ...).

    api_key and base_url default to the OPENAI_API_KEY and OPENAI_BASE_URL environment variables.
    json_mode controls how JSON output is requested, since servers differ in what they support:
      "schema": response_format json_schema with strict=True (best, when supported)
      "object": response_format json_object, with the schema described in the system prompt
      "prompt": no response_format; the schema is described in the system prompt only
    Extra keyword arguments (temperature, reasoning_effort, extra_body, ...) are passed to
    chat.completions.create on every request.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        json_mode: str = "schema",
        max_tokens: int | None = None,
        client: Any = None,
        **request_kwargs,
    ):
        if json_mode not in ("schema", "object", "prompt"):
            raise ValueError(f"json_mode must be 'schema', 'object' or 'prompt', not {json_mode!r}")
        if client is None:
            import openai

            client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.client = client
        self.model = model
        self.json_mode = json_mode
        self.max_tokens = max_tokens
        self.request_kwargs = request_kwargs

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        kwargs = dict(self.request_kwargs)
        if self.json_mode == "schema":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "metagene_description", "schema": schema, "strict": True},
            }
        else:
            system = system + _schema_instruction(schema)
            if self.json_mode == "object":
                kwargs["response_format"] = {"type": "json_object"}
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kwargs,
        )
        choice = response.choices[0]
        if getattr(choice.message, "refusal", None):
            raise BackendError(f"model refused: {choice.message.refusal}")
        if choice.finish_reason == "length":
            raise BackendError("response truncated at max_tokens")
        if not choice.message.content:
            raise BackendError("response is empty")
        return parse_json(choice.message.content)


def make_backend(
    provider: str = "anthropic",
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs,
) -> Backend:
    """Construct a backend by name: "anthropic" or "openai" (any OpenAI-compatible endpoint)."""
    if provider == "anthropic":
        return AnthropicBackend(
            model or DEFAULT_ANTHROPIC_MODEL, api_key=api_key, base_url=base_url, **kwargs
        )
    if provider == "openai":
        if model is None:
            raise ValueError("the openai backend needs a model name")
        return OpenAIBackend(model, api_key=api_key, base_url=base_url, **kwargs)
    raise ValueError(f"unknown provider {provider!r}; expected 'anthropic' or 'openai'")
