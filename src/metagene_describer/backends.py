"""LLM backends. Each one turns (system, user, JSON schema) into a parsed JSON object, optionally
letting the model call client-side tools first."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-5-5"


@dataclass
class Tool:
    """A client-side tool: run(input) is called with the model's arguments and returns text."""

    name: str
    description: str
    input_schema: dict
    run: Callable[[dict], str]


class Backend(Protocol):
    def complete_json(
        self, system: str, user: str, schema: dict, tools: Sequence[Tool] = ()
    ) -> dict: ...


class BackendError(RuntimeError):
    """The model returned no usable response: a refusal, truncation, or unparseable JSON."""


def _run_tool(tools: Sequence[Tool], name: str, arguments: dict) -> tuple[str, bool]:
    """Run a tool call, returning (result text, is_error)."""
    tool = next((t for t in tools if t.name == name), None)
    if tool is None:
        return f"unknown tool {name!r}", True
    try:
        return tool.run(arguments), False
    except Exception as e:
        return f"{type(e).__name__}: {e}", True


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


def validate(data: dict, schema: dict) -> dict:
    """Check a parsed response has the schema's required fields and no placeholder values.

    Some models, offered tools while a JSON format is enforced, answer with stand-in values like
    "placeholder" instead of calling the tool.
    """
    if not isinstance(data, dict):
        raise BackendError("response is not a JSON object")
    missing = [k for k in schema.get("required", []) if k not in data]
    if missing:
        raise BackendError(f"response is missing {', '.join(missing)}")
    for k, v in data.items():
        if isinstance(v, str) and v.strip().lower().startswith("placeholder"):
            raise BackendError(f"response has a placeholder {k}: {v[:60]!r}")
    return data


def _schema_instruction(schema: dict) -> str:
    return (
        "\n\nRespond with only a JSON object matching this JSON schema, with no other text:\n"
        + json.dumps(schema, indent=2)
    )


class AnthropicBackend:
    """Claude through the official Anthropic SDK, with schema-constrained structured output.

    api_key and base_url default to the ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL environment
    variables (or an `ant auth login` profile). Extra keyword arguments are passed to
    messages.create on every request. With tools, the model may make up to max_tool_rounds
    rounds of tool calls before it must give its final answer.
    """

    def __init__(
        self,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        effort: str | None = None,
        max_tokens: int = 16000,
        max_tool_rounds: int = 3,
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
        self.max_tool_rounds = max_tool_rounds
        self.request_kwargs = request_kwargs

    def complete_json(
        self, system: str, user: str, schema: dict, tools: Sequence[Tool] = ()
    ) -> dict:
        output_config: dict = {"format": {"type": "json_schema", "schema": schema}}
        if self.effort is not None:
            output_config["effort"] = self.effort
        kwargs = dict(self.request_kwargs)
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ]
        messages: list[dict] = [{"role": "user", "content": user}]
        for round in range(self.max_tool_rounds + 1):
            if tools and round == self.max_tool_rounds:
                kwargs["tool_choice"] = {"type": "none"}  # out of rounds: answer now
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=messages,
                output_config=output_config,
                **kwargs,
            )
            if response.stop_reason != "tool_use":
                break
            # Append the assistant turn unchanged (it may carry thinking blocks), then the results.
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    text, is_error = _run_tool(tools, block.name, block.input)
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": text,
                            "is_error": is_error,
                        }
                    )
            messages.append({"role": "user", "content": results})
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise BackendError(f"model refused ({getattr(details, 'category', None)})")
        if response.stop_reason == "max_tokens":
            raise BackendError("response truncated at max_tokens")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise BackendError("response has no text block")
        return validate(parse_json(text), schema)


class OpenAIBackend:
    """Any OpenAI-compatible chat completions endpoint (OpenAI, OpenRouter, vLLM, Ollama, ...).

    api_key and base_url default to the OPENAI_API_KEY and OPENAI_BASE_URL environment variables.
    The schema is always described in the system prompt, since some servers accept
    response_format but don't enforce it. json_mode controls what is requested on top of that:
      "schema": response_format json_schema with strict=True (best, when supported)
      "object": response_format json_object
      "prompt": no response_format
    Extra keyword arguments (temperature, reasoning_effort, extra_body, ...) are passed to
    chat.completions.create on every request.

    With tools (which the server must support), the model may make up to max_tool_rounds rounds
    of tool calls. response_format isn't sent during those rounds, because with it some models
    answer with placeholders instead of calling the tool. If the model's answer after the tool
    rounds isn't valid JSON for the schema, one more request asks for it with tools off and
    response_format on.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        json_mode: str = "schema",
        max_tokens: int | None = None,
        max_tool_rounds: int = 3,
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
        self.max_tool_rounds = max_tool_rounds
        self.request_kwargs = request_kwargs

    def complete_json(
        self, system: str, user: str, schema: dict, tools: Sequence[Tool] = ()
    ) -> dict:
        kwargs = dict(self.request_kwargs)
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        format_kwargs = {}
        if self.json_mode == "schema":
            format_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "metagene_description", "schema": schema, "strict": True},
            }
        elif self.json_mode == "object":
            format_kwargs["response_format"] = {"type": "json_object"}
        messages: list[dict] = [
            {"role": "system", "content": system + _schema_instruction(schema)},
            {"role": "user", "content": user},
        ]
        if not tools:
            return validate(parse_json(self._content(messages, **kwargs, **format_kwargs)), schema)

        tool_kwargs = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in tools
            ]
        }
        for _ in range(self.max_tool_rounds):
            choice = self._create(messages, **kwargs, **tool_kwargs)
            calls = getattr(choice.message, "tool_calls", None)
            if not calls:
                break
            messages.append(choice.message.model_dump(exclude_none=True))
            for call in calls:
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    text = "arguments are not valid JSON"
                else:
                    text, _ = _run_tool(tools, call.function.name, arguments)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
        else:
            choice = None  # out of tool rounds without an answer
        if choice is not None and choice.message.content:
            try:
                return validate(parse_json(self._check(choice)), schema)
            except BackendError:
                messages.append({"role": "assistant", "content": choice.message.content})
        # Ask for the final answer with tools off and the format enforced.
        messages.append(
            {"role": "user", "content": "Give your final answer now, as only the JSON object."}
        )
        content = self._content(
            messages, **kwargs, **tool_kwargs, tool_choice="none", **format_kwargs
        )
        return validate(parse_json(content), schema)

    def _create(self, messages: list[dict], **kwargs):
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, **kwargs
        )
        return response.choices[0]

    def _check(self, choice) -> str:
        if getattr(choice.message, "refusal", None):
            raise BackendError(f"model refused: {choice.message.refusal}")
        if choice.finish_reason == "length":
            raise BackendError("response truncated at max_tokens")
        if not choice.message.content:
            raise BackendError("response is empty")
        return choice.message.content

    def _content(self, messages: list[dict], **kwargs) -> str:
        return self._check(self._create(messages, **kwargs))


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
