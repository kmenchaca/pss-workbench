"""Bounded, text-only OpenRouter adapter for the local workbench.

The general research CLI retains its original providers. This adapter deliberately
does not expose tools other than PSS checkpoints, embeddings, retries or endpoints.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import httpx

from pss.providers import Provider
from pss.types import TokenUsage
from pss.web.gpu_runtime import local_url


class RunLimitReached(RuntimeError):
    """An explicit run bound or user stop was reached."""


class ProviderFailure(RuntimeError):
    """Safe, user-facing provider failure without raw response/credential data."""


class WorkbenchProvider(Provider):
    MAX_CALLS = 18
    MAX_OUTPUT_TOKENS = 768
    LOCAL_OUTPUT_TOKENS = 192
    MAX_SECONDS = 300
    MAX_REQUEST_BYTES = 48_000

    def __init__(
        self,
        model: str,
        api_key: str,
        stop: threading.Event,
        transport: httpx.BaseTransport | None = None,
        backend: str = "openrouter",
    ):
        if backend not in ("openrouter", "ollama", "llamacpp"):
            raise ValueError("Unsupported workbench backend")
        self.backend = backend
        local = backend != "openrouter"
        self.model = model
        self.stop = stop
        self.started = time.monotonic()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.lock = threading.Lock()
        self.client = httpx.Client(
            base_url=(local_url() + "/v1") if backend == "llamacpp" else {"ollama": "http://127.0.0.1:11434/v1", "openrouter": "https://openrouter.ai/api/v1"}[backend],
            headers={} if backend == "ollama" or (backend == "llamacpp" and not api_key) else {"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(120.0 if local else 45.0, connect=10.0),
            trust_env=not local,
            follow_redirects=False,
            transport=transport,
        )

    def close(self):
        self.client.close()

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> tuple[str, list[dict], int, TokenUsage]:
        with self.lock:
            if self.stop.is_set():
                raise RunLimitReached("Stopped by you. No further requests were sent.")
            if time.monotonic() - self.started >= self.MAX_SECONDS:
                raise RunLimitReached("The five-minute request window ended.")
            if self.calls >= self.MAX_CALLS:
                raise RunLimitReached(
                    "The 18-request limit was reached. Your partial exploration is preserved."
                )
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.LOCAL_OUTPUT_TOKENS if self.backend == "ollama" else self.MAX_OUTPUT_TOKENS,
                "temperature": 0.7,
            }
            if tools:
                if any(t.get("name") != "checkpoint_decision" for t in tools):
                    raise ProviderFailure(
                        "This workbench does not allow filesystem or execution tools."
                    )
                payload["tools"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": t["name"],
                            "description": t.get("description", ""),
                            "parameters": t["input_schema"],
                        },
                    }
                    for t in tools
                ]
                payload["tool_choice"] = (
                    {"type": "function", "function": {"name": "checkpoint_decision"}}
                    if tool_choice == "required"
                    else "auto"
                )
            if self.backend in ("ollama", "llamacpp"):
                payload["reasoning_effort"] = "none"
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            if len(encoded) > self.MAX_REQUEST_BYTES:
                raise RunLimitReached("A branch reached the 48 KB request-size limit.")
            # Reserve before dispatch. Failed requests count, and are never retried.
            self.calls += 1
        try:
            response = self.client.post(
                "/chat/completions",
                content=encoded,
                headers={"Content-Type": "application/json"},
            )
        except httpx.TimeoutException as exc:
            raise ProviderFailure(
                "The model timed out. Nothing was retried; partial work is preserved."
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderFailure(
                "Could not reach the configured model. Check local setup and try a new run."
            ) from exc
        if response.status_code in (401, 403):
            raise ProviderFailure(
                "OpenRouter rejected the API key or model permission. Check your local setup."
            )
        if response.status_code == 402:
            raise ProviderFailure(
                "Your OpenRouter credit or spending limit was reached."
            )
        if response.status_code == 429:
            raise ProviderFailure(
                "OpenRouter is rate limiting requests. Wait before starting another run."
            )
        if response.status_code != 200:
            raise ProviderFailure(
                f"The provider returned HTTP {response.status_code}. No retry was made."
            )
        if len(response.content) > 1_000_000:
            raise ProviderFailure("The provider returned an oversized response.")
        try:
            data = response.json()
            message = data["choices"][0]["message"]
            text = message.get("content") or ""
            if not isinstance(text, str):
                raise TypeError("Non-text response")
            calls = []
            for item in message.get("tool_calls") or []:
                fn = item["function"]
                if fn["name"] != "checkpoint_decision":
                    raise ValueError("Unexpected tool")
                args = json.loads(fn["arguments"])
                if not isinstance(args, dict):
                    raise TypeError("Invalid arguments")
                calls.append({"id": item["id"], "name": fn["name"], "arguments": args})
            if not text.strip() and not calls:
                raise ProviderFailure(
                    "The model returned no usable text or checkpoint. It may have used "
                    "the output limit on internal reasoning. Choose a compatible model "
                    "with reasoning disabled; no retry was made."
                )
            raw_usage = data.get("usage") or {}
            usage = TokenUsage(
                input_tokens=max(0, int(raw_usage.get("prompt_tokens") or 0)),
                output_tokens=max(0, int(raw_usage.get("completion_tokens") or 0)),
                cost=0.0,  # Do not manufacture a dollar estimate from stale model prices.
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderFailure(
                "The model returned an unsupported response. Partial work is preserved."
            ) from exc
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        return text, calls, usage.total_tokens, usage
