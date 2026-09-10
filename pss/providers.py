"""LLM provider abstraction for PSS."""

import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import wraps
from typing import Any

from dotenv import load_dotenv

from pss.types import TokenUsage

load_dotenv()

logger = logging.getLogger(__name__)

# Retryable exception types (checked by class name to avoid import dependency)
RETRYABLE_ERROR_NAMES = {
    "RateLimitError",
    "APIConnectionError",
    "InternalServerError",
    "APITimeoutError",
    "ServiceUnavailableError",
}


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is retryable based on its class name."""
    return type(exc).__name__ in RETRYABLE_ERROR_NAMES


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator that retries on transient API errors with exponential backoff."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    if not _is_retryable(e) or attempt == max_retries:
                        raise
                    last_exc = e
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"[PSS] API call failed ({type(e).__name__}), "
                        f"retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
            raise last_exc  # unreachable, but satisfies type checker
        return wrapper
    return decorator


def retry_with_backoff_async(max_retries: int = 3, base_delay: float = 1.0):
    """Async decorator that retries on transient API errors with exponential backoff."""
    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return await fn(*args, **kwargs)
                except Exception as e:
                    if not _is_retryable(e) or attempt == max_retries:
                        raise
                    last_exc = e
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"[PSS] API call failed ({type(e).__name__}), "
                        f"retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
            raise last_exc  # unreachable, but satisfies type checker
        return wrapper
    return decorator


# Pricing per 1M tokens (USD) - updated 2024
MODEL_PRICING = {
    # Anthropic
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    "claude-opus-4-6": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    # OpenRouter - Llama
    "meta-llama/llama-3.1-8b-instruct": {"input": 0.055, "output": 0.055},
    "meta-llama/llama-3.1-70b-instruct": {"input": 0.35, "output": 0.40},
    # OpenRouter - Others
    "anthropic/claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

# Default pricing for unknown models
DEFAULT_PRICING = {"input": 1.00, "output": 5.00}


@dataclass
class StreamEvent:
    """Event from streaming response."""

    type: str  # "text", "tool_call", "done", "error"
    text: str | None = None  # For text events
    tool_call: dict | None = None  # For tool_call events
    usage: TokenUsage | None = None  # For done events


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD for token usage."""
    pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


class Provider(ABC):
    """Abstract base class for LLM providers."""

    model: str  # subclasses must set this

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> tuple[str, list[dict], int, TokenUsage]:
        """
        Send a chat completion request.

        Returns:
            Tuple of (text_response, tool_calls, tokens_used, usage)
            - text_response: The assistant's text reply
            - tool_calls: List of tool calls made by the model
            - tokens_used: Total tokens (input + output) used (legacy)
            - usage: Detailed TokenUsage with cost
        """
        ...

    async def chat_async(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> tuple[str, list[dict], int, TokenUsage]:
        """Async version of chat. Default implementation wraps sync version."""
        import asyncio

        return await asyncio.to_thread(self.chat, messages, tools, tool_choice)

    async def chat_stream_async(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Streaming version of chat. Yields StreamEvents as content arrives.

        Default implementation falls back to non-streaming and yields all at once.
        """
        text, tool_calls, tokens_used, usage = await self.chat_async(
            messages, tools, tool_choice
        )
        if text:
            yield StreamEvent(type="text", text=text)
        for tc in tool_calls:
            yield StreamEvent(type="tool_call", tool_call=tc)
        yield StreamEvent(type="done", usage=usage)


class AnthropicProvider(Provider):
    """Provider for Anthropic Claude models."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        import anthropic

        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.async_client = anthropic.AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        self.model = model

    @retry_with_backoff()
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> tuple[str, list[dict], int]:
        kwargs = self._build_kwargs(messages, tools, tool_choice)
        response = self.client.messages.create(**kwargs)
        return self._parse_response(response)

    @retry_with_backoff_async()
    async def chat_async(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> tuple[str, list[dict], int]:
        kwargs = self._build_kwargs(messages, tools, tool_choice)
        response = await self.async_client.messages.create(**kwargs)
        return self._parse_response(response)

    def _build_kwargs(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        tool_choice: str | dict | None,
    ) -> dict[str, Any]:
        anthropic_tools = None
        if tools:
            anthropic_tools = [self._convert_tool(t) for t in tools]

        anthropic_tool_choice = None
        if tool_choice == "required" and tools:
            anthropic_tool_choice = {"type": "tool", "name": tools[0]["name"]}
        elif tool_choice == "auto":
            anthropic_tool_choice = {"type": "auto"}
        elif tool_choice == "none":
            anthropic_tool_choice = {"type": "none"}
        elif isinstance(tool_choice, dict):
            anthropic_tool_choice = tool_choice

        # Extract system messages - Anthropic requires them as top-level parameter
        system_content = None
        filtered_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                # Combine multiple system messages if any
                if system_content is None:
                    system_content = msg.get("content", "")
                else:
                    system_content += "\n\n" + msg.get("content", "")
            else:
                filtered_messages.append(msg)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 8192,
            "messages": filtered_messages,
        }
        if system_content:
            kwargs["system"] = system_content
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
        if anthropic_tool_choice:
            kwargs["tool_choice"] = anthropic_tool_choice
        return kwargs

    def _parse_response(self, response) -> tuple[str, list[dict], int, TokenUsage]:
        text = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "arguments": block.input,
                    }
                )

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        tokens_used = input_tokens + output_tokens
        cost = calculate_cost(self.model, input_tokens, output_tokens)
        usage = TokenUsage(
            input_tokens=input_tokens, output_tokens=output_tokens, cost=cost
        )
        return text, tool_calls, tokens_used, usage

    async def chat_stream_async(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream response from Anthropic, yielding text chunks as they arrive."""
        kwargs = self._build_kwargs(messages, tools, tool_choice)

        # Collect tool calls as they come in
        current_tool_calls: dict[int, dict] = {}  # index -> partial tool call
        input_tokens = 0
        output_tokens = 0

        async with self.async_client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "message_start":
                    if hasattr(event.message, "usage"):
                        input_tokens = event.message.usage.input_tokens
                elif event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        current_tool_calls[event.index] = {
                            "id": event.content_block.id,
                            "name": event.content_block.name,
                            "arguments": "",
                        }
                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        yield StreamEvent(type="text", text=event.delta.text)
                    elif event.delta.type == "input_json_delta":
                        # Accumulate tool call arguments
                        if event.index in current_tool_calls:
                            current_tool_calls[event.index][
                                "arguments"
                            ] += event.delta.partial_json
                elif event.type == "content_block_stop":
                    # Emit completed tool call
                    if event.index in current_tool_calls:
                        tc = current_tool_calls[event.index]
                        import json

                        try:
                            tc["arguments"] = json.loads(tc["arguments"])
                        except json.JSONDecodeError:
                            tc["arguments"] = {}
                        yield StreamEvent(type="tool_call", tool_call=tc)
                elif event.type == "message_delta":
                    if hasattr(event.usage, "output_tokens"):
                        output_tokens = event.usage.output_tokens

        # Final done event with usage
        cost = calculate_cost(self.model, input_tokens, output_tokens)
        usage = TokenUsage(
            input_tokens=input_tokens, output_tokens=output_tokens, cost=cost
        )
        yield StreamEvent(type="done", usage=usage)

    def _convert_tool(self, tool: dict) -> dict:
        """Convert OpenAI-style tool to Anthropic format."""
        return {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "input_schema": tool.get("input_schema", tool.get("parameters", {})),
        }


class OpenRouterProvider(Provider):
    """Provider for OpenRouter (OpenAI-compatible API)."""

    def __init__(self, model: str = "meta-llama/llama-3.1-8b-instruct"):
        from openai import AsyncOpenAI, OpenAI

        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
        self.async_client = AsyncOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
        self.model = model

    @retry_with_backoff()
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> tuple[str, list[dict], int]:
        kwargs = self._build_kwargs(messages, tools, tool_choice)
        response = self.client.chat.completions.create(**kwargs)
        return self._parse_response(response)

    @retry_with_backoff_async()
    async def chat_async(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> tuple[str, list[dict], int]:
        kwargs = self._build_kwargs(messages, tools, tool_choice)
        response = await self.async_client.chat.completions.create(**kwargs)
        return self._parse_response(response)

    def _build_kwargs(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        tool_choice: str | dict | None,
    ) -> dict[str, Any]:
        openai_tools = None
        if tools:
            openai_tools = [self._convert_tool(t) for t in tools]

        openai_tool_choice = None
        if tool_choice == "required" and tools:
            openai_tool_choice = {
                "type": "function",
                "function": {"name": tools[0]["name"]},
            }
        elif tool_choice in ("auto", "none"):
            openai_tool_choice = tool_choice
        elif isinstance(tool_choice, dict):
            openai_tool_choice = tool_choice

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if openai_tools:
            kwargs["tools"] = openai_tools
        if openai_tool_choice:
            kwargs["tool_choice"] = openai_tool_choice
        return kwargs

    def _parse_response(self, response) -> tuple[str, list[dict], int, TokenUsage]:
        import json

        message = response.choices[0].message
        text = message.content or ""
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments),
                    }
                )

        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        tokens_used = input_tokens + output_tokens
        cost = calculate_cost(self.model, input_tokens, output_tokens)
        usage = TokenUsage(
            input_tokens=input_tokens, output_tokens=output_tokens, cost=cost
        )
        return text, tool_calls, tokens_used, usage

    async def chat_stream_async(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream response from OpenRouter, yielding text chunks as they arrive."""
        import json

        kwargs = self._build_kwargs(messages, tools, tool_choice)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        # Collect tool calls as they come in
        current_tool_calls: dict[int, dict] = {}  # index -> partial tool call
        input_tokens = 0
        output_tokens = 0

        stream = await self.async_client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta

                # Text content
                if delta.content:
                    yield StreamEvent(type="text", text=delta.content)

                # Tool calls
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in current_tool_calls:
                            current_tool_calls[idx] = {
                                "id": tc_delta.id or "",
                                "name": "",
                                "arguments": "",
                            }
                        if tc_delta.id:
                            current_tool_calls[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                current_tool_calls[idx]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                current_tool_calls[idx][
                                    "arguments"
                                ] += tc_delta.function.arguments

            # Usage info (comes at the end)
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens

        # Emit completed tool calls
        for tc in current_tool_calls.values():
            try:
                tc["arguments"] = json.loads(tc["arguments"])
            except json.JSONDecodeError:
                tc["arguments"] = {}
            yield StreamEvent(type="tool_call", tool_call=tc)

        # Final done event with usage
        cost = calculate_cost(self.model, input_tokens, output_tokens)
        usage = TokenUsage(
            input_tokens=input_tokens, output_tokens=output_tokens, cost=cost
        )
        yield StreamEvent(type="done", usage=usage)

    def _convert_tool(self, tool: dict) -> dict:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", tool.get("parameters", {})),
            },
        }


def create_provider(provider_type: str, model: str) -> Provider:
    """Factory function to create a provider."""
    if provider_type == "anthropic":
        return AnthropicProvider(model=model)
    elif provider_type == "openrouter":
        return OpenRouterProvider(model=model)
    else:
        raise ValueError(f"Unknown provider: {provider_type}")
