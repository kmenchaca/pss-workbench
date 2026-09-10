"""Exploration loops and gate detection for PSS.

This module contains the core exploration logic that runs until
a token gate is hit.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any, Literal

from pss.adaptive_gates import (
    AdaptiveGateConfig,
    GateTimingState,
    should_trigger_gate,
)
from pss.lazy_branching import LazyBranchingConfig, check_early_convergence
from pss.providers import Provider, StreamEvent
from pss.types import Context, GateResult, TokenUsage, ToolTrace
from pss.velocity import VelocityTracker

# Type for streaming callback: (context_id, text_chunk) -> None
StreamCallback = Callable[[str, str], None]


def _format_tool_calls_for_openai(tool_calls: list[dict]) -> list[dict]:
    """Convert normalized tool calls to OpenAI message format."""
    return [
        {
            "id": tc["id"],
            "type": "function",
            "function": {
                "name": tc["name"],
                "arguments": json.dumps(tc["arguments"]) if isinstance(tc["arguments"], dict) else tc["arguments"],
            },
        }
        for tc in tool_calls
    ]


def _format_tool_results_for_openai(tool_results: list[dict]) -> list[dict]:
    """Convert tool results to OpenAI message format (separate messages)."""
    return [
        {
            "role": "tool",
            "tool_call_id": tr["tool_use_id"],
            "content": tr["content"],
        }
        for tr in tool_results
    ]


# Continuation prompts - varied to avoid pattern detection
CONTINUATION_PROMPTS = [
    "Continue your analysis. What else should you consider?",
    "Keep exploring. Are there other angles to examine?",
    "What other factors might be relevant here?",
    "Continue. What haven't you considered yet?",
    "Go deeper. What are the implications?",
]


def determine_gate_type(
    ctx: Context,
    new_total: int,
    soft_gate: int,
    hard_gate: int,
) -> str:
    """
    Determine what type of gate (if any) was hit.

    Gates fire at regular intervals:
    - First gate (surprise): at soft_gate tokens
    - Subsequent gates: every soft_gate tokens thereafter
    - Hard gate: at hard_gate tokens (forces decision)

    Args:
        ctx: The context being evaluated
        new_total: Total token count after latest response
        soft_gate: Soft gate threshold
        hard_gate: Hard gate threshold

    Returns:
        "hard", "soft", or "none"
    """
    if new_total >= hard_gate:
        return "hard"

    # Calculate the threshold for the next soft gate
    # gates_seen=0: threshold = soft_gate (first gate, surprise)
    # gates_seen=1: threshold = soft_gate * 2
    # gates_seen=2: threshold = soft_gate * 3
    # etc.
    next_soft_threshold = soft_gate * (ctx.gates_seen + 1)

    if new_total >= next_soft_threshold:
        return "soft"

    return "none"


def determine_gate_type_adaptive(
    ctx: Context,
    velocity_tracker: VelocityTracker,
    timing_state: GateTimingState,
    config: AdaptiveGateConfig,
    new_text: str,
    prev_window_text: str,
    tokens_in_chunk: int,
    total_tokens: int,
) -> tuple[str, VelocityTracker, GateTimingState]:
    """
    Determine gate type using adaptive velocity-based timing.

    This measures velocity from the new text and uses it to decide
    whether to fire a gate earlier, later, or at the normal time.

    Args:
        ctx: The context being evaluated
        velocity_tracker: Tracker with velocity history
        timing_state: State tracking for this context
        config: Adaptive gate configuration
        new_text: The latest response text
        prev_window_text: Previous window text for comparison
        tokens_in_chunk: Tokens in the new response
        total_tokens: Total tokens used so far (ctx.token_count + new tokens)

    Returns:
        Tuple of (gate_type, velocity_tracker, timing_state)
    """
    # Measure velocity if enough tokens have passed
    if velocity_tracker.should_measure(tokens_in_chunk):
        snapshot = velocity_tracker.measure(new_text, tokens_in_chunk, prev_window_text)
        ctx.velocity_history.append(snapshot)

    # Use adaptive gate logic with explicit token count
    state, gate_type = should_trigger_gate(
        ctx, velocity_tracker, timing_state, config, token_count=total_tokens
    )

    return (gate_type, velocity_tracker, timing_state)


def run_until_gate(
    ctx: Context,
    provider: Provider,
    soft_gate: int,
    hard_gate: int,
    tools: list[dict[str, Any]] | None = None,
    tool_executor: Callable[[str, dict[str, Any]], tuple[str, bool]] | None = None,
    adaptive_config: AdaptiveGateConfig | None = None,
    capture_velocity: bool = True,
    lazy_branching_config: LazyBranchingConfig | None = None,
) -> GateResult:
    """
    Run a context until it hits a token gate (sync version).

    Loops making API calls until the cumulative token count crosses
    a gate threshold. Uses continuation prompts to keep exploration going.

    Args:
        ctx: The context to run
        provider: LLM provider
        soft_gate: Soft gate token threshold (used if adaptive_config is None)
        hard_gate: Hard gate token threshold (used if adaptive_config is None)
        tools: Optional list of tool definitions for agentic exploration
        tool_executor: Optional function to execute tools. Takes (tool_name, arguments)
                       and returns (result_string, success_bool)
        adaptive_config: Optional config for adaptive velocity-based gating
        capture_velocity: Whether to capture velocity metrics (default True for tracing)
        lazy_branching_config: Optional config for lazy convergence detection

    Returns:
        GateResult with gate type and token usage
    """
    total_tokens_this_run = 0
    total_usage = TokenUsage()
    last_text = ""
    last_tool_calls: list[dict] = []
    prompt_idx = 0
    checked_lazy_convergence = False  # Only check once, on first response

    # Always create velocity tracker for data capture; adaptive timing is optional
    velocity_tracker = VelocityTracker() if (capture_velocity or adaptive_config) else None
    timing_state = GateTimingState() if adaptive_config else None
    prev_window_text = ""

    while True:
        # Pass tools if agentic mode enabled
        text, tool_calls, tokens_used, usage = provider.chat(
            ctx.messages,
            tools=tools,
            tool_choice="auto" if tools else None,
        )
        total_tokens_this_run += tokens_used
        total_usage = total_usage + usage

        # Handle tool calls if any
        if tool_calls and tool_executor:
            # Append the assistant message with tool calls
            # Use OpenAI format for OpenRouter compatibility
            ctx.messages.append({
                "role": "assistant",
                "content": text or "",
                "tool_calls": _format_tool_calls_for_openai(tool_calls),
            })

            # Execute each tool and collect results
            for tc in tool_calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("arguments", {})
                tool_id = tc.get("id", "")

                result_str, success = tool_executor(tool_name, tool_args)

                # Record in tool traces
                ctx.tool_traces.append(ToolTrace(
                    tool_name=tool_name,
                    arguments=tool_args,
                    result=result_str[:5000],  # Truncate for storage
                    success=success,
                    timestamp=time.time(),
                    tokens_used=len(result_str) // 4,  # Rough token estimate
                ))

                # Append tool result as separate message (OpenAI format)
                ctx.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result_str,
                })

            # Check gate even during tool calling to prevent runaway usage
            new_total = ctx.token_count + total_tokens_this_run

            if adaptive_config and velocity_tracker and timing_state:
                gate_type, velocity_tracker, timing_state = determine_gate_type_adaptive(
                    ctx, velocity_tracker, timing_state, adaptive_config,
                    text or "", prev_window_text, tokens_used, new_total
                )
                prev_window_text = text or ""
            else:
                # v1.1: Always capture velocity for tracing, even without adaptive gates
                if velocity_tracker and text and velocity_tracker.should_measure(tokens_used):
                    snapshot = velocity_tracker.measure(text or "", tokens_used, prev_window_text)
                    ctx.velocity_history.append(snapshot)
                    prev_window_text = text or ""
                gate_type = determine_gate_type(ctx, new_total, soft_gate, hard_gate)

            if gate_type != "none":
                # Hit a gate during tool use - return to inject checkpoint
                return GateResult(
                    gate_type=gate_type,
                    tokens_used=total_tokens_this_run,
                    response_text=text or f"[Used {len(tool_calls)} tool(s)]",
                    tool_calls=tool_calls,
                    usage=total_usage,
                )

            # Continue the loop - model will process tool results
            continue

        # No tool calls - regular text response
        ctx.messages.append({"role": "assistant", "content": text})
        last_text = text
        last_tool_calls = tool_calls

        new_total = ctx.token_count + total_tokens_this_run

        # Update velocity tracking (needed for both adaptive gates and lazy branching)
        if adaptive_config and velocity_tracker and timing_state:
            gate_type, velocity_tracker, timing_state = determine_gate_type_adaptive(
                ctx, velocity_tracker, timing_state, adaptive_config,
                text, prev_window_text, tokens_used, new_total
            )
            prev_window_text = text
        else:
            # v1.1: Always capture velocity for tracing, even without adaptive gates
            if velocity_tracker and text and velocity_tracker.should_measure(tokens_used):
                snapshot = velocity_tracker.measure(text, tokens_used, prev_window_text)
                ctx.velocity_history.append(snapshot)
                prev_window_text = text
            gate_type = determine_gate_type(ctx, new_total, soft_gate, hard_gate)

        # v1.3: Lazy branching - check for early convergence BEFORE gate check
        # This runs only on the first response, before any continuation prompts
        if (lazy_branching_config
            and not checked_lazy_convergence
            and ctx.id == "root"
            and last_text
            and prompt_idx == 0):  # First response only
            checked_lazy_convergence = True

            # Get velocity if available
            current_velocity = None
            if ctx.velocity_history:
                current_velocity = ctx.velocity_history[-1].overall_velocity

            is_converged = check_early_convergence(
                last_text, new_total, current_velocity, lazy_branching_config
            )

            if is_converged:
                print(f"[PSS] Lazy convergence detected in first response ({new_total} tokens)")
                return GateResult(
                    gate_type="lazy_converged",
                    tokens_used=total_tokens_this_run,
                    response_text=last_text,
                    tool_calls=[],
                    usage=total_usage,
                )

        if gate_type != "none":
            # Hit a gate - return so the outer code can inject checkpoint
            return GateResult(
                gate_type=gate_type,
                tokens_used=total_tokens_this_run,
                response_text=last_text,
                tool_calls=last_tool_calls,
                usage=total_usage,
            )

        # Haven't hit gate yet - prompt to continue exploring
        # The model doesn't know about gates (Heisenberg trick)
        continuation = CONTINUATION_PROMPTS[prompt_idx % len(CONTINUATION_PROMPTS)]
        ctx.messages.append({"role": "user", "content": continuation})
        prompt_idx += 1

        # Safety valve: don't loop forever if something is wrong
        if prompt_idx > 50:
            print(f"[PSS] Warning: Context {ctx.id} hit continuation limit")
            return GateResult(
                gate_type="hard",  # Force a decision
                tokens_used=total_tokens_this_run,
                response_text=last_text,
                tool_calls=last_tool_calls,
                usage=total_usage,
            )


async def run_until_gate_async(
    ctx: Context,
    provider: Provider,
    soft_gate: int,
    hard_gate: int,
    tools: list[dict[str, Any]] | None = None,
    tool_executor: Callable[[str, dict[str, Any]], tuple[str, bool]] | None = None,
    adaptive_config: AdaptiveGateConfig | None = None,
    capture_velocity: bool = True,
) -> GateResult:
    """
    Run a context until it hits a token gate (async version).

    Loops making API calls until the cumulative token count crosses
    a gate threshold. Uses continuation prompts to keep exploration going.

    Args:
        ctx: The context to run
        provider: LLM provider (must support chat_async)
        soft_gate: Soft gate token threshold (used if adaptive_config is None)
        hard_gate: Hard gate token threshold (used if adaptive_config is None)
        tools: Optional list of tool definitions for agentic exploration
        tool_executor: Optional function to execute tools. Takes (tool_name, arguments)
                       and returns (result_string, success_bool)
        adaptive_config: Optional config for adaptive velocity-based gating
        capture_velocity: Whether to capture velocity metrics (default True for tracing)

    Returns:
        GateResult with gate type and token usage
    """
    total_tokens_this_run = 0
    total_usage = TokenUsage()
    last_text = ""
    last_tool_calls: list[dict] = []
    prompt_idx = 0

    # Always create velocity tracker for data capture; adaptive timing is optional
    velocity_tracker = VelocityTracker() if (capture_velocity or adaptive_config) else None
    timing_state = GateTimingState() if adaptive_config else None
    prev_window_text = ""

    while True:
        # Pass tools if agentic mode enabled
        text, tool_calls, tokens_used, usage = await provider.chat_async(
            ctx.messages,
            tools=tools,
            tool_choice="auto" if tools else None,
        )
        total_tokens_this_run += tokens_used
        total_usage = total_usage + usage

        # Handle tool calls if any
        if tool_calls and tool_executor:
            # Append the assistant message with tool calls
            # Use OpenAI format for OpenRouter compatibility
            ctx.messages.append({
                "role": "assistant",
                "content": text or "",
                "tool_calls": _format_tool_calls_for_openai(tool_calls),
            })

            # Execute each tool and collect results
            for tc in tool_calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("arguments", {})
                tool_id = tc.get("id", "")

                result_str, success = tool_executor(tool_name, tool_args)

                # Record in tool traces
                ctx.tool_traces.append(ToolTrace(
                    tool_name=tool_name,
                    arguments=tool_args,
                    result=result_str[:5000],  # Truncate for storage
                    success=success,
                    timestamp=time.time(),
                    tokens_used=len(result_str) // 4,  # Rough token estimate
                ))

                # Append tool result as separate message (OpenAI format)
                ctx.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result_str,
                })

            # Check gate even during tool calling to prevent runaway usage
            new_total = ctx.token_count + total_tokens_this_run

            if adaptive_config and velocity_tracker and timing_state:
                gate_type, velocity_tracker, timing_state = determine_gate_type_adaptive(
                    ctx, velocity_tracker, timing_state, adaptive_config,
                    text or "", prev_window_text, tokens_used, new_total
                )
                prev_window_text = text or ""
            else:
                # v1.1: Always capture velocity for tracing, even without adaptive gates
                if velocity_tracker and text and velocity_tracker.should_measure(tokens_used):
                    snapshot = velocity_tracker.measure(text or "", tokens_used, prev_window_text)
                    ctx.velocity_history.append(snapshot)
                    prev_window_text = text or ""
                gate_type = determine_gate_type(ctx, new_total, soft_gate, hard_gate)

            if gate_type != "none":
                # Hit a gate during tool use - return to inject checkpoint
                return GateResult(
                    gate_type=gate_type,
                    tokens_used=total_tokens_this_run,
                    response_text=text or f"[Used {len(tool_calls)} tool(s)]",
                    tool_calls=tool_calls,
                    usage=total_usage,
                )

            # Continue the loop - model will process tool results
            continue

        # No tool calls - regular text response
        ctx.messages.append({"role": "assistant", "content": text})
        last_text = text
        last_tool_calls = tool_calls

        new_total = ctx.token_count + total_tokens_this_run

        if adaptive_config and velocity_tracker and timing_state:
            gate_type, velocity_tracker, timing_state = determine_gate_type_adaptive(
                ctx, velocity_tracker, timing_state, adaptive_config,
                text, prev_window_text, tokens_used, new_total
            )
            prev_window_text = text
        else:
            # v1.1: Always capture velocity for tracing, even without adaptive gates
            if velocity_tracker and text and velocity_tracker.should_measure(tokens_used):
                snapshot = velocity_tracker.measure(text, tokens_used, prev_window_text)
                ctx.velocity_history.append(snapshot)
                prev_window_text = text
            gate_type = determine_gate_type(ctx, new_total, soft_gate, hard_gate)

        if gate_type != "none":
            # Hit a gate - return so the outer code can inject checkpoint
            return GateResult(
                gate_type=gate_type,
                tokens_used=total_tokens_this_run,
                response_text=last_text,
                tool_calls=last_tool_calls,
                usage=total_usage,
            )

        # Haven't hit gate yet - prompt to continue exploring
        # The model doesn't know about gates (Heisenberg trick)
        continuation = CONTINUATION_PROMPTS[prompt_idx % len(CONTINUATION_PROMPTS)]
        ctx.messages.append({"role": "user", "content": continuation})
        prompt_idx += 1

        # Safety valve: don't loop forever if something is wrong
        if prompt_idx > 50:
            print(f"[PSS] Warning: Context {ctx.id} hit continuation limit")
            return GateResult(
                gate_type="hard",  # Force a decision
                tokens_used=total_tokens_this_run,
                response_text=last_text,
                tool_calls=last_tool_calls,
                usage=total_usage,
            )


async def run_until_gate_streaming(
    ctx: Context,
    provider: Provider,
    soft_gate: int,
    hard_gate: int,
    on_token: StreamCallback | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_executor: Callable[[str, dict[str, Any]], tuple[str, bool]] | None = None,
    adaptive_config: AdaptiveGateConfig | None = None,
    capture_velocity: bool = True,
) -> GateResult:
    """
    Run a context until it hits a token gate (async streaming version).

    Like run_until_gate_async but streams tokens via callback for real-time display.

    Args:
        ctx: The context to run
        provider: LLM provider (must support chat_stream_async)
        soft_gate: Soft gate token threshold (used if adaptive_config is None)
        hard_gate: Hard gate token threshold (used if adaptive_config is None)
        on_token: Callback for each text token: (ctx_id, text) -> None
        tools: Optional list of tool definitions for agentic exploration
        tool_executor: Optional function to execute tools
        adaptive_config: Optional config for adaptive velocity-based gating
        capture_velocity: Whether to capture velocity metrics (default True for tracing)

    Returns:
        GateResult with gate type and token usage
    """
    total_tokens_this_run = 0
    total_usage = TokenUsage()
    last_text = ""
    last_tool_calls: list[dict] = []
    prompt_idx = 0

    # Always create velocity tracker for data capture; adaptive timing is optional
    velocity_tracker = VelocityTracker() if (capture_velocity or adaptive_config) else None
    timing_state = GateTimingState() if adaptive_config else None
    prev_window_text = ""

    while True:
        # Use streaming API
        accumulated_text = ""
        tool_calls: list[dict] = []
        usage = TokenUsage()

        async for event in provider.chat_stream_async(
            ctx.messages,
            tools=tools,
            tool_choice="auto" if tools else None,
        ):
            if event.type == "text" and event.text:
                accumulated_text += event.text
                # Stream token to callback
                if on_token:
                    on_token(ctx.id, event.text)
            elif event.type == "tool_call" and event.tool_call:
                tool_calls.append(event.tool_call)
            elif event.type == "done" and event.usage:
                usage = event.usage

        tokens_used = usage.input_tokens + usage.output_tokens
        total_tokens_this_run += tokens_used
        total_usage = total_usage + usage

        # Handle tool calls if any
        if tool_calls and tool_executor:
            # Append the assistant message with tool calls
            ctx.messages.append({
                "role": "assistant",
                "content": accumulated_text or "",
                "tool_calls": _format_tool_calls_for_openai(tool_calls),
            })

            # Execute each tool and collect results
            for tc in tool_calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("arguments", {})
                tool_id = tc.get("id", "")

                result_str, success = tool_executor(tool_name, tool_args)

                # Record in tool traces
                ctx.tool_traces.append(ToolTrace(
                    tool_name=tool_name,
                    arguments=tool_args,
                    result=result_str[:5000],
                    success=success,
                    timestamp=time.time(),
                    tokens_used=len(result_str) // 4,
                ))

                # Append tool result
                ctx.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result_str,
                })

            # Check gate during tool calling
            new_total = ctx.token_count + total_tokens_this_run

            if adaptive_config and velocity_tracker and timing_state:
                gate_type, velocity_tracker, timing_state = determine_gate_type_adaptive(
                    ctx, velocity_tracker, timing_state, adaptive_config,
                    accumulated_text or "", prev_window_text, tokens_used, new_total
                )
                prev_window_text = accumulated_text or ""
            else:
                # v1.1: Always capture velocity for tracing, even without adaptive gates
                if velocity_tracker and accumulated_text and velocity_tracker.should_measure(tokens_used):
                    snapshot = velocity_tracker.measure(accumulated_text or "", tokens_used, prev_window_text)
                    ctx.velocity_history.append(snapshot)
                    prev_window_text = accumulated_text or ""
                gate_type = determine_gate_type(ctx, new_total, soft_gate, hard_gate)

            if gate_type != "none":
                return GateResult(
                    gate_type=gate_type,
                    tokens_used=total_tokens_this_run,
                    response_text=accumulated_text or f"[Used {len(tool_calls)} tool(s)]",
                    tool_calls=tool_calls,
                    usage=total_usage,
                )
            continue

        # No tool calls - regular text response
        ctx.messages.append({"role": "assistant", "content": accumulated_text})
        last_text = accumulated_text
        last_tool_calls = tool_calls

        new_total = ctx.token_count + total_tokens_this_run

        if adaptive_config and velocity_tracker and timing_state:
            gate_type, velocity_tracker, timing_state = determine_gate_type_adaptive(
                ctx, velocity_tracker, timing_state, adaptive_config,
                accumulated_text, prev_window_text, tokens_used, new_total
            )
            prev_window_text = accumulated_text
        else:
            # v1.1: Always capture velocity for tracing, even without adaptive gates
            if velocity_tracker and accumulated_text and velocity_tracker.should_measure(tokens_used):
                snapshot = velocity_tracker.measure(accumulated_text, tokens_used, prev_window_text)
                ctx.velocity_history.append(snapshot)
                prev_window_text = accumulated_text
            gate_type = determine_gate_type(ctx, new_total, soft_gate, hard_gate)

        if gate_type != "none":
            return GateResult(
                gate_type=gate_type,
                tokens_used=total_tokens_this_run,
                response_text=last_text,
                tool_calls=last_tool_calls,
                usage=total_usage,
            )

        # Prompt to continue exploring
        continuation = CONTINUATION_PROMPTS[prompt_idx % len(CONTINUATION_PROMPTS)]
        ctx.messages.append({"role": "user", "content": continuation})
        prompt_idx += 1

        if prompt_idx > 50:
            print(f"[PSS] Warning: Context {ctx.id} hit continuation limit")
            return GateResult(
                gate_type="hard",
                tokens_used=total_tokens_this_run,
                response_text=last_text,
                tool_calls=last_tool_calls,
                usage=total_usage,
            )
