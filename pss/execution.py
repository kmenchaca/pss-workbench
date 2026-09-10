"""Async execution orchestration for PSS.

This module contains the parallel execution logic using asyncio.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Callable

from pss.config import PSSConfig
from pss.exploration import run_until_gate_async, run_until_gate_streaming
from pss.gates import (
    CHECKPOINT_DECISION_TOOL,
    create_gate_prompt,
    create_gate_retry_prompt,
    parse_decision_from_text,
    parse_gate_decision,
)
from pss.providers import Provider
from pss.tools import ToolExecutor, get_tool_definitions
from pss.types import Context, DiversityState, GateDecision, SearchTree, SpawnReason, TerminationReason

if TYPE_CHECKING:
    from pss.interactive import InteractiveController

# Default tools when agentic mode is enabled
DEFAULT_AGENTIC_TOOLS = ["read_file", "list_directory", "search_files"]


def _create_tool_executor(config: PSSConfig) -> tuple[list[dict[str, Any]] | None, Callable[[str, dict], tuple[str, bool]] | None]:
    """Create tool definitions and executor based on config."""
    if not config.agentic_enabled:
        return None, None

    tool_names = config.agentic_tools or DEFAULT_AGENTIC_TOOLS
    tool_definitions = get_tool_definitions(tool_names)

    executor = ToolExecutor(
        working_dir=config.agentic_working_dir,
        timeout=config.agentic_tool_timeout,
        read_only=config.parallel,  # Block writes in parallel mode
    )

    def execute_tool(tool_name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        result = executor.execute(tool_name, arguments)
        return executor.format_result_for_message(result), result.success

    return tool_definitions, execute_tool


async def run_pss_async(
    initial_prompt: str,
    config: PSSConfig,
    provider: Provider,
    eval_fn: Callable[[str], float] | None = None,
    on_context_update: Callable[[Context], None] | None = None,
    process_decision_fn: Callable = None,
    spawn_branch_fn: Callable = None,
    collect_leaves_fn: Callable = None,
    interactive_controller: "InteractiveController | None" = None,
    on_token: Callable[[str, str], None] | None = None,
) -> list[Context]:
    """
    Run Pepe Silvia Search with parallel context execution.

    Uses asyncio.Semaphore to limit concurrent API calls.

    Args:
        initial_prompt: The initial prompt to explore
        config: PSS configuration
        provider: LLM provider
        eval_fn: Optional evaluation function for early exit
        on_context_update: Optional callback for context updates
        process_decision_fn: Function to process gate decisions (injected from harness)
        spawn_branch_fn: Function to spawn branches (injected from harness)
        collect_leaves_fn: Function to collect leaves (injected from harness)
        interactive_controller: Optional controller for interactive mode
        on_token: Optional callback for streaming tokens: (ctx_id, text) -> None

    Returns:
        List of leaf contexts with outputs
    """
    tree = SearchTree()
    total_tokens = 0
    total_tokens_lock = asyncio.Lock()
    tree_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(config.max_parallel)
    early_exit = asyncio.Event()

    # v0.5: Initialize diversity state for real-time tracking
    diversity_state: DiversityState | None = None
    diversity_lock = asyncio.Lock()
    if config.diversity_aware_enabled:
        diversity_state = DiversityState()
        print("[PSS] Diversity-aware spawning enabled")

    # Build initial messages, optionally with system prompt
    initial_messages = []
    if config.system_prompt:
        initial_messages.append({"role": "system", "content": config.system_prompt})
    initial_messages.append({"role": "user", "content": initial_prompt})

    root = Context(
        id="root",
        parent_id=None,
        messages=initial_messages,
        token_budget=config.per_context_max,
    )
    tree.contexts["root"] = root
    tree.root_id = "root"

    # v0.4: Create tool executor for agentic mode
    tools, tool_executor = _create_tool_executor(config)
    if tools:
        print(f"[PSS] Agentic mode enabled with tools: {[t['name'] for t in tools]}")

    async def process_context(ctx: Context) -> None:
        nonlocal total_tokens

        async with semaphore:
            if early_exit.is_set():
                return

            # v0.8: Interactive mode checks
            if interactive_controller:
                if interactive_controller.is_quit_requested():
                    early_exit.set()
                    return
                if interactive_controller.should_kill(ctx.id):
                    ctx.status = "terminated"
                    ctx.termination_reason = TerminationReason.KILLED
                    print(f"[PSS] Context {ctx.id} killed by user")
                    return

            # Check budget
            async with total_tokens_lock:
                if total_tokens >= config.total_max:
                    return

            # v0.8: Check for injected messages
            if interactive_controller:
                injection = interactive_controller.get_injection(ctx.id)
                if injection:
                    ctx.messages.append({
                        "role": "user",
                        "content": f"\n---USER GUIDANCE---\n{injection}\n---END GUIDANCE---\n\nPlease take this guidance into account and continue.",
                    })
                    print(f"[PSS] Injected guidance into {ctx.id}")

            # Run until gate (use streaming if callback provided)
            if on_token:
                result = await run_until_gate_streaming(
                    ctx, provider, config.soft_gate_tokens, config.hard_gate_tokens,
                    on_token=on_token, tools=tools, tool_executor=tool_executor,
                )
            else:
                result = await run_until_gate_async(
                    ctx, provider, config.soft_gate_tokens, config.hard_gate_tokens,
                    tools=tools, tool_executor=tool_executor,
                )

            async with total_tokens_lock:
                total_tokens += result.tokens_used
            ctx.token_count += result.tokens_used
            ctx.usage = ctx.usage + result.usage

            if on_context_update:
                on_context_update(ctx)

            # Get gate decision (v0.5: pass diversity state)
            decision = await _run_gate_decision_async(
                ctx, provider, force=(result.gate_type == "hard"),
                diversity_state=diversity_state, config=config,
            )

            async with total_tokens_lock:
                total_tokens += 100

            if on_context_update:
                on_context_update(ctx)

            # v0.8: Handle promote - spawn extra branches like this context
            if interactive_controller and interactive_controller.should_promote(ctx.id):
                promote_edit = f"Continue exploring like context {ctx.id}, which was promoted"
                if ctx.branch_reason:
                    promote_edit = f"Explore similar to: {ctx.branch_reason}"
                async with tree_lock:
                    spawn_branch_fn(
                        tree,
                        ctx,
                        promote_edit,
                        config.per_context_max,
                        spawn_reason=SpawnReason.PROMOTION,
                        spawn_confidence=0.8,  # promoted branches get high confidence
                        provider=provider,
                        config=config,
                    )
                print(f"[PSS] Promoted {ctx.id} - spawned similar branch")

            # Process decision (v0.5: pass diversity state under lock)
            async with tree_lock:
                async with diversity_lock:
                    process_decision_fn(tree, ctx, decision, config, eval_fn, diversity_state, provider=provider)

            # v0.8: Notify interactive controller
            if interactive_controller:
                if decision.action == "terminate" and decision.output:
                    interactive_controller.on_leaf_added(ctx.id)
                elif decision.action == "branch":
                    for edit in decision.branches:
                        async with tree_lock:
                            for cid in tree.contexts:
                                if cid.startswith(f"{ctx.id}->") and tree.contexts[cid].branch_reason == edit:
                                    interactive_controller.on_branch_spawned(ctx.id, cid, edit)
                                    break

            print(
                f"[PSS] Context {ctx.id}: {decision.action} "
                f"(tokens: {ctx.token_count}, branches: {len(decision.branches)})"
            )

            # Check for early exit
            if decision.action == "terminate" and decision.output and eval_fn:
                score = eval_fn(decision.output)
                if score >= config.good_enough_threshold:
                    print(f"[PSS] Found good enough output (score={score:.2f})")
                    early_exit.set()

    # Main loop
    while not early_exit.is_set():
        # v0.8: Interactive mode - check for quit
        if interactive_controller and interactive_controller.is_quit_requested():
            print("[PSS] Quit requested by user")
            break

        # v0.8: Interactive mode - wait while paused (in thread-safe way)
        if interactive_controller and interactive_controller.is_paused():
            await asyncio.sleep(0.1)
            continue

        async with tree_lock:
            running = [c for c in tree.contexts.values() if c.status == "running"]
            # v0.8: Filter out killed contexts
            if interactive_controller:
                running = [c for c in running if not interactive_controller.should_kill(c.id)]

        if not running:
            break

        async with total_tokens_lock:
            if total_tokens >= config.total_max:
                print(f"[PSS] Hit total token budget ({config.total_max})")
                break

            # v1.5: Priority scheduling — sort by confidence, skip low-confidence when budget tight
            # NOTE: LLM self-reported confidence is miscalibrated. Low-confidence branches
            # may be exploring unfamiliar territory — exactly what PSS is for. This only
            # affects ordering (which contexts get the semaphore first) and budget-constrained
            # skipping. Monitor whether skipped branches would have produced high-diversity leaves.
            if config.priority_scheduling and running:
                budget_ratio = total_tokens / config.total_max if config.total_max > 0 else 0
                running.sort(
                    key=lambda c: (
                        -(c.spawn_metadata.spawn_confidence if c.spawn_metadata else 0.5),
                        c.id.count("->"),  # depth as tie-breaker (shallower first)
                    )
                )
                if budget_ratio >= config.priority_budget_threshold:
                    # Budget is tight — only run high-confidence branches.
                    # Use a low bar (0.3) rather than 0.5 to avoid killing divergent exploration.
                    before_count = len(running)
                    running = [
                        c for c in running
                        if (c.spawn_metadata and c.spawn_metadata.spawn_confidence >= 0.3)
                        or c.id == "root"
                    ]
                    if before_count > len(running):
                        print(f"[PSS] Priority: skipped {before_count - len(running)} low-confidence branches (budget {budget_ratio:.0%})")
                    if not running:
                        break

        # Process all running contexts in parallel
        tasks = [process_context(ctx) for ctx in running]
        await asyncio.gather(*tasks)

    return collect_leaves_fn(tree)


async def _run_gate_decision_async(
    ctx: Context,
    provider: Provider,
    force: bool = False,
    max_retries: int = 3,
    diversity_state: DiversityState | None = None,
    config: PSSConfig | None = None,
) -> GateDecision:
    """Inject a gate and get the context's decision (async version)."""
    gate_prompt = create_gate_prompt(
        ctx, force=force, diversity_state=diversity_state, config=config
    )
    ctx.messages.append({"role": "user", "content": gate_prompt})
    ctx.gates_seen += 1

    for attempt in range(max_retries):
        text, tool_calls, tokens_used, usage = await provider.chat_async(
            ctx.messages,
            tools=[CHECKPOINT_DECISION_TOOL],
            tool_choice="required",
        )
        ctx.usage = ctx.usage + usage  # Track gate decision tokens too

        decision = parse_gate_decision(tool_calls)
        if decision:
            # Enforce no-terminate rule for early gates (gates 1 and 2)
            # gates_seen is already incremented, so check <= 2
            if ctx.gates_seen <= 2 and decision.action == "terminate":
                print(f"[PSS] Context {ctx.id}: blocked early termination (gate {ctx.gates_seen}), forcing continue")
                decision = GateDecision(action="continue")
            if text:
                ctx.messages.append({"role": "assistant", "content": text})
            return decision

        decision = parse_decision_from_text(text)
        if decision:
            print(f"[PSS] Context {ctx.id}: parsed decision from text (fallback)")
            # Enforce no-terminate rule for early gates here too
            if ctx.gates_seen <= 2 and decision.action == "terminate":
                print(f"[PSS] Context {ctx.id}: blocked early termination (gate {ctx.gates_seen}), forcing continue")
                decision = GateDecision(action="continue")
            if text:
                ctx.messages.append({"role": "assistant", "content": text})
            return decision

        ctx.messages.append({"role": "assistant", "content": text})
        ctx.messages.append({"role": "user", "content": create_gate_retry_prompt()})

    print(f"[PSS] Context {ctx.id} failed to make gate decision, force terminating")
    return GateDecision(action="terminate", output=None, confidence=0.0)
