"""Main PSS harness - manages the exploration tree."""

from __future__ import annotations

import asyncio
import copy
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from pss.adaptive_gates import AdaptiveGateConfig
from pss.config import PSSConfig
from pss.convergence import ConvergenceConfig, TaskType, detect_task_type, should_branch
from pss.exploration import determine_gate_type, run_until_gate
from pss.lazy_branching import (
    LazyBranchingConfig,
    BranchSignal,
    analyze_initial_response,
    check_early_convergence,
)
from pss.sibling_diversity import (
    check_sibling_diversity,
    format_diversity_rejection_log,
)
from pss.failures import (
    FailureRegistry,
    extract_failure_from_context,
    format_failures_for_injection,
)
from pss.signals import (
    BulletinBoard,
    extract_signals_from_response,
    format_signals_for_injection,
)
from pss.gates import (
    CHECKPOINT_DECISION_TOOL,
    create_gate_prompt,
    create_gate_retry_prompt,
    parse_decision_from_text,
    parse_gate_decision,
)
from pss.providers import Provider
from pss.synthesis import check_incremental_consensus, synthesize
from pss.tools import ToolExecutor, get_tool_definitions
from pss.types import (
    Context,
    DiversityState,
    GateDecision,
    PSSResult,
    SearchTree,
    SpawnMetadata,
    SpawnReason,
    TerminationReason,
    TokenUsage,
)

if TYPE_CHECKING:
    from pss.genealogy import GenealogyTree
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


def _create_adaptive_config(config: PSSConfig) -> AdaptiveGateConfig | None:
    """Create AdaptiveGateConfig from PSSConfig if enabled."""
    if not config.adaptive_gates_enabled:
        return None

    return AdaptiveGateConfig(
        min_tokens=config.adaptive_min_tokens,
        max_tokens=config.adaptive_max_tokens,
        high_velocity_threshold=config.adaptive_high_velocity_threshold,
        low_velocity_threshold=config.adaptive_low_velocity_threshold,
        soft_gate_base_tokens=config.adaptive_soft_gate_base_tokens,
        hard_gate_base_tokens=config.adaptive_hard_gate_base_tokens,
        velocity_window_tokens=config.adaptive_velocity_window_tokens,
        measurement_interval_tokens=config.adaptive_measurement_interval_tokens,
        # v1.1: Momentum-based early gates
        enable_momentum_detection=config.adaptive_momentum_enabled,
        momentum_loss_threshold=config.adaptive_momentum_loss_threshold,
        conclusion_velocity=config.adaptive_conclusion_velocity,
    )


def _create_convergence_config(config: PSSConfig) -> ConvergenceConfig | None:
    """Create ConvergenceConfig from PSSConfig if enabled."""
    if not config.convergence_enabled:
        return None

    return ConvergenceConfig(
        enabled=True,
        branch_on_uncertain=config.convergence_branch_on_uncertain,
        convergent_confidence_threshold=config.convergence_confidence_threshold,
    )


def _check_convergence(
    prompt: str,
    config: PSSConfig,
) -> tuple[bool, str]:
    """Check if prompt is convergent and should skip branching.

    Returns:
        Tuple of (is_convergent, reason)
    """
    if not config.convergence_enabled:
        return (False, "convergence_disabled")

    convergence_config = _create_convergence_config(config)
    do_branch, reason = should_branch(prompt, convergence_config)

    # Invert: if should_branch is False, task is convergent
    return (not do_branch, reason)


def _create_lazy_branching_config(config: PSSConfig) -> LazyBranchingConfig | None:
    """Create LazyBranchingConfig from PSSConfig if enabled."""
    if not config.lazy_branching_enabled:
        return None

    return LazyBranchingConfig(
        enabled=True,
        min_tokens_to_judge=config.lazy_branching_min_tokens,
        convergence_window_tokens=config.lazy_branching_window_tokens,
        convergence_velocity=config.lazy_branching_convergence_velocity,
        max_converged_length=config.lazy_branching_max_converged_length,
    )


def _check_lazy_convergence(
    response_text: str,
    token_count: int,
    velocity: float | None,
    config: PSSConfig,
) -> tuple[bool, str]:
    """Check if initial response shows convergence.

    This is the 'lazy' check - it analyzes the actual LLM response
    rather than the prompt to determine if branching is needed.

    Returns:
        Tuple of (is_converged, reason)
    """
    if not config.lazy_branching_enabled:
        return (False, "lazy_branching_disabled")

    lazy_config = _create_lazy_branching_config(config)
    signal, confidence, reason = analyze_initial_response(
        response_text, token_count, velocity, lazy_config
    )

    if signal == BranchSignal.CONVERGED and confidence >= 0.5:
        return (True, f"lazy_converged:{reason}")

    return (False, f"not_converged:{reason}")


def run_pss(
    initial_prompt: str,
    config: PSSConfig,
    provider: Provider,
    eval_fn: Callable[[str], float] | None = None,
    on_context_update: Callable[[Context], None] | None = None,
    interactive_controller: "InteractiveController | None" = None,
    on_token: Callable[[str, str], None] | None = None,
) -> list[Context] | PSSResult:
    """
    Run Pepe Silvia Search.

    This is the main entry point for PSS. It runs exploration contexts
    until they hit token gates, then processes their decisions (continue,
    branch, or terminate).

    Args:
        initial_prompt: The prompt to explore
        config: PSS configuration
        provider: LLM provider
        eval_fn: Optional evaluation function for early exit
        on_context_update: Optional callback for context updates
        interactive_controller: Optional controller for interactive mode
        on_token: Optional callback for streaming tokens: (ctx_id, text) -> None

    Returns:
        If synthesis_enabled: PSSResult with leaves and synthesis
        Otherwise: List of leaf contexts (those that terminated with output)
    """
    # Track trace data
    trace_data: dict | None = None

    if config.parallel:
        # Run the async version
        from pss.execution import run_pss_async

        leaves = asyncio.run(
            run_pss_async(
                initial_prompt,
                config,
                provider,
                eval_fn,
                on_context_update,
                process_decision_fn=_process_decision,
                spawn_branch_fn=_spawn_branch,
                collect_leaves_fn=_collect_leaves,
                interactive_controller=interactive_controller,
                on_token=on_token,
            )
        )
        # Parallel mode doesn't capture trace data (yet)
        tree = SearchTree()
        for leaf in leaves:
            tree.contexts[leaf.id] = leaf
            tree.leaves.append(leaf.id)
    else:
        seq_result = _run_pss_sequential(
            initial_prompt, config, provider, eval_fn, on_context_update,
            interactive_controller=interactive_controller,
        )
        leaves = seq_result.leaves
        tree = seq_result.tree
        # Capture trace data for later
        trace_data = {
            "genealogy": seq_result.genealogy,
            "failure_registry": seq_result.failure_registry,
            "bulletin": seq_result.bulletin,
            "diversity_state": seq_result.diversity_state,
        }

    # v0.4: Run synthesis if enabled
    synthesis_result = None
    if config.synthesis_enabled and len(leaves) >= 1:
        print(f"[PSS] Running synthesis on {len(leaves)} leaves...")
        synthesis_result = synthesize(leaves, initial_prompt, config)

    # Calculate total usage across all leaves
    total_usage = TokenUsage()
    for leaf in leaves:
        total_usage = total_usage + leaf.usage
    if synthesis_result:
        total_usage = total_usage + synthesis_result.usage

    return PSSResult(
        leaves=leaves,
        tree=tree,
        total_usage=total_usage,
        synthesis=synthesis_result,
        _trace_data=trace_data,
    )


@dataclass
class _SequentialResult:
    """Internal result from sequential execution with trace data."""
    leaves: list[Context]
    tree: SearchTree
    genealogy: "GenealogyTree | None" = None
    failure_registry: "FailureRegistry | None" = None
    bulletin: "BulletinBoard | None" = None
    diversity_state: "DiversityState | None" = None


def _run_single_shot(
    prompt: str,
    config: PSSConfig,
    provider: Provider,
    eval_fn: Callable[[str], float] | None = None,
    on_context_update: Callable[[Context], None] | None = None,
) -> _SequentialResult:
    """Run a single-shot completion without branching.

    Used for convergent tasks that have single correct answers.
    Skips all branching logic and just gets one response.
    """
    tree = SearchTree()

    # Build messages
    messages = []
    if config.system_prompt:
        messages.append({"role": "system", "content": config.system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Create single context
    ctx = Context(
        id="root",
        parent_id=None,
        messages=messages,
        token_budget=config.per_context_max,
    )
    tree.contexts["root"] = ctx
    tree.root_id = "root"

    # Single LLM call - no gates, no branching
    text, tool_calls, tokens_used, usage = provider.chat(
        ctx.messages,
        tools=None,
        tool_choice=None,
    )

    ctx.token_count = tokens_used
    ctx.usage = usage
    ctx.output = text
    ctx.status = "terminated"
    ctx.termination_reason = TerminationReason.COMPLETED

    if text:
        ctx.messages.append({"role": "assistant", "content": text})

    tree.leaves.append(ctx.id)

    if on_context_update:
        on_context_update(ctx)

    print(f"[PSS] Single-shot complete: {tokens_used} tokens")

    return _SequentialResult(
        leaves=[ctx],
        tree=tree,
        genealogy=None,
        failure_registry=None,
        bulletin=None,
        diversity_state=None,
    )


def _run_pss_sequential(
    initial_prompt: str,
    config: PSSConfig,
    provider: Provider,
    eval_fn: Callable[[str], float] | None = None,
    on_context_update: Callable[[Context], None] | None = None,
    interactive_controller: "InteractiveController | None" = None,
) -> _SequentialResult:
    """Sequential (non-parallel) PSS execution."""
    # v1.2: Check for convergent task
    is_convergent, convergence_reason = _check_convergence(initial_prompt, config)
    if is_convergent:
        print(f"[PSS] Convergent task detected ({convergence_reason}), running single-shot")
        return _run_single_shot(initial_prompt, config, provider, eval_fn, on_context_update)

    tree = SearchTree()

    # v0.5: Initialize diversity state for real-time tracking
    diversity_state: DiversityState | None = None
    if config.diversity_aware_enabled:
        diversity_state = DiversityState()
        print("[PSS] Diversity-aware spawning enabled")

    # v1.0: Initialize failure registry
    failure_registry: FailureRegistry | None = None
    genealogy: "GenealogyTree | None" = None
    if config.failure_propagation_enabled:
        failure_registry = FailureRegistry(
            stale_after_branches=config.failure_stale_after_branches,
            archive_after_branches=config.failure_archive_after_branches,
        )
        # Import here to avoid circular imports
        from pss.genealogy import GenealogyTree
        genealogy = GenealogyTree(root_id="root")
        print("[PSS] Failure propagation enabled")

    # v1.0: Initialize bulletin board
    bulletin: BulletinBoard | None = None
    if config.bulletin_enabled:
        bulletin = BulletinBoard()
        print("[PSS] Bulletin board enabled")

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

    # v1.0: Create adaptive gate config
    adaptive_config = _create_adaptive_config(config)
    if adaptive_config:
        print(f"[PSS] Adaptive gates enabled (momentum detection: {adaptive_config.enable_momentum_detection})")

    # v1.3: Create lazy branching config
    lazy_branching_config = _create_lazy_branching_config(config)
    if lazy_branching_config:
        print(f"[PSS] Lazy branching enabled (min: {lazy_branching_config.min_tokens_to_judge} tokens)")

    total_tokens = 0
    budget_exhausted = False

    while True:
        # v0.8: Interactive mode - check for pause/quit
        if interactive_controller:
            if interactive_controller.is_quit_requested():
                print("[PSS] Quit requested by user")
                break
            if not interactive_controller.wait_while_paused():
                print("[PSS] Quit requested while paused")
                break

        # Filter out killed contexts
        running = [c for c in tree.contexts.values() if c.status == "running"]
        if interactive_controller:
            running = [c for c in running if not interactive_controller.should_kill(c.id)]
            # Mark killed contexts
            for ctx in tree.contexts.values():
                if ctx.status == "running" and interactive_controller.should_kill(ctx.id):
                    ctx.status = "terminated"
                    ctx.termination_reason = TerminationReason.KILLED
                    print(f"[PSS] Context {ctx.id} killed by user")

        if not running:
            break

        if total_tokens >= config.total_max:
            print(f"[PSS] Hit total token budget ({config.total_max})")
            budget_exhausted = True
            break

        # v1.5: Priority scheduling — sort by confidence, skip low-confidence when budget tight
        # NOTE: LLM self-reported confidence is miscalibrated. Low-confidence branches
        # may be exploring unfamiliar territory. Use a low bar to avoid killing divergence.
        if config.priority_scheduling:
            budget_ratio = total_tokens / config.total_max if config.total_max > 0 else 0
            running.sort(
                key=lambda c: (
                    -(c.spawn_metadata.spawn_confidence if c.spawn_metadata else 0.5),
                    c.id.count("->"),
                )
            )
            if budget_ratio >= config.priority_budget_threshold:
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

        for ctx in running:
            # v0.8: Check again for kill/pause (context could be killed while loop runs)
            if interactive_controller:
                if interactive_controller.is_quit_requested():
                    break
                if interactive_controller.should_kill(ctx.id):
                    ctx.status = "terminated"
                    ctx.termination_reason = TerminationReason.KILLED
                    print(f"[PSS] Context {ctx.id} killed by user")
                    continue

            if len(tree.contexts) >= config.max_contexts:
                print(
                    f"[PSS] Hit max contexts ({config.max_contexts}), not spawning more"
                )

            # v0.8: Check for injected messages before running
            if interactive_controller:
                injection = interactive_controller.get_injection(ctx.id)
                if injection:
                    ctx.messages.append({
                        "role": "user",
                        "content": f"\n---USER GUIDANCE---\n{injection}\n---END GUIDANCE---\n\nPlease take this guidance into account and continue.",
                    })
                    print(f"[PSS] Injected guidance into {ctx.id}")

            result = run_until_gate(
                ctx, provider, config.soft_gate_tokens, config.hard_gate_tokens,
                tools=tools, tool_executor=tool_executor,
                adaptive_config=adaptive_config,
                lazy_branching_config=lazy_branching_config,
            )
            total_tokens += result.tokens_used
            ctx.token_count += result.tokens_used
            ctx.usage = ctx.usage + result.usage

            # v1.3: Handle lazy convergence detected in exploration
            if result.gate_type == "lazy_converged":
                print(f"[PSS] Lazy convergence - returning early with {ctx.token_count} tokens")
                ctx.output = result.response_text
                ctx.status = "terminated"
                ctx.termination_reason = TerminationReason.COMPLETED
                tree.leaves.append(ctx.id)
                if on_context_update:
                    on_context_update(ctx)
                return _SequentialResult(
                    leaves=[ctx],
                    tree=tree,
                    genealogy=genealogy,
                    failure_registry=failure_registry,
                    bulletin=bulletin,
                    diversity_state=diversity_state,
                )

            # v1.0: Extract signals from response
            if bulletin and config.bulletin_enabled and result.response_text:
                signals = extract_signals_from_response(
                    result.response_text,
                    ctx.id,
                    total_tokens,
                    base_confidence=config.bulletin_min_confidence_to_post,
                )
                for signal in signals:
                    bulletin.post(signal)
                if signals:
                    print(f"[PSS] Extracted {len(signals)} signal(s) from {ctx.id}")

            if on_context_update:
                on_context_update(ctx)

            # v1.0: Deliver signals at gate checkpoint
            signals_to_inject = []
            if bulletin and config.bulletin_enabled:
                signals_to_inject = bulletin.get_signals_for_branch(
                    ctx.id,
                    ctx.branch_reason or "",
                    total_tokens,
                    max_signals=config.bulletin_max_signals_per_injection,
                    min_confidence=config.bulletin_min_confidence_to_deliver,
                    half_life_tokens=config.bulletin_signal_half_life_tokens,
                    dead_source_penalty=config.bulletin_dead_source_penalty,
                )

            decision = _run_gate_decision(
                ctx, provider, force=(result.gate_type == "hard"),
                diversity_state=diversity_state, config=config,
                signals_to_inject=signals_to_inject,
            )
            total_tokens += 100

            if on_context_update:
                on_context_update(ctx)

            # v0.8: Handle promote - spawn extra branches like this context
            if interactive_controller and interactive_controller.should_promote(ctx.id):
                promote_edit = f"Continue exploring like context {ctx.id}, which was promoted"
                if ctx.branch_reason:
                    promote_edit = f"Explore similar to: {ctx.branch_reason}"
                # Get failures for promoted branch
                promote_failures = []
                if failure_registry and config.failure_propagation_enabled:
                    promote_failures = failure_registry.get_relevant_failures(
                        promote_edit, max_failures=config.failure_max_to_inject
                    )
                _spawn_branch(
                    tree,
                    ctx,
                    promote_edit,
                    config.per_context_max,
                    spawn_reason=SpawnReason.PROMOTION,
                    spawn_confidence=0.8,  # promoted branches get high confidence
                    failure_registry=failure_registry,
                    failures_to_inject=promote_failures,
                    provider=provider,
                    config=config,
                )
                print(f"[PSS] Promoted {ctx.id} - spawned similar branch")
                if interactive_controller:
                    interactive_controller.on_branch_spawned(ctx.id, f"{ctx.id}->promoted", promote_edit)

            _process_decision(
                tree, ctx, decision, config, eval_fn, diversity_state,
                failure_registry=failure_registry, genealogy=genealogy,
                bulletin=bulletin, provider=provider,
            )

            # v1.0: Sync genealogy after decision processing
            if genealogy:
                genealogy.sync_from_search_tree(tree)

            # v0.8: Notify interactive controller of leaves
            if interactive_controller and decision.action == "terminate" and decision.output:
                interactive_controller.on_leaf_added(ctx.id)

            # v0.8: Notify interactive controller of branches
            if interactive_controller and decision.action == "branch":
                for i, edit in enumerate(decision.branches):
                    if i < config.max_contexts - len(tree.contexts) + len(decision.branches):
                        # Find the spawned branch ID
                        for cid in tree.contexts:
                            if cid.startswith(f"{ctx.id}->") and tree.contexts[cid].branch_reason == edit:
                                interactive_controller.on_branch_spawned(ctx.id, cid, edit)
                                break

            print(
                f"[PSS] Context {ctx.id}: {decision.action} "
                f"(tokens: {ctx.token_count}, branches: {len(decision.branches or [])})"
            )

            # Check for early exit
            if decision.action == "terminate" and decision.output and eval_fn:
                score = eval_fn(decision.output)
                if score >= config.good_enough_threshold:
                    print(f"[PSS] Found good enough output (score={score:.2f})")
                    return _SequentialResult(
                        leaves=_collect_leaves(tree),
                        tree=tree,
                        genealogy=genealogy,
                        failure_registry=failure_registry,
                        bulletin=bulletin,
                        diversity_state=diversity_state,
                    )

            # v1.5: Incremental synthesis — check consensus as leaves arrive
            if (config.incremental_synthesis
                    and decision.action == "terminate"
                    and decision.output):
                current_leaves = _collect_leaves(tree)
                if (len(current_leaves) >= 2
                        and len(current_leaves) % config.incremental_check_interval == 0):
                    should_stop, consensus = check_incremental_consensus(
                        current_leaves, config, provider
                    )
                    print(f"[PSS] Incremental consensus check: {consensus:.2f}")
                    if should_stop:
                        print(f"[PSS] Strong consensus ({consensus:.2f}) — stopping early")
                        return _SequentialResult(
                            leaves=current_leaves,
                            tree=tree,
                            genealogy=genealogy,
                            failure_registry=failure_registry,
                            bulletin=bulletin,
                            diversity_state=diversity_state,
                        )

            # Check if we've exceeded budget mid-loop
            if total_tokens >= config.total_max:
                budget_exhausted = True
                break

    return _SequentialResult(
        leaves=_collect_leaves(tree, force_include_running=budget_exhausted),
        tree=tree,
        genealogy=genealogy,
        failure_registry=failure_registry,
        bulletin=bulletin,
        diversity_state=diversity_state,
    )


def _get_existing_sibling_edits(tree: SearchTree, parent_id: str) -> list[str]:
    """Get branch_reason from all running siblings.

    This returns edit strings from branches that share the same parent
    and are still running, so we can compare new proposed edits against them.
    """
    sibling_edits = []
    for ctx in tree.contexts.values():
        if ctx.parent_id == parent_id and ctx.status == "running":
            if ctx.branch_reason:
                sibling_edits.append(ctx.branch_reason)
    return sibling_edits


def _process_decision(
    tree: SearchTree,
    ctx: Context,
    decision: GateDecision,
    config: PSSConfig,
    eval_fn: Callable[[str], float] | None,
    diversity_state: DiversityState | None = None,
    failure_registry: FailureRegistry | None = None,
    genealogy: "GenealogyTree | None" = None,
    bulletin: BulletinBoard | None = None,
    provider: Provider | None = None,
) -> None:
    """Process a gate decision (shared by sync and async versions)."""
    if decision.action == "terminate":
        ctx.status = "terminated"
        if decision.output:
            ctx.output = decision.output
            tree.leaves.append(ctx.id)
            # v1.0: Set termination reason
            ctx.termination_reason = TerminationReason.COMPLETED

            # v0.5: Update diversity on new leaf
            if diversity_state and config.diversity_aware_enabled:
                from pss.diversity import update_diversity_incremental

                success = update_diversity_incremental(ctx, diversity_state, config)
                if success and diversity_state.num_leaves >= 2:
                    print(
                        f"[PSS] Diversity: {diversity_state.current_score:.2f} "
                        f"({diversity_state.interpretation}) - {diversity_state.num_leaves} leaves"
                    )

            # v1.0: Check for contradicted failures on successful completion
            if failure_registry and genealogy:
                genealogy.sync_from_search_tree(tree)
                branch_path = genealogy.get_branch_path(ctx.id)
                contradicted = failure_registry.handle_contradiction(ctx.id, branch_path)
                if contradicted:
                    print(f"[PSS] Branch {ctx.id} contradicted {len(contradicted)} failure(s)")
        else:
            # Terminated without output = stuck
            ctx.termination_reason = TerminationReason.STUCK

            # v1.0: Extract failure summary
            if failure_registry and config.failure_propagation_enabled:
                failure = extract_failure_from_context(ctx, genealogy)
                if failure and failure.confidence >= config.failure_min_confidence:
                    failure_registry.add_failure(failure)
                    print(f"[PSS] Extracted failure from {ctx.id}: {failure.what_was_attempted[:50]}")

        # v1.0: Mark signals from this branch as terminated
        if bulletin:
            reason = "completed" if ctx.termination_reason == TerminationReason.COMPLETED else "stuck"
            bulletin.mark_branch_terminated(ctx.id, reason)

    elif decision.action == "continue":
        pass

    elif decision.action == "branch":
        can_spawn = config.max_contexts - len(tree.contexts)
        branches_to_spawn = decision.branches[:can_spawn]

        # v1.4: Sibling diversity enforcement
        if (config.sibling_diversity_enabled and
                len(branches_to_spawn) >= config.sibling_diversity_min_edits):
            existing_siblings = []
            if config.sibling_diversity_include_existing:
                existing_siblings = _get_existing_sibling_edits(tree, ctx.id)

            try:
                diversity_result = check_sibling_diversity(
                    branches_to_spawn,
                    similarity_threshold=config.sibling_similarity_threshold,
                    existing_siblings=existing_siblings,
                )
                # Log rejections
                for edit in diversity_result.rejected_edits:
                    reason = diversity_result.rejection_reasons.get(edit, "too similar")
                    print(format_diversity_rejection_log(edit, reason))

                # Use only accepted edits
                branches_to_spawn = diversity_result.accepted_edits

                if diversity_result.rejected_edits:
                    print(f"[PSS] Sibling diversity: {len(diversity_result.accepted_edits)} accepted, "
                          f"{len(diversity_result.rejected_edits)} rejected")
            except Exception as e:
                # Don't block on diversity check failure - just log and continue
                print(f"[PSS] Sibling diversity check failed: {e}")

        # Get confidence for spawned branches (default 0.5 if not provided)
        spawn_confidence = decision.confidence if decision.confidence is not None else 0.5

        # v1.0: Get relevant failures for injection
        failures_to_inject: list = []
        if failure_registry and config.failure_propagation_enabled:
            # Use first branch edit as direction hint
            direction = branches_to_spawn[0] if branches_to_spawn else ""
            failures_to_inject = failure_registry.get_relevant_failures(
                direction, max_failures=config.failure_max_to_inject
            )

        for edit in branches_to_spawn:
            _spawn_branch(
                tree,
                ctx,
                edit,
                config.per_context_max,
                spawn_reason=SpawnReason.GATE_BRANCH,
                spawn_confidence=spawn_confidence,
                failure_registry=failure_registry,
                failures_to_inject=failures_to_inject,
                provider=provider,
                config=config,
            )

        if decision.also_continue:
            ctx.status = "running"
        else:
            ctx.status = "branched"


def _run_gate_decision(
    ctx: Context,
    provider: Provider,
    force: bool = False,
    max_retries: int = 3,
    diversity_state: DiversityState | None = None,
    config: PSSConfig | None = None,
    signals_to_inject: list | None = None,
) -> GateDecision:
    """Inject a gate and get the context's decision (sync version)."""
    # v1.0: Inject signals before gate prompt
    if signals_to_inject:
        signal_text = format_signals_for_injection(signals_to_inject)
        if signal_text:
            ctx.messages.append({"role": "user", "content": signal_text})

    gate_prompt = create_gate_prompt(
        ctx, force=force, diversity_state=diversity_state, config=config
    )
    ctx.messages.append({"role": "user", "content": gate_prompt})
    ctx.gates_seen += 1

    for attempt in range(max_retries):
        text, tool_calls, tokens_used, usage = provider.chat(
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


def _compress_messages(
    messages: list[dict],
    provider: Provider,
    max_messages: int = 20,
) -> list[dict]:
    """Compress parent message history for branch inheritance.

    Keeps the system prompt, last N messages, and gate decisions.
    Summarizes everything else into a single context note.
    """
    if len(messages) <= max_messages:
        return copy.deepcopy(messages)

    # Separate system messages from conversation
    system_msgs = [m for m in messages if m.get("role") == "system"]
    conv_msgs = [m for m in messages if m.get("role") != "system"]

    if len(conv_msgs) <= max_messages:
        return copy.deepcopy(messages)

    # Split: older messages to summarize, recent messages to keep
    keep_count = max_messages - len(system_msgs)
    older = conv_msgs[:-keep_count]
    recent = conv_msgs[-keep_count:]

    # Preserve gate decisions and branch directives verbatim
    preserved = []
    to_summarize = []
    for m in older:
        content = str(m.get("content", ""))
        if "---CHECKPOINT---" in content or "---BRANCH---" in content:
            preserved.append(copy.deepcopy(m))
        else:
            to_summarize.append(m)

    # Build summary of the exploration content (not gate/branch messages)
    older_text = "\n".join(
        f"[{m.get('role', '?')}]: {str(m.get('content', ''))[:200]}"
        for m in to_summarize
    )

    try:
        summary_prompt = (
            "Summarize this exploration history in 2-3 sentences. "
            "Preserve: specific findings, conclusions reached, evidence discovered. "
            "Drop: thinking-out-loud, hedging, repetitive analysis.\n\n"
            f"{older_text[:4000]}"
        )
        text, _, _, _ = provider.chat(
            [{"role": "user", "content": summary_prompt}],
        )
        summary = text.strip()
    except Exception:
        # Fallback: just truncate without LLM summarization
        summary = f"[Prior exploration: {len(to_summarize)} messages covering initial analysis]"

    # Reconstruct: system + summary + preserved gate decisions + recent
    compressed = list(system_msgs)
    compressed.append({
        "role": "user",
        "content": f"[Context from prior exploration]\n{summary}\n[End context]",
    })
    compressed.extend(preserved)
    compressed.extend(copy.deepcopy(recent))

    return compressed


def _spawn_branch(
    tree: SearchTree,
    parent: Context,
    edit: str,
    budget: int,
    spawn_reason: SpawnReason = SpawnReason.GATE_BRANCH,
    spawn_confidence: float = 0.5,
    failure_registry: FailureRegistry | None = None,
    failures_to_inject: list | None = None,
    provider: Provider | None = None,
    config: PSSConfig | None = None,
) -> Context:
    """Create a new branch from a parent context.

    Args:
        tree: The search tree to add the branch to
        parent: The parent context to branch from
        edit: The edit/direction for this branch
        budget: Token budget for the new branch
        spawn_reason: Why this branch was created
        spawn_confidence: Confidence from parent's gate decision (0-1)
        failure_registry: Optional registry to age on spawn
        failures_to_inject: Optional list of failures to inject into branch
        provider: Optional provider for context compression
        config: Optional config for compression settings
    """
    branch_id = f"{parent.id}->{uuid.uuid4().hex[:6]}"

    # Get parent's current velocity if available (for adaptive gates)
    parent_velocity = None
    if parent.velocity_history:
        parent_velocity = parent.velocity_history[-1].overall_velocity

    # Create spawn metadata
    spawn_metadata = SpawnMetadata(
        spawn_reason=spawn_reason,
        spawn_edit=edit,
        spawn_gate_number=parent.gates_seen,
        spawn_confidence=spawn_confidence,
        parent_velocity=parent_velocity,
    )

    # v1.5: Optionally compress parent history
    if config and config.context_compression and provider:
        branch_messages = _compress_messages(
            parent.messages, provider, config.compression_max_messages
        )
    else:
        branch_messages = copy.deepcopy(parent.messages)

    branch = Context(
        id=branch_id,
        parent_id=parent.id,
        messages=branch_messages,
        state=copy.deepcopy(parent.state),
        token_count=0,
        token_budget=budget,
        branch_reason=edit,
        gates_seen=1,
        spawn_metadata=spawn_metadata,
    )

    # Build branch prompt with optional failure injection
    branch_prompt = f"\n---BRANCH---\n\nYou're exploring an alternative: {edit}"

    # v1.0: Inject failures if any
    if failures_to_inject:
        failure_text = format_failures_for_injection(failures_to_inject)
        if failure_text:
            branch_prompt += f"\n\n{failure_text}"

    branch_prompt += "\n\nContinue from here."

    branch.messages.append({"role": "user", "content": branch_prompt})

    # v1.0: Age all failures on spawn
    if failure_registry:
        failure_registry.age_all()

    tree.contexts[branch_id] = branch
    print(f"[PSS] Spawned branch {branch_id}: {edit[:50]}...")
    return branch


def _collect_leaves(tree: SearchTree, force_include_running: bool = False) -> list[Context]:
    """Collect all leaf contexts (those with output).

    Args:
        tree: The search tree
        force_include_running: If True, also include running contexts
            that have messages (useful when budget is exhausted)
    """
    leaves = [
        tree.contexts[leaf_id] for leaf_id in tree.leaves if leaf_id in tree.contexts
    ]

    if force_include_running:
        # Include any running/branched contexts that haven't terminated
        # Try to extract partial output from their message history
        for ctx in tree.contexts.values():
            if ctx.id in tree.leaves:
                continue  # Already collected
            if ctx.status in ("running", "branched"):
                # Try to get last assistant message as partial output
                for msg in reversed(ctx.messages):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        ctx.output = f"[Partial - budget exhausted]\n{msg['content']}"
                        ctx.status = "terminated"
                        ctx.termination_reason = TerminationReason.BUDGET
                        leaves.append(ctx)
                        break

    return leaves


# Backward compatibility aliases for tests
_determine_gate_type = determine_gate_type
_run_until_gate = run_until_gate
