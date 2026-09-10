"""CLI entry point for Pepe Silvia Search."""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, skip

# Handle Windows encoding issues
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pss.config import load_config, PSSConfig
from pss.harness import run_pss
from pss.presets import get_preset, list_presets, apply_preset_to_config, PRESETS
from pss.providers import Provider, create_provider
from pss.types import Context, PSSResult

# Evaluation imports (lazy loaded when needed)
EVAL_TASKS = ["game_of_24", "gsm8k", "humaneval", "brainstorm", "bugfind", "all"]


@dataclass
class BaselineResult:
    """Result of a single-shot baseline run."""

    outputs: list[str]
    total_cost: float
    total_tokens: int


def run_single_shot_baseline(
    prompt: str,
    provider: Provider,
    n: int,
) -> BaselineResult:
    """
    Run the same prompt N times single-shot (no PSS).

    Returns list of outputs for diversity comparison.
    """
    outputs = []
    total_cost = 0.0
    total_tokens = 0

    for i in range(n):
        print(f"[Baseline] Running single-shot {i + 1}/{n}...")
        messages = [{"role": "user", "content": prompt}]
        text, _, tokens_used, usage = provider.chat(messages)
        outputs.append(text)
        total_cost += usage.cost
        total_tokens += tokens_used

    return BaselineResult(
        outputs=outputs,
        total_cost=total_cost,
        total_tokens=total_tokens,
    )


def build_tree_visualization(leaves: list[Context]) -> str:
    """Build an ASCII tree visualization from leaf contexts."""
    if not leaves:
        return "  (no leaves)"

    # Build tree structure from leaf IDs
    # IDs look like: "root", "root->abc123", "root->abc123->def456"
    nodes: dict[str, dict] = {}

    for leaf in leaves:
        # Parse the path from the ID
        parts = leaf.id.split("->")
        path = ""
        for i, part in enumerate(parts):
            parent = path
            path = "->".join(parts[: i + 1]) if i > 0 else part
            if path not in nodes:
                nodes[path] = {
                    "parent": parent if parent else None,
                    "children": [],
                    "is_leaf": False,
                    "leaf": None,
                }
        # Mark the final node as a leaf
        nodes[leaf.id]["is_leaf"] = True
        nodes[leaf.id]["leaf"] = leaf

    # Build parent-child relationships
    for node_id, node in nodes.items():
        if node["parent"] and node["parent"] in nodes:
            if node_id not in nodes[node["parent"]]["children"]:
                nodes[node["parent"]]["children"].append(node_id)

    # Render tree
    lines = []

    def render_node(node_id: str, prefix: str = "", is_last: bool = True):
        node = nodes[node_id]
        connector = "└── " if is_last else "├── "

        # Format node label
        short_id = node_id.split("->")[-1] if "->" in node_id else node_id
        if node["is_leaf"]:
            leaf = node["leaf"]
            label = f"{short_id} [LEAF] ${leaf.usage.cost:.4f}"
        else:
            label = short_id

        lines.append(f"{prefix}{connector}{label}")

        # Render children
        children = sorted(node["children"])
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child_id in enumerate(children):
            render_node(child_id, child_prefix, i == len(children) - 1)

    # Start from root
    if "root" in nodes:
        root = nodes["root"]
        if root["is_leaf"]:
            lines.append(f"root [LEAF] ${root['leaf'].usage.cost:.4f}")
        else:
            lines.append("root")

        children = sorted(root["children"])
        for i, child_id in enumerate(children):
            render_node(child_id, "", i == len(children) - 1)

    return "\n".join(lines) if lines else "  (empty tree)"


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments shared by all commands."""
    parser.add_argument(
        "--config",
        default="pss_config.yaml",
        help="Path to config file (default: pss_config.yaml)",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openrouter"],
        help="Override provider",
    )
    parser.add_argument(
        "--model",
        help="Override model",
    )
    parser.add_argument(
        "--soft-gate",
        type=int,
        help="Override soft gate token threshold",
    )
    parser.add_argument(
        "--hard-gate",
        type=int,
        help="Override hard gate token threshold",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Override total token budget",
    )
    parser.add_argument(
        "--max-contexts",
        type=int,
        help="Override max contexts",
    )
    parser.add_argument(
        "--parallel",
        "-p",
        action="store_true",
        help="Enable parallel context execution",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        help="Max concurrent API calls (default: 5)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--no-diversity",
        action="store_true",
        help="Skip diversity analysis (saves embedding API cost)",
    )
    parser.add_argument(
        "--diversity-matrix",
        action="store_true",
        help="Show full pairwise distance matrix in diversity output",
    )
    parser.add_argument(
        "--compare",
        type=int,
        metavar="N",
        help="Compare PSS diversity against N single-shot runs",
    )
    # v0.5: Diversity-aware spawning
    parser.add_argument(
        "--diversity-aware",
        action="store_true",
        help="Enable diversity-aware branching (v0.5)",
    )
    parser.add_argument(
        "--diversity-target",
        type=float,
        metavar="SCORE",
        help="Target diversity score (0.0-1.0) - stop spawning when reached",
    )
    parser.add_argument(
        "--diversity-min",
        type=float,
        metavar="SCORE",
        help="Minimum diversity threshold - encourage branching below this",
    )
    # v0.4: Agentic branches
    parser.add_argument(
        "--agentic",
        action="store_true",
        help="Enable agentic mode with tool use during exploration",
    )
    parser.add_argument(
        "--working-dir",
        help="Working directory for agentic tools (default: current directory)",
    )
    # v0.4: Synthesis
    parser.add_argument(
        "--synthesis",
        action="store_true",
        help="Enable synthesis phase to merge branch findings",
    )
    parser.add_argument(
        "--synthesis-model",
        help="Model to use for synthesis (default: same as exploration)",
    )
    parser.add_argument(
        "--synthesis-strategy",
        choices=["merge", "vote", "deliberate", "plan"],
        help="Synthesis strategy (default: merge)",
    )
    # v1.1: Tracing
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable trace collection for post-hoc analysis",
    )
    parser.add_argument(
        "--trace-dir",
        help="Directory to save traces (default: ./traces)",
    )


def apply_cli_overrides(config: PSSConfig, args: argparse.Namespace) -> None:
    """Apply CLI argument overrides to config."""
    if args.provider:
        config.provider = args.provider
    if args.model:
        config.model = args.model
    if args.soft_gate:
        config.soft_gate_tokens = args.soft_gate
    if args.hard_gate:
        config.hard_gate_tokens = args.hard_gate
    if args.max_tokens:
        config.total_max = args.max_tokens
    if args.max_contexts:
        config.max_contexts = args.max_contexts
    if args.parallel:
        config.parallel = True
    if args.max_parallel:
        config.max_parallel = args.max_parallel
    if args.no_diversity:
        config.diversity_enabled = False
    if args.diversity_matrix:
        config.diversity_show_matrix = True
    # v0.5: Diversity-aware spawning
    if hasattr(args, "diversity_aware") and args.diversity_aware:
        config.diversity_aware_enabled = True
    if hasattr(args, "diversity_target") and args.diversity_target is not None:
        config.diversity_target = args.diversity_target
    if hasattr(args, "diversity_min") and args.diversity_min is not None:
        config.diversity_min_threshold = args.diversity_min
    # v0.4: Agentic and synthesis
    if hasattr(args, "agentic") and args.agentic:
        config.agentic_enabled = True
    if hasattr(args, "working_dir") and args.working_dir:
        config.agentic_working_dir = args.working_dir
    if hasattr(args, "synthesis") and args.synthesis:
        config.synthesis_enabled = True
    if hasattr(args, "synthesis_model") and args.synthesis_model:
        config.synthesis_model = args.synthesis_model
    if hasattr(args, "synthesis_strategy") and args.synthesis_strategy:
        config.synthesis_strategy = args.synthesis_strategy


def load_prompt(prompt_arg: str) -> str:
    """Load prompt from argument or file."""
    if prompt_arg.startswith("@"):
        filepath = Path(prompt_arg[1:])
        if not filepath.exists():
            print(f"Error: File not found: {filepath}", file=sys.stderr)
            sys.exit(1)
        return filepath.read_text()
    return prompt_arg


def run_pss_with_output(
    prompt: str,
    config: PSSConfig,
    args: argparse.Namespace,
) -> int:
    """Run PSS and display results. Returns exit code."""
    import time

    # Create provider
    print(f"[PSS] Using {config.provider} with model {config.model}")
    print(
        f"[PSS] Gates: soft={config.soft_gate_tokens}, hard={config.hard_gate_tokens}"
    )
    print(
        f"[PSS] Budget: {config.total_max} total tokens, {config.max_contexts} max contexts"
    )
    if config.parallel:
        print(
            f"[PSS] Parallel execution enabled (max {config.max_parallel} concurrent)"
        )
    if config.system_prompt:
        print(f"[PSS] Using system prompt ({len(config.system_prompt)} chars)")
    # v0.4: Show agentic and synthesis settings
    if config.agentic_enabled:
        tools = config.agentic_tools or ["read_file", "list_directory", "search_files"]
        print(f"[PSS] Agentic mode: tools={tools}")
    if config.synthesis_enabled:
        synth_model = config.synthesis_model or config.model
        print(f"[PSS] Synthesis: strategy={config.synthesis_strategy}, model={synth_model}")
    # v0.5: Show diversity-aware settings
    if config.diversity_aware_enabled:
        print(f"[PSS] Diversity-aware: target={config.diversity_target}, min={config.diversity_min_threshold}")
    # v1.1: Show trace settings
    trace_enabled = getattr(args, "trace", False)
    trace_dir = getattr(args, "trace_dir", None)
    if trace_enabled:
        print(f"[PSS] Tracing enabled (dir: {trace_dir or 'traces/'})")
    print()

    provider = create_provider(config.provider, config.model)

    # Callback for verbose mode
    def on_update(ctx):
        if args.verbose:
            print(f"  [{ctx.id}] status={ctx.status}, tokens={ctx.token_count}")

    # Run PSS with timing
    start_time = time.time()
    try:
        result = run_pss(
            initial_prompt=prompt,
            config=config,
            provider=provider,
            on_context_update=on_update if args.verbose else None,
        )
    except KeyboardInterrupt:
        print("\n[PSS] Interrupted")
        return 130
    elapsed_seconds = time.time() - start_time

    # Handle both PSSResult and list[Context] return types
    synthesis_result = None
    if isinstance(result, PSSResult):
        leaves = result.leaves
        synthesis_result = result.synthesis
    else:
        leaves = result

    # Output results
    print()
    print("=" * 60)
    print(f"RESULTS: {len(leaves)} leaf context(s)")
    print("=" * 60)

    # Calculate totals
    total_input = sum(leaf.usage.input_tokens for leaf in leaves)
    total_output = sum(leaf.usage.output_tokens for leaf in leaves)
    total_cost = sum(leaf.usage.cost for leaf in leaves)

    for i, leaf in enumerate(leaves, 1):
        print(f"\n--- Leaf {i}: {leaf.id} ---")
        print(
            f"Tokens: {leaf.usage.input_tokens:,} in / {leaf.usage.output_tokens:,} out ({leaf.usage.total_tokens:,} total)"
        )
        print(f"Cost: ${leaf.usage.cost:.4f}")
        if leaf.branch_reason:
            print(f"Branch reason: {leaf.branch_reason}")
        print()
        if leaf.output:
            print(leaf.output)
        else:
            # Show last assistant message if no explicit output
            for msg in reversed(leaf.messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    print(msg["content"][:2000])
                    if len(msg["content"]) > 2000:
                        print("... (truncated)")
                    break
        print()

    # Tree visualization
    print("=" * 60)
    print("EXPLORATION TREE")
    print("=" * 60)
    print(build_tree_visualization(leaves))
    print()

    # Cost summary
    print("=" * 60)
    print("COST SUMMARY")
    print("=" * 60)
    print(
        f"Total tokens: {total_input:,} in / {total_output:,} out ({total_input + total_output:,} total)"
    )
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Cost per leaf: ${total_cost / len(leaves):.4f}" if leaves else "N/A")
    print()

    # v0.4: Synthesis results
    if synthesis_result:
        print("=" * 60)
        print("SYNTHESIS RESULT")
        print("=" * 60)
        print(f"Confidence: {synthesis_result.confidence:.2f}")
        print(f"Synthesis cost: ${synthesis_result.usage.cost:.4f}")
        total_cost += synthesis_result.usage.cost
        print()

        # Show evidence summary if available
        if synthesis_result.evidence_summary:
            print("Cross-branch evidence:")
            for finding, count in sorted(
                synthesis_result.evidence_summary.items(),
                key=lambda x: -x[1]
            ):
                print(f"  - {finding}: {count} branches")
            print()

        # Show action plan if available
        if synthesis_result.action_plan:
            print("Action plan:")
            for i, step in enumerate(synthesis_result.action_plan, 1):
                print(f"  {i}. {step}")
            print()

        # Show dissenting views if any
        if synthesis_result.dissenting_views:
            print("Dissenting views:")
            for view in synthesis_result.dissenting_views:
                print(f"  - {view}")
            print()

        # Show unified output
        print("Unified output:")
        print("-" * 40)
        print(synthesis_result.unified_output)
        print("-" * 40)
        print()

    # Diversity analysis
    pss_diversity = None
    if config.diversity_enabled and len(leaves) >= 2:
        from pss.diversity import compute_diversity, format_diversity_report

        print("=" * 60)
        print("DIVERSITY ANALYSIS")
        print("=" * 60)

        try:
            pss_diversity = compute_diversity(
                leaves,
                provider=config.diversity_provider,
                model=config.diversity_model,
                include_matrix=config.diversity_show_matrix,
            )
            if pss_diversity:
                print(format_diversity_report(pss_diversity, leaves))
                # Add embedding cost to total
                total_cost += pss_diversity.embedding_cost
            else:
                print("  (diversity analysis unavailable)")
        except Exception as e:
            print(f"  Diversity analysis failed: {e}")
            print("  Use --no-diversity to skip, or set OPENAI_API_KEY for embeddings")

        print()

    # Baseline comparison
    if args.compare and args.compare >= 2:
        from pss.diversity import compute_diversity_from_texts

        print("=" * 60)
        print(f"BASELINE COMPARISON (single-shot x{args.compare})")
        print("=" * 60)

        try:
            baseline = run_single_shot_baseline(prompt, provider, args.compare)
            print(
                f"Baseline cost: ${baseline.total_cost:.4f} ({baseline.total_tokens:,} tokens)"
            )
            print()

            # Compute baseline diversity
            baseline_diversity = compute_diversity_from_texts(
                baseline.outputs,
                provider=config.diversity_provider,
                model=config.diversity_model,
            )

            if baseline_diversity and pss_diversity:
                print("=" * 60)
                print("DIVERSITY COMPARISON")
                print("=" * 60)
                pss_score = pss_diversity.score
                baseline_score = baseline_diversity.score

                # Calculate improvement
                if baseline_score > 0:
                    improvement = ((pss_score - baseline_score) / baseline_score) * 100
                    improvement_str = (
                        f"+{improvement:.0f}%"
                        if improvement > 0
                        else f"{improvement:.0f}%"
                    )
                else:
                    improvement_str = "N/A"

                print(
                    f"  Single-shot x{args.compare}:  {baseline_score:.2f} ({baseline_diversity.interpretation})"
                )
                print(
                    f"  PSS {len(leaves)} leaves:      {pss_score:.2f} ({pss_diversity.interpretation})"
                )
                print(f"  Improvement:          {improvement_str}")
                print()

                # Cost comparison
                cost_ratio = (
                    total_cost / baseline.total_cost if baseline.total_cost > 0 else 0
                )
                print(f"  Single-shot cost: ${baseline.total_cost:.4f}")
                print(f"  PSS cost:         ${total_cost:.4f} ({cost_ratio:.1f}x)")
                print()

            elif baseline_diversity:
                print(
                    f"Baseline diversity: {baseline_diversity.score:.2f} ({baseline_diversity.interpretation})"
                )
                print("  (PSS diversity unavailable for comparison)")
            else:
                print("  (baseline diversity computation failed)")

        except Exception as e:
            print(f"  Baseline comparison failed: {e}")

        print()

    # v1.1: Save trace if enabled
    if trace_enabled and isinstance(result, PSSResult):
        from pathlib import Path
        from pss.trace import (
            build_trace_from_result,
            save_trace,
        )

        print("=" * 60)
        print("SAVING TRACE")
        print("=" * 60)

        # Extract trace data from result
        trace_data = result._trace_data or {}
        genealogy = trace_data.get("genealogy")
        failure_registry = trace_data.get("failure_registry")
        bulletin = trace_data.get("bulletin")
        diversity_state_obj = trace_data.get("diversity_state")

        # Build the trace
        trace = build_trace_from_result(
            prompt=prompt,
            config=config,
            result=result,
            elapsed_seconds=elapsed_seconds,
            tree=result.tree,
            genealogy=genealogy,
            failure_registry=failure_registry,
            bulletin=bulletin,
            diversity_state=diversity_state_obj,
        )

        # Save to specified or default directory
        trace_path = save_trace(
            trace,
            trace_dir=Path(trace_dir) if trace_dir else None,
        )
        print(f"Trace saved: {trace_path}")
        print(f"  ID: {trace.id}")
        print(f"  Leaves: {trace.num_leaves}")
        print(f"  Duration: {trace.elapsed_seconds:.1f}s")
        print()

    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Handle the default 'run' command (or bare pss invocation)."""
    config = load_config(args.config)

    # Apply preset if specified
    if hasattr(args, "preset") and args.preset:
        preset = get_preset(args.preset)
        if not preset:
            print(f"Error: Unknown preset '{args.preset}'", file=sys.stderr)
            print(f"Available presets: {', '.join(PRESETS.keys())}", file=sys.stderr)
            return 1
        apply_preset_to_config(preset, config)
        print(f"[PSS] Using preset: {preset.name} - {preset.description}")

    apply_cli_overrides(config, args)
    prompt = load_prompt(args.prompt)
    return run_pss_with_output(prompt, config, args)


def cmd_explore(args: argparse.Namespace) -> int:
    """Handle the 'explore' subcommand."""
    config = load_config(args.config)
    preset = get_preset("explore")
    apply_preset_to_config(preset, config)
    print(f"[PSS] Using preset: explore - {preset.description}")
    apply_cli_overrides(config, args)
    prompt = load_prompt(args.prompt)
    return run_pss_with_output(prompt, config, args)


def cmd_write(args: argparse.Namespace) -> int:
    """Handle the 'write' subcommand."""
    config = load_config(args.config)
    preset = get_preset("write")
    apply_preset_to_config(preset, config)
    print(f"[PSS] Using preset: write - {preset.description}")
    apply_cli_overrides(config, args)
    prompt = load_prompt(args.prompt)
    return run_pss_with_output(prompt, config, args)


def cmd_research(args: argparse.Namespace) -> int:
    """Handle the 'research' subcommand."""
    config = load_config(args.config)
    preset = get_preset("research")
    apply_preset_to_config(preset, config)
    print(f"[PSS] Using preset: research - {preset.description}")
    apply_cli_overrides(config, args)
    prompt = load_prompt(args.prompt)
    return run_pss_with_output(prompt, config, args)


def cmd_draft(args: argparse.Namespace) -> int:
    """Handle the 'draft' subcommand."""
    config = load_config(args.config)
    preset = get_preset("draft")
    apply_preset_to_config(preset, config)
    print(f"[PSS] Using preset: draft - {preset.description}")
    apply_cli_overrides(config, args)
    prompt = load_prompt(args.prompt)
    return run_pss_with_output(prompt, config, args)


def cmd_investigate(args: argparse.Namespace) -> int:
    """Handle the 'investigate' subcommand (v0.4)."""
    config = load_config(args.config)
    preset = get_preset("investigate")
    apply_preset_to_config(preset, config)
    print(f"[PSS] Using preset: investigate - {preset.description}")
    apply_cli_overrides(config, args)
    prompt = load_prompt(args.prompt)
    return run_pss_with_output(prompt, config, args)


def cmd_debug(args: argparse.Namespace) -> int:
    """Handle the 'debug' subcommand (v0.4)."""
    config = load_config(args.config)
    preset = get_preset("debug")
    apply_preset_to_config(preset, config)
    print(f"[PSS] Using preset: debug - {preset.description}")
    apply_cli_overrides(config, args)
    prompt = load_prompt(args.prompt)
    return run_pss_with_output(prompt, config, args)


def cmd_reasoning(args: argparse.Namespace) -> int:
    """Handle the 'reasoning' subcommand (v0.6)."""
    config = load_config(args.config)
    preset = get_preset("reasoning")
    apply_preset_to_config(preset, config)
    print(f"[PSS] Using preset: reasoning - {preset.description}")
    apply_cli_overrides(config, args)
    prompt = load_prompt(args.prompt)
    return run_pss_with_output(prompt, config, args)


def cmd_web(args: argparse.Namespace) -> int:
    """Handle the 'web' subcommand (v0.9)."""
    try:
        import uvicorn
        from pss.web.server import app
    except ImportError as e:
        print("Web UI requires additional dependencies.", file=sys.stderr)
        print("Install with: pip install pss[web]", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        return 1

    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 4311)
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("The workbench supports loopback use only.", file=sys.stderr)
        return 1

    print(f"[PSS] Starting Web UI server")
    print(f"[PSS] Local: http://localhost:{port}")
    print()
    print("Local workbench. Live model calls are explicitly opt-in.")
    print("Press Ctrl+C to stop the server")
    print()

    try:
        uvicorn.run(app, host=host, port=port)
    except KeyboardInterrupt:
        print("\n[PSS] Server stopped")

    return 0


def cmd_interactive(args: argparse.Namespace) -> int:
    """Handle the 'interactive' subcommand (v0.8)."""
    from pss.interactive import InteractiveController
    from pss.providers import create_provider

    config = load_config(args.config)

    # Apply preset if specified
    if hasattr(args, "preset") and args.preset:
        preset = get_preset(args.preset)
        if not preset:
            print(f"Error: Unknown preset '{args.preset}'", file=sys.stderr)
            return 1
        apply_preset_to_config(preset, config)

    apply_cli_overrides(config, args)
    prompt = load_prompt(args.prompt)

    print(f"[PSS] Interactive mode - {config.provider}/{config.model}")
    print(f"[PSS] Gates: soft={config.soft_gate_tokens}, hard={config.hard_gate_tokens}")
    print()
    print("Commands: k <id> (kill), i <id> <msg> (inject), p <id> (promote)")
    print("          pause, resume, q (quit), ? (help)")
    print()

    provider = create_provider(config.provider, config.model)
    controller = InteractiveController(config)

    # Run in interactive mode
    from pss.harness import run_pss

    result_holder: list[Context] = []

    def exploration():
        result = run_pss(
            initial_prompt=prompt,
            config=config,
            provider=provider,
            on_context_update=controller.on_context_update,
            interactive_controller=controller,
        )
        if isinstance(result, PSSResult):
            result_holder.extend(result.leaves)
        else:
            result_holder.extend(result)

    try:
        controller.run_display(exploration)
    except KeyboardInterrupt:
        print("\n[PSS] Interrupted")
        return 130

    return 0


def cmd_synthesize(args: argparse.Namespace) -> int:
    """Handle the 'synthesize' subcommand — standalone synthesis of text files."""
    from pss.synthesis import synthesize
    from pss.types import Context, TokenUsage

    config = load_config(args.config)
    apply_cli_overrides(config, args)

    # Override synthesis settings
    config.synthesis_enabled = True
    if args.synthesis_strategy:
        config.synthesis_strategy = args.synthesis_strategy
    if args.synthesis_model:
        config.synthesis_model = args.synthesis_model

    # Read input files
    input_files = args.files
    if not input_files:
        print("Error: No input files specified", file=sys.stderr)
        return 1

    leaves = []
    for i, filepath in enumerate(input_files):
        path = Path(filepath)
        if not path.exists():
            print(f"Error: File not found: {path}", file=sys.stderr)
            return 1

        text = path.read_text(encoding="utf-8", errors="replace")
        ctx = Context(
            id=f"input_{i}",
            parent_id=None,
            messages=[],
            output=text,
            status="terminated",
            branch_reason=path.name,
        )
        leaves.append(ctx)

    print(f"[PSS] Synthesizing {len(leaves)} inputs with strategy '{config.synthesis_strategy}'")

    provider = create_provider(
        config.synthesis_provider or config.provider,
        config.synthesis_model or config.model,
    )

    result = synthesize(leaves, args.prompt or "Synthesize these inputs.", config, provider)

    # Output
    print()
    print("=" * 60)
    print("SYNTHESIS RESULT")
    print("=" * 60)
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Cost: ${result.usage.cost:.4f}")
    print()

    if result.dissenting_views:
        print("Dissenting views:")
        for view in result.dissenting_views:
            print(f"  - {view}")
        print()

    if result.action_plan:
        print("Action plan:")
        for i, step in enumerate(result.action_plan, 1):
            print(f"  {i}. {step}")
        print()

    print(result.unified_output)
    return 0


def cmd_presets(args: argparse.Namespace) -> int:
    """List available presets."""
    print("Available presets:\n")
    for preset in list_presets():
        print(f"  {preset.name}")
        print(f"    {preset.description}")
        print(f"    Model: {preset.provider}/{preset.model}")
        print(f"    Gates: soft={preset.soft_gate_tokens}, hard={preset.hard_gate_tokens}")
        print(f"    Budget: {preset.total_max} tokens, {preset.max_contexts} contexts")
        # v0.4: Show agentic/synthesis info
        if preset.agentic_enabled:
            print(f"    Agentic: tools={preset.agentic_tools}")
        if preset.synthesis_enabled:
            print(f"    Synthesis: {preset.synthesis_strategy} via {preset.synthesis_model}")
        print()
    return 0


def cmd_traces(args: argparse.Namespace) -> int:
    """Handle the 'traces' subcommand."""
    from pathlib import Path
    from pss.trace import (
        list_traces,
        ExplorationTrace,
        generate_trace_summary,
        analyze_velocity_patterns,
        analyze_failure_effectiveness,
        analyze_signal_utility,
        analyze_spawn_confidence,
    )

    trace_dir = Path(args.dir)

    # Handle rating update
    if args.rate:
        if not args.rating:
            print("Error: --rate requires --rating value (1-5)", file=sys.stderr)
            return 1

        # Find the trace
        trace_path = None
        for p in list_traces(trace_dir):
            if args.rate in str(p):
                trace_path = p
                break

        if not trace_path:
            print(f"Error: No trace found matching '{args.rate}'", file=sys.stderr)
            return 1

        # Load, update, and save
        trace = ExplorationTrace.load(trace_path)
        trace.human_rating = args.rating
        if args.notes:
            trace.human_notes = args.notes
        trace.save(trace_path)
        print(f"Updated trace {trace.id} with rating {args.rating}/5")
        return 0

    # Handle detailed analysis
    if args.analyze:
        # Find the trace
        trace_path = None
        for p in list_traces(trace_dir):
            if args.analyze in str(p):
                trace_path = p
                break

        if not trace_path:
            print(f"Error: No trace found matching '{args.analyze}'", file=sys.stderr)
            return 1

        trace = ExplorationTrace.load(trace_path)

        print("=" * 60)
        print("TRACE ANALYSIS")
        print("=" * 60)
        print(generate_trace_summary(trace))
        print()

        # Velocity analysis
        print("-" * 40)
        print("Velocity Patterns")
        print("-" * 40)
        velocity = analyze_velocity_patterns(trace)
        print(f"  Branches with velocity data: {velocity['branches_with_velocity']}")
        print(f"  Avg final velocity: {velocity['avg_final_velocity']:.3f}")
        if velocity['successful_branch_avg_velocity'] > 0:
            print(f"  Successful branches avg: {velocity['successful_branch_avg_velocity']:.3f}")
        if velocity['stuck_branch_avg_velocity'] > 0:
            print(f"  Stuck branches avg: {velocity['stuck_branch_avg_velocity']:.3f}")
        print()

        # Failure analysis
        print("-" * 40)
        print("Failure Propagation")
        print("-" * 40)
        failures = analyze_failure_effectiveness(trace)
        print(f"  Total failures extracted: {failures['total_failures_extracted']}")
        if failures['failures_by_type']:
            print("  By type:")
            for ftype, count in failures['failures_by_type'].items():
                print(f"    {ftype}: {count}")
        print()

        # Signal analysis
        print("-" * 40)
        print("Signal Utility")
        print("-" * 40)
        signals = analyze_signal_utility(trace)
        print(f"  Total signals: {signals['total_signals']}")
        print(f"  Delivery rate: {signals['delivery_rate']:.1%}")
        if signals['signals_by_type']:
            print("  By type:")
            for stype, count in signals['signals_by_type'].items():
                print(f"    {stype}: {count}")
        print()

        # Spawn confidence
        print("-" * 40)
        print("Spawn Confidence Calibration")
        print("-" * 40)
        confidence = analyze_spawn_confidence(trace)
        print(f"  Branches with confidence: {confidence['branches_with_confidence']}")
        if confidence['successful_avg_confidence'] > 0:
            print(f"  Successful branches avg: {confidence['successful_avg_confidence']:.3f}")
        if confidence['stuck_avg_confidence'] > 0:
            print(f"  Stuck branches avg: {confidence['stuck_avg_confidence']:.3f}")
        print()

        return 0

    # List traces
    traces = list_traces(trace_dir)
    if not traces:
        print(f"No traces found in {trace_dir}/")
        print("Run with --trace to collect traces: pss run 'prompt' --trace")
        return 0

    print(f"Traces in {trace_dir}/ ({len(traces)} total):\n")
    print(f"{'ID':<14} {'Date':<20} {'Leaves':<7} {'Cost':<10} {'Rating':<8} {'Prompt':<30}")
    print("-" * 95)

    from datetime import datetime
    for path in traces[:args.limit]:
        try:
            trace = ExplorationTrace.load(path)
            ts = datetime.fromtimestamp(trace.timestamp)
            date_str = ts.strftime("%Y-%m-%d %H:%M")
            cost_str = f"${trace.total_usage['cost']:.4f}"
            rating_str = f"{trace.human_rating}/5" if trace.human_rating else "-"
            prompt_preview = trace.prompt[:28] + ".." if len(trace.prompt) > 30 else trace.prompt
            print(f"{trace.id:<14} {date_str:<20} {trace.num_leaves:<7} {cost_str:<10} {rating_str:<8} {prompt_preview:<30}")
        except Exception as e:
            print(f"Error loading {path.name}: {e}")

    if len(traces) > args.limit:
        print(f"\n... and {len(traces) - args.limit} more (use --limit to show more)")

    print()
    print("Commands:")
    print("  pss traces --analyze <ID>         Detailed analysis of a trace")
    print("  pss traces --rate <ID> --rating 4 Add a human rating (1-5)")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Handle the 'eval' subcommand."""
    from pss.eval import (
        EvalResult,
        evaluate_outputs,
        pass_at_k_from_results,
        self_bleu,
        distinct_n,
        format_eval_result,
        compare_results,
        leaves_to_outputs,
    )
    from pss.benchmarks.game_of_24 import (
        GAME_OF_24_PROBLEMS,
        format_game24_prompt,
        verify_24_solution,
        get_problems_by_difficulty as get_game24_by_difficulty,
    )
    from pss.benchmarks.brainstorm import (
        BRAINSTORM_PROBLEMS,
        format_brainstorm_prompt,
        evaluate_brainstorm_diversity,
        extract_ideas,
        get_problems_by_category,
    )
    from pss.benchmarks.bugfind import (
        BUGFIND_PROBLEMS,
        format_bugfind_prompt,
        evaluate_bugfind,
        check_bug_found,
        get_problems_by_difficulty as get_bugfind_by_difficulty,
    )

    task = args.task
    runs = getattr(args, "runs", 3)
    difficulty = getattr(args, "difficulty", None)
    dry_run = getattr(args, "dry_run", False)
    verbose = getattr(args, "verbose", False)

    # Load config
    config = load_config(args.config)

    # Apply preset if specified
    if hasattr(args, "preset") and args.preset:
        preset = get_preset(args.preset)
        if preset:
            apply_preset_to_config(preset, config)
            print(f"[Eval] Using preset: {args.preset}")

    apply_cli_overrides(config, args)

    if dry_run:
        print("[Eval] Dry run mode - showing benchmark info only")
        print()

    tasks_to_run = []
    if task == "all":
        tasks_to_run = ["game_of_24", "gsm8k", "humaneval", "brainstorm", "bugfind"]
    else:
        tasks_to_run = [task]

    all_results = []

    for current_task in tasks_to_run:
        print("=" * 60)
        print(f"BENCHMARK: {current_task}")
        print("=" * 60)

        if current_task == "game_of_24":
            results = _run_game24_eval(
                config, args, runs, difficulty, dry_run, verbose
            )
            all_results.extend(results)

        elif current_task == "brainstorm":
            results = _run_brainstorm_eval(
                config, args, runs, difficulty, dry_run, verbose
            )
            all_results.extend(results)

        elif current_task == "bugfind":
            results = _run_bugfind_eval(
                config, args, runs, difficulty, dry_run, verbose
            )
            all_results.extend(results)

        elif current_task == "gsm8k":
            results = _run_gsm8k_eval(
                config, args, runs, difficulty, dry_run, verbose
            )
            all_results.extend(results)

        elif current_task == "humaneval":
            results = _run_humaneval_eval(
                config, args, runs, difficulty, dry_run, verbose
            )
            all_results.extend(results)

        print()

    # Summary
    if all_results and not dry_run:
        print("=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        for result in all_results:
            print(format_eval_result(result))
            print()

    return 0


def _run_game24_eval(
    config: PSSConfig,
    args: argparse.Namespace,
    runs: int,
    difficulty: str | None,
    dry_run: bool,
    verbose: bool,
) -> list:
    """Run Game of 24 evaluation."""
    from pss.benchmarks.game_of_24 import (
        GAME_OF_24_PROBLEMS,
        format_game24_prompt,
        verify_24_solution,
        get_problems_by_difficulty,
    )
    from pss.eval import evaluate_outputs, EvalResult

    # Show current gate settings
    print(f"[Eval] Gates: soft={config.soft_gate_tokens}, hard={config.hard_gate_tokens}")

    # Filter problems
    if difficulty:
        problems = get_problems_by_difficulty(difficulty)
    else:
        # Default: use easy and medium for quick eval
        problems = [p for p in GAME_OF_24_PROBLEMS if p.difficulty in ("easy", "medium")]

    print(f"Problems: {len(problems)}")
    for p in problems[:5]:  # Show first 5
        print(f"  - {p.numbers} ({p.difficulty})")
    if len(problems) > 5:
        print(f"  ... and {len(problems) - 5} more")

    if dry_run:
        return []

    # Run evaluation
    results = []
    provider = create_provider(config.provider, config.model)

    for i, problem in enumerate(problems):
        print(f"\n[{i+1}/{len(problems)}] Problem: {problem.numbers}")
        prompt = format_game24_prompt(problem)

        # Run PSS
        pss_outputs = []
        pss_cost = 0.0
        pss_tokens = 0

        for run in range(runs):
            print(f"  Run {run+1}/{runs}...")
            try:
                result = run_pss(
                    initial_prompt=prompt,
                    config=config,
                    provider=provider,
                )
                if isinstance(result, PSSResult):
                    leaves = result.leaves
                else:
                    leaves = result

                # Guard against None or empty leaves
                if not leaves:
                    print(f"    No leaves returned")
                    continue

                for leaf in leaves:
                    text = leaf.output or ""
                    if not text:
                        for msg in reversed(leaf.messages):
                            if msg.get("role") == "assistant":
                                text = msg.get("content", "")
                                break
                    # Only add non-empty outputs
                    if text:
                        pss_outputs.append(text)
                    pss_cost += leaf.usage.cost
                    pss_tokens += leaf.usage.total_tokens
            except Exception as e:
                import traceback
                print(f"    Error: {e}")
                traceback.print_exc()

        if pss_outputs:
            # Verify each output
            verifier = lambda output: verify_24_solution(output, problem)
            pss_result = evaluate_outputs(
                pss_outputs,
                verifier,
                task=f"game_of_24_{problem.difficulty}",
                method="pss",
                total_cost=pss_cost,
                total_tokens=pss_tokens,
            )
            results.append(pss_result)

            if verbose:
                print(f"  PSS Pass@1: {pss_result.pass_at_1:.2%}, Pass@{pss_result.num_samples}: {pss_result.pass_at_k:.2%}")

        # Run baseline comparison if requested
        compare_n = getattr(args, "compare", None)
        if compare_n and compare_n >= 1:
            print(f"  Running {compare_n} baseline comparisons...")
            baseline_outputs = []
            baseline_cost = 0.0
            baseline_tokens = 0

            for j in range(compare_n):
                try:
                    messages = [{"role": "user", "content": prompt}]
                    text, _, tokens_used, usage = provider.chat(messages)
                    if text:
                        baseline_outputs.append(text)
                    baseline_cost += usage.cost
                    baseline_tokens += tokens_used
                except Exception as e:
                    print(f"    Baseline {j+1} error: {e}")

            if baseline_outputs:
                baseline_result = evaluate_outputs(
                    baseline_outputs,
                    verifier,
                    task=f"game_of_24_{problem.difficulty}",
                    method="baseline",
                    total_cost=baseline_cost,
                    total_tokens=baseline_tokens,
                )
                results.append(baseline_result)

                if verbose:
                    print(f"  Baseline Pass@1: {baseline_result.pass_at_1:.2%}, Pass@{baseline_result.num_samples}: {baseline_result.pass_at_k:.2%}")

                # Show comparison
                if pss_outputs:
                    pss_pk = pss_result.pass_at_k
                    baseline_pk = baseline_result.pass_at_k
                    if baseline_pk > 0:
                        gain = ((pss_pk - baseline_pk) / baseline_pk) * 100
                        print(f"  Comparison: PSS {pss_pk:.0%} vs Baseline {baseline_pk:.0%} ({'+' if gain > 0 else ''}{gain:.0f}%)")
                    else:
                        print(f"  Comparison: PSS {pss_pk:.0%} vs Baseline {baseline_pk:.0%}")

    return results


def _run_brainstorm_eval(
    config: PSSConfig,
    args: argparse.Namespace,
    runs: int,
    difficulty: str | None,
    dry_run: bool,
    verbose: bool,
) -> list:
    """Run brainstorm evaluation."""
    from pss.benchmarks.brainstorm import (
        BRAINSTORM_PROBLEMS,
        format_brainstorm_prompt,
        evaluate_brainstorm_diversity,
        get_problems_by_category,
    )
    from pss.eval import EvalResult, self_bleu, distinct_n

    # Filter problems by category (using difficulty as category)
    if difficulty:
        problems = get_problems_by_category(difficulty)
    else:
        # Default: use 2 from each category
        problems = BRAINSTORM_PROBLEMS[:4]

    print(f"Problems: {len(problems)}")
    for p in problems:
        print(f"  - {p.topic} ({p.category})")

    if dry_run:
        return []

    # Run evaluation
    results = []
    provider = create_provider(config.provider, config.model)

    for i, problem in enumerate(problems):
        print(f"\n[{i+1}/{len(problems)}] Topic: {problem.topic}")
        prompt = format_brainstorm_prompt(problem)

        # Run PSS
        pss_outputs = []
        pss_cost = 0.0
        pss_tokens = 0

        for run in range(runs):
            print(f"  Run {run+1}/{runs}...")
            try:
                result = run_pss(
                    initial_prompt=prompt,
                    config=config,
                    provider=provider,
                )
                if isinstance(result, PSSResult):
                    leaves = result.leaves
                else:
                    leaves = result

                # Guard against None or empty leaves
                if not leaves:
                    print(f"    No leaves returned")
                    continue

                for leaf in leaves:
                    text = leaf.output or ""
                    if not text:
                        for msg in reversed(leaf.messages):
                            if msg.get("role") == "assistant":
                                text = msg.get("content", "")
                                break
                    pss_outputs.append(text)
                    pss_cost += leaf.usage.cost
                    pss_tokens += leaf.usage.total_tokens
            except Exception as e:
                print(f"    Error: {e}")

        if pss_outputs:
            # Evaluate diversity
            diversity_metrics = evaluate_brainstorm_diversity(pss_outputs, problem)

            # Create result
            result = EvalResult(
                task=f"brainstorm_{problem.topic}",
                method="pss",
                num_samples=len(pss_outputs),
                num_correct=1 if diversity_metrics["meets_minimum"] else 0,
                pass_at_1=1.0 if diversity_metrics["meets_minimum"] else 0.0,
                pass_at_k=1.0 if diversity_metrics["meets_minimum"] else 0.0,
                self_bleu=self_bleu(pss_outputs),
                distinct_1=distinct_n(pss_outputs, 1),
                distinct_2=distinct_n(pss_outputs, 2),
                total_cost=pss_cost,
                total_tokens=pss_tokens,
                metadata=diversity_metrics,
            )
            results.append(result)

            if verbose:
                print(f"  Unique ideas: {diversity_metrics['unique_ideas']}")
                print(f"  Self-BLEU: {result.self_bleu:.2f}")

    return results


def _run_bugfind_eval(
    config: PSSConfig,
    args: argparse.Namespace,
    runs: int,
    difficulty: str | None,
    dry_run: bool,
    verbose: bool,
) -> list:
    """Run bug finding evaluation."""
    from pss.benchmarks.bugfind import (
        BUGFIND_PROBLEMS,
        format_bugfind_prompt,
        evaluate_bugfind,
        get_problems_by_difficulty,
    )
    from pss.eval import EvalResult

    # Filter problems
    if difficulty:
        problems = get_problems_by_difficulty(difficulty)
    else:
        problems = BUGFIND_PROBLEMS

    print(f"Problems: {len(problems)}")
    for p in problems:
        print(f"  - {p.name}: {len(p.bugs)} bugs ({p.difficulty})")

    if dry_run:
        return []

    # Run evaluation
    results = []
    provider = create_provider(config.provider, config.model)

    for i, problem in enumerate(problems):
        print(f"\n[{i+1}/{len(problems)}] Problem: {problem.name}")
        prompt = format_bugfind_prompt(problem)

        # Run PSS
        pss_outputs = []
        pss_cost = 0.0
        pss_tokens = 0

        for run in range(runs):
            print(f"  Run {run+1}/{runs}...")
            try:
                result = run_pss(
                    initial_prompt=prompt,
                    config=config,
                    provider=provider,
                )
                if isinstance(result, PSSResult):
                    leaves = result.leaves
                else:
                    leaves = result

                # Guard against None or empty leaves
                if not leaves:
                    print(f"    No leaves returned")
                    continue

                for leaf in leaves:
                    text = leaf.output or ""
                    if not text:
                        for msg in reversed(leaf.messages):
                            if msg.get("role") == "assistant":
                                text = msg.get("content", "")
                                break
                    pss_outputs.append(text)
                    pss_cost += leaf.usage.cost
                    pss_tokens += leaf.usage.total_tokens
            except Exception as e:
                print(f"    Error: {e}")

        if pss_outputs:
            # Evaluate bug detection
            bugfind_metrics = evaluate_bugfind(pss_outputs, problem)

            # Create result
            result = EvalResult(
                task=f"bugfind_{problem.name}",
                method="pss",
                num_samples=len(pss_outputs),
                num_correct=bugfind_metrics["bugs_found"],
                pass_at_1=bugfind_metrics["detection_rate"],
                pass_at_k=1.0 if bugfind_metrics["any_output_found_all"] else bugfind_metrics["detection_rate"],
                total_cost=pss_cost,
                total_tokens=pss_tokens,
                metadata=bugfind_metrics,
            )
            results.append(result)

            if verbose:
                print(f"  Detection rate: {bugfind_metrics['detection_rate']:.2%}")
                print(f"  Bugs found: {bugfind_metrics['bugs_found']}/{bugfind_metrics['total_bugs']}")

    return results


def _run_gsm8k_eval(
    config: PSSConfig,
    args: argparse.Namespace,
    runs: int,
    difficulty: str | None,
    dry_run: bool,
    verbose: bool,
) -> list:
    """Run GSM8K (math word problems) evaluation."""
    from pss.benchmarks.gsm8k import (
        GSM8K_PROBLEMS,
        format_gsm8k_prompt,
        verify_gsm8k_solution,
        get_problems_by_difficulty,
        estimate_cost,
    )
    from pss.eval import evaluate_outputs, EvalResult

    # Show current gate settings
    print(f"[Eval] Gates: soft={config.soft_gate_tokens}, hard={config.hard_gate_tokens}")

    # Filter problems
    if difficulty:
        problems = get_problems_by_difficulty(difficulty)
    else:
        # Default: use easy and medium for quick eval
        problems = [p for p in GSM8K_PROBLEMS if p.difficulty in ("easy", "medium")]

    print(f"Problems: {len(problems)}")
    for p in problems[:5]:
        print(f"  - {p.question[:60]}... ({p.difficulty})")
    if len(problems) > 5:
        print(f"  ... and {len(problems) - 5} more")

    # Show cost estimate
    cost_est = estimate_cost(len(problems))
    print(f"Estimated cost: ${cost_est['total_cost']:.4f}")

    if dry_run:
        return []

    # Run evaluation
    results = []
    provider = create_provider(config.provider, config.model)

    for i, problem in enumerate(problems):
        print(f"\n[{i+1}/{len(problems)}] Answer: {problem.answer}")
        prompt = format_gsm8k_prompt(problem)

        # Run PSS
        pss_outputs = []
        pss_cost = 0.0
        pss_tokens = 0

        for run in range(runs):
            print(f"  Run {run+1}/{runs}...")
            try:
                result = run_pss(
                    initial_prompt=prompt,
                    config=config,
                    provider=provider,
                )
                if isinstance(result, PSSResult):
                    leaves = result.leaves
                else:
                    leaves = result

                # Guard against None or empty leaves
                if not leaves:
                    print(f"    No leaves returned")
                    continue

                for leaf in leaves:
                    text = leaf.output or ""
                    if not text:
                        for msg in reversed(leaf.messages):
                            if msg.get("role") == "assistant":
                                text = msg.get("content", "")
                                break
                    if text:
                        pss_outputs.append(text)
                    pss_cost += leaf.usage.cost
                    pss_tokens += leaf.usage.total_tokens
            except Exception as e:
                print(f"    Error: {e}")

        if pss_outputs:
            verifier = lambda output, p=problem: verify_gsm8k_solution(output, p)
            pss_result = evaluate_outputs(
                pss_outputs,
                verifier,
                task=f"gsm8k_{problem.difficulty}",
                method="pss",
                total_cost=pss_cost,
                total_tokens=pss_tokens,
            )
            results.append(pss_result)

            if verbose:
                print(f"  PSS Pass@1: {pss_result.pass_at_1:.2%}, Pass@{pss_result.num_samples}: {pss_result.pass_at_k:.2%}")

        # Baseline comparison
        compare_n = getattr(args, "compare", None)
        if compare_n and compare_n >= 1:
            print(f"  Running {compare_n} baseline comparisons...")
            baseline_outputs = []
            baseline_cost = 0.0
            baseline_tokens = 0

            for j in range(compare_n):
                try:
                    messages = [{"role": "user", "content": prompt}]
                    text, _, tokens_used, usage = provider.chat(messages)
                    if text:
                        baseline_outputs.append(text)
                    baseline_cost += usage.cost
                    baseline_tokens += tokens_used
                except Exception as e:
                    print(f"    Baseline {j+1} error: {e}")

            if baseline_outputs:
                baseline_result = evaluate_outputs(
                    baseline_outputs,
                    verifier,
                    task=f"gsm8k_{problem.difficulty}",
                    method="baseline",
                    total_cost=baseline_cost,
                    total_tokens=baseline_tokens,
                )
                results.append(baseline_result)

                if pss_outputs and verbose:
                    pss_pk = pss_result.pass_at_k
                    baseline_pk = baseline_result.pass_at_k
                    if baseline_pk > 0:
                        gain = ((pss_pk - baseline_pk) / baseline_pk) * 100
                        print(f"  PSS {pss_pk:.0%} vs Baseline {baseline_pk:.0%} ({'+' if gain > 0 else ''}{gain:.0f}%)")

    return results


def _run_humaneval_eval(
    config: PSSConfig,
    args: argparse.Namespace,
    runs: int,
    difficulty: str | None,
    dry_run: bool,
    verbose: bool,
) -> list:
    """Run HumanEval (code generation) evaluation."""
    from pss.benchmarks.humaneval import (
        HUMANEVAL_PROBLEMS,
        format_humaneval_prompt,
        verify_humaneval_solution,
        get_problems_by_difficulty,
        estimate_cost,
    )
    from pss.eval import evaluate_outputs, EvalResult

    # Show current gate settings
    print(f"[Eval] Gates: soft={config.soft_gate_tokens}, hard={config.hard_gate_tokens}")

    # Filter problems
    if difficulty:
        problems = get_problems_by_difficulty(difficulty)
    else:
        # Default: use easy and medium
        problems = [p for p in HUMANEVAL_PROBLEMS if p.difficulty in ("easy", "medium")]

    print(f"Problems: {len(problems)}")
    for p in problems[:5]:
        print(f"  - {p.task_id}: {p.entry_point}() ({p.difficulty})")
    if len(problems) > 5:
        print(f"  ... and {len(problems) - 5} more")

    # Show cost estimate
    cost_est = estimate_cost(len(problems))
    print(f"Estimated cost: ${cost_est['total_cost']:.4f}")

    if dry_run:
        return []

    # Run evaluation
    results = []
    provider = create_provider(config.provider, config.model)

    for i, problem in enumerate(problems):
        print(f"\n[{i+1}/{len(problems)}] {problem.task_id}: {problem.entry_point}()")
        prompt = format_humaneval_prompt(problem)

        # Run PSS
        pss_outputs = []
        pss_cost = 0.0
        pss_tokens = 0

        for run in range(runs):
            print(f"  Run {run+1}/{runs}...")
            try:
                result = run_pss(
                    initial_prompt=prompt,
                    config=config,
                    provider=provider,
                )
                if isinstance(result, PSSResult):
                    leaves = result.leaves
                else:
                    leaves = result

                # Guard against None or empty leaves
                if not leaves:
                    print(f"    No leaves returned")
                    continue

                for leaf in leaves:
                    text = leaf.output or ""
                    if not text:
                        for msg in reversed(leaf.messages):
                            if msg.get("role") == "assistant":
                                text = msg.get("content", "")
                                break
                    if text:
                        pss_outputs.append(text)
                    pss_cost += leaf.usage.cost
                    pss_tokens += leaf.usage.total_tokens
            except Exception as e:
                print(f"    Error: {e}")

        if pss_outputs:
            verifier = lambda output, p=problem: verify_humaneval_solution(output, p)
            pss_result = evaluate_outputs(
                pss_outputs,
                verifier,
                task=f"humaneval_{problem.task_id}",
                method="pss",
                total_cost=pss_cost,
                total_tokens=pss_tokens,
            )
            results.append(pss_result)

            if verbose:
                print(f"  PSS Pass@1: {pss_result.pass_at_1:.2%}, Pass@{pss_result.num_samples}: {pss_result.pass_at_k:.2%}")

        # Baseline comparison
        compare_n = getattr(args, "compare", None)
        if compare_n and compare_n >= 1:
            print(f"  Running {compare_n} baseline comparisons...")
            baseline_outputs = []
            baseline_cost = 0.0
            baseline_tokens = 0

            for j in range(compare_n):
                try:
                    messages = [{"role": "user", "content": prompt}]
                    text, _, tokens_used, usage = provider.chat(messages)
                    if text:
                        baseline_outputs.append(text)
                    baseline_cost += usage.cost
                    baseline_tokens += tokens_used
                except Exception as e:
                    print(f"    Baseline {j+1} error: {e}")

            if baseline_outputs:
                baseline_result = evaluate_outputs(
                    baseline_outputs,
                    verifier,
                    task=f"humaneval_{problem.task_id}",
                    method="baseline",
                    total_cost=baseline_cost,
                    total_tokens=baseline_tokens,
                )
                results.append(baseline_result)

                if pss_outputs and verbose:
                    pss_pk = pss_result.pass_at_k
                    baseline_pk = baseline_result.pass_at_k
                    if baseline_pk > 0:
                        gain = ((pss_pk - baseline_pk) / baseline_pk) * 100
                        print(f"  PSS {pss_pk:.0%} vs Baseline {baseline_pk:.0%} ({'+' if gain > 0 else ''}{gain:.0f}%)")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pepe Silvia Search - Forward-only branching exploration for LLM agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pss "What are the implications of X?"          Run with default settings
  pss explore "Map out approaches to Y"          Wide, shallow exploration
  pss write "A story about a lighthouse keeper"  Creative writing mode
  pss research "How does Z work?"                Deep, thorough analysis
  pss reasoning "Solve: 3 + 4 * 5 = ?"           Math/logic with early gates
  pss investigate "Find bugs in src/"            Agentic code investigation
  pss debug "Test failures in auth module"       Parallel bug hunting
  pss interactive "Complex problem"              Live tree view with controls
  pss web                                        Web UI for phone access
  pss --preset explore "topic"                   Explicit preset selection
  pss --agentic --synthesis "Analyze codebase"   Enable agentic + synthesis
  pss presets                                    List available presets
  pss eval --task game_of_24                     Run Game of 24 benchmark
  pss eval --task all --dry-run                  Show all benchmarks info
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # 'explore' subcommand
    explore_parser = subparsers.add_parser(
        "explore",
        help="Map out a topic space (cheap, wide, shallow)",
    )
    explore_parser.add_argument("prompt", help="Topic to explore")
    add_common_args(explore_parser)
    explore_parser.set_defaults(func=cmd_explore)

    # 'write' subcommand
    write_parser = subparsers.add_parser(
        "write",
        help="Creative/professional writing (moderate depth)",
    )
    write_parser.add_argument("prompt", help="Writing prompt")
    add_common_args(write_parser)
    write_parser.set_defaults(func=cmd_write)

    # 'research' subcommand
    research_parser = subparsers.add_parser(
        "research",
        help="Deep analysis (thorough, careful)",
    )
    research_parser.add_argument("prompt", help="Research question")
    add_common_args(research_parser)
    research_parser.set_defaults(func=cmd_research)

    # 'draft' subcommand
    draft_parser = subparsers.add_parser(
        "draft",
        help="Polished outputs (focused, low branching)",
    )
    draft_parser.add_argument("prompt", help="Draft prompt")
    add_common_args(draft_parser)
    draft_parser.set_defaults(func=cmd_draft)

    # v0.4: 'investigate' subcommand - agentic code investigation
    investigate_parser = subparsers.add_parser(
        "investigate",
        help="Agentic code investigation with synthesis (v0.4)",
    )
    investigate_parser.add_argument("prompt", help="Investigation prompt")
    add_common_args(investigate_parser)
    investigate_parser.set_defaults(func=cmd_investigate)

    # v0.4: 'debug' subcommand - parallel bug hunting
    debug_parser = subparsers.add_parser(
        "debug",
        help="Parallel bug hunting with evidence voting (v0.4)",
    )
    debug_parser.add_argument("prompt", help="Bug description or area to investigate")
    add_common_args(debug_parser)
    debug_parser.set_defaults(func=cmd_debug)

    # v0.6: 'reasoning' subcommand - math/logic problems
    reasoning_parser = subparsers.add_parser(
        "reasoning",
        help="Mathematical and logical reasoning (early checkpoints)",
    )
    reasoning_parser.add_argument("prompt", help="Math or logic problem")
    add_common_args(reasoning_parser)
    reasoning_parser.set_defaults(func=cmd_reasoning)

    # v0.8: 'interactive' subcommand - human-in-the-loop control
    interactive_parser = subparsers.add_parser(
        "interactive",
        help="Interactive mode with live tree view and branch control (v0.8)",
    )
    interactive_parser.add_argument("prompt", help="Initial prompt to explore")
    interactive_parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        help="Use a preset configuration",
    )
    add_common_args(interactive_parser)
    interactive_parser.set_defaults(func=cmd_interactive)

    # v0.9: 'web' subcommand - web UI server
    web_parser = subparsers.add_parser(
        "web",
        help="Start the loopback-only exploration workbench",
    )
    web_parser.add_argument(
        "--host",
        default="127.0.0.1",
        choices=["127.0.0.1", "localhost", "::1"],
        help="Loopback host (default: 127.0.0.1)",
    )
    web_parser.add_argument(
        "--port",
        type=int,
        default=4311,
        help="Port to bind to (default: 4311)",
    )
    web_parser.set_defaults(func=cmd_web)

    # 'presets' subcommand - list available presets
    presets_parser = subparsers.add_parser(
        "presets",
        help="List available presets",
    )
    presets_parser.set_defaults(func=cmd_presets)

    # 'eval' subcommand - run benchmark evaluations
    eval_parser = subparsers.add_parser(
        "eval",
        help="Run benchmark evaluations",
    )
    eval_parser.add_argument(
        "--task",
        choices=EVAL_TASKS,
        default="all",
        help="Benchmark task to run (default: all)",
    )
    eval_parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of PSS runs per problem (default: 1)",
    )
    eval_parser.add_argument(
        "--difficulty",
        help="Filter problems by difficulty (task-specific)",
    )
    eval_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show benchmark info without running",
    )
    eval_parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        help="Use a preset configuration",
    )
    add_common_args(eval_parser)
    eval_parser.set_defaults(func=cmd_eval)

    # 'run' subcommand (explicit version of default)
    run_parser = subparsers.add_parser(
        "run",
        help="Run PSS with a prompt (same as bare invocation)",
    )
    run_parser.add_argument("prompt", help="Initial prompt")
    run_parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        help="Use a preset configuration",
    )
    add_common_args(run_parser)
    run_parser.set_defaults(func=cmd_run)

    # 'synthesize' subcommand - standalone synthesis of text files
    synthesize_parser = subparsers.add_parser(
        "synthesize",
        help="Synthesize N text files using a strong model (no branching)",
    )
    synthesize_parser.add_argument(
        "files",
        nargs="+",
        help="Input text files to synthesize",
    )
    synthesize_parser.add_argument(
        "--prompt",
        help="Context prompt for synthesis (default: generic)",
    )
    # --synthesis-strategy and --synthesis-model are provided by add_common_args
    add_common_args(synthesize_parser)
    synthesize_parser.set_defaults(func=cmd_synthesize)

    # 'traces' subcommand - list and analyze saved traces
    traces_parser = subparsers.add_parser(
        "traces",
        help="List and analyze saved exploration traces (v1.1)",
    )
    traces_parser.add_argument(
        "--dir",
        default="traces",
        help="Trace directory (default: traces/)",
    )
    traces_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max traces to show (default: 20)",
    )
    traces_parser.add_argument(
        "--analyze",
        metavar="TRACE_ID",
        help="Analyze a specific trace by ID or filename",
    )
    traces_parser.add_argument(
        "--rate",
        metavar="TRACE_ID",
        help="Add a human rating to a trace",
    )
    traces_parser.add_argument(
        "--rating",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Rating value (1-5) when using --rate",
    )
    traces_parser.add_argument(
        "--notes",
        help="Notes to add when using --rate",
    )
    traces_parser.set_defaults(func=cmd_traces)

    # For backwards compatibility: if first arg isn't a subcommand, treat as prompt
    # We do this by checking sys.argv manually
    import sys as _sys

    known_commands = {"explore", "write", "research", "draft", "investigate", "debug", "reasoning", "interactive", "web", "presets", "eval", "run", "traces", "synthesize", "-h", "--help"}

    # If there are args and the first non-flag arg isn't a known command, insert "run"
    if len(_sys.argv) > 1:
        first_arg = _sys.argv[1]
        if first_arg not in known_commands and not first_arg.startswith("-"):
            # Insert "run" as the command
            _sys.argv.insert(1, "run")
        elif first_arg.startswith("--preset"):
            # Handle --preset flag before prompt
            _sys.argv.insert(1, "run")
        elif first_arg == "-p" or first_arg == "-v":
            # Short flags that might come before prompt
            # Find the first non-flag arg
            for i, arg in enumerate(_sys.argv[1:], 1):
                if not arg.startswith("-") and arg not in known_commands:
                    _sys.argv.insert(1, "run")
                    break

    args = parser.parse_args()

    # Handle subcommands
    if hasattr(args, "func"):
        return args.func(args)

    # No command specified - show help
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
