"""Synthesis phase for PSS v0.4.

This module implements the synthesis phase that merges findings from
multiple exploration branches into a unified output.

The key insight: use cheap models for parallel exploration, then an
expensive model for synthesis. This externalizes the search that
deep reasoning does internally.
"""

from dataclasses import dataclass
from typing import Any, Literal

from pss.config import PSSConfig
from pss.providers import Provider, create_provider
from pss.types import Context, SynthesisResult, TokenUsage, ToolTrace


@dataclass
class BranchSummary:
    """Summary of one branch for synthesis."""

    branch_id: str
    branch_reason: str | None
    tool_traces: list[ToolTrace]
    final_output: str | None
    confidence: float | None
    tool_summary: str  # Human-readable summary of tool usage


def _summarize_tool_traces(traces: list[ToolTrace], max_chars: int = 2000) -> str:
    """Create a human-readable summary of tool traces."""
    if not traces:
        return "(no tools used)"

    lines = []
    char_count = 0

    for trace in traces:
        # Format the tool call
        args_str = ", ".join(f"{k}={repr(v)[:50]}" for k, v in trace.arguments.items())
        status = "OK" if trace.success else "FAILED"

        # Truncate result for summary
        result_preview = trace.result[:200].replace("\n", " ")
        if len(trace.result) > 200:
            result_preview += "..."

        line = f"- {trace.tool_name}({args_str}) [{status}]: {result_preview}"

        if char_count + len(line) > max_chars:
            lines.append(f"... and {len(traces) - len(lines)} more tool calls")
            break

        lines.append(line)
        char_count += len(line)

    return "\n".join(lines)


def _build_branch_summary(ctx: Context) -> BranchSummary:
    """Build a summary of a branch for synthesis."""
    return BranchSummary(
        branch_id=ctx.id,
        branch_reason=ctx.branch_reason,
        tool_traces=ctx.tool_traces,
        final_output=ctx.output,
        confidence=None,  # Could extract from gate decision if preserved
        tool_summary=_summarize_tool_traces(ctx.tool_traces),
    )


def _extract_cross_branch_evidence(summaries: list[BranchSummary]) -> dict[str, list[str]]:
    """
    Analyze branch outputs to find common findings.

    This is a simple keyword-based approach. A more sophisticated version
    could use embeddings to cluster similar findings.

    Returns:
        Dict of {finding_keyword: [branch_ids that mentioned it]}
    """
    # Simple approach: look for common substrings in outputs
    # More sophisticated: use embeddings to cluster
    evidence: dict[str, list[str]] = {}

    for summary in summaries:
        if not summary.final_output:
            continue

        output_lower = summary.final_output.lower()

        # Look for common patterns
        patterns = [
            ("bug", "found bug"),
            ("error", "found error"),
            ("issue", "found issue"),
            ("fix", "proposed fix"),
            ("correct", "verified correct"),
            ("test", "ran tests"),
            ("fail", "test failed"),
            ("pass", "test passed"),
        ]

        for keyword, finding in patterns:
            if keyword in output_lower:
                if finding not in evidence:
                    evidence[finding] = []
                if summary.branch_id not in evidence[finding]:
                    evidence[finding].append(summary.branch_id)

    return evidence


def _format_evidence_summary(evidence: dict[str, list[str]], total_branches: int) -> str:
    """Format cross-branch evidence for the synthesis prompt."""
    if not evidence:
        return "(no cross-branch evidence patterns detected)"

    lines = []
    for finding, branch_ids in sorted(evidence.items(), key=lambda x: -len(x[1])):
        count = len(branch_ids)
        lines.append(f"- \"{finding}\": {count}/{total_branches} branches ({', '.join(branch_ids)})")

    return "\n".join(lines)


SYNTHESIS_STRATEGIES: dict[str, str] = {
    "merge": """Synthesize findings into a DECISION FRAMEWORK, not a summary.

1. IDENTIFY TENSIONS: Where do branches fundamentally contradict? Name them explicitly.
   "Branch X argues for A. Branch Y argues for B. These are mutually exclusive because..."

2. NAME THE CRUX: What underlying variable determines which branch is right?
   "The decision hinges on whether [specific condition]."

3. MAP EVIDENCE TO CRUX: What did each branch find that informs the crux?
   "Branch X found [evidence] suggesting [conclusion about crux]."
   "Branch Y found [evidence] suggesting [opposite conclusion]."

4. GIVE CONDITIONAL ADVICE: Don't hedge with "consider both." Give clear conditionals.
   "If [condition A], then [specific action] because [reasoning from branches]."
   "If [condition B], then [different action] because [reasoning from branches]."

DO NOT say "consider a mix of both" or "balance these factors." That's not synthesis.
Real synthesis identifies the CRUX that determines which path is correct.""",

    "vote": """Weight evidence by how many branches independently found each thing.
Findings supported by multiple branches are more reliable.
But note: unanimous agreement may mean all branches made the same mistake.
Dissenting branches that found contradictory evidence are especially valuable.""",

    "deliberate": """Reason through disagreements between branches as a structured debate.

For each disagreement:
1. State the positions clearly (Branch X says A, Branch Y says B)
2. Identify the crux - what fact would settle this?
3. Check if any branch found evidence bearing on the crux
4. Render a verdict with reasoning, or identify what information is missing

Do not split the difference. Take a position.""",

    "plan": """Produce an action plan based on branch findings.

Structure:
1. DECISION POINT: What's the key choice that determines the plan?
2. IF [condition from exploration]: Take path A
   - Step 1...
   - Step 2...
3. IF [opposite condition]: Take path B
   - Step 1...
   - Step 2...
4. FIRST ACTION regardless of path: [lowest-risk way to learn which condition applies]

Do not give a generic plan. Give a conditional plan that uses branch insights.""",
}


def _build_synthesis_prompt(
    original_prompt: str,
    summaries: list[BranchSummary],
    evidence: dict[str, list[str]],
    strategy: Literal["merge", "vote", "deliberate", "plan"],
) -> str:
    """Build the synthesis prompt."""
    branch_sections = []

    for i, summary in enumerate(summaries, 1):
        reason = f" (spawned: \"{summary.branch_reason}\")" if summary.branch_reason else ""
        section = f"""=== BRANCH {i}: "{summary.branch_id}"{reason} ===
Tool calls:
{summary.tool_summary}

Final output:
{summary.final_output or "(no output)"}
"""
        branch_sections.append(section)

    branches_text = "\n".join(branch_sections)
    evidence_text = _format_evidence_summary(evidence, len(summaries))
    strategy_instruction = SYNTHESIS_STRATEGIES.get(strategy, SYNTHESIS_STRATEGIES["merge"])

    return f"""You are synthesizing findings from {len(summaries)} parallel exploration branches.

Original task: {original_prompt}

{branches_text}

=== CROSS-BRANCH EVIDENCE ===
{evidence_text}

=== YOUR TASK ===
{strategy_instruction}

Your synthesis MUST:
1. Name specific tensions between branches (not "some branches found X")
2. Identify the CRUX - the key question that determines which branch is right
3. Give CONDITIONAL advice: "If X, do A. If Y, do B."
4. NOT hedge with "consider both" or "balance these factors"

The value of parallel exploration is seeing contradictions. Your job is to
make those contradictions useful by identifying what determines the answer."""


SYNTHESIS_TOOL = {
    "name": "synthesis_result",
    "description": "Submit the synthesis result.",
    "input_schema": {
        "type": "object",
        "properties": {
            "unified_output": {
                "type": "string",
                "description": "The synthesized output combining all branch findings",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence in the synthesis (0-1)",
            },
            "key_findings": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of key findings supported by evidence",
            },
            "disagreements": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Areas where branches disagreed",
            },
            "action_plan": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ordered list of actions to take (for plan strategy)",
            },
        },
        "required": ["unified_output", "confidence"],
    },
}


def _parse_synthesis_result(
    text: str,
    tool_calls: list[dict],
    usage: TokenUsage,
    evidence: dict[str, list[str]],
) -> SynthesisResult:
    """Parse the synthesis result from the model response."""
    # Try to parse from tool call first
    for call in tool_calls:
        if call.get("name") == "synthesis_result":
            args = call.get("arguments", {})

            # Handle action_plan - may be string or list
            action_plan = args.get("action_plan")
            if isinstance(action_plan, str):
                # Model returned a string, split into lines
                action_plan = [
                    line.strip().lstrip("-").strip()
                    for line in action_plan.split("\n")
                    if line.strip() and line.strip() not in ("-", "")
                ]

            # Handle disagreements - may be string or list
            disagreements = args.get("disagreements", [])
            if isinstance(disagreements, str):
                disagreements = [
                    line.strip().lstrip("-").strip()
                    for line in disagreements.split("\n")
                    if line.strip() and line.strip() not in ("-", "")
                ]

            # Ensure confidence is a float
            confidence = args.get("confidence", 0.5)
            if isinstance(confidence, str):
                try:
                    confidence = float(confidence)
                except ValueError:
                    confidence = 0.5

            return SynthesisResult(
                unified_output=args.get("unified_output", text),
                confidence=confidence,
                evidence_summary={k: len(v) for k, v in evidence.items()},
                dissenting_views=disagreements,
                action_plan=action_plan if action_plan else None,
                usage=usage,
            )

    # Fall back to text response
    return SynthesisResult(
        unified_output=text,
        confidence=0.5,
        evidence_summary={k: len(v) for k, v in evidence.items()},
        dissenting_views=[],
        action_plan=None,
        usage=usage,
    )


def get_synthesis_provider(config: PSSConfig) -> Provider:
    """Get the provider for synthesis (may be different from exploration)."""
    provider_name = config.synthesis_provider or config.provider
    model = config.synthesis_model or config.model

    return create_provider(provider_name, model)


def synthesize(
    leaves: list[Context],
    original_prompt: str,
    config: PSSConfig,
    provider: Provider | None = None,
) -> SynthesisResult:
    """
    Synthesize findings from multiple branches.

    This is the "expensive" phase - uses a strong model to merge
    the cheap parallel exploration.

    Args:
        leaves: Leaf contexts from exploration
        original_prompt: The original user prompt
        config: PSS configuration
        provider: Optional provider override (defaults to synthesis provider from config)

    Returns:
        SynthesisResult with unified output and metadata
    """
    if not leaves:
        return SynthesisResult(
            unified_output="No branches produced output.",
            confidence=0.0,
            evidence_summary={},
            dissenting_views=[],
            action_plan=None,
            usage=TokenUsage(),
        )

    # Get the synthesis provider
    if provider is None:
        provider = get_synthesis_provider(config)

    # Build branch summaries
    summaries = [_build_branch_summary(leaf) for leaf in leaves]

    # Extract cross-branch evidence
    evidence = _extract_cross_branch_evidence(summaries)

    # Build synthesis prompt
    synthesis_prompt = _build_synthesis_prompt(
        original_prompt,
        summaries,
        evidence,
        config.synthesis_strategy,
    )

    # Call the synthesis model
    messages = [{"role": "user", "content": synthesis_prompt}]

    text, tool_calls, tokens_used, usage = provider.chat(
        messages,
        tools=[SYNTHESIS_TOOL],
        tool_choice="auto",
    )

    # Parse result
    result = _parse_synthesis_result(text, tool_calls, usage, evidence)

    print(f"[PSS] Synthesis complete: {len(leaves)} branches -> unified output")
    print(f"[PSS] Synthesis confidence: {result.confidence:.2f}")
    if result.action_plan:
        print(f"[PSS] Action plan: {len(result.action_plan)} steps")

    return result


def _fast_similarity_check(leaves: list[Context]) -> float | None:
    """Quick embedding-based similarity check to gate expensive LLM consensus call.

    Returns mean pairwise similarity (0-1), or None if embeddings unavailable.
    High similarity (>0.7) suggests consensus worth checking with LLM.
    Low similarity (<0.4) means branches are divergent — skip the LLM call.
    """
    try:
        from pss.diversity import compute_diversity, get_leaf_text
    except ImportError:
        return None

    texts = [get_leaf_text(leaf) for leaf in leaves[:6]]
    if not all(texts):
        return None

    try:
        from pss.diversity import get_embeddings_openai, cosine_distance
        import numpy as np

        embeddings, _ = get_embeddings_openai(texts)
        arrays = [np.array(e) for e in embeddings]
        similarities = []
        for i in range(len(arrays)):
            for j in range(i + 1, len(arrays)):
                similarities.append(1.0 - cosine_distance(arrays[i], arrays[j]))
        return float(np.mean(similarities)) if similarities else None
    except Exception:
        return None


def check_incremental_consensus(
    leaves: list[Context],
    config: PSSConfig,
    provider: Provider | None = None,
) -> tuple[bool, float]:
    """Check if completed leaves show strong consensus (early stopping signal).

    Uses a fast embedding similarity heuristic to gate the expensive LLM call.
    Only runs the full LLM consensus check if embeddings suggest high similarity.

    Returns (should_stop, consensus_score).
    """
    if len(leaves) < 2:
        return False, 0.0

    # Fast heuristic: check embedding similarity first
    similarity = _fast_similarity_check(leaves)
    if similarity is not None:
        if similarity < 0.4:
            # Branches are clearly divergent — no consensus, skip LLM call
            return False, similarity
        # similarity >= 0.4: worth checking with LLM

    if provider is None:
        provider = get_synthesis_provider(config)

    # Build a lightweight summary of leaf outputs
    leaf_summaries = []
    for i, leaf in enumerate(leaves[:6], 1):  # Cap at 6 to keep prompt small
        output_preview = (leaf.output or "")[:500]
        leaf_summaries.append(f"Branch {i}: {output_preview}")

    summaries_text = "\n\n".join(leaf_summaries)

    consensus_prompt = f"""Given these {len(leaves)} exploration branch outputs, rate the consensus level.

{summaries_text}

Rate from 0.0 to 1.0:
- 0.0 = completely different conclusions, strong disagreement
- 0.5 = partial overlap but significant differences
- 1.0 = strong consensus, all branches converge on the same answer

Respond with ONLY a number between 0.0 and 1.0."""

    try:
        text, _, _, _ = provider.chat(
            [{"role": "user", "content": consensus_prompt}],
        )
        # Parse the score
        score = float(text.strip().split()[0])
        score = max(0.0, min(1.0, score))
        return score >= config.incremental_consensus_threshold, score
    except Exception:
        return False, 0.0


async def synthesize_async(
    leaves: list[Context],
    original_prompt: str,
    config: PSSConfig,
    provider: Provider | None = None,
) -> SynthesisResult:
    """
    Async version of synthesize.

    Args:
        leaves: Leaf contexts from exploration
        original_prompt: The original user prompt
        config: PSS configuration
        provider: Optional provider override

    Returns:
        SynthesisResult with unified output and metadata
    """
    if not leaves:
        return SynthesisResult(
            unified_output="No branches produced output.",
            confidence=0.0,
            evidence_summary={},
            dissenting_views=[],
            action_plan=None,
            usage=TokenUsage(),
        )

    # Get the synthesis provider
    if provider is None:
        provider = get_synthesis_provider(config)

    # Build branch summaries
    summaries = [_build_branch_summary(leaf) for leaf in leaves]

    # Extract cross-branch evidence
    evidence = _extract_cross_branch_evidence(summaries)

    # Build synthesis prompt
    synthesis_prompt = _build_synthesis_prompt(
        original_prompt,
        summaries,
        evidence,
        config.synthesis_strategy,
    )

    # Call the synthesis model
    messages = [{"role": "user", "content": synthesis_prompt}]

    text, tool_calls, tokens_used, usage = await provider.chat_async(
        messages,
        tools=[SYNTHESIS_TOOL],
        tool_choice="auto",
    )

    # Parse result
    result = _parse_synthesis_result(text, tool_calls, usage, evidence)

    print(f"[PSS] Synthesis complete: {len(leaves)} branches -> unified output")
    print(f"[PSS] Synthesis confidence: {result.confidence:.2f}")
    if result.action_plan:
        print(f"[PSS] Action plan: {len(result.action_plan)} steps")

    return result
