"""Gate injection and checkpoint decision handling."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from pss.types import Context, GateDecision

if TYPE_CHECKING:
    from pss.config import PSSConfig
    from pss.types import DiversityState

# The checkpoint decision tool definition
CHECKPOINT_DECISION_TOOL = {
    "name": "checkpoint_decision",
    "description": "Make a decision at a checkpoint. You must call this to continue.",
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["continue", "branch", "terminate"],
                "description": (
                    "continue = keep exploring this path, "
                    "branch = spawn alternatives, "
                    "terminate = this path is done"
                ),
            },
            "branches": {
                "type": "array",
                "items": {"type": "string"},
                "description": "If branching: describe each alternative to explore (1-5 branches)",
            },
            "also_continue": {
                "type": "boolean",
                "description": "If branching: should the current path also continue alongside new branches?",
            },
            "output": {
                "type": "string",
                "description": "If terminating with a result: the final output of this path",
            },
            "confidence": {
                "type": "number",
                "description": "Self-assessed confidence in this decision (0.0-1.0)",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of why you're making this decision",
            },
        },
        "required": ["action"],
    },
}


def create_gate_prompt(
    ctx: Context,
    force: bool = False,
    diversity_state: "DiversityState | None" = None,
    config: "PSSConfig | None" = None,
) -> str:
    """
    Generate the gate prompt based on context state.

    The first gate is a surprise (Heisenberg trick).
    Subsequent gates are expected.

    v0.5: If diversity_state is provided and diversity_aware_enabled,
    adds diversity guidance to subsequent gates (not the first one).
    """
    if ctx.gates_seen < 2:
        # First TWO gates - inject as interrupt with explicit examples
        # NOTE: NO diversity info here - Heisenberg trick preserved
        # IMPORTANT: No TERMINATE option on first two checkpoints - force exploration
        prompt = """

---CHECKPOINT---

You've been exploring for a while. Time to decide.

You MUST respond with EXACTLY ONE of these two actions:

**CONTINUE** - Keep going on this path if you're making progress
Example: {"action": "continue"}

**BRANCH** - Spawn alternative approaches to explore different strategies in parallel
Example: {"action": "branch", "branches": ["try approach A", "try approach B"], "also_continue": true}

This is an early checkpoint. You cannot terminate yet - explore more first.

Call the checkpoint_decision tool with your choice. If you cannot use tools, respond with valid JSON matching one of the examples above.
"""
    else:
        # Subsequent gates - agent knows the drill
        prompt = """

---CHECKPOINT---

Decision point. Choose one action:
- {"action": "continue"} - keep exploring
- {"action": "branch", "branches": ["alt 1", "alt 2"]} - spawn alternatives
- {"action": "terminate", "output": "final result"} - done with output

Call checkpoint_decision or respond with JSON.
"""
        # v0.5: Add diversity guidance (only after first gate)
        if diversity_state and config and config.diversity_aware_enabled:
            from pss.diversity import create_diversity_guidance

            guidance = create_diversity_guidance(diversity_state, config)
            if guidance:
                prompt += guidance

    if force:
        prompt += "\nYou MUST decide now. No more exploration without a decision.\n"

    return prompt


def parse_gate_decision(tool_calls: list[dict]) -> GateDecision | None:
    """
    Parse a checkpoint_decision tool call into a GateDecision.

    Returns None if no valid checkpoint_decision was found.
    """
    for call in tool_calls:
        if call.get("name") == "checkpoint_decision":
            args = call.get("arguments", {})
            # Handle None values explicitly (model might return null)
            branches = args.get("branches")
            if branches is None:
                branches = []
            # Filter out invalid branches (must be non-empty strings)
            branches = [b for b in branches if isinstance(b, str) and len(b.strip()) > 5]

            action = args.get("action", "continue")
            also_continue = args.get("also_continue", False)

            # Auto-fix: if model says "continue" but provides branches,
            # treat as "branch" with also_continue=True
            # This handles models that want alternatives but use wrong action
            if action == "continue" and branches:
                action = "branch"
                also_continue = True

            return GateDecision(
                action=action,
                branches=branches,
                also_continue=also_continue,
                output=args.get("output"),
                confidence=args.get("confidence"),
                reasoning=args.get("reasoning"),
            )
    return None


def parse_decision_from_text(text: str) -> GateDecision | None:
    """
    Fallback parser for models that don't use tools properly.

    Attempts to extract a decision from natural language or embedded JSON.
    """
    if not text:
        return None

    text_lower = text.lower()

    # Try to find JSON in the response
    json_match = re.search(r'\{[^{}]*"action"\s*:\s*"[^"]+"[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if "action" in data:
                branches = data.get("branches", [])
                # Validate branches is a list of strings (not iterating over a string)
                if isinstance(branches, str):
                    branches = [branches] if len(branches) > 5 else []
                elif isinstance(branches, list):
                    branches = [
                        b for b in branches if isinstance(b, str) and len(b) > 5
                    ][:5]
                else:
                    branches = []

                action = data.get("action", "continue")
                also_continue = data.get("also_continue", False)

                # Auto-fix: continue + branches → branch + also_continue
                if action == "continue" and branches:
                    action = "branch"
                    also_continue = True

                return GateDecision(
                    action=action,
                    branches=branches,
                    also_continue=also_continue,
                    output=data.get("output"),
                    confidence=data.get("confidence"),
                    reasoning=data.get("reasoning"),
                )
        except json.JSONDecodeError:
            pass

    # Try to find JSON with single quotes (common mistake)
    json_like = re.search(r"\{[^{}]*'action'\s*:\s*'([^']+)'[^{}]*\}", text, re.DOTALL)
    if json_like:
        action = json_like.group(1).lower()
        if action in ("continue", "branch", "terminate"):
            branches = []
            branch_match = re.search(r"'branches'\s*:\s*\[([^\]]+)\]", text)
            if branch_match:
                raw_branches = [
                    b.strip().strip("'\"") for b in branch_match.group(1).split(",")
                ]
                branches = [b for b in raw_branches if len(b) > 5][:5]
            return GateDecision(
                action=action,
                branches=branches,
                also_continue=False,
                output=None,
            )

    # Look for explicit action keywords at the start of lines or after colons
    action_patterns = [
        (
            r"(?:^|\n|:)\s*\**\s*(?:action|decision|choice)\s*[:=]?\s*\**\s*(continue|branch|terminate)",
            1,
        ),
        (
            r"(?:^|\n)\s*(?:i (?:will|choose to|decide to)|my (?:decision|choice) is to)\s+(continue|branch|terminate)",
            1,
        ),
        (r"(?:^|\n)\s*\*?\*?(continue|branch|terminate)\*?\*?\s*[-:.]", 1),
    ]

    for pattern, group in action_patterns:
        match = re.search(pattern, text_lower)
        if match:
            action = match.group(group)
            branches = []

            # If branching, try to extract branch descriptions
            if action == "branch":
                # Look for numbered lists or bullet points after "branch"
                branch_section = text[match.end() :]
                branch_items = re.findall(
                    r"(?:^|\n)\s*(?:\d+[.):]|\*|-)\s*(.+?)(?=\n|$)",
                    branch_section[:500],
                )
                if branch_items:
                    # Filter to reasonable branch descriptions (>10 chars, <200 chars)
                    branches = [
                        b.strip() for b in branch_items if 10 < len(b.strip()) < 200
                    ][:5]

            # If terminating, try to extract output
            output = None
            if action == "terminate":
                # Look for output/result/answer section
                output_match = re.search(
                    r"(?:output|result|answer|conclusion)\s*[:=]\s*(.+?)(?:\n\n|$)",
                    text[match.end() :],
                    re.IGNORECASE | re.DOTALL,
                )
                if output_match:
                    output = output_match.group(1).strip()

            return GateDecision(
                action=action,
                branches=branches,
                also_continue=False,
                output=output,
            )

    # Last resort: look for the keywords anywhere, with some context
    if "terminate" in text_lower and (
        "done" in text_lower or "complete" in text_lower or "finish" in text_lower
    ):
        return GateDecision(action="terminate", output=text[:1000])

    if "branch" in text_lower and (
        "alternative" in text_lower
        or "different" in text_lower
        or "explore" in text_lower
    ):
        # Try to extract what to branch into
        branches = []
        alt_match = re.findall(
            r"(?:alternative|option|approach|path)\s*\d*\s*[:=]?\s*(.+?)(?:\n|$)",
            text,
            re.IGNORECASE,
        )
        if alt_match:
            branches = [b.strip() for b in alt_match if 10 < len(b.strip()) < 200][:5]
        return GateDecision(
            action="branch", branches=branches or ["explore alternative approach"]
        )

    if "continue" in text_lower and (
        "progress" in text_lower or "keep" in text_lower or "more" in text_lower
    ):
        return GateDecision(action="continue")

    return None


def create_gate_retry_prompt() -> str:
    """Prompt to re-request a gate decision if the model didn't call the tool."""
    return """

You must make a checkpoint decision. Respond with ONLY valid JSON:
- To continue: {"action": "continue"}
- To branch: {"action": "branch", "branches": ["description 1", "description 2"]}
- To terminate: {"action": "terminate", "output": "your final result"}

No other text. Just the JSON.
"""
