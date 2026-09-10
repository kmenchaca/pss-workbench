"""Failure propagation for PSS v1.0.

When a branch fails, extract a structured failure summary and inject it into
new branches so they don't repeat the same dead ends. This is forward
propagation—failed branches stay dead, but their knowledge lives on.

Key design decisions:
- No explicit "challenge" mechanism. Branches can go against warnings by just
  going that way. If they succeed, the harness notices and downgrades/archives
  the failure.
- Failures have a lifecycle: active → stale → archived
- Multiple siblings failing at the same point escalates scope to DIRECTION
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pss.genealogy import GenealogyTree
    from pss.types import Context


class FailureType(Enum):
    """Classification of why a branch failed."""

    DEAD_END = "dead_end"  # path leads nowhere
    INVALID_ASSUMPTION = "invalid_assumption"  # started from wrong premise
    RESOURCE_EXHAUSTED = "resource_exhausted"  # ran out of context/time
    CIRCULAR = "circular"  # kept returning to same point
    CONTRADICTED = "contradicted"  # found evidence against the approach
    BLOCKED = "blocked"  # external constraint prevents progress


class FailureScope(Enum):
    """How broadly a failure applies."""

    LOCAL = "local"  # only this specific approach
    DIRECTION = "direction"  # this whole direction (e.g., "OAuth won't work")
    GLOBAL = "global"  # applies to all branches (e.g., "API is down")


@dataclass
class FailureSummary:
    """Structured summary of why a branch failed."""

    branch_id: str
    failure_type: FailureType
    what_was_attempted: str  # high-level description
    why_it_failed: str  # diagnosis
    evidence: list[str]  # specific observations that led to failure
    suggested_avoidance: str  # what to do differently
    confidence: float  # 0-1, how sure are we this is a real dead end
    scope: FailureScope

    # Genealogy context
    branch_depth: int = 0
    branch_path: list[str] = field(default_factory=list)
    sibling_outcomes: dict[str, str] = field(default_factory=dict)

    # Lifecycle tracking
    age_branches: int = 0  # incremented as new branches spawn
    status: Literal["active", "stale", "archived"] = "active"


@dataclass
class FailureRegistry:
    """Registry of known failures for propagation to new branches."""

    failures: list[FailureSummary] = field(default_factory=list)

    # Config (can be overridden)
    stale_after_branches: int = 5
    archive_after_branches: int = 15
    merge_similarity_threshold: float = 0.85

    def add_failure(self, failure: FailureSummary) -> None:
        """Add a failure, merging if similar to existing."""
        # Check for merge candidates
        for existing in self.failures:
            if existing.status == "archived":
                continue
            if self._should_merge(existing, failure):
                self._merge_into(existing, failure)
                return

        self.failures.append(failure)

    def get_relevant_failures(
        self,
        branch_direction: str,
        max_failures: int = 5,
    ) -> list[FailureSummary]:
        """Return failures relevant to a branch's current direction.

        Uses simple keyword matching. Could be upgraded to embedding similarity.
        """
        active_failures = [
            f for f in self.failures if f.status in ("active", "stale")
        ]

        if not branch_direction:
            # No direction to match against, return global failures only
            return [f for f in active_failures if f.scope == FailureScope.GLOBAL][
                :max_failures
            ]

        # Score failures by relevance
        scored = []
        direction_lower = branch_direction.lower()
        direction_words = set(re.findall(r"\w+", direction_lower))

        for failure in active_failures:
            score = 0.0

            # Global failures always relevant
            if failure.scope == FailureScope.GLOBAL:
                score = 1.0
            else:
                # Check keyword overlap with what_was_attempted
                attempted_lower = failure.what_was_attempted.lower()
                attempted_words = set(re.findall(r"\w+", attempted_lower))

                overlap = direction_words & attempted_words
                if overlap:
                    # More overlap = more relevant
                    score = len(overlap) / max(
                        len(direction_words), len(attempted_words)
                    )

                # Direction-level failures get a boost
                if failure.scope == FailureScope.DIRECTION:
                    score *= 1.5

            # Weight by confidence and freshness
            freshness = 1.0 if failure.status == "active" else 0.7
            score *= failure.confidence * freshness

            if score > 0.1:  # threshold for relevance
                scored.append((score, failure))

        # Sort by score descending
        scored.sort(key=lambda x: -x[0])
        return [f for _, f in scored[:max_failures]]

    def get_global_failures(self) -> list[FailureSummary]:
        """Return failures that apply to all branches."""
        return [
            f
            for f in self.failures
            if f.scope == FailureScope.GLOBAL and f.status != "archived"
        ]

    def age_all(self) -> None:
        """Age all failures by one branch spawn. Call on each new branch."""
        for failure in self.failures:
            if failure.status == "archived":
                continue

            failure.age_branches += 1

            if failure.age_branches >= self.archive_after_branches:
                failure.status = "archived"
            elif failure.age_branches >= self.stale_after_branches:
                failure.status = "stale"

    def handle_contradiction(
        self,
        contradicting_branch_id: str,
        contradicting_path: list[str],
    ) -> list[FailureSummary]:
        """Handle a branch succeeding despite failure warnings.

        Returns list of failures that were contradicted.
        """
        contradicted = []

        for failure in self.failures:
            if failure.status == "archived":
                continue

            # Check if this success contradicts the failure
            if self._paths_overlap(contradicting_path, failure.branch_path):
                contradicted.append(failure)
                self._downgrade_failure(failure, contradicting_branch_id)

        return contradicted

    def _should_merge(self, f1: FailureSummary, f2: FailureSummary) -> bool:
        """Determine if two failures should be merged."""
        # Must be same failure type
        if f1.failure_type != f2.failure_type:
            return False

        # Check textual similarity of what_was_attempted
        words1 = set(re.findall(r"\w+", f1.what_was_attempted.lower()))
        words2 = set(re.findall(r"\w+", f2.what_was_attempted.lower()))

        if not words1 or not words2:
            return False

        overlap = len(words1 & words2)
        union = len(words1 | words2)
        similarity = overlap / union if union > 0 else 0

        return similarity >= self.merge_similarity_threshold

    def _merge_into(self, existing: FailureSummary, new: FailureSummary) -> None:
        """Merge a new failure into an existing one."""
        # Combine evidence
        for evidence in new.evidence:
            if evidence not in existing.evidence:
                existing.evidence.append(evidence)

        # Track multiple branch sources
        existing.branch_id = f"{existing.branch_id}+{new.branch_id}"

        # Boost confidence if multiple branches hit same failure
        existing.confidence = min(existing.confidence + 0.1, 1.0)

        # Update sibling outcomes
        existing.sibling_outcomes.update(new.sibling_outcomes)

        # If multiple siblings failed at same thing, escalate scope
        if (
            len(existing.branch_id.split("+")) >= 2
            and existing.scope == FailureScope.LOCAL
        ):
            existing.scope = FailureScope.DIRECTION
            print(
                f"[PSS] Failure escalated to DIRECTION scope: {existing.what_was_attempted[:50]}"
            )

        # Reset age since it's been reinforced
        existing.age_branches = 0
        existing.status = "active"

    def _paths_overlap(self, path1: list[str], path2: list[str]) -> bool:
        """Check if two branch paths have significant overlap."""
        if not path1 or not path2:
            return False

        # Convert to word sets for comparison
        words1 = set()
        for p in path1:
            words1.update(re.findall(r"\w+", p.lower()))

        words2 = set()
        for p in path2:
            words2.update(re.findall(r"\w+", p.lower()))

        if not words1 or not words2:
            return False

        overlap = len(words1 & words2)
        min_size = min(len(words1), len(words2))

        # Require at least 50% overlap
        return overlap / min_size >= 0.5 if min_size > 0 else False

    def _downgrade_failure(
        self, failure: FailureSummary, contradicting_branch_id: str
    ) -> None:
        """Downgrade or archive a contradicted failure."""
        note = f"Contradicted by {contradicting_branch_id}"

        if failure.scope == FailureScope.LOCAL:
            # Archive with note
            failure.status = "archived"
            failure.evidence.append(note)
            print(f"[PSS] Failure archived (contradicted): {failure.what_was_attempted[:50]}")
        else:
            # Downgrade scope
            if failure.scope == FailureScope.GLOBAL:
                failure.scope = FailureScope.DIRECTION
            elif failure.scope == FailureScope.DIRECTION:
                failure.scope = FailureScope.LOCAL
            failure.confidence *= 0.5
            failure.evidence.append(note)
            print(
                f"[PSS] Failure downgraded to {failure.scope.value}: {failure.what_was_attempted[:50]}"
            )


def extract_failure_from_context(
    ctx: "Context",
    genealogy: "GenealogyTree | None" = None,
) -> FailureSummary | None:
    """Extract a failure summary from a terminated context.

    Uses pattern matching on the context's messages to identify failure signals.
    This is a simple heuristic approach—could be upgraded to LLM-based extraction.

    Returns None if no clear failure pattern is detected.
    """
    from pss.types import TerminationReason

    # Only extract from stuck/killed contexts
    if ctx.termination_reason not in (
        TerminationReason.STUCK,
        TerminationReason.KILLED,
    ):
        return None

    # Get the last few assistant messages for analysis
    last_messages = []
    for msg in reversed(ctx.messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            last_messages.append(msg["content"])
            if len(last_messages) >= 3:
                break
    last_messages.reverse()

    if not last_messages:
        return None

    # Combine for analysis
    text = "\n".join(last_messages)
    text_lower = text.lower()

    # Detect failure type based on patterns
    failure_type = _detect_failure_type(text_lower)

    # Extract what was being attempted (from branch reason or context)
    what_attempted = ctx.branch_reason or "unknown approach"

    # Extract why it failed
    why_failed = _extract_failure_reason(text_lower)

    # Extract evidence snippets
    evidence = _extract_evidence(text)

    # Generate avoidance suggestion
    avoidance = _generate_avoidance(failure_type, what_attempted, why_failed)

    # Get genealogy info if available
    branch_depth = 0
    branch_path: list[str] = []
    sibling_outcomes: dict[str, str] = {}

    if genealogy and ctx.id in genealogy.nodes:
        branch_depth = genealogy.get_depth(ctx.id)
        branch_path = genealogy.get_branch_path(ctx.id)

        # Check sibling outcomes
        siblings = genealogy.get_siblings(ctx.id)
        for sib in siblings:
            if sib.termination_reason:
                sibling_outcomes[sib.id] = sib.termination_reason.value

    # Determine scope based on context
    scope = FailureScope.LOCAL
    if failure_type == FailureType.BLOCKED:
        # Blocked issues might be global
        if any(
            kw in text_lower
            for kw in ["api down", "service unavailable", "connection refused"]
        ):
            scope = FailureScope.GLOBAL

    # Calculate confidence based on how clear the failure signals are
    confidence = _calculate_failure_confidence(text_lower, failure_type)

    return FailureSummary(
        branch_id=ctx.id,
        failure_type=failure_type,
        what_was_attempted=what_attempted,
        why_it_failed=why_failed,
        evidence=evidence,
        suggested_avoidance=avoidance,
        confidence=confidence,
        scope=scope,
        branch_depth=branch_depth,
        branch_path=branch_path,
        sibling_outcomes=sibling_outcomes,
    )


def _detect_failure_type(text_lower: str) -> FailureType:
    """Detect failure type from text patterns."""
    # Check for circular/repetitive patterns
    if any(
        kw in text_lower
        for kw in [
            "going in circles",
            "tried this before",
            "same result",
            "keeps failing",
            "stuck in loop",
        ]
    ):
        return FailureType.CIRCULAR

    # Check for blocked/external issues
    if any(
        kw in text_lower
        for kw in [
            "permission denied",
            "access denied",
            "api error",
            "connection",
            "timeout",
            "unavailable",
            "blocked",
        ]
    ):
        return FailureType.BLOCKED

    # Check for invalid assumptions
    if any(
        kw in text_lower
        for kw in [
            "assumption was wrong",
            "doesn't exist",
            "not supported",
            "incompatible",
            "misunderstood",
        ]
    ):
        return FailureType.INVALID_ASSUMPTION

    # Check for contradictions
    if any(
        kw in text_lower
        for kw in [
            "contradicts",
            "inconsistent",
            "conflicts with",
            "cannot both",
        ]
    ):
        return FailureType.CONTRADICTED

    # Check for resource exhaustion
    if any(
        kw in text_lower
        for kw in [
            "out of",
            "exhausted",
            "limit reached",
            "too complex",
            "too long",
        ]
    ):
        return FailureType.RESOURCE_EXHAUSTED

    # Default to dead end
    return FailureType.DEAD_END


def _extract_failure_reason(text_lower: str) -> str:
    """Extract why the approach failed."""
    # Look for explicit failure explanations
    patterns = [
        r"(?:failed because|fails because|doesn't work because|won't work because)\s+(.{20,100})",
        r"(?:the problem is|the issue is|the error is)\s+(.{20,100})",
        r"(?:cannot|can't|unable to)\s+(.{20,80})",
        r"(?:this approach|this method|this strategy)\s+(.{20,100})",
    ]

    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1).strip()

    return "Approach did not lead to a solution"


def _extract_evidence(text: str) -> list[str]:
    """Extract specific evidence snippets from the text."""
    evidence = []

    # Look for error messages
    error_patterns = [
        r"error[:\s]+([^\n]{10,100})",
        r"exception[:\s]+([^\n]{10,100})",
        r"failed[:\s]+([^\n]{10,100})",
    ]

    for pattern in error_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches[:2]:  # limit to 2 per pattern
            evidence.append(match.strip())

    # Look for explicit observations
    obs_patterns = [
        r"(?:I found that|I noticed that|I observed that)\s+([^\n]{10,100})",
        r"(?:It turns out|It appears that)\s+([^\n]{10,100})",
    ]

    for pattern in obs_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches[:2]:
            evidence.append(match.strip())

    return evidence[:5]  # limit total evidence


def _generate_avoidance(
    failure_type: FailureType, what_attempted: str, why_failed: str
) -> str:
    """Generate a suggestion for avoiding this failure."""
    avoidance_templates = {
        FailureType.DEAD_END: f"Avoid pursuing '{what_attempted}' - it leads nowhere",
        FailureType.INVALID_ASSUMPTION: f"Don't assume '{what_attempted}' is viable without verification",
        FailureType.RESOURCE_EXHAUSTED: f"'{what_attempted}' requires too many resources - try simpler alternatives",
        FailureType.CIRCULAR: f"'{what_attempted}' leads to circular reasoning - try a fundamentally different approach",
        FailureType.CONTRADICTED: f"Evidence contradicts '{what_attempted}' - explore opposite direction",
        FailureType.BLOCKED: f"'{what_attempted}' is blocked by external constraints - find workarounds",
    }

    return avoidance_templates.get(
        failure_type, f"Avoid '{what_attempted}' - {why_failed}"
    )


def _calculate_failure_confidence(text_lower: str, failure_type: FailureType) -> float:
    """Calculate confidence that this is a real failure."""
    confidence = 0.5  # base confidence

    # Explicit failure language increases confidence
    if any(
        kw in text_lower
        for kw in [
            "definitely",
            "clearly",
            "certainly",
            "confirmed",
            "verified",
        ]
    ):
        confidence += 0.2

    # Hedging language decreases confidence
    if any(
        kw in text_lower
        for kw in ["maybe", "might", "possibly", "not sure", "uncertain"]
    ):
        confidence -= 0.15

    # Multiple failure signals increase confidence
    failure_signals = sum(
        1
        for kw in ["failed", "error", "cannot", "won't work", "doesn't work"]
        if kw in text_lower
    )
    confidence += min(failure_signals * 0.1, 0.25)

    # Blocked failures are usually high confidence
    if failure_type == FailureType.BLOCKED:
        confidence += 0.1

    return max(0.2, min(0.95, confidence))


def format_failures_for_injection(
    failures: list[FailureSummary],
    max_chars: int = 1500,
) -> str:
    """Format failures for injection into branch prompts."""
    if not failures:
        return ""

    lines = ["── Known Dead Ends ──"]

    char_count = len(lines[0])

    for failure in failures:
        scope_str = failure.scope.value.upper()
        conf_str = "high" if failure.confidence >= 0.7 else "medium" if failure.confidence >= 0.4 else "low"

        header = f"\n[{scope_str}] {failure.what_was_attempted}"
        body = f"\n  Why: {failure.why_it_failed}"
        avoid = f"\n  Avoid: {failure.suggested_avoidance}"
        conf = f"\n  Confidence: {conf_str}"

        entry = header + body + avoid + conf

        if char_count + len(entry) > max_chars:
            lines.append("\n... (additional failures omitted)")
            break

        lines.append(entry)
        char_count += len(entry)

    lines.append("\n─────────────────────")

    return "".join(lines)
