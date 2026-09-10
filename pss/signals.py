"""Cross-branch signal sharing for PSS v1.0 (Bulletin Board).

This module implements a bulletin board where branches can post and receive
signals about discoveries, warnings, constraints, and hypotheses. Signals
enable coordination between branches without breaking the "myopia" that
makes PSS effective.

Key design decisions:
- Signals are posted as "harness observations" (no sibling attribution)
- Signals have confidence decay based on token distance from source
- Signals are demoted when source branch terminates unsuccessfully
- Delivery is batched at gates to avoid overwhelming branches
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class SignalType(Enum):
    """Types of signals that can be posted to the bulletin board."""

    DISCOVERY = "discovery"  # Found something useful/interesting
    WARNING = "warning"  # Something to avoid or be careful of
    CONSTRAINT = "constraint"  # A hard requirement or limitation discovered
    HYPOTHESIS = "hypothesis"  # A theory worth testing


@dataclass
class Signal:
    """A signal posted to the bulletin board.

    Signals have confidence decay based on token distance and source outcome.
    """

    id: str
    source_branch: str
    timestamp: float
    signal_type: SignalType
    content: str
    confidence: float
    tags: list[str] = field(default_factory=list)

    # Genealogy weight (inherited from source branch)
    source_weight: float = 0.5

    # Decay tracking
    created_at_total_tokens: int = 0  # Global token count when posted
    source_terminated: bool = False
    source_termination_reason: str | None = None

    # Delivery tracking
    delivered_to: list[str] = field(default_factory=list)  # Branch IDs

    def effective_confidence(
        self,
        current_total_tokens: int,
        half_life_tokens: int = 10000,
        dead_source_penalty: float = 0.5,
    ) -> float:
        """Compute confidence with decay applied.

        Args:
            current_total_tokens: Current global token count.
            half_life_tokens: Tokens until confidence halves.
            dead_source_penalty: Multiplier when source died unsuccessfully.

        Returns:
            Effective confidence in range [0, 1].
        """
        age_tokens = current_total_tokens - self.created_at_total_tokens
        decay_factor = 0.5 ** (age_tokens / half_life_tokens)

        # Additional penalty if source died unsuccessfully
        if self.source_terminated and self.source_termination_reason in (
            "stuck",
            "killed",
        ):
            decay_factor *= dead_source_penalty

        return self.confidence * decay_factor * self.source_weight

    def mark_source_terminated(self, reason: str) -> None:
        """Mark that the source branch has terminated."""
        self.source_terminated = True
        self.source_termination_reason = reason


@dataclass
class BulletinBoard:
    """Shared bulletin board for cross-branch signals.

    Branches post signals here, and the harness delivers relevant signals
    at gate checkpoints.
    """

    signals: list[Signal] = field(default_factory=list)
    _total_tokens: int = 0  # Track global token count for decay

    def post(self, signal: Signal) -> None:
        """Post a signal to the bulletin board.

        Deduplicates similar signals automatically.
        """
        # Check for duplicates
        for existing in self.signals:
            if self._are_similar(signal, existing):
                # Merge: boost confidence, add tags
                existing.confidence = min(
                    existing.confidence + signal.confidence * 0.3, 0.95
                )
                existing.tags = list(set(existing.tags + signal.tags))
                return

        self.signals.append(signal)

    def _are_similar(self, s1: Signal, s2: Signal) -> bool:
        """Check if two signals are similar enough to merge."""
        if s1.signal_type != s2.signal_type:
            return False

        # Simple word overlap check
        words1 = set(s1.content.lower().split())
        words2 = set(s2.content.lower().split())

        if not words1 or not words2:
            return False

        overlap = len(words1 & words2) / min(len(words1), len(words2))
        return overlap > 0.6

    def get_signals_for_branch(
        self,
        branch_id: str,
        branch_direction: str,
        current_total_tokens: int,
        max_signals: int = 5,
        min_confidence: float = 0.3,
        half_life_tokens: int = 10000,
        dead_source_penalty: float = 0.5,
    ) -> list[Signal]:
        """Get relevant signals for a branch.

        Args:
            branch_id: The branch requesting signals.
            branch_direction: The branch's current direction/focus.
            current_total_tokens: Current global token count.
            max_signals: Maximum signals to return.
            min_confidence: Minimum effective confidence to include.
            half_life_tokens: Half-life for confidence decay.
            dead_source_penalty: Penalty for signals from dead branches.

        Returns:
            List of relevant signals, sorted by effective confidence.
        """
        self._total_tokens = current_total_tokens

        candidates = []
        for signal in self.signals:
            # Skip signals from this branch (don't echo back)
            if signal.source_branch == branch_id:
                continue

            # Skip already delivered
            if branch_id in signal.delivered_to:
                continue

            # Compute effective confidence with decay
            eff_conf = signal.effective_confidence(
                current_total_tokens, half_life_tokens, dead_source_penalty
            )

            if eff_conf < min_confidence:
                continue

            # Compute relevance to branch direction
            relevance = self._compute_relevance(signal, branch_direction)

            if relevance > 0.2:  # Minimum relevance threshold
                candidates.append((signal, eff_conf * relevance))

        # Sort by combined score (confidence * relevance)
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Return top signals
        result = [c[0] for c in candidates[:max_signals]]

        # Mark as delivered
        for signal in result:
            signal.delivered_to.append(branch_id)

        return result

    def _compute_relevance(self, signal: Signal, direction: str) -> float:
        """Compute relevance of signal to branch direction.

        Uses simple keyword matching. Could be upgraded to embeddings later.
        """
        if not direction:
            return 0.5  # Neutral relevance for unknown direction

        signal_words = set(signal.content.lower().split())
        signal_words.update(tag.lower() for tag in signal.tags)

        direction_words = set(direction.lower().split())

        if not direction_words:
            return 0.5

        overlap = len(signal_words & direction_words)

        # Normalize by direction length
        return min(overlap / len(direction_words), 1.0)

    def get_global_signals(
        self,
        current_total_tokens: int,
        min_confidence: float = 0.5,
    ) -> list[Signal]:
        """Get high-confidence signals that apply globally.

        These are typically CONSTRAINT or high-confidence WARNING signals.
        """
        result = []
        for signal in self.signals:
            eff_conf = signal.effective_confidence(current_total_tokens)

            if eff_conf >= min_confidence:
                if signal.signal_type in (SignalType.CONSTRAINT, SignalType.WARNING):
                    result.append(signal)

        return result

    def mark_branch_terminated(self, branch_id: str, reason: str) -> None:
        """Update all signals from a branch when it terminates."""
        for signal in self.signals:
            if signal.source_branch == branch_id:
                signal.mark_source_terminated(reason)

    def prune_stale(
        self,
        current_total_tokens: int,
        min_effective_confidence: float = 0.1,
    ) -> int:
        """Remove signals whose effective confidence is too low.

        Returns number of signals pruned.
        """
        original_count = len(self.signals)

        self.signals = [
            s
            for s in self.signals
            if s.effective_confidence(current_total_tokens) >= min_effective_confidence
        ]

        return original_count - len(self.signals)

    def get_stats(self) -> dict:
        """Get statistics about the bulletin board."""
        by_type = {}
        for signal in self.signals:
            by_type[signal.signal_type.value] = (
                by_type.get(signal.signal_type.value, 0) + 1
            )

        return {
            "total_signals": len(self.signals),
            "by_type": by_type,
            "total_tokens_tracked": self._total_tokens,
        }


# Signal extraction patterns
DISCOVERY_PATTERNS = [
    r"\b(I found|discovered|realized|noticed|observed)\b[^.!?]{10,}[.!?]",
    r"\bimportant(?:ly)?:\s*[^.!?]+[.!?]",
    r"\bkey (?:finding|insight|observation):\s*[^.!?]+[.!?]",
]

WARNING_PATTERNS = [
    r"\b(don't|do not|avoid|careful|warning|caution)\b[^.!?]{10,}[.!?]",
    r"\bthis (?:won't|doesn't|can't) work\b[^.!?]*[.!?]",
    r"\b(?:error|exception|failure)\b[^.!?]*[.!?]",
]

CONSTRAINT_PATTERNS = [
    r"\b(must|required|necessary|cannot|impossible)\b[^.!?]{10,}[.!?]",
    r"\bthe (?:only|sole) way\b[^.!?]+[.!?]",
    r"\blimitation:\s*[^.!?]+[.!?]",
]

HYPOTHESIS_PATTERNS = [
    r"\b(hypothesis|theory|conjecture):\s*[^.!?]+[.!?]",
    r"\bwhat if\b[^.!?]+[.!?]",
    r"\bperhaps\b[^.!?]{15,}[.!?]",
]


def extract_signals_from_response(
    text: str,
    branch_id: str,
    total_tokens: int,
    base_confidence: float = 0.5,
    source_weight: float = 0.5,
) -> list[Signal]:
    """Extract signals from an LLM response using pattern matching.

    Args:
        text: The LLM response text.
        branch_id: ID of the branch that produced this response.
        total_tokens: Current global token count.
        base_confidence: Base confidence for extracted signals.
        source_weight: Weight of the source branch.

    Returns:
        List of extracted signals.
    """
    signals = []
    timestamp = time.time()

    # Extract discoveries
    for pattern in DISCOVERY_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            content = match.group(0).strip()
            signals.append(
                Signal(
                    id=str(uuid.uuid4())[:8],
                    source_branch=branch_id,
                    timestamp=timestamp,
                    signal_type=SignalType.DISCOVERY,
                    content=_clean_signal_content(content),
                    confidence=base_confidence,
                    source_weight=source_weight,
                    created_at_total_tokens=total_tokens,
                    tags=_extract_tags(content),
                )
            )

    # Extract warnings
    for pattern in WARNING_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            content = match.group(0).strip()
            signals.append(
                Signal(
                    id=str(uuid.uuid4())[:8],
                    source_branch=branch_id,
                    timestamp=timestamp,
                    signal_type=SignalType.WARNING,
                    content=_clean_signal_content(content),
                    confidence=base_confidence + 0.1,  # Warnings get slight boost
                    source_weight=source_weight,
                    created_at_total_tokens=total_tokens,
                    tags=_extract_tags(content),
                )
            )

    # Extract constraints
    for pattern in CONSTRAINT_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            content = match.group(0).strip()
            signals.append(
                Signal(
                    id=str(uuid.uuid4())[:8],
                    source_branch=branch_id,
                    timestamp=timestamp,
                    signal_type=SignalType.CONSTRAINT,
                    content=_clean_signal_content(content),
                    confidence=base_confidence + 0.15,  # Constraints get more boost
                    source_weight=source_weight,
                    created_at_total_tokens=total_tokens,
                    tags=_extract_tags(content),
                )
            )

    # Extract hypotheses
    for pattern in HYPOTHESIS_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            content = match.group(0).strip()
            signals.append(
                Signal(
                    id=str(uuid.uuid4())[:8],
                    source_branch=branch_id,
                    timestamp=timestamp,
                    signal_type=SignalType.HYPOTHESIS,
                    content=_clean_signal_content(content),
                    confidence=base_confidence - 0.1,  # Hypotheses are less certain
                    source_weight=source_weight,
                    created_at_total_tokens=total_tokens,
                    tags=_extract_tags(content),
                )
            )

    return signals


def _clean_signal_content(content: str) -> str:
    """Clean extracted signal content."""
    # Remove leading pattern words
    content = re.sub(
        r"^(I found|discovered|realized|noticed|observed|important(?:ly)?:)\s*",
        "",
        content,
        flags=re.IGNORECASE,
    )
    # Truncate to reasonable length
    if len(content) > 200:
        content = content[:197] + "..."
    return content.strip()


def _extract_tags(content: str) -> list[str]:
    """Extract potential tags from content."""
    # Look for technical terms (capitalized words, acronyms)
    tags = []

    # Acronyms (2-5 uppercase letters)
    acronyms = re.findall(r"\b[A-Z]{2,5}\b", content)
    tags.extend(acronyms)

    # Technical terms (CamelCase)
    camel_case = re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b", content)
    tags.extend(camel_case)

    return list(set(tags))[:5]  # Max 5 tags


def format_signals_for_injection(
    signals: list[Signal],
    max_chars: int = 1000,
) -> str:
    """Format signals for injection into branch context.

    Signals are presented as "harness observations" without attribution
    to preserve branch myopia.

    Args:
        signals: List of signals to format.
        max_chars: Maximum character limit.

    Returns:
        Formatted string for injection, or empty string if no signals.
    """
    if not signals:
        return ""

    lines = ["[Harness Observations]"]
    lines.append("The following observations may be relevant to your exploration:")
    lines.append("")

    char_count = sum(len(line) for line in lines)

    for signal in signals:
        type_emoji = {
            SignalType.DISCOVERY: "+",
            SignalType.WARNING: "!",
            SignalType.CONSTRAINT: "#",
            SignalType.HYPOTHESIS: "?",
        }.get(signal.signal_type, "-")

        line = f"  {type_emoji} [{signal.signal_type.value.upper()}] {signal.content}"

        if char_count + len(line) > max_chars:
            lines.append(f"  ... ({len(signals) - len(lines) + 3} more observations)")
            break

        lines.append(line)
        char_count += len(line)

    lines.append("")
    lines.append("Consider these observations as you proceed, but verify independently.")

    return "\n".join(lines)


def create_signal(
    branch_id: str,
    signal_type: SignalType,
    content: str,
    total_tokens: int,
    confidence: float = 0.5,
    source_weight: float = 0.5,
    tags: list[str] | None = None,
) -> Signal:
    """Helper to create a signal manually.

    Args:
        branch_id: ID of the source branch.
        signal_type: Type of signal.
        content: Signal content.
        total_tokens: Current global token count.
        confidence: Confidence level.
        source_weight: Weight of source branch.
        tags: Optional tags.

    Returns:
        New Signal instance.
    """
    return Signal(
        id=str(uuid.uuid4())[:8],
        source_branch=branch_id,
        timestamp=time.time(),
        signal_type=signal_type,
        content=content,
        confidence=confidence,
        source_weight=source_weight,
        created_at_total_tokens=total_tokens,
        tags=tags or [],
    )
