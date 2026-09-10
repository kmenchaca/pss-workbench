"""Velocity tracking for adaptive gates in PSS v1.0.

Velocity measures how "productively" a branch is exploring. High velocity means
new concepts, assertions, forward progress. Low velocity means repetition,
uncertainty, circular reasoning.

This module uses simple regex-based heuristics (not embedding-based) per the
design decision to "timebox and ship simple."
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field

from pss.types import VelocitySnapshot


# Regex patterns for velocity metrics
ASSERTION_PATTERNS = [
    r"\b(is|are|was|were|will be|must be|should be|can be|has|have|had)\b",
    r"\b(therefore|thus|hence|consequently|as a result)\b",
    r"\b(clearly|obviously|certainly|definitely|undoubtedly)\b",
    r"\b(the \w+ is|this \w+ is|that \w+ is)\b",
    r"\b(I found|I discovered|I determined|I concluded)\b",
]

HEDGING_PATTERNS = [
    r"\b(maybe|perhaps|possibly|might|could|may)\b",
    r"\b(I think|I believe|I suppose|I guess)\b",
    r"\b(not sure|uncertain|unclear|unsure)\b",
    r"\b(seems|appears|looks like)\b",
    r"\b(probably|likely|unlikely)\b",
]

QUESTION_PATTERNS = [
    r"\?",
    r"\b(what|why|how|when|where|which|who)\b[^.!?]*\?",
    r"\b(should I|could I|can I|would it)\b",
]

CIRCULAR_PATTERNS = [
    r"\b(as I (said|mentioned) (before|earlier|previously))\b",
    r"\b(going in circles|tried this before|same approach)\b",
    r"\b(again|once more|another attempt)\b",
    r"\b(let me try again|starting over|back to)\b",
]

# Technical terms that indicate novelty (simple heuristic)
TECHNICAL_TERM_PATTERN = r"\b([A-Z][a-zA-Z]*(?:Error|Exception|Handler|Manager|Service|Client|Server|API|URL|HTTP|JSON|SQL|DB|File|Path|Config|Auth|Token|Key|Hash|Cache|Queue|Event|Stream|Buffer|Worker|Thread|Process|Memory|Disk|Network|Socket|Port|Host|Request|Response|Header|Body|Status|Code|Data|Schema|Model|View|Controller|Router|Route|Middleware|Hook|Plugin|Module|Package|Import|Export|Class|Function|Method|Variable|Constant|Parameter|Argument|Return|Async|Await|Promise|Callback|Lambda|Iterator|Generator|Decorator|Interface|Type|Generic|Enum|Struct|Union|Pointer|Reference|Instance|Object|Array|List|Dict|Map|Set|Tuple|String|Int|Float|Bool|Null|Void|Any))\b"


@dataclass
class VelocityTracker:
    """Tracks velocity metrics over a sliding window.

    Call measure() after each LLM response to compute a new snapshot.
    The tracker maintains history for trend analysis.
    """

    window_size_tokens: int = 2000
    measurement_interval_tokens: int = 500
    snapshots: list[VelocitySnapshot] = field(default_factory=list)

    # Internal state
    _total_tokens_seen: int = 0
    _last_measurement_tokens: int = 0
    _all_text: str = ""  # accumulated text for repetition detection

    def measure(
        self,
        new_text: str,
        tokens_in_chunk: int,
        prev_window_text: str = "",
    ) -> VelocitySnapshot:
        """Compute velocity metrics for a chunk of text.

        Args:
            new_text: The new LLM response text.
            tokens_in_chunk: Approximate token count for new_text.
            prev_window_text: Text from previous window for comparison.

        Returns:
            VelocitySnapshot with computed metrics.
        """
        self._total_tokens_seen += tokens_in_chunk
        self._all_text += " " + new_text

        # Compute individual metrics
        novelty = _compute_novelty_rate(new_text, prev_window_text)
        assertions = _compute_assertion_rate(new_text)
        uncertainty = _compute_uncertainty_trend(new_text)
        repetition = _compute_repetition_score(new_text, prev_window_text)
        questions = _compute_question_density(new_text)

        snapshot = VelocitySnapshot(
            timestamp=time.time(),
            tokens_in_window=tokens_in_chunk,
            novelty_rate=novelty,
            assertion_rate=assertions,
            uncertainty_trend=uncertainty,
            repetition_score=repetition,
            question_density=questions,
        )

        self.snapshots.append(snapshot)
        self._last_measurement_tokens = self._total_tokens_seen

        return snapshot

    def should_measure(self, tokens_since_last: int) -> bool:
        """Check if enough tokens have passed for a new measurement."""
        return tokens_since_last >= self.measurement_interval_tokens

    def get_current_velocity(self) -> float:
        """Get the most recent velocity score."""
        if not self.snapshots:
            return 0.5  # neutral default
        return self.snapshots[-1].overall_velocity

    def get_trend(self, window: int = 3) -> float:
        """Get velocity trend over recent snapshots.

        Returns:
            -1 to 1: negative = declining, positive = improving
        """
        if len(self.snapshots) < 2:
            return 0.0

        recent = self.snapshots[-window:] if len(self.snapshots) >= window else self.snapshots

        if len(recent) < 2:
            return 0.0

        # Simple linear trend
        velocities = [s.overall_velocity for s in recent]
        first_half = sum(velocities[:len(velocities)//2]) / max(len(velocities)//2, 1)
        second_half = sum(velocities[len(velocities)//2:]) / max(len(velocities) - len(velocities)//2, 1)

        trend = second_half - first_half
        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, trend * 2))

    def get_average_velocity(self, window: int = 5) -> float:
        """Get average velocity over recent snapshots."""
        if not self.snapshots:
            return 0.5

        recent = self.snapshots[-window:] if len(self.snapshots) >= window else self.snapshots
        return sum(s.overall_velocity for s in recent) / len(recent)

    def is_plateauing(
        self,
        threshold: float = 0.1,
        window: int = 3,
    ) -> bool:
        """Check if velocity has been low and stable (plateau)."""
        if len(self.snapshots) < window:
            return False

        recent = self.snapshots[-window:]
        velocities = [s.overall_velocity for s in recent]

        avg = sum(velocities) / len(velocities)
        variance = sum((v - avg) ** 2 for v in velocities) / len(velocities)

        return avg < 0.4 and variance < threshold

    def reset(self) -> None:
        """Clear all tracking state."""
        self.snapshots.clear()
        self._total_tokens_seen = 0
        self._last_measurement_tokens = 0
        self._all_text = ""


def _compute_novelty_rate(text: str, prev_text: str) -> float:
    """Compute rate of new technical terms/concepts.

    Higher = more new concepts being introduced.
    """
    if not text.strip():
        return 0.0

    # Extract technical terms
    current_terms = set(re.findall(TECHNICAL_TERM_PATTERN, text, re.IGNORECASE))
    prev_terms = set(re.findall(TECHNICAL_TERM_PATTERN, prev_text, re.IGNORECASE)) if prev_text else set()

    # Also look for capitalized multi-word phrases (potential concepts)
    current_phrases = set(re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text))
    prev_phrases = set(re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", prev_text)) if prev_text else set()

    all_current = current_terms | current_phrases
    all_prev = prev_terms | prev_phrases

    if not all_current:
        # No technical content - use word-level novelty
        current_words = set(text.lower().split())
        prev_words = set(prev_text.lower().split()) if prev_text else set()
        new_words = current_words - prev_words
        if not current_words:
            return 0.0
        return min(len(new_words) / len(current_words), 1.0) * 0.5  # cap at 0.5 for word novelty

    new_terms = all_current - all_prev

    # Normalize: high novelty if >50% of terms are new
    novelty = len(new_terms) / len(all_current) if all_current else 0.0
    return min(novelty, 1.0)


def _compute_assertion_rate(text: str) -> float:
    """Compute rate of assertive/declarative statements.

    Higher = more confident claims being made.
    """
    if not text.strip():
        return 0.0

    # Count sentences (rough)
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return 0.0

    assertion_count = 0
    for sentence in sentences:
        for pattern in ASSERTION_PATTERNS:
            if re.search(pattern, sentence, re.IGNORECASE):
                assertion_count += 1
                break

    return min(assertion_count / len(sentences), 1.0)


def _compute_uncertainty_trend(text: str) -> float:
    """Compute uncertainty level in text.

    Returns:
        -1 to 1: negative = uncertain, positive = confident
    """
    if not text.strip():
        return 0.0

    text_lower = text.lower()

    # Count hedging patterns
    hedge_count = sum(
        len(re.findall(pattern, text_lower))
        for pattern in HEDGING_PATTERNS
    )

    # Count confident patterns (subset of assertion patterns)
    confident_patterns = [
        r"\b(definitely|certainly|clearly|obviously|undoubtedly)\b",
        r"\b(confirmed|verified|proven|established)\b",
        r"\b(must|will|always|never)\b",
    ]
    confident_count = sum(
        len(re.findall(pattern, text_lower))
        for pattern in confident_patterns
    )

    # Normalize
    total = hedge_count + confident_count
    if total == 0:
        return 0.0

    # Return value from -1 (all hedging) to 1 (all confident)
    return (confident_count - hedge_count) / total


def _compute_repetition_score(text: str, prev_text: str) -> float:
    """Compute how repetitive the text is compared to previous.

    Higher = more repetitive (bad for velocity).
    """
    if not text.strip() or not prev_text.strip():
        return 0.0

    # Check for circular patterns explicitly
    circular_count = sum(
        len(re.findall(pattern, text, re.IGNORECASE))
        for pattern in CIRCULAR_PATTERNS
    )

    if circular_count > 0:
        return min(0.5 + circular_count * 0.1, 1.0)

    # N-gram overlap
    def get_ngrams(s: str, n: int = 3) -> Counter:
        words = s.lower().split()
        return Counter(tuple(words[i:i+n]) for i in range(len(words) - n + 1))

    current_ngrams = get_ngrams(text)
    prev_ngrams = get_ngrams(prev_text)

    if not current_ngrams or not prev_ngrams:
        return 0.0

    # Jaccard-like overlap
    overlap = sum((current_ngrams & prev_ngrams).values())
    total = sum(current_ngrams.values())

    if total == 0:
        return 0.0

    return min(overlap / total, 1.0)


def _compute_question_density(text: str) -> float:
    """Compute density of questions in text.

    Higher = more questions (can indicate uncertainty or exploration).
    """
    if not text.strip():
        return 0.0

    # Count sentences
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return 0.0

    # Count questions
    question_count = text.count("?")

    # Also count interrogative patterns without question marks
    interrogative_count = sum(
        len(re.findall(pattern, text, re.IGNORECASE))
        for pattern in [r"\b(what|why|how|when|where|which|who)\b[^.!?]{10,}"]
    )

    total_questions = question_count + interrogative_count * 0.5

    return min(total_questions / len(sentences), 1.0)


def format_velocity_display(snapshot: VelocitySnapshot) -> str:
    """Format velocity for display in terminal UI.

    Returns a multi-line string with progress bars.
    """
    def bar(value: float, width: int = 10) -> str:
        filled = int(value * width)
        return "█" * filled + "░" * (width - filled)

    overall = snapshot.overall_velocity

    # Determine status
    if overall >= 0.7:
        status = "high"
    elif overall >= 0.4:
        status = "medium"
    else:
        status = "low"

    lines = [
        f"Velocity: {bar(overall)} {overall:.2f} ({status})",
        f"├─ Novelty:    {bar(snapshot.novelty_rate)} {snapshot.novelty_rate:.2f}",
        f"├─ Assertions: {bar(snapshot.assertion_rate)} {snapshot.assertion_rate:.2f}",
        f"├─ Repetition: {bar(snapshot.repetition_score)} {snapshot.repetition_score:.2f} {'(good)' if snapshot.repetition_score < 0.3 else '(bad)' if snapshot.repetition_score > 0.6 else ''}",
        f"└─ Questions:  {bar(snapshot.question_density)} {snapshot.question_density:.2f}",
    ]

    return "\n".join(lines)
