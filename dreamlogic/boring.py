"""
Dream Logic Generator - Boring Propagation

Inverted failure system: detect and propagate "boring" to push exploration
toward stranger territory. Unlike traditional failure propagation that
shares what doesn't work, this shares what's too predictable.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from .types import DreamBranch, DreamFragment


# Patterns that indicate boring/predictable content
BORING_PATTERNS: list[tuple[str, float, str]] = [
    # Pattern, severity, category
    (r'\b(once upon a time)\b', 0.8, "cliche_opening"),
    (r'\b(happily ever after)\b', 0.8, "cliche_ending"),
    (r'\b(in conclusion|to summarize)\b', 0.7, "essay_structure"),
    (r'\b(first|second|third|finally)\b', 0.4, "list_structure"),
    (r'^\d+\.\s', 0.5, "numbered_list"),
    (r'\b(the moral of the story)\b', 0.9, "didactic"),
    (r'\b(it was all a dream)\b', 0.95, "meta_cop_out"),
    (r'\b(suddenly|all of a sudden)\b', 0.3, "lazy_transition"),
    (r'\b(very|really|quite)\b', 0.2, "weak_modifier"),
    (r'\b(interesting|nice|good|bad)\b', 0.3, "vague_adjective"),
    (r'\b(said|walked|went)\b', 0.2, "basic_verb"),
    (r'\b(beautiful|amazing|wonderful)\b', 0.4, "generic_positive"),
    (r'\b(terrible|horrible|awful)\b', 0.4, "generic_negative"),
    (r'\b(in my opinion|I think that)\b', 0.5, "hedging"),
    (r'\b(as we all know|obviously)\b', 0.6, "assumption"),
    (r'\b(at the end of the day)\b', 0.7, "business_cliche"),
    (r'\b(synergy|leverage|optimize)\b', 0.8, "corporate_speak"),
    (r'\b(paradigm shift)\b', 0.85, "buzzword"),
]

# Structural patterns that indicate boring content
BORING_STRUCTURES: list[tuple[str, float]] = [
    ("five_paragraph_essay", 0.7),
    ("simple_cause_effect", 0.5),
    ("predictable_arc", 0.6),
    ("neat_resolution", 0.7),
    ("balanced_argument", 0.5),
]

# Avoidance suggestions when boring is detected
AVOIDANCE_SUGGESTIONS: dict[str, list[str]] = {
    "cliche_opening": [
        "Start mid-action, mid-thought, mid-dissolution",
        "Begin with a contradiction",
        "Start from the ending and work backwards",
    ],
    "cliche_ending": [
        "End before resolution",
        "End with a new question",
        "End by zooming out to cosmic scale",
    ],
    "essay_structure": [
        "Abandon structure entirely",
        "Let form follow the chaos of content",
        "Fragment the argument across impossible spaces",
    ],
    "list_structure": [
        "Let items bleed into each other",
        "Number things that aren't a sequence",
        "Create a list where each item contradicts the previous",
    ],
    "didactic": [
        "Remove all lessons",
        "Make the meaning actively resist extraction",
        "Let confusion be the point",
    ],
    "meta_cop_out": [
        "If it's a dream, make the waking stranger",
        "Nest dreams infinitely with no escape",
        "Dissolve the distinction between dream and wake",
    ],
    "lazy_transition": [
        "Cut abruptly between scenes",
        "Use white space as a character",
        "Let transitions be their own micro-stories",
    ],
    "weak_modifier": [
        "Use specific, unexpected modifiers",
        "Stack contradictory modifiers",
        "Let the noun modify itself",
    ],
    "corporate_speak": [
        "Use language that corporations would fear",
        "Make the jargon literal and terrifying",
        "Speak like something older than commerce",
    ],
}


@dataclass
class BoringDetection:
    """Result of boring detection analysis."""
    score: float  # Overall boring score (0 = fascinating, 1 = utterly predictable)
    patterns_found: list[tuple[str, float, str]]  # (match, severity, category)
    suggestions: list[str]  # How to be less boring
    structural_issues: list[str]  # Structural problems detected


@dataclass
class BoringDetector:
    """
    Identifies predictable, uninteresting content.

    Tracks patterns across branches to build a model of what's "boring"
    in this particular exploration.
    """
    # Accumulated boring patterns from exploration
    seen_patterns: dict[str, int] = field(default_factory=dict)
    # Content that was marked boring
    boring_content: list[str] = field(default_factory=list)
    # Custom patterns learned during exploration
    learned_patterns: list[tuple[str, float, str]] = field(default_factory=list)

    def detect(self, content: str) -> BoringDetection:
        """
        Analyze content for boring patterns.

        Args:
            content: Content to analyze.

        Returns:
            BoringDetection with score and details.
        """
        content_lower = content.lower()
        patterns_found = []
        categories_found = set()

        # Check built-in patterns
        for pattern, severity, category in BORING_PATTERNS:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            if matches:
                for match in matches:
                    patterns_found.append((match, severity, category))
                    categories_found.add(category)
                    # Track for learning
                    self.seen_patterns[pattern] = self.seen_patterns.get(pattern, 0) + 1

        # Check learned patterns
        for pattern, severity, category in self.learned_patterns:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            if matches:
                for match in matches:
                    patterns_found.append((match, severity, category))
                    categories_found.add(category)

        # Check structural issues
        structural_issues = self._detect_structural_issues(content)

        # Calculate overall score
        if not patterns_found and not structural_issues:
            score = 0.0
        else:
            pattern_score = sum(p[1] for p in patterns_found) / max(len(patterns_found), 1)
            structural_score = len(structural_issues) * 0.2
            score = min(1.0, (pattern_score + structural_score) / 2)

        # Gather suggestions
        suggestions = []
        for category in categories_found:
            if category in AVOIDANCE_SUGGESTIONS:
                suggestions.extend(AVOIDANCE_SUGGESTIONS[category][:2])

        return BoringDetection(
            score=score,
            patterns_found=patterns_found,
            suggestions=suggestions[:5],  # Limit suggestions
            structural_issues=structural_issues,
        )

    def _detect_structural_issues(self, content: str) -> list[str]:
        """Detect structural boring patterns."""
        issues = []
        lines = content.split('\n')
        paragraphs = [p for p in content.split('\n\n') if p.strip()]

        # Five paragraph essay detection
        if len(paragraphs) == 5:
            issues.append("Suspiciously essay-like structure (5 paragraphs)")

        # Neat numbered list
        numbered_lines = sum(1 for line in lines if re.match(r'^\d+\.', line.strip()))
        if numbered_lines >= 3:
            issues.append("Ordered list suggests linear thinking")

        # Balanced argument (on one hand... on the other)
        if 'on one hand' in content.lower() or 'on the other hand' in content.lower():
            issues.append("Balanced argument structure detected")

        # Too-neat resolution
        if re.search(r'(and so|in the end|finally|thus|therefore).{0,50}$', content.lower()):
            issues.append("Content resolves too neatly")

        return issues

    def score_predictability(self, content: str, context: Optional[str] = None) -> float:
        """
        Calculate how predictable content is.

        Args:
            content: Content to score.
            context: Optional preceding context.

        Returns:
            Predictability score from 0 to 1.
        """
        detection = self.detect(content)
        base_score = detection.score

        # Context-based predictability
        if context:
            # If content closely follows patterns in context, more predictable
            context_words = set(context.lower().split())
            content_words = set(content.lower().split())
            overlap = len(context_words & content_words)
            if context_words:
                continuation_score = overlap / len(context_words)
                # Heavy continuation = predictable
                if continuation_score > 0.5:
                    base_score = min(1.0, base_score + 0.2)

        # Compare to boring content history
        if self.boring_content:
            similarity = self._similarity_to_boring(content)
            base_score = min(1.0, base_score + similarity * 0.3)

        return base_score

    def _similarity_to_boring(self, content: str) -> float:
        """Calculate similarity to previously boring content."""
        content_words = set(content.lower().split())
        max_similarity = 0.0

        for boring in self.boring_content[-20:]:
            boring_words = set(boring.lower().split())
            if boring_words:
                intersection = content_words & boring_words
                union = content_words | boring_words
                if union:
                    similarity = len(intersection) / len(union)
                    max_similarity = max(max_similarity, similarity)

        return max_similarity

    def mark_boring(self, content: str) -> None:
        """Mark content as boring for future reference."""
        self.boring_content.append(content)
        # Keep bounded
        if len(self.boring_content) > 50:
            self.boring_content = self.boring_content[-50:]

    def learn_pattern(self, pattern: str, severity: float, category: str) -> None:
        """Learn a new boring pattern from observation."""
        self.learned_patterns.append((pattern, severity, category))


def propagate_boring(branches: list[DreamBranch], detector: Optional[BoringDetector] = None) -> dict[str, float]:
    """
    Calculate boring scores for branches and propagate knowledge.

    Branches with high boring scores should be pushed toward stranger territory.

    Args:
        branches: List of dream branches to analyze.
        detector: Optional detector (creates new if not provided).

    Returns:
        Dictionary of branch_id -> boring_score.
    """
    if detector is None:
        detector = BoringDetector()

    scores: dict[str, float] = {}

    for branch in branches:
        if not branch.fragments:
            scores[branch.id] = 0.0
            continue

        # Score each fragment
        fragment_scores = []
        for fragment in branch.fragments:
            detection = detector.detect(fragment.content)
            fragment_scores.append(detection.score)

            # Mark if boring
            if detection.score > 0.6:
                detector.mark_boring(fragment.content)

        # Branch boring score is average
        branch_score = sum(fragment_scores) / len(fragment_scores)
        scores[branch.id] = branch_score
        branch.boring_score = branch_score

    return scores


def avoid_patterns(content: str, detector: Optional[BoringDetector] = None) -> list[str]:
    """
    Get suggestions for patterns to avoid based on content analysis.

    Args:
        content: Content to analyze.
        detector: Optional detector.

    Returns:
        List of patterns/concepts to avoid.
    """
    if detector is None:
        detector = BoringDetector()

    detection = detector.detect(content)
    patterns_to_avoid = []

    for match, _, category in detection.patterns_found:
        patterns_to_avoid.append(f"Avoid: '{match}' ({category})")

    return patterns_to_avoid


def boring_score_for_fragment(fragment: DreamFragment, detector: Optional[BoringDetector] = None) -> float:
    """
    Get boring score for a single fragment.

    Args:
        fragment: The fragment to score.
        detector: Optional detector.

    Returns:
        Boring score from 0 to 1.
    """
    if detector is None:
        detector = BoringDetector()

    detection = detector.detect(fragment.content)
    return detection.score


def push_toward_strange(
    boring_content: str,
    detector: Optional[BoringDetector] = None
) -> str:
    """
    Generate a prompt to push away from boring content toward strangeness.

    Args:
        boring_content: The boring content to escape.
        detector: Optional detector for analysis.

    Returns:
        A prompt instruction for generating stranger content.
    """
    if detector is None:
        detector = BoringDetector()

    detection = detector.detect(boring_content)

    suggestions = detection.suggestions
    if not suggestions:
        suggestions = [
            "Make it weirder",
            "Break the pattern",
            "Introduce something impossible",
        ]

    prompt_parts = [
        "The previous content was too predictable.",
        "To escape the boring:",
    ]

    for suggestion in suggestions[:3]:
        prompt_parts.append(f"  - {suggestion}")

    prompt_parts.append("\nNow, generate something that defies expectation.")

    return "\n".join(prompt_parts)


def is_too_boring(content: str, threshold: float = 0.5) -> bool:
    """
    Check if content exceeds boring threshold.

    Args:
        content: Content to check.
        threshold: Maximum acceptable boring score.

    Returns:
        True if content is too boring.
    """
    detector = BoringDetector()
    detection = detector.detect(content)
    return detection.score > threshold
