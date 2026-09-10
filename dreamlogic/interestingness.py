"""
Dream Logic Generator - Interestingness Scoring

Scores content for novelty, surprise, and aesthetic value.
Penalizes coherence (too-logical content is boring).
"""

import re
import math
from dataclasses import dataclass, field
from typing import Optional

from .types import InterestingnessScore, DreamFragment


# Patterns that indicate "boring" logical content
BORING_PATTERNS: list[tuple[str, float]] = [
    (r'\b(therefore|thus|hence|consequently)\b', 0.15),
    (r'\b(first|second|third|finally)\b', 0.1),
    (r'\b(in conclusion|to summarize|in summary)\b', 0.2),
    (r'\b(because|since|as a result)\b', 0.1),
    (r'\b(clearly|obviously|of course)\b', 0.15),
    (r'\b(step \d|point \d)\b', 0.2),
    (r'\b(pros and cons|advantages and disadvantages)\b', 0.25),
    (r'\b(on the other hand|however|but)\b', 0.05),  # Some contrast is ok
    (r'\b(it is important to note|it should be noted)\b', 0.2),
    (r'^\d+\.\s', 0.15),  # Numbered lists
]

# Patterns that indicate interesting/surprising content
INTERESTING_PATTERNS: list[tuple[str, float]] = [
    (r'\b(whisper|echo|shadow|mirror|dream)\b', 0.1),
    (r'\b(impossible|paradox|contradiction)\b', 0.15),
    (r'\b(infinite|eternal|void|abyss)\b', 0.1),
    (r'\b(dissolve|melt|transform|metamorphose)\b', 0.1),
    (r'\b(forgotten|lost|hidden|secret)\b', 0.08),
    (r'\b(taste|smell|texture)\s+of\s+\w+', 0.12),  # Synesthesia
    (r'(the color of|the sound of|the shape of)\s+\w+', 0.15),
    (r'\b(before time|after everything|between moments)\b', 0.15),
    (r'\b(upside down|inside out|backwards)\b', 0.1),
    (r'[A-Z][a-z]+\s+that\s+(cannot|should not|refuses to)\b', 0.12),
]

# Common/cliche phrases to penalize
CLICHE_PHRASES: list[str] = [
    "once upon a time",
    "it was a dark and stormy",
    "the moral of the story",
    "at the end of the day",
    "when all is said and done",
    "think outside the box",
    "paradigm shift",
    "synergy",
    "best practices",
    "leverage",
    "moving forward",
]


@dataclass
class InterestingnessModel:
    """
    Model for scoring content interestingness.

    Can be trained on human ratings to improve over time.
    """
    # Learned weights from human feedback
    feature_weights: dict[str, float] = field(default_factory=lambda: {
        "novelty": 1.0,
        "surprise": 1.2,  # Slightly prefer surprise
        "aesthetic": 0.9,
        "coherence_penalty": 0.8,
    })
    # History of seen content for novelty calculation
    seen_content: list[str] = field(default_factory=list)
    # Human ratings for training
    ratings: list[tuple[str, float]] = field(default_factory=list)

    def score(self, content: str, context: Optional[str] = None) -> InterestingnessScore:
        """
        Score content for interestingness.

        Args:
            content: The content to score.
            context: Optional context for surprise calculation.

        Returns:
            InterestingnessScore with all component scores.
        """
        novelty = self._novelty_score(content)
        surprise = self._surprise_score(content, context)
        aesthetic = self._aesthetic_score(content)
        coherence_penalty = self._coherence_penalty(content)

        return InterestingnessScore(
            novelty=novelty,
            surprise=surprise,
            aesthetic=aesthetic,
            coherence_penalty=coherence_penalty
        )

    def _novelty_score(self, content: str) -> float:
        """
        Calculate how novel the content is compared to history.

        Args:
            content: Content to evaluate.

        Returns:
            Novelty score from 0 to 1.
        """
        if not self.seen_content:
            return 0.8  # First content gets high novelty

        # Calculate similarity to seen content
        content_words = set(content.lower().split())
        max_similarity = 0.0

        for seen in self.seen_content[-50:]:  # Only check recent history
            seen_words = set(seen.lower().split())
            if not seen_words:
                continue
            intersection = content_words & seen_words
            union = content_words | seen_words
            similarity = len(intersection) / len(union) if union else 0.0
            max_similarity = max(max_similarity, similarity)

        # Novelty is inverse of similarity
        return 1.0 - max_similarity

    def _surprise_score(self, content: str, context: Optional[str]) -> float:
        """
        Calculate how surprising the content is given context.

        Args:
            content: Content to evaluate.
            context: Optional context that preceded this content.

        Returns:
            Surprise score from 0 to 1.
        """
        base_score = 0.5  # Neutral starting point

        # Check for interesting patterns
        content_lower = content.lower()
        for pattern, bonus in INTERESTING_PATTERNS:
            if re.search(pattern, content_lower):
                base_score += bonus

        # Context-based surprise
        if context:
            context_words = set(context.lower().split())
            content_words = set(content_lower.split())

            # More surprising if content introduces many new concepts
            new_words = content_words - context_words
            common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'of'}
            significant_new = new_words - common_words

            if content_words:
                new_ratio = len(significant_new) / len(content_words)
                base_score += new_ratio * 0.2

        # Check for unexpected juxtapositions
        if self._has_juxtaposition(content):
            base_score += 0.15

        return min(1.0, max(0.0, base_score))

    def _has_juxtaposition(self, content: str) -> bool:
        """Check if content contains unexpected juxtapositions."""
        # Look for patterns like "X meets Y" or "X and Y" with unrelated concepts
        juxtaposition_patterns = [
            r'(\w+)\s+meets?\s+(\w+)',
            r'(\w+)\s+and\s+(\w+)\s+collide',
            r'(\w+)\s+inside\s+(\w+)',
            r'the\s+(\w+)\s+of\s+(\w+)',
        ]

        for pattern in juxtaposition_patterns:
            match = re.search(pattern, content.lower())
            if match:
                word1, word2 = match.groups()
                # Very simple heuristic: different lengths suggest different concepts
                if abs(len(word1) - len(word2)) > 3:
                    return True

        return False

    def _aesthetic_score(self, content: str) -> float:
        """
        Score the aesthetic/evocative quality of content.

        Args:
            content: Content to evaluate.

        Returns:
            Aesthetic score from 0 to 1.
        """
        score = 0.5  # Baseline

        # Reward varied sentence lengths
        sentences = re.split(r'[.!?]+', content)
        if len(sentences) > 1:
            lengths = [len(s.split()) for s in sentences if s.strip()]
            if lengths:
                variance = self._calculate_variance(lengths)
                # Some variance is good, too much is chaotic
                if 5 < variance < 50:
                    score += 0.1

        # Reward sensory language
        sensory_words = ['taste', 'smell', 'touch', 'sound', 'feel', 'texture',
                        'warm', 'cold', 'soft', 'sharp', 'bright', 'dark']
        sensory_count = sum(1 for word in sensory_words if word in content.lower())
        score += min(0.2, sensory_count * 0.05)

        # Reward rhythm (roughly equal word counts in phrases)
        words_per_sentence = [len(s.split()) for s in sentences if s.strip()]
        if len(words_per_sentence) >= 3:
            # Check for rhythmic patterns
            diffs = [abs(words_per_sentence[i] - words_per_sentence[i+1])
                    for i in range(len(words_per_sentence)-1)]
            if diffs and sum(diffs) / len(diffs) < 3:
                score += 0.1

        # Penalize cliches
        for cliche in CLICHE_PHRASES:
            if cliche in content.lower():
                score -= 0.15

        return min(1.0, max(0.0, score))

    def _calculate_variance(self, values: list[int]) -> float:
        """Calculate variance of a list of values."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

    def _coherence_penalty(self, content: str) -> float:
        """
        Calculate penalty for overly logical/coherent content.

        Args:
            content: Content to evaluate.

        Returns:
            Penalty score from 0 to 1 (higher = more boring).
        """
        penalty = 0.0
        content_lower = content.lower()

        # Check for boring patterns
        for pattern, weight in BORING_PATTERNS:
            if re.search(pattern, content_lower):
                penalty += weight

        # Penalize very structured content
        lines = content.split('\n')
        if len(lines) > 3:
            # Check if lines follow a pattern (like all starting the same way)
            first_words = [line.split()[0] if line.split() else '' for line in lines]
            if len(set(first_words)) < len(first_words) / 2:
                penalty += 0.1

        # Penalize excessive length without weirdness
        if len(content) > 500:
            interesting_matches = sum(
                1 for pattern, _ in INTERESTING_PATTERNS
                if re.search(pattern, content_lower)
            )
            if interesting_matches < 2:
                penalty += 0.1

        return min(1.0, penalty)

    def add_to_history(self, content: str) -> None:
        """Add content to the seen history for novelty tracking."""
        self.seen_content.append(content)
        # Keep history bounded
        if len(self.seen_content) > 100:
            self.seen_content = self.seen_content[-100:]

    def add_rating(self, content: str, rating: float) -> None:
        """
        Add a human rating for training.

        Args:
            content: The content that was rated.
            rating: Human rating from 0 to 1.
        """
        self.ratings.append((content, rating))

    def learn_from_ratings(self) -> None:
        """
        Adjust model weights based on human ratings.

        Simple learning: compare predicted vs actual and adjust.
        """
        if len(self.ratings) < 10:
            return  # Need more data

        # Calculate prediction errors for each feature
        feature_errors: dict[str, list[float]] = {
            "novelty": [],
            "surprise": [],
            "aesthetic": [],
            "coherence_penalty": [],
        }

        for content, actual_rating in self.ratings[-50:]:
            score = self.score(content)
            predicted = score.total

            # Simple attribution: which features correlated with error?
            error = actual_rating - predicted

            # If we underestimated and feature was high, increase weight
            # If we overestimated and feature was high, decrease weight
            feature_errors["novelty"].append(error * score.novelty)
            feature_errors["surprise"].append(error * score.surprise)
            feature_errors["aesthetic"].append(error * score.aesthetic)
            feature_errors["coherence_penalty"].append(-error * score.coherence_penalty)

        # Adjust weights
        learning_rate = 0.1
        for feature, errors in feature_errors.items():
            if errors:
                adjustment = sum(errors) / len(errors) * learning_rate
                self.feature_weights[feature] += adjustment
                # Keep weights reasonable
                self.feature_weights[feature] = max(0.1, min(2.0, self.feature_weights[feature]))


# Global default model
_default_model: Optional[InterestingnessModel] = None


def get_default_model() -> InterestingnessModel:
    """Get or create the default interestingness model."""
    global _default_model
    if _default_model is None:
        _default_model = InterestingnessModel()
    return _default_model


def score_interestingness(
    content: str,
    context: Optional[str] = None,
    model: Optional[InterestingnessModel] = None
) -> InterestingnessScore:
    """
    Score content for interestingness.

    Args:
        content: The content to score.
        context: Optional context for surprise calculation.
        model: Optional custom model (uses default if not provided).

    Returns:
        InterestingnessScore with all components.
    """
    if model is None:
        model = get_default_model()
    return model.score(content, context)


def novelty_score(content: str, history: list[str]) -> float:
    """
    Calculate novelty score given history.

    Args:
        content: Content to evaluate.
        history: Previously seen content.

    Returns:
        Novelty score from 0 to 1.
    """
    model = InterestingnessModel()
    model.seen_content = history
    score = model.score(content)
    return score.novelty


def surprise_score(content: str, context: str) -> float:
    """
    Calculate surprise score given context.

    Args:
        content: Content to evaluate.
        context: Context that preceded this content.

    Returns:
        Surprise score from 0 to 1.
    """
    model = get_default_model()
    score = model.score(content, context)
    return score.surprise


def aesthetic_score(content: str) -> float:
    """
    Calculate aesthetic score.

    Args:
        content: Content to evaluate.

    Returns:
        Aesthetic score from 0 to 1.
    """
    model = get_default_model()
    score = model.score(content)
    return score.aesthetic


def coherence_penalty(content: str) -> float:
    """
    Calculate coherence penalty (being too logical).

    Args:
        content: Content to evaluate.

    Returns:
        Penalty from 0 to 1.
    """
    model = get_default_model()
    score = model.score(content)
    return score.coherence_penalty


def score_fragment(fragment: DreamFragment) -> InterestingnessScore:
    """
    Score a dream fragment for interestingness.

    Args:
        fragment: The fragment to score.

    Returns:
        InterestingnessScore for the fragment.
    """
    return score_interestingness(fragment.content)


def is_interesting_enough(content: str, threshold: float = 0.4) -> bool:
    """
    Check if content meets minimum interestingness threshold.

    Args:
        content: Content to check.
        threshold: Minimum acceptable score.

    Returns:
        True if content is interesting enough.
    """
    score = score_interestingness(content)
    return score.total >= threshold
