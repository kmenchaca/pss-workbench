"""Persona Performance Tracking - Learning which personas add value.

Tracks persona effectiveness over time to improve future
persona selection and identify which perspectives are most valuable
for different problem types.
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from .types import PersonaPerformance


# ============================================================================
# Persona Tracker
# ============================================================================


class PersonaTracker:
    """Tracks persona performance metrics over time.

    Records usage, success rates, unique insights, and redundancy
    to learn which personas are most valuable for which problems.
    """

    def __init__(self, persistence_path: Optional[Path] = None):
        """Initialize the tracker.

        Args:
            persistence_path: Path to save/load performance data (optional)
        """
        self._data: dict[str, PersonaPerformance] = {}
        self._persistence_path = persistence_path

        if persistence_path and persistence_path.exists():
            self._load()

    def _get_or_create(self, persona_id: str) -> PersonaPerformance:
        """Get existing performance data or create new entry.

        Args:
            persona_id: ID of the persona

        Returns:
            PersonaPerformance for the persona
        """
        if persona_id not in self._data:
            self._data[persona_id] = PersonaPerformance(persona_id=persona_id)
        return self._data[persona_id]

    def record_use(
        self,
        persona_id: str,
        success: bool = True,
        unique_insights: int = 0,
        was_redundant: bool = False,
        problem_type: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> None:
        """Record a persona usage event.

        Args:
            persona_id: ID of the persona used
            success: Whether the persona provided useful output
            unique_insights: Count of unique insights provided
            was_redundant: Whether persona duplicated other responses
            problem_type: Type of problem (for category tracking)
            confidence: Confidence score of response
        """
        perf = self._get_or_create(persona_id)

        # Update totals
        perf.total_uses += 1
        perf.unique_insights += unique_insights

        # Update success rate (rolling average)
        old_successes = perf.success_rate * (perf.total_uses - 1)
        perf.success_rate = (old_successes + (1.0 if success else 0.0)) / perf.total_uses

        # Update redundancy rate
        old_redundant = perf.redundancy_rate * (perf.total_uses - 1)
        perf.redundancy_rate = (old_redundant + (1.0 if was_redundant else 0.0)) / perf.total_uses

        # Update problem type performance
        if problem_type:
            if problem_type not in perf.problem_types:
                perf.problem_types[problem_type] = 0.0
            old_type_score = perf.problem_types[problem_type]
            # Simple exponential moving average
            perf.problem_types[problem_type] = 0.8 * old_type_score + 0.2 * (
                1.0 if success else 0.0
            )

        # Update confidence average
        if confidence is not None:
            old_conf_sum = perf.avg_confidence * (perf.total_uses - 1)
            perf.avg_confidence = (old_conf_sum + confidence) / perf.total_uses

        # Update timestamp
        perf.last_used = datetime.now().isoformat()

        # Persist if enabled
        if self._persistence_path:
            self._save()

    def get_performance(self, persona_id: str) -> Optional[PersonaPerformance]:
        """Get performance data for a persona.

        Args:
            persona_id: ID of the persona

        Returns:
            PersonaPerformance if available, None otherwise
        """
        return self._data.get(persona_id)

    def persona_value_score(self, persona_id: str) -> float:
        """Calculate overall value score for a persona.

        Combines success rate, unique insights, and redundancy into
        a single value metric.

        Args:
            persona_id: ID of the persona

        Returns:
            Value score from 0.0 to 1.0
        """
        perf = self._data.get(persona_id)
        if not perf or perf.total_uses == 0:
            return 0.5  # Unknown = neutral

        # Weight components
        success_weight = 0.4
        uniqueness_weight = 0.4
        redundancy_penalty = 0.2

        # Calculate uniqueness ratio (unique insights per use)
        uniqueness = min(1.0, perf.unique_insights / max(perf.total_uses, 1))

        # Combine into value score
        value = (
            success_weight * perf.success_rate
            + uniqueness_weight * uniqueness
            - redundancy_penalty * perf.redundancy_rate
        )

        return max(0.0, min(1.0, value))

    def recommend_personas(
        self,
        problem_type: Optional[str] = None,
        count: int = 3,
    ) -> list[str]:
        """Recommend personas based on historical performance.

        Args:
            problem_type: Optional problem type to optimize for
            count: Number of recommendations

        Returns:
            List of recommended persona IDs
        """
        if not self._data:
            return []

        scores: dict[str, float] = {}

        for persona_id, perf in self._data.items():
            if problem_type and problem_type in perf.problem_types:
                # Use problem-specific performance
                scores[persona_id] = perf.problem_types[problem_type]
            else:
                # Use general value score
                scores[persona_id] = self.persona_value_score(persona_id)

        # Sort by score descending
        sorted_personas = sorted(scores.items(), key=lambda x: -x[1])

        return [p[0] for p in sorted_personas[:count]]

    def get_top_performers(self, count: int = 5) -> list[tuple[str, float]]:
        """Get top performing personas by value score.

        Args:
            count: Number of top performers to return

        Returns:
            List of (persona_id, score) tuples
        """
        scores = [(pid, self.persona_value_score(pid)) for pid in self._data]
        scores.sort(key=lambda x: -x[1])
        return scores[:count]

    def get_underperformers(self, threshold: float = 0.3) -> list[str]:
        """Get personas performing below threshold.

        Args:
            threshold: Value score threshold

        Returns:
            List of underperforming persona IDs
        """
        return [
            pid for pid in self._data if self.persona_value_score(pid) < threshold
        ]

    def get_redundant_personas(self, threshold: float = 0.5) -> list[str]:
        """Get personas with high redundancy rates.

        Args:
            threshold: Redundancy rate threshold

        Returns:
            List of frequently redundant persona IDs
        """
        return [
            pid
            for pid, perf in self._data.items()
            if perf.redundancy_rate >= threshold and perf.total_uses >= 3
        ]

    def get_statistics(self) -> dict:
        """Get overall tracking statistics.

        Returns:
            Dictionary of statistics
        """
        if not self._data:
            return {
                "total_personas": 0,
                "total_uses": 0,
                "avg_success_rate": 0.0,
                "total_unique_insights": 0,
            }

        total_uses = sum(p.total_uses for p in self._data.values())
        total_insights = sum(p.unique_insights for p in self._data.values())
        avg_success = (
            sum(p.success_rate * p.total_uses for p in self._data.values()) / total_uses
            if total_uses > 0
            else 0.0
        )

        return {
            "total_personas": len(self._data),
            "total_uses": total_uses,
            "avg_success_rate": avg_success,
            "total_unique_insights": total_insights,
            "avg_value_score": sum(self.persona_value_score(pid) for pid in self._data)
            / len(self._data),
        }

    def reset(self) -> None:
        """Reset all tracking data."""
        self._data.clear()
        if self._persistence_path and self._persistence_path.exists():
            self._persistence_path.unlink()

    def _save(self) -> None:
        """Save performance data to disk."""
        if not self._persistence_path:
            return

        data = {pid: asdict(perf) for pid, perf in self._data.items()}
        self._persistence_path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        """Load performance data from disk."""
        if not self._persistence_path or not self._persistence_path.exists():
            return

        try:
            data = json.loads(self._persistence_path.read_text())
            for pid, perf_dict in data.items():
                self._data[pid] = PersonaPerformance(**perf_dict)
        except (json.JSONDecodeError, TypeError):
            # Invalid data, start fresh
            self._data.clear()


# ============================================================================
# Analysis Functions
# ============================================================================


def analyze_persona_effectiveness(
    tracker: PersonaTracker,
) -> dict[str, dict]:
    """Analyze effectiveness of all tracked personas.

    Args:
        tracker: PersonaTracker with data

    Returns:
        Dictionary of analysis per persona
    """
    analysis = {}

    for persona_id in tracker._data:
        perf = tracker.get_performance(persona_id)
        if not perf:
            continue

        value = tracker.persona_value_score(persona_id)

        # Determine effectiveness category
        if value >= 0.7:
            category = "high_value"
        elif value >= 0.4:
            category = "moderate_value"
        else:
            category = "low_value"

        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []

        if perf.success_rate >= 0.7:
            strengths.append("high_success_rate")
        elif perf.success_rate < 0.4:
            weaknesses.append("low_success_rate")

        if perf.unique_insights / max(perf.total_uses, 1) >= 0.5:
            strengths.append("provides_unique_insights")
        elif perf.unique_insights == 0 and perf.total_uses >= 3:
            weaknesses.append("no_unique_insights")

        if perf.redundancy_rate >= 0.5:
            weaknesses.append("often_redundant")
        elif perf.redundancy_rate < 0.2:
            strengths.append("low_redundancy")

        analysis[persona_id] = {
            "value_score": value,
            "category": category,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "total_uses": perf.total_uses,
            "best_problem_types": sorted(
                perf.problem_types.items(), key=lambda x: -x[1]
            )[:3],
        }

    return analysis


def compare_personas(
    tracker: PersonaTracker,
    persona_a: str,
    persona_b: str,
) -> dict:
    """Compare two personas' performance.

    Args:
        tracker: PersonaTracker with data
        persona_a: First persona ID
        persona_b: Second persona ID

    Returns:
        Comparison dictionary
    """
    perf_a = tracker.get_performance(persona_a)
    perf_b = tracker.get_performance(persona_b)

    if not perf_a or not perf_b:
        return {"error": "One or both personas not found"}

    value_a = tracker.persona_value_score(persona_a)
    value_b = tracker.persona_value_score(persona_b)

    return {
        "persona_a": {
            "id": persona_a,
            "value_score": value_a,
            "success_rate": perf_a.success_rate,
            "unique_insights": perf_a.unique_insights,
            "total_uses": perf_a.total_uses,
        },
        "persona_b": {
            "id": persona_b,
            "value_score": value_b,
            "success_rate": perf_b.success_rate,
            "unique_insights": perf_b.unique_insights,
            "total_uses": perf_b.total_uses,
        },
        "comparison": {
            "higher_value": persona_a if value_a > value_b else persona_b,
            "value_difference": abs(value_a - value_b),
            "more_unique_insights": (
                persona_a
                if perf_a.unique_insights > perf_b.unique_insights
                else persona_b
            ),
        },
    }


def suggest_persona_improvements(tracker: PersonaTracker) -> list[str]:
    """Generate suggestions for improving persona ensemble.

    Args:
        tracker: PersonaTracker with data

    Returns:
        List of improvement suggestions
    """
    suggestions = []
    stats = tracker.get_statistics()

    if stats["total_uses"] < 10:
        suggestions.append(
            "Insufficient data. Run more ensembles to get meaningful recommendations."
        )
        return suggestions

    # Check for underperformers
    underperformers = tracker.get_underperformers(0.3)
    if underperformers:
        suggestions.append(
            f"Consider replacing low-value personas: {', '.join(underperformers)}"
        )

    # Check for redundancy
    redundant = tracker.get_redundant_personas(0.5)
    if redundant:
        suggestions.append(
            f"Personas frequently redundant with others: {', '.join(redundant)}. "
            "Consider replacing with more diverse perspectives."
        )

    # Check overall success rate
    if stats["avg_success_rate"] < 0.5:
        suggestions.append(
            "Overall success rate is low. Review problem framing and persona prompts."
        )

    # Check for missing insights
    avg_insights = stats["total_unique_insights"] / max(stats["total_uses"], 1)
    if avg_insights < 0.3:
        suggestions.append(
            "Low unique insight rate. Add more contrasting personas for diversity."
        )

    # Check for imbalanced usage
    if stats["total_personas"] > 0:
        uses_per_persona = [
            tracker.get_performance(pid).total_uses
            for pid in tracker._data
            if tracker.get_performance(pid)
        ]
        if uses_per_persona:
            max_uses = max(uses_per_persona)
            min_uses = min(uses_per_persona)
            if max_uses > 3 * min_uses and min_uses > 0:
                suggestions.append(
                    "Persona usage is imbalanced. Consider using assignment strategies "
                    "that ensure even coverage."
                )

    if not suggestions:
        suggestions.append("Persona ensemble is performing well. No changes needed.")

    return suggestions
