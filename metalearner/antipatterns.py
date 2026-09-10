"""Anti-patterns database for the meta-learner system.

This module tracks strategies that don't work, preventing the system
from repeatedly trying failed approaches.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

from .types import Strategy


@dataclass
class AntiPattern:
    """A strategy that doesn't work for a problem type.

    Attributes:
        strategy_id: ID of the failing strategy.
        strategy_name: Name of the strategy.
        problem_type: Problem type where it fails.
        reason: Why it fails.
        failure_count: Number of times it has failed.
        example_problems: Example problems where it failed.
        discovered_at: When this anti-pattern was discovered.
        last_failed_at: Most recent failure.
        severity: How bad the failures are (0.0-1.0).
    """
    strategy_id: str
    strategy_name: str
    problem_type: str
    reason: str
    failure_count: int = 1
    example_problems: list[str] = field(default_factory=list)
    discovered_at: datetime = field(default_factory=datetime.now)
    last_failed_at: datetime = field(default_factory=datetime.now)
    severity: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "problem_type": self.problem_type,
            "reason": self.reason,
            "failure_count": self.failure_count,
            "example_problems": self.example_problems,
            "discovered_at": self.discovered_at.isoformat(),
            "last_failed_at": self.last_failed_at.isoformat(),
            "severity": self.severity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AntiPattern":
        """Deserialize from dictionary."""
        discovered_at = data.get("discovered_at")
        if isinstance(discovered_at, str):
            discovered_at = datetime.fromisoformat(discovered_at)
        else:
            discovered_at = datetime.now()

        last_failed_at = data.get("last_failed_at")
        if isinstance(last_failed_at, str):
            last_failed_at = datetime.fromisoformat(last_failed_at)
        else:
            last_failed_at = datetime.now()

        return cls(
            strategy_id=data["strategy_id"],
            strategy_name=data["strategy_name"],
            problem_type=data["problem_type"],
            reason=data["reason"],
            failure_count=data.get("failure_count", 1),
            example_problems=data.get("example_problems", []),
            discovered_at=discovered_at,
            last_failed_at=last_failed_at,
            severity=data.get("severity", 0.5),
        )


class AntiPatternDB:
    """Database for tracking strategy anti-patterns."""

    def __init__(self, db_path: str | Path | None = None):
        """Initialize the anti-pattern database.

        Args:
            db_path: Path to store the database. None for in-memory.
        """
        self.db_path = Path(db_path) if db_path else None
        self._lock = Lock()

        # In-memory storage: (strategy_id, problem_type) -> AntiPattern
        self._patterns: dict[tuple[str, str], AntiPattern] = {}

        # Threshold for considering something an anti-pattern
        self.failure_threshold = 3
        self.severity_threshold = 0.3

        if self.db_path and self.db_path.exists():
            self._load()

    def _load(self) -> None:
        """Load from disk."""
        if not self.db_path:
            return

        try:
            with open(self.db_path) as f:
                data = json.load(f)

            for pattern_data in data.get("patterns", []):
                pattern = AntiPattern.from_dict(pattern_data)
                key = (pattern.strategy_id, pattern.problem_type)
                self._patterns[key] = pattern

        except (json.JSONDecodeError, KeyError):
            self._patterns = {}

    def _save(self) -> None:
        """Save to disk."""
        if not self.db_path:
            return

        data = {
            "patterns": [p.to_dict() for p in self._patterns.values()],
            "metadata": {
                "last_updated": datetime.now().isoformat(),
                "total_patterns": len(self._patterns),
            },
        }

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "w") as f:
            json.dump(data, f, indent=2)

    def record_antipattern(
        self,
        strategy: Strategy,
        problem_type: str,
        reason: str,
        example_problem: str | None = None,
        severity: float = 0.5,
    ) -> AntiPattern:
        """Record a strategy that doesn't work.

        Args:
            strategy: The failing strategy.
            problem_type: Type of problem it fails on.
            reason: Why it fails.
            example_problem: Example problem where it failed.
            severity: How severe the failure is.

        Returns:
            The recorded or updated anti-pattern.
        """
        with self._lock:
            key = (strategy.id, problem_type)

            if key in self._patterns:
                # Update existing
                pattern = self._patterns[key]
                pattern.failure_count += 1
                pattern.last_failed_at = datetime.now()
                pattern.severity = max(pattern.severity, severity)
                if example_problem and example_problem not in pattern.example_problems:
                    pattern.example_problems.append(example_problem)
                    # Keep only recent examples
                    pattern.example_problems = pattern.example_problems[-5:]
            else:
                # Create new
                pattern = AntiPattern(
                    strategy_id=strategy.id,
                    strategy_name=strategy.name,
                    problem_type=problem_type,
                    reason=reason,
                    example_problems=[example_problem] if example_problem else [],
                    severity=severity,
                )
                self._patterns[key] = pattern

            self._save()
            return pattern

    def is_antipattern(
        self,
        strategy: Strategy,
        problem_type: str,
    ) -> bool:
        """Check if a strategy is an anti-pattern for a problem type.

        Args:
            strategy: The strategy to check.
            problem_type: The problem type.

        Returns:
            True if this is a known anti-pattern.
        """
        with self._lock:
            key = (strategy.id, problem_type)
            pattern = self._patterns.get(key)

            if not pattern:
                return False

            return (
                pattern.failure_count >= self.failure_threshold
                or pattern.severity >= self.severity_threshold
            )

    def get_antipattern(
        self,
        strategy_id: str,
        problem_type: str,
    ) -> AntiPattern | None:
        """Get an anti-pattern if it exists.

        Args:
            strategy_id: Strategy ID.
            problem_type: Problem type.

        Returns:
            The anti-pattern or None.
        """
        with self._lock:
            return self._patterns.get((strategy_id, problem_type))

    def get_antipatterns_for_strategy(
        self,
        strategy_id: str,
    ) -> list[AntiPattern]:
        """Get all anti-patterns for a strategy.

        Args:
            strategy_id: Strategy ID.

        Returns:
            List of anti-patterns.
        """
        with self._lock:
            return [
                p for (sid, _), p in self._patterns.items()
                if sid == strategy_id
            ]

    def get_antipatterns_for_problem_type(
        self,
        problem_type: str,
    ) -> list[AntiPattern]:
        """Get all anti-patterns for a problem type.

        Args:
            problem_type: Problem type.

        Returns:
            List of anti-patterns.
        """
        with self._lock:
            return [
                p for (_, pt), p in self._patterns.items()
                if pt == problem_type
            ]

    def get_all_antipatterns(self) -> list[AntiPattern]:
        """Get all recorded anti-patterns.

        Returns:
            List of all anti-patterns.
        """
        with self._lock:
            return list(self._patterns.values())

    def get_worst_antipatterns(self, top_n: int = 10) -> list[AntiPattern]:
        """Get the worst anti-patterns by severity and failure count.

        Args:
            top_n: Number to return.

        Returns:
            List of worst anti-patterns.
        """
        with self._lock:
            patterns = list(self._patterns.values())
            # Score by severity * log(failure_count + 1)
            import math
            patterns.sort(
                key=lambda p: p.severity * math.log(p.failure_count + 1),
                reverse=True,
            )
            return patterns[:top_n]

    def get_strategy_blacklist(
        self,
        problem_type: str,
    ) -> list[str]:
        """Get list of strategy IDs to avoid for a problem type.

        Args:
            problem_type: Problem type.

        Returns:
            List of strategy IDs to avoid.
        """
        patterns = self.get_antipatterns_for_problem_type(problem_type)
        return [
            p.strategy_id for p in patterns
            if p.failure_count >= self.failure_threshold
            or p.severity >= self.severity_threshold
        ]

    def remove_antipattern(
        self,
        strategy_id: str,
        problem_type: str,
    ) -> bool:
        """Remove an anti-pattern (e.g., after strategy improvement).

        Args:
            strategy_id: Strategy ID.
            problem_type: Problem type.

        Returns:
            True if removed.
        """
        with self._lock:
            key = (strategy_id, problem_type)
            if key in self._patterns:
                del self._patterns[key]
                self._save()
                return True
            return False

    def decay_antipatterns(
        self,
        decay_factor: float = 0.9,
    ) -> int:
        """Decay anti-pattern severity over time.

        Args:
            decay_factor: Multiplier for severity.

        Returns:
            Number of anti-patterns decayed.
        """
        with self._lock:
            decayed = 0
            to_remove = []

            for key, pattern in self._patterns.items():
                pattern.severity *= decay_factor
                decayed += 1

                # Remove if severity drops too low
                if pattern.severity < 0.1 and pattern.failure_count < self.failure_threshold:
                    to_remove.append(key)

            for key in to_remove:
                del self._patterns[key]

            self._save()
            return decayed

    def get_statistics(self) -> dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with statistics.
        """
        with self._lock:
            if not self._patterns:
                return {
                    "total_patterns": 0,
                    "unique_strategies": 0,
                    "unique_problem_types": 0,
                    "avg_severity": 0.0,
                    "avg_failure_count": 0.0,
                }

            strategies = set()
            problem_types = set()
            total_severity = 0.0
            total_failures = 0

            for (sid, pt), pattern in self._patterns.items():
                strategies.add(sid)
                problem_types.add(pt)
                total_severity += pattern.severity
                total_failures += pattern.failure_count

            n = len(self._patterns)

            return {
                "total_patterns": n,
                "unique_strategies": len(strategies),
                "unique_problem_types": len(problem_types),
                "avg_severity": total_severity / n,
                "avg_failure_count": total_failures / n,
            }

    def clear(self) -> None:
        """Clear all anti-patterns."""
        with self._lock:
            self._patterns = {}
            if self.db_path and self.db_path.exists():
                self.db_path.unlink()


# Global instance
_default_db: AntiPatternDB | None = None


def get_antipattern_db(db_path: str | Path | None = None) -> AntiPatternDB:
    """Get the default anti-pattern database.

    Args:
        db_path: Path for database (only used on first call).

    Returns:
        The database instance.
    """
    global _default_db
    if _default_db is None:
        _default_db = AntiPatternDB(db_path)
    return _default_db


def record_antipattern(
    strategy: Strategy,
    problem_type: str,
    reason: str,
    example_problem: str | None = None,
) -> AntiPattern:
    """Record an anti-pattern.

    Args:
        strategy: The failing strategy.
        problem_type: Problem type.
        reason: Why it fails.
        example_problem: Example problem.

    Returns:
        The recorded anti-pattern.
    """
    return get_antipattern_db().record_antipattern(
        strategy, problem_type, reason, example_problem
    )


def is_antipattern(strategy: Strategy, problem_type: str) -> bool:
    """Check if a strategy is an anti-pattern.

    Args:
        strategy: Strategy to check.
        problem_type: Problem type.

    Returns:
        True if it's an anti-pattern.
    """
    return get_antipattern_db().is_antipattern(strategy, problem_type)


def get_strategy_blacklist(problem_type: str) -> list[str]:
    """Get blacklisted strategies for a problem type.

    Args:
        problem_type: Problem type.

    Returns:
        List of strategy IDs to avoid.
    """
    return get_antipattern_db().get_strategy_blacklist(problem_type)
