"""Performance database for the meta-learner system.

This module provides persistent storage for strategy performance data,
enabling learning from historical results.
"""

import json
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

from .types import (
    PerformanceRecord,
    Strategy,
    StrategyResult,
)


class PerformanceDB:
    """Database for tracking strategy performance over time."""

    def __init__(self, db_path: str | Path | None = None):
        """Initialize the database.

        Args:
            db_path: Path to store the database. None for in-memory only.
        """
        self.db_path = Path(db_path) if db_path else None
        self._lock = Lock()

        # In-memory storage
        self._records: dict[str, dict[str, PerformanceRecord]] = {}  # strategy_id -> problem_type -> record
        self._strategies: dict[str, Strategy] = {}
        self._results_count = 0

        # Load existing data if path provided
        if self.db_path and self.db_path.exists():
            self._load()

    def _load(self) -> None:
        """Load database from disk."""
        if not self.db_path:
            return

        try:
            with open(self.db_path) as f:
                data = json.load(f)

            # Load strategies
            for s_data in data.get("strategies", []):
                strategy = Strategy.from_dict(s_data)
                self._strategies[strategy.id] = strategy

            # Load records
            for record_data in data.get("records", []):
                record = PerformanceRecord.from_dict(record_data)
                if record.strategy_id not in self._records:
                    self._records[record.strategy_id] = {}
                self._records[record.strategy_id][record.problem_type] = record
                self._results_count += len(record.results)

        except (json.JSONDecodeError, KeyError) as e:
            # Start fresh if file is corrupted
            self._records = {}
            self._strategies = {}

    def _save(self) -> None:
        """Save database to disk."""
        if not self.db_path:
            return

        # Collect all records
        all_records = []
        for strategy_records in self._records.values():
            for record in strategy_records.values():
                all_records.append(record.to_dict())

        data = {
            "strategies": [s.to_dict() for s in self._strategies.values()],
            "records": all_records,
            "metadata": {
                "last_updated": datetime.now().isoformat(),
                "total_results": self._results_count,
            },
        }

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "w") as f:
            json.dump(data, f, indent=2)

    def record_result(
        self,
        strategy: Strategy,
        problem_type: str,
        result: StrategyResult,
    ) -> None:
        """Record a strategy result.

        Args:
            strategy: The strategy that was used.
            problem_type: Type of problem solved.
            result: The result to record.
        """
        with self._lock:
            # Store strategy if not seen
            if strategy.id not in self._strategies:
                self._strategies[strategy.id] = strategy

            # Initialize records for strategy
            if strategy.id not in self._records:
                self._records[strategy.id] = {}

            # Get or create record for this strategy/problem_type combo
            if problem_type not in self._records[strategy.id]:
                self._records[strategy.id][problem_type] = PerformanceRecord(
                    strategy_id=strategy.id,
                    problem_type=problem_type,
                )

            # Add result
            self._records[strategy.id][problem_type].results.append(result)
            self._results_count += 1

            # Auto-save periodically
            if self._results_count % 10 == 0:
                self._save()

    def query_performance(
        self,
        strategy_id: str,
        problem_type: str | None = None,
    ) -> PerformanceRecord | dict[str, PerformanceRecord] | None:
        """Query performance data for a strategy.

        Args:
            strategy_id: The strategy to query.
            problem_type: Specific problem type, or None for all.

        Returns:
            PerformanceRecord if problem_type specified, dict of records otherwise.
        """
        with self._lock:
            if strategy_id not in self._records:
                return None

            records = self._records[strategy_id]

            if problem_type:
                return records.get(problem_type)
            return dict(records)

    def best_strategy_for(
        self,
        problem_type: str,
        min_samples: int = 3,
        metric: str = "success_rate",
    ) -> tuple[Strategy | None, float]:
        """Find the best strategy for a problem type.

        Args:
            problem_type: The type of problem.
            min_samples: Minimum samples required.
            metric: Metric to optimize ("success_rate", "avg_quality", "avg_tokens").

        Returns:
            Tuple of (best_strategy, score) or (None, 0.0).
        """
        with self._lock:
            candidates: list[tuple[Strategy, float]] = []

            for strategy_id, type_records in self._records.items():
                if problem_type not in type_records:
                    continue

                record = type_records[problem_type]
                if record.sample_count < min_samples:
                    continue

                strategy = self._strategies.get(strategy_id)
                if not strategy:
                    continue

                if metric == "success_rate":
                    score = record.success_rate
                elif metric == "avg_quality":
                    score = record.avg_quality
                elif metric == "avg_tokens":
                    # Lower is better for tokens, so invert
                    score = 1.0 / (1.0 + record.avg_tokens / 1000)
                else:
                    score = record.success_rate

                candidates.append((strategy, score))

            if not candidates:
                return None, 0.0

            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0]

    def strategy_rankings(
        self,
        problem_type: str | None = None,
        metric: str = "success_rate",
        min_samples: int = 1,
    ) -> list[tuple[Strategy, float]]:
        """Get ranked list of strategies.

        Args:
            problem_type: Filter by problem type, or None for overall.
            metric: Metric to rank by.
            min_samples: Minimum samples required.

        Returns:
            List of (strategy, score) tuples, sorted descending.
        """
        with self._lock:
            scores: dict[str, list[float]] = {}

            for strategy_id, type_records in self._records.items():
                for pt, record in type_records.items():
                    if problem_type and pt != problem_type:
                        continue
                    if record.sample_count < min_samples:
                        continue

                    if strategy_id not in scores:
                        scores[strategy_id] = []

                    if metric == "success_rate":
                        scores[strategy_id].append(record.success_rate)
                    elif metric == "avg_quality":
                        scores[strategy_id].append(record.avg_quality)
                    elif metric == "avg_tokens":
                        scores[strategy_id].append(
                            1.0 / (1.0 + record.avg_tokens / 1000)
                        )

            # Average scores across problem types
            rankings = []
            for strategy_id, score_list in scores.items():
                strategy = self._strategies.get(strategy_id)
                if strategy and score_list:
                    avg_score = sum(score_list) / len(score_list)
                    rankings.append((strategy, avg_score))

            rankings.sort(key=lambda x: x[1], reverse=True)
            return rankings

    def get_strategy(self, strategy_id: str) -> Strategy | None:
        """Get a strategy by ID.

        Args:
            strategy_id: The strategy ID.

        Returns:
            The strategy if found.
        """
        return self._strategies.get(strategy_id)

    def get_all_strategies(self) -> list[Strategy]:
        """Get all known strategies.

        Returns:
            List of all strategies.
        """
        return list(self._strategies.values())

    def get_problem_types(self) -> list[str]:
        """Get all problem types with recorded results.

        Returns:
            List of problem type strings.
        """
        types: set[str] = set()
        for type_records in self._records.values():
            types.update(type_records.keys())
        return sorted(types)

    def get_statistics(self) -> dict[str, Any]:
        """Get overall database statistics.

        Returns:
            Dictionary with statistics.
        """
        with self._lock:
            total_results = 0
            total_successes = 0
            total_tokens = 0

            for type_records in self._records.values():
                for record in type_records.values():
                    for result in record.results:
                        total_results += 1
                        if result.success:
                            total_successes += 1
                        total_tokens += result.tokens_used

            return {
                "total_strategies": len(self._strategies),
                "total_problem_types": len(self.get_problem_types()),
                "total_results": total_results,
                "overall_success_rate": (
                    total_successes / total_results if total_results else 0.0
                ),
                "total_tokens_used": total_tokens,
            }

    def clear(self) -> None:
        """Clear all data from the database."""
        with self._lock:
            self._records = {}
            self._strategies = {}
            self._results_count = 0
            if self.db_path and self.db_path.exists():
                self.db_path.unlink()

    def save(self) -> None:
        """Force save to disk."""
        with self._lock:
            self._save()

    def export_json(self) -> str:
        """Export database as JSON string.

        Returns:
            JSON representation of the database.
        """
        with self._lock:
            all_records = []
            for strategy_records in self._records.values():
                for record in strategy_records.values():
                    all_records.append(record.to_dict())

            data = {
                "strategies": [s.to_dict() for s in self._strategies.values()],
                "records": all_records,
            }
            return json.dumps(data, indent=2)

    def import_json(self, json_str: str) -> int:
        """Import data from JSON string.

        Args:
            json_str: JSON data to import.

        Returns:
            Number of records imported.
        """
        data = json.loads(json_str)
        count = 0

        with self._lock:
            for s_data in data.get("strategies", []):
                strategy = Strategy.from_dict(s_data)
                if strategy.id not in self._strategies:
                    self._strategies[strategy.id] = strategy

            for record_data in data.get("records", []):
                record = PerformanceRecord.from_dict(record_data)
                if record.strategy_id not in self._records:
                    self._records[record.strategy_id] = {}

                existing = self._records[record.strategy_id].get(record.problem_type)
                if existing:
                    # Merge results
                    existing_timestamps = {
                        r.timestamp.isoformat() for r in existing.results
                    }
                    for result in record.results:
                        if result.timestamp.isoformat() not in existing_timestamps:
                            existing.results.append(result)
                            count += 1
                else:
                    self._records[record.strategy_id][record.problem_type] = record
                    count += len(record.results)

            self._save()

        return count

    def prune_old_results(
        self,
        max_age_days: int = 30,
        keep_min_samples: int = 10,
    ) -> int:
        """Remove old results while keeping minimum samples.

        Args:
            max_age_days: Maximum age in days.
            keep_min_samples: Minimum samples to keep per record.

        Returns:
            Number of results removed.
        """
        cutoff = datetime.now().timestamp() - (max_age_days * 86400)
        removed = 0

        with self._lock:
            for strategy_records in self._records.values():
                for record in strategy_records.values():
                    if len(record.results) <= keep_min_samples:
                        continue

                    # Sort by timestamp descending
                    record.results.sort(key=lambda r: r.timestamp, reverse=True)

                    # Keep recent and minimum samples
                    new_results = []
                    for result in record.results:
                        if (
                            len(new_results) < keep_min_samples
                            or result.timestamp.timestamp() > cutoff
                        ):
                            new_results.append(result)
                        else:
                            removed += 1

                    record.results = new_results

            self._results_count -= removed
            self._save()

        return removed


# Global database instance
_default_db: PerformanceDB | None = None


def get_db(db_path: str | Path | None = None) -> PerformanceDB:
    """Get the default database instance.

    Args:
        db_path: Path for the database. Only used on first call.

    Returns:
        The database instance.
    """
    global _default_db
    if _default_db is None:
        _default_db = PerformanceDB(db_path)
    return _default_db


def record_result(
    strategy: Strategy,
    problem_type: str,
    result: StrategyResult,
) -> None:
    """Record a result to the default database.

    Args:
        strategy: The strategy used.
        problem_type: Type of problem.
        result: The result.
    """
    get_db().record_result(strategy, problem_type, result)


def query_performance(
    strategy_id: str,
    problem_type: str | None = None,
) -> PerformanceRecord | dict[str, PerformanceRecord] | None:
    """Query performance from the default database.

    Args:
        strategy_id: Strategy to query.
        problem_type: Optional problem type filter.

    Returns:
        Performance record(s).
    """
    return get_db().query_performance(strategy_id, problem_type)


def best_strategy_for(
    problem_type: str,
    min_samples: int = 3,
) -> tuple[Strategy | None, float]:
    """Get best strategy for a problem type.

    Args:
        problem_type: The problem type.
        min_samples: Minimum samples required.

    Returns:
        Tuple of (strategy, score).
    """
    return get_db().best_strategy_for(problem_type, min_samples)


def strategy_rankings(
    problem_type: str | None = None,
    metric: str = "success_rate",
) -> list[tuple[Strategy, float]]:
    """Get strategy rankings.

    Args:
        problem_type: Optional problem type filter.
        metric: Metric to rank by.

    Returns:
        List of (strategy, score) tuples.
    """
    return get_db().strategy_rankings(problem_type, metric)
