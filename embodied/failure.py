"""
Failure sharing across exploration branches.

Prevents repeated mistakes by tracking and propagating failure information.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import hashlib
import json

from .types import Action, ActionResult, Observation
from .observation import observation_hash


@dataclass
class FailureRecord:
    """Record of a failed action.

    Attributes:
        action_type: Type of action that failed
        action_params: Parameters of failed action
        state_hash: Hash of state where failure occurred
        error: Error message
        timestamp: When failure occurred
        count: Number of times this failure has occurred
        context: Additional context about the failure
    """
    action_type: str
    action_params: dict[str, Any]
    state_hash: str
    error: str
    timestamp: datetime = field(default_factory=datetime.now)
    count: int = 1
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Generate unique key for this failure type."""
        params_str = json.dumps(self.action_params, sort_keys=True)
        key_str = f"{self.action_type}:{params_str}:{self.state_hash}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def matches(self, action: Action, state_hash: str) -> bool:
        """Check if this failure matches an action in a state.

        Args:
            action: Action to check
            state_hash: Hash of current state

        Returns:
            True if this failure matches
        """
        if action.action_type != self.action_type:
            return False

        # Check state similarity (could be relaxed for generalization)
        if state_hash != self.state_hash:
            return False

        # Check parameter overlap
        for key, value in self.action_params.items():
            if key in action.parameters and action.parameters[key] == value:
                return True

        return False


class FailureMemory:
    """Tracks failed actions across exploration.

    Provides methods to record failures and check if an action
    should be avoided based on past failures.

    Attributes:
        failures: Dict of failure key to FailureRecord
        max_failures: Maximum failures to track
        decay_factor: How much to decay old failures
    """

    def __init__(
        self,
        max_failures: int = 1000,
        decay_factor: float = 0.9,
    ):
        self.failures: dict[str, FailureRecord] = {}
        self.max_failures = max_failures
        self.decay_factor = decay_factor
        self._total_recorded = 0

    def record_failure(
        self,
        action: Action,
        state: Observation,
        error: str,
        context: Optional[dict[str, Any]] = None,
    ) -> FailureRecord:
        """Log a failed action.

        Args:
            action: Action that failed
            state: State where failure occurred
            error: Error message
            context: Additional context

        Returns:
            FailureRecord created or updated
        """
        state_hash = observation_hash(state)

        record = FailureRecord(
            action_type=action.action_type,
            action_params=action.parameters.copy(),
            state_hash=state_hash,
            error=error,
            context=context or {},
        )

        key = record.key

        if key in self.failures:
            # Update existing record
            existing = self.failures[key]
            existing.count += 1
            existing.timestamp = datetime.now()
            if context:
                existing.context.update(context)
            return existing
        else:
            # Add new record
            self.failures[key] = record
            self._total_recorded += 1

            # Prune if over limit
            if len(self.failures) > self.max_failures:
                self._prune_old_failures()

            return record

    def should_avoid(
        self,
        action: Action,
        state: Observation,
        threshold: int = 1,
    ) -> tuple[bool, Optional[str]]:
        """Check if an action should be avoided based on failure history.

        Args:
            action: Action to check
            state: Current state
            threshold: Minimum failure count to trigger avoidance

        Returns:
            Tuple of (should_avoid, reason)
        """
        state_hash = observation_hash(state)

        for record in self.failures.values():
            if record.matches(action, state_hash) and record.count >= threshold:
                reason = f"Action '{action}' failed {record.count} times in similar state: {record.error}"
                return True, reason

        return False, None

    def get_failures_for_state(
        self,
        state: Observation,
    ) -> list[FailureRecord]:
        """Get all failures that occurred in a similar state.

        Args:
            state: State to check

        Returns:
            List of matching failure records
        """
        state_hash = observation_hash(state)
        return [
            record for record in self.failures.values()
            if record.state_hash == state_hash
        ]

    def get_failures_for_action(
        self,
        action_type: str,
    ) -> list[FailureRecord]:
        """Get all failures for a specific action type.

        Args:
            action_type: Action type to check

        Returns:
            List of matching failure records
        """
        return [
            record for record in self.failures.values()
            if record.action_type == action_type
        ]

    def _prune_old_failures(self) -> None:
        """Remove old, infrequent failures."""
        if not self.failures:
            return

        # Sort by (count * recency)
        scored = []
        now = datetime.now()

        for key, record in self.failures.items():
            age_hours = (now - record.timestamp).total_seconds() / 3600
            recency = self.decay_factor ** age_hours
            score = record.count * recency
            scored.append((key, score))

        # Keep top max_failures/2
        scored.sort(key=lambda x: x[1], reverse=True)
        keep_keys = {key for key, _ in scored[:self.max_failures // 2]}

        self.failures = {
            key: record
            for key, record in self.failures.items()
            if key in keep_keys
        }

    def clear(self) -> None:
        """Clear all failure records."""
        self.failures.clear()
        self._total_recorded = 0

    @property
    def total_failures(self) -> int:
        """Total number of unique failures recorded."""
        return len(self.failures)

    def summarize(self) -> dict[str, Any]:
        """Get summary statistics of failure memory.

        Returns:
            Dict with summary statistics
        """
        if not self.failures:
            return {
                "total_unique": 0,
                "total_recorded": self._total_recorded,
                "action_types": {},
            }

        action_counts: dict[str, int] = {}
        for record in self.failures.values():
            action_counts[record.action_type] = \
                action_counts.get(record.action_type, 0) + record.count

        return {
            "total_unique": len(self.failures),
            "total_recorded": self._total_recorded,
            "action_types": action_counts,
            "most_common_action": max(action_counts, key=action_counts.get) if action_counts else None,
        }


def propagate_failure(
    source_memory: FailureMemory,
    target_memories: list[FailureMemory],
    filter_recent: bool = True,
    max_age_hours: float = 1.0,
) -> int:
    """Share failure information with sibling branches.

    Args:
        source_memory: Memory to copy from
        target_memories: Memories to copy to
        filter_recent: Only propagate recent failures
        max_age_hours: Maximum age of failures to propagate

    Returns:
        Number of failures propagated
    """
    propagated = 0
    now = datetime.now()

    for record in source_memory.failures.values():
        # Check age if filtering
        if filter_recent:
            age_hours = (now - record.timestamp).total_seconds() / 3600
            if age_hours > max_age_hours:
                continue

        # Propagate to each target
        for target in target_memories:
            if record.key not in target.failures:
                # Copy record (with reduced count for propagated failures)
                new_record = FailureRecord(
                    action_type=record.action_type,
                    action_params=record.action_params.copy(),
                    state_hash=record.state_hash,
                    error=record.error,
                    timestamp=record.timestamp,
                    count=max(1, record.count // 2),  # Reduce weight for propagated
                    context={**record.context, "propagated": True},
                )
                target.failures[new_record.key] = new_record
                propagated += 1

    return propagated


class FailureAwareActionFilter:
    """Filters actions based on failure memory.

    Provides a way to filter out actions that have failed too many times.
    """

    def __init__(
        self,
        memory: FailureMemory,
        threshold: int = 2,
        suggest_alternatives: bool = True,
    ):
        self.memory = memory
        self.threshold = threshold
        self.suggest_alternatives = suggest_alternatives

    def filter_actions(
        self,
        actions: list[Action],
        state: Observation,
    ) -> list[Action]:
        """Filter out actions that should be avoided.

        Args:
            actions: Actions to filter
            state: Current state

        Returns:
            Filtered list of actions
        """
        filtered = []

        for action in actions:
            should_avoid, reason = self.memory.should_avoid(
                action, state, self.threshold
            )

            if not should_avoid:
                filtered.append(action)

        return filtered

    def annotate_actions(
        self,
        actions: list[Action],
        state: Observation,
    ) -> list[tuple[Action, Optional[str]]]:
        """Annotate actions with failure warnings.

        Args:
            actions: Actions to annotate
            state: Current state

        Returns:
            List of (action, warning) tuples
        """
        annotated = []

        for action in actions:
            should_avoid, reason = self.memory.should_avoid(
                action, state, self.threshold
            )

            annotated.append((action, reason if should_avoid else None))

        return annotated
