"""
Action execution utilities.

Safe execution of actions with retry logic, timeouts, and constraints.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional
import time

from .environment import Environment
from .types import Action, ActionResult, Observation


@dataclass
class ExecutionConstraints:
    """Constraints for safe action execution.

    Attributes:
        max_retries: Maximum retry attempts
        timeout_seconds: Timeout for action execution
        forbidden_actions: Actions that should not be executed
        require_preconditions: Whether to check preconditions
        max_cost: Maximum cost/resource usage allowed
    """
    max_retries: int = 3
    timeout_seconds: float = 30.0
    forbidden_actions: list[str] = field(default_factory=list)
    require_preconditions: bool = True
    max_cost: Optional[float] = None


@dataclass
class ExecutionStats:
    """Statistics about action execution.

    Attributes:
        total_actions: Total actions attempted
        successful_actions: Actions that succeeded
        failed_actions: Actions that failed
        retried_actions: Actions that required retries
        total_time: Total execution time
    """
    total_actions: int = 0
    successful_actions: int = 0
    failed_actions: int = 0
    retried_actions: int = 0
    total_time: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_actions == 0:
            return 0.0
        return self.successful_actions / self.total_actions


def execute_action(env: Environment, action: Action) -> ActionResult:
    """Execute an action in the environment.

    Args:
        env: Environment to execute in
        action: Action to execute

    Returns:
        Result of the action
    """
    # Validate action
    valid, error = env.validate_action(action)
    if not valid:
        return ActionResult(
            action_id=action.id,
            success=False,
            new_observation=env.observe(),
            reward=-0.1,
            error=error,
        )

    # Execute
    return env.step(action)


def retry_with_backoff(
    env: Environment,
    action: Action,
    max_retries: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 2.0,
) -> ActionResult:
    """Execute action with exponential backoff retry.

    Args:
        env: Environment to execute in
        action: Action to execute
        max_retries: Maximum number of retries
        base_delay: Initial delay between retries
        max_delay: Maximum delay between retries

    Returns:
        Result of the action
    """
    delay = base_delay
    last_result = None

    for attempt in range(max_retries + 1):
        result = execute_action(env, action)
        last_result = result

        if result.success:
            return result

        # Check if error is retryable
        if not _is_retryable_error(result.error):
            return result

        if attempt < max_retries:
            time.sleep(delay)
            delay = min(delay * 2, max_delay)

    return last_result or ActionResult(
        action_id=action.id,
        success=False,
        new_observation=env.observe(),
        reward=-0.1,
        error="Max retries exceeded",
    )


def _is_retryable_error(error: Optional[str]) -> bool:
    """Check if an error is retryable."""
    if error is None:
        return False

    error_lower = error.lower()

    # Retryable errors
    retryable_patterns = [
        "timeout",
        "temporarily",
        "try again",
        "busy",
        "unavailable",
        "rate limit",
    ]

    for pattern in retryable_patterns:
        if pattern in error_lower:
            return True

    return False


def timeout_action(
    env: Environment,
    action: Action,
    timeout: float,
) -> ActionResult:
    """Execute action with timeout.

    For synchronous environments, this uses a simple time check.
    For async environments, proper async timeout should be used.

    Args:
        env: Environment to execute in
        action: Action to execute
        timeout: Timeout in seconds

    Returns:
        Result of the action
    """
    start_time = time.time()

    # For simple sync execution, we can't truly timeout
    # but we record the time and fail if it exceeds
    result = execute_action(env, action)

    elapsed = time.time() - start_time

    if elapsed > timeout:
        return ActionResult(
            action_id=action.id,
            success=False,
            new_observation=result.new_observation,
            reward=-0.1,
            error=f"Action timed out after {elapsed:.2f}s (limit: {timeout}s)",
        )

    return result


async def async_timeout_action(
    env: Environment,
    action: Action,
    timeout: float,
) -> ActionResult:
    """Execute action with async timeout.

    Args:
        env: Environment to execute in
        action: Action to execute
        timeout: Timeout in seconds

    Returns:
        Result of the action
    """
    try:
        # Run in executor since env.step might be blocking
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: execute_action(env, action)),
            timeout=timeout,
        )
        return result
    except asyncio.TimeoutError:
        return ActionResult(
            action_id=action.id,
            success=False,
            new_observation=env.observe(),
            reward=-0.1,
            error=f"Action timed out after {timeout}s",
        )


def safe_execute(
    env: Environment,
    action: Action,
    constraints: ExecutionConstraints,
) -> ActionResult:
    """Execute action with safety constraints.

    Args:
        env: Environment to execute in
        action: Action to execute
        constraints: Safety constraints

    Returns:
        Result of the action
    """
    # Check forbidden actions
    if action.action_type in constraints.forbidden_actions:
        return ActionResult(
            action_id=action.id,
            success=False,
            new_observation=env.observe(),
            reward=-1.0,
            error=f"Action '{action.action_type}' is forbidden",
        )

    # Check preconditions
    if constraints.require_preconditions and action.preconditions:
        obs = env.observe()
        for precondition in action.preconditions:
            if not _check_precondition(precondition, obs):
                return ActionResult(
                    action_id=action.id,
                    success=False,
                    new_observation=obs,
                    reward=-0.1,
                    error=f"Precondition not met: {precondition}",
                )

    # Execute with retry and timeout
    if constraints.max_retries > 0:
        return retry_with_backoff(
            env,
            action,
            max_retries=constraints.max_retries,
        )
    else:
        return timeout_action(env, action, constraints.timeout_seconds)


def _check_precondition(precondition: str, obs: Observation) -> bool:
    """Check if a precondition is satisfied.

    Simple precondition checker - in practice this would be more sophisticated.

    Args:
        precondition: Precondition string (e.g., "has_item:key")
        obs: Current observation

    Returns:
        True if precondition is met
    """
    if ":" not in precondition:
        return True

    condition_type, value = precondition.split(":", 1)

    if condition_type == "has_item":
        inventory = obs.state.get("inventory", [])
        return value in inventory

    if condition_type == "at_position":
        pos = obs.state.get("position")
        if pos:
            try:
                target = tuple(map(int, value.split(",")))
                return tuple(pos) == target
            except (ValueError, TypeError):
                pass

    if condition_type == "in_room":
        room = obs.state.get("room", "")
        return room.lower() == value.lower()

    # Unknown precondition type - assume satisfied
    return True


class ActionExecutor:
    """Stateful action executor with tracking.

    Tracks execution statistics and provides batch execution.
    """

    def __init__(self, env: Environment, constraints: Optional[ExecutionConstraints] = None):
        self.env = env
        self.constraints = constraints or ExecutionConstraints()
        self.stats = ExecutionStats()
        self._history: list[tuple[Action, ActionResult]] = []

    def execute(self, action: Action) -> ActionResult:
        """Execute a single action.

        Args:
            action: Action to execute

        Returns:
            Result of the action
        """
        start_time = time.time()

        result = safe_execute(self.env, action, self.constraints)

        elapsed = time.time() - start_time

        # Update stats
        self.stats.total_actions += 1
        self.stats.total_time += elapsed

        if result.success:
            self.stats.successful_actions += 1
        else:
            self.stats.failed_actions += 1

        # Record history
        self._history.append((action, result))

        return result

    def execute_sequence(
        self,
        actions: list[Action],
        stop_on_failure: bool = True,
    ) -> list[ActionResult]:
        """Execute a sequence of actions.

        Args:
            actions: Actions to execute
            stop_on_failure: Whether to stop on first failure

        Returns:
            List of results
        """
        results = []

        for action in actions:
            result = self.execute(action)
            results.append(result)

            if not result.success and stop_on_failure:
                break

        return results

    def get_history(self) -> list[tuple[Action, ActionResult]]:
        """Get execution history."""
        return self._history.copy()

    def clear_history(self) -> None:
        """Clear execution history."""
        self._history.clear()

    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self.stats = ExecutionStats()


def rollback_on_failure(
    env: Environment,
    action: Action,
    checkpoint_before: bool = True,
) -> tuple[ActionResult, bool]:
    """Execute action and rollback on failure.

    Args:
        env: Environment to execute in
        action: Action to execute
        checkpoint_before: Whether to checkpoint before execution

    Returns:
        Tuple of (result, was_rolled_back)
    """
    state = None
    if checkpoint_before:
        state = env.checkpoint()

    result = execute_action(env, action)

    if not result.success and state is not None:
        env.restore(state)
        return result, True

    return result, False
