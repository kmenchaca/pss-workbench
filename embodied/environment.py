"""
Environment abstraction for embodied agents.

Provides base classes for environments that agents can interact with,
including action spaces and state management.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from .types import Action, ActionResult, EnvironmentState, Observation


@dataclass
class ActionSpace:
    """Defines valid actions for an environment.

    Attributes:
        action_types: List of valid action type names
        parameter_specs: Specification for parameters of each action type
        continuous: Whether actions have continuous parameters
    """
    action_types: list[str] = field(default_factory=list)
    parameter_specs: dict[str, dict[str, Any]] = field(default_factory=dict)
    continuous: bool = False

    def is_valid(self, action: Action) -> bool:
        """Check if an action is valid in this action space."""
        if action.action_type not in self.action_types:
            return False

        if action.action_type in self.parameter_specs:
            spec = self.parameter_specs[action.action_type]
            required = spec.get("required", [])
            for param in required:
                if param not in action.parameters:
                    return False

            allowed = spec.get("allowed", None)
            if allowed is not None:
                for param in action.parameters:
                    if param not in allowed:
                        return False

        return True

    def get_actions(self) -> list[str]:
        """Return list of available action types."""
        return self.action_types.copy()

    def describe(self) -> str:
        """Return human-readable description of action space."""
        lines = ["Available actions:"]
        for action_type in self.action_types:
            if action_type in self.parameter_specs:
                spec = self.parameter_specs[action_type]
                params = spec.get("required", [])
                if params:
                    lines.append(f"  - {action_type}({', '.join(params)})")
                else:
                    lines.append(f"  - {action_type}")
            else:
                lines.append(f"  - {action_type}")
        return "\n".join(lines)


class Environment(ABC):
    """Base class for environments that agents interact with.

    Environments provide:
    - Observation of current state
    - Ability to take actions and receive results
    - State checkpointing for branching exploration
    - Terminal state detection
    """

    def __init__(self):
        self._action_space: Optional[ActionSpace] = None
        self._current_observation: Optional[Observation] = None

    @property
    def action_space(self) -> ActionSpace:
        """Return the action space for this environment."""
        if self._action_space is None:
            self._action_space = self._create_action_space()
        return self._action_space

    @abstractmethod
    def _create_action_space(self) -> ActionSpace:
        """Create and return the action space. Override in subclasses."""
        pass

    @abstractmethod
    def reset(self) -> Observation:
        """Reset environment to initial state and return initial observation.

        Returns:
            Initial observation after reset
        """
        pass

    @abstractmethod
    def observe(self) -> Observation:
        """Get current observation without taking an action.

        Returns:
            Current observation
        """
        pass

    @abstractmethod
    def step(self, action: Action) -> ActionResult:
        """Take an action and return the result.

        Args:
            action: Action to execute

        Returns:
            Result of the action including new observation and reward
        """
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """Check if current state is terminal (goal reached or stuck).

        Returns:
            True if in terminal state
        """
        pass

    @abstractmethod
    def checkpoint(self) -> EnvironmentState:
        """Save current state for later restoration.

        Returns:
            Serializable state snapshot
        """
        pass

    @abstractmethod
    def restore(self, state: EnvironmentState) -> None:
        """Restore environment to a previously saved state.

        Args:
            state: State snapshot to restore
        """
        pass

    def validate_action(self, action: Action) -> tuple[bool, str]:
        """Validate an action against the action space.

        Args:
            action: Action to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.action_space.is_valid(action):
            return False, f"Invalid action: {action}"
        return True, ""

    def get_available_actions(self) -> list[str]:
        """Get list of currently available action types.

        Override in subclasses for context-dependent actions.

        Returns:
            List of action type names
        """
        return self.action_space.get_actions()


class EnvironmentWrapper(Environment):
    """Wrapper that adds functionality to an existing environment.

    Base class for decorators that add logging, safety checks, etc.
    """

    def __init__(self, env: Environment):
        super().__init__()
        self._wrapped_env = env

    def _create_action_space(self) -> ActionSpace:
        return self._wrapped_env.action_space

    def reset(self) -> Observation:
        return self._wrapped_env.reset()

    def observe(self) -> Observation:
        return self._wrapped_env.observe()

    def step(self, action: Action) -> ActionResult:
        return self._wrapped_env.step(action)

    def is_terminal(self) -> bool:
        return self._wrapped_env.is_terminal()

    def checkpoint(self) -> EnvironmentState:
        return self._wrapped_env.checkpoint()

    def restore(self, state: EnvironmentState) -> None:
        self._wrapped_env.restore(state)


class LoggingEnvironment(EnvironmentWrapper):
    """Environment wrapper that logs all actions and results."""

    def __init__(self, env: Environment, log_func=None):
        super().__init__(env)
        self._log_func = log_func or print
        self._action_history: list[tuple[Action, ActionResult]] = []

    def step(self, action: Action) -> ActionResult:
        self._log_func(f"Action: {action}")
        result = self._wrapped_env.step(action)
        self._log_func(f"Result: {result}")
        self._action_history.append((action, result))
        return result

    def get_history(self) -> list[tuple[Action, ActionResult]]:
        """Return history of all actions and results."""
        return self._action_history.copy()


class SafetyEnvironment(EnvironmentWrapper):
    """Environment wrapper that enforces safety constraints."""

    def __init__(self, env: Environment, forbidden_actions: Optional[list[str]] = None):
        super().__init__(env)
        self._forbidden_actions = set(forbidden_actions or [])

    def step(self, action: Action) -> ActionResult:
        if action.action_type in self._forbidden_actions:
            return ActionResult(
                action_id=action.id,
                success=False,
                new_observation=self.observe(),
                reward=-1.0,
                error=f"Action '{action.action_type}' is forbidden"
            )
        return self._wrapped_env.step(action)
