"""
Type definitions for the Embodied Action Planner.

Dataclasses representing actions, observations, results, and goals
for embodied agents that interact with environments.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional
import uuid


@dataclass
class Action:
    """An action to take in an environment.

    Attributes:
        id: Unique identifier for this action instance
        action_type: The type/name of the action (e.g., "move", "click", "type")
        parameters: Action-specific parameters (e.g., {"direction": "north"})
        preconditions: Optional conditions that must be true for action to succeed
    """
    action_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    preconditions: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __str__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        return f"{self.action_type}({params})" if params else self.action_type


@dataclass
class Observation:
    """What the agent perceives from the environment.

    Attributes:
        id: Unique identifier for this observation
        state: Structured state information
        timestamp: When observation was made
        raw_data: Unprocessed observation data from environment
    """
    state: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    raw_data: Any = None
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __str__(self) -> str:
        state_str = ", ".join(f"{k}={v}" for k, v in list(self.state.items())[:3])
        if len(self.state) > 3:
            state_str += ", ..."
        return f"Observation({state_str})"


@dataclass
class ActionResult:
    """Outcome of executing an action.

    Attributes:
        action_id: ID of the action that was executed
        success: Whether the action succeeded
        new_observation: Observation after action execution
        reward: Numerical reward signal from environment
        error: Error message if action failed
    """
    action_id: str
    success: bool
    new_observation: Observation
    reward: float = 0.0
    error: Optional[str] = None

    def __str__(self) -> str:
        status = "success" if self.success else f"failed: {self.error}"
        return f"ActionResult({self.action_id}, {status}, reward={self.reward})"


@dataclass
class ActionSequence:
    """A series of actions forming a plan.

    Attributes:
        actions: List of actions in sequence
        total_reward: Cumulative reward from executing all actions
        terminal: Whether sequence reaches a terminal state
    """
    actions: list[Action] = field(default_factory=list)
    total_reward: float = 0.0
    terminal: bool = False

    def __len__(self) -> int:
        return len(self.actions)

    def __iter__(self):
        return iter(self.actions)

    def append(self, action: Action) -> None:
        """Add an action to the sequence."""
        self.actions.append(action)

    def __str__(self) -> str:
        action_strs = [str(a) for a in self.actions[:3]]
        if len(self.actions) > 3:
            action_strs.append(f"... +{len(self.actions) - 3} more")
        return f"ActionSequence([{', '.join(action_strs)}], reward={self.total_reward})"


@dataclass
class EnvironmentState:
    """Serializable snapshot of environment state for checkpointing.

    Attributes:
        id: Unique identifier for this state snapshot
        data: Serializable state data
        timestamp: When snapshot was taken
    """
    data: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __str__(self) -> str:
        return f"EnvironmentState({self.id}, keys={list(self.data.keys())})"


@dataclass
class Goal:
    """What the agent is trying to achieve.

    Attributes:
        description: Human-readable goal description
        success_condition: Function that checks if goal is achieved given observation
        reward_function: Function that computes reward given observation
    """
    description: str
    success_condition: Optional[Callable[[Observation], bool]] = None
    reward_function: Optional[Callable[[Observation, Action, Observation], float]] = None

    def is_achieved(self, observation: Observation) -> bool:
        """Check if goal is achieved given current observation."""
        if self.success_condition is None:
            return False
        return self.success_condition(observation)

    def compute_reward(
        self,
        prev_obs: Observation,
        action: Action,
        new_obs: Observation
    ) -> float:
        """Compute reward for a state transition."""
        if self.reward_function is None:
            return 0.0
        return self.reward_function(prev_obs, action, new_obs)

    def __str__(self) -> str:
        return f"Goal({self.description})"


@dataclass
class ExplorationConfig:
    """Configuration for exploration behavior.

    Attributes:
        max_branches: Maximum number of parallel branches
        max_depth: Maximum action sequence length per branch
        branch_on_uncertainty: Whether to branch when action choice is uncertain
        failure_threshold: Number of failures before abandoning branch
        loop_detection_window: Number of recent states to check for loops
    """
    max_branches: int = 4
    max_depth: int = 20
    branch_on_uncertainty: bool = True
    failure_threshold: int = 3
    loop_detection_window: int = 5
