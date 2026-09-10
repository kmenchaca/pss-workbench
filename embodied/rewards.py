"""
Reward functions for embodied agents.

Provides flexible reward specification for different task types.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .types import Action, Goal, Observation


class RewardFunction(ABC):
    """Base class for reward functions.

    Reward functions compute numerical rewards based on
    state transitions and goal progress.
    """

    @abstractmethod
    def compute(
        self,
        prev_obs: Observation,
        action: Action,
        new_obs: Observation,
    ) -> float:
        """Compute reward for a state transition.

        Args:
            prev_obs: Observation before action
            action: Action taken
            new_obs: Observation after action

        Returns:
            Numerical reward
        """
        pass

    def __call__(
        self,
        prev_obs: Observation,
        action: Action,
        new_obs: Observation,
    ) -> float:
        """Shorthand for compute."""
        return self.compute(prev_obs, action, new_obs)


class GoalReward(RewardFunction):
    """Reward based on goal completion.

    Provides a large reward when the goal is achieved,
    and optional shaping rewards for progress.

    Attributes:
        goal: Goal to achieve
        completion_reward: Reward for achieving goal
        failure_penalty: Penalty for explicit failure states
    """

    def __init__(
        self,
        goal: Goal,
        completion_reward: float = 10.0,
        failure_penalty: float = -5.0,
    ):
        self.goal = goal
        self.completion_reward = completion_reward
        self.failure_penalty = failure_penalty

    def compute(
        self,
        prev_obs: Observation,
        action: Action,
        new_obs: Observation,
    ) -> float:
        # Check goal achievement
        if self.goal.is_achieved(new_obs):
            return self.completion_reward

        # Check if goal has custom reward function
        if self.goal.reward_function:
            return self.goal.reward_function(prev_obs, action, new_obs)

        # Default: small negative reward for each step
        return -0.01


class ProgressReward(RewardFunction):
    """Reward for incremental progress toward goal.

    Uses a progress metric to reward getting closer to the goal.

    Attributes:
        progress_fn: Function that measures progress (0 to 1)
        scale: Scale factor for progress reward
        step_penalty: Small penalty for each step
    """

    def __init__(
        self,
        progress_fn: Callable[[Observation], float],
        scale: float = 1.0,
        step_penalty: float = 0.01,
    ):
        self.progress_fn = progress_fn
        self.scale = scale
        self.step_penalty = step_penalty

    def compute(
        self,
        prev_obs: Observation,
        action: Action,
        new_obs: Observation,
    ) -> float:
        prev_progress = self.progress_fn(prev_obs)
        new_progress = self.progress_fn(new_obs)

        progress_delta = new_progress - prev_progress
        reward = progress_delta * self.scale

        # Subtract step penalty
        reward -= self.step_penalty

        return reward


class ExplorationReward(RewardFunction):
    """Reward for visiting new states.

    Provides intrinsic motivation to explore the state space.

    Attributes:
        novelty_bonus: Reward for visiting a new state
        revisit_penalty: Penalty for revisiting states
    """

    def __init__(
        self,
        novelty_bonus: float = 0.5,
        revisit_penalty: float = 0.1,
    ):
        self.novelty_bonus = novelty_bonus
        self.revisit_penalty = revisit_penalty
        self._visited_states: set[str] = set()

    def compute(
        self,
        prev_obs: Observation,
        action: Action,
        new_obs: Observation,
    ) -> float:
        from .observation import observation_hash

        state_hash = observation_hash(new_obs)

        if state_hash not in self._visited_states:
            self._visited_states.add(state_hash)
            return self.novelty_bonus
        else:
            return -self.revisit_penalty

    def reset(self) -> None:
        """Reset visited state tracking."""
        self._visited_states.clear()

    @property
    def states_visited(self) -> int:
        """Number of unique states visited."""
        return len(self._visited_states)


class CompositeReward(RewardFunction):
    """Combine multiple reward functions.

    Allows weighted combination of different reward signals.

    Attributes:
        components: List of (reward_function, weight) tuples
    """

    def __init__(
        self,
        components: Optional[list[tuple[RewardFunction, float]]] = None,
    ):
        self.components: list[tuple[RewardFunction, float]] = components or []

    def add(self, reward_fn: RewardFunction, weight: float = 1.0) -> "CompositeReward":
        """Add a reward component.

        Args:
            reward_fn: Reward function to add
            weight: Weight for this component

        Returns:
            Self for chaining
        """
        self.components.append((reward_fn, weight))
        return self

    def compute(
        self,
        prev_obs: Observation,
        action: Action,
        new_obs: Observation,
    ) -> float:
        total = 0.0

        for reward_fn, weight in self.components:
            reward = reward_fn.compute(prev_obs, action, new_obs)
            total += reward * weight

        return total


class ActionCostReward(RewardFunction):
    """Penalize actions based on their cost.

    Different actions can have different costs.

    Attributes:
        action_costs: Dict mapping action type to cost
        default_cost: Default cost for unlisted actions
    """

    def __init__(
        self,
        action_costs: Optional[dict[str, float]] = None,
        default_cost: float = 0.01,
    ):
        self.action_costs = action_costs or {}
        self.default_cost = default_cost

    def compute(
        self,
        prev_obs: Observation,
        action: Action,
        new_obs: Observation,
    ) -> float:
        cost = self.action_costs.get(action.action_type, self.default_cost)
        return -cost


class SuccessReward(RewardFunction):
    """Reward based on action success.

    Simple reward that gives positive reward for successful actions
    and negative for failures.

    Attributes:
        success_reward: Reward for successful action
        failure_penalty: Penalty for failed action
    """

    def __init__(
        self,
        success_reward: float = 0.1,
        failure_penalty: float = 0.1,
    ):
        self.success_reward = success_reward
        self.failure_penalty = failure_penalty
        self._last_result_success: Optional[bool] = None

    def set_result(self, success: bool) -> None:
        """Set the result of the last action.

        Called by the execution layer after action execution.
        """
        self._last_result_success = success

    def compute(
        self,
        prev_obs: Observation,
        action: Action,
        new_obs: Observation,
    ) -> float:
        if self._last_result_success is None:
            return 0.0

        success = self._last_result_success
        self._last_result_success = None

        if success:
            return self.success_reward
        else:
            return -self.failure_penalty


class DistanceReward(RewardFunction):
    """Reward based on distance to goal.

    Commonly used for navigation tasks.

    Attributes:
        goal_key: Key in state for goal position
        position_key: Key in state for current position
        scale: Scale factor for distance change
    """

    def __init__(
        self,
        goal_key: str = "goal",
        position_key: str = "position",
        scale: float = 0.1,
    ):
        self.goal_key = goal_key
        self.position_key = position_key
        self.scale = scale

    def compute(
        self,
        prev_obs: Observation,
        action: Action,
        new_obs: Observation,
    ) -> float:
        prev_pos = prev_obs.state.get(self.position_key)
        new_pos = new_obs.state.get(self.position_key)
        goal = new_obs.state.get(self.goal_key)

        if prev_pos is None or new_pos is None or goal is None:
            return 0.0

        # Calculate distances (Manhattan)
        prev_dist = abs(prev_pos[0] - goal[0]) + abs(prev_pos[1] - goal[1])
        new_dist = abs(new_pos[0] - goal[0]) + abs(new_pos[1] - goal[1])

        # Reward for getting closer
        improvement = prev_dist - new_dist
        return improvement * self.scale


class TimeReward(RewardFunction):
    """Penalize based on time/steps taken.

    Encourages efficiency by penalizing each step.

    Attributes:
        step_penalty: Penalty per step
        max_steps: Maximum steps (larger penalty after this)
    """

    def __init__(
        self,
        step_penalty: float = 0.01,
        max_steps: int = 100,
        overtime_multiplier: float = 2.0,
    ):
        self.step_penalty = step_penalty
        self.max_steps = max_steps
        self.overtime_multiplier = overtime_multiplier
        self._step_count = 0

    def compute(
        self,
        prev_obs: Observation,
        action: Action,
        new_obs: Observation,
    ) -> float:
        self._step_count += 1

        if self._step_count > self.max_steps:
            return -self.step_penalty * self.overtime_multiplier
        else:
            return -self.step_penalty

    def reset(self) -> None:
        """Reset step count."""
        self._step_count = 0

    @property
    def steps(self) -> int:
        """Current step count."""
        return self._step_count


def create_goal_reward(
    goal_description: str,
    success_condition: Callable[[Observation], bool],
    completion_reward: float = 10.0,
    progress_fn: Optional[Callable[[Observation], float]] = None,
) -> RewardFunction:
    """Create a complete reward function for a goal.

    Args:
        goal_description: Human-readable goal description
        success_condition: Function that checks goal achievement
        completion_reward: Reward for achieving goal
        progress_fn: Optional function measuring progress

    Returns:
        Configured reward function
    """
    goal = Goal(
        description=goal_description,
        success_condition=success_condition,
    )

    goal_reward = GoalReward(goal, completion_reward=completion_reward)

    if progress_fn:
        progress_reward = ProgressReward(progress_fn)
        return CompositeReward([
            (goal_reward, 1.0),
            (progress_reward, 0.5),
        ])

    return goal_reward


def create_exploration_reward(
    novelty_bonus: float = 0.5,
    step_penalty: float = 0.01,
) -> CompositeReward:
    """Create reward function that encourages exploration.

    Args:
        novelty_bonus: Bonus for visiting new states
        step_penalty: Penalty per step to encourage efficiency

    Returns:
        Configured composite reward
    """
    return CompositeReward([
        (ExplorationReward(novelty_bonus=novelty_bonus), 1.0),
        (TimeReward(step_penalty=step_penalty), 1.0),
    ])
