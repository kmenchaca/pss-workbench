"""
Mock environment for testing embodied agents.

Provides a configurable grid-world environment for testing
action planning and exploration without external dependencies.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from ..environment import ActionSpace, Environment
from ..types import Action, ActionResult, EnvironmentState, Observation


@dataclass
class GridPosition:
    """Position in a 2D grid."""
    x: int
    y: int

    def __eq__(self, other) -> bool:
        if not isinstance(other, GridPosition):
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))


class MockEnvironment(Environment):
    """Configurable mock environment for testing.

    A simple grid world where the agent can move, interact with objects,
    and try to reach a goal position.

    Attributes:
        width: Grid width
        height: Grid height
        start_pos: Starting position
        goal_pos: Goal position
        obstacles: Set of obstacle positions
        items: Dict of item positions to item names
    """

    def __init__(
        self,
        width: int = 5,
        height: int = 5,
        start_pos: tuple[int, int] = (0, 0),
        goal_pos: tuple[int, int] = (4, 4),
        obstacles: Optional[list[tuple[int, int]]] = None,
        items: Optional[dict[tuple[int, int], str]] = None,
        max_steps: int = 100,
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.start_pos = GridPosition(*start_pos)
        self.goal_pos = GridPosition(*goal_pos)
        self.obstacles = {GridPosition(*p) for p in (obstacles or [])}
        self.items = {GridPosition(*k): v for k, v in (items or {}).items()}

        self.max_steps = max_steps
        self._position: GridPosition = self.start_pos
        self._inventory: list[str] = []
        self._steps_taken: int = 0
        self._reached_goal: bool = False

    def _create_action_space(self) -> ActionSpace:
        return ActionSpace(
            action_types=["move", "pickup", "use", "wait"],
            parameter_specs={
                "move": {
                    "required": ["direction"],
                    "allowed": ["direction"],
                },
                "pickup": {
                    "required": [],
                    "allowed": ["item"],
                },
                "use": {
                    "required": ["item"],
                    "allowed": ["item", "target"],
                },
                "wait": {
                    "required": [],
                    "allowed": [],
                },
            },
        )

    def reset(self) -> Observation:
        """Reset to initial state."""
        self._position = GridPosition(self.start_pos.x, self.start_pos.y)
        self._inventory = []
        self._steps_taken = 0
        self._reached_goal = False
        return self.observe()

    def observe(self) -> Observation:
        """Get current observation."""
        # Find nearby items and obstacles
        nearby = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                pos = GridPosition(self._position.x + dx, self._position.y + dy)
                if pos in self.obstacles:
                    nearby.append(f"obstacle at ({pos.x}, {pos.y})")
                if pos in self.items:
                    nearby.append(f"{self.items[pos]} at ({pos.x}, {pos.y})")

        return Observation(
            state={
                "position": (self._position.x, self._position.y),
                "goal": (self.goal_pos.x, self.goal_pos.y),
                "inventory": self._inventory.copy(),
                "nearby": nearby,
                "steps": self._steps_taken,
            },
            raw_data={
                "grid_size": (self.width, self.height),
                "obstacles": [(p.x, p.y) for p in self.obstacles],
                "items": {(p.x, p.y): v for p, v in self.items.items()},
            },
        )

    def step(self, action: Action) -> ActionResult:
        """Execute an action."""
        self._steps_taken += 1
        prev_obs = self.observe()

        if action.action_type == "move":
            return self._handle_move(action, prev_obs)
        elif action.action_type == "pickup":
            return self._handle_pickup(action, prev_obs)
        elif action.action_type == "use":
            return self._handle_use(action, prev_obs)
        elif action.action_type == "wait":
            return self._handle_wait(action)
        else:
            return ActionResult(
                action_id=action.id,
                success=False,
                new_observation=self.observe(),
                reward=-0.1,
                error=f"Unknown action type: {action.action_type}",
            )

    def _handle_move(self, action: Action, prev_obs: Observation) -> ActionResult:
        """Handle move action."""
        direction = action.parameters.get("direction", "")
        dx, dy = 0, 0

        if direction == "north":
            dy = 1
        elif direction == "south":
            dy = -1
        elif direction == "east":
            dx = 1
        elif direction == "west":
            dx = -1
        else:
            return ActionResult(
                action_id=action.id,
                success=False,
                new_observation=self.observe(),
                reward=-0.1,
                error=f"Invalid direction: {direction}",
            )

        new_pos = GridPosition(self._position.x + dx, self._position.y + dy)

        # Check bounds
        if not (0 <= new_pos.x < self.width and 0 <= new_pos.y < self.height):
            return ActionResult(
                action_id=action.id,
                success=False,
                new_observation=self.observe(),
                reward=-0.1,
                error="Cannot move outside grid",
            )

        # Check obstacles
        if new_pos in self.obstacles:
            return ActionResult(
                action_id=action.id,
                success=False,
                new_observation=self.observe(),
                reward=-0.1,
                error="Blocked by obstacle",
            )

        # Move successful
        self._position = new_pos
        new_obs = self.observe()

        # Calculate reward
        reward = -0.01  # Small step cost

        # Bonus for reaching goal
        if self._position == self.goal_pos:
            self._reached_goal = True
            reward = 1.0

        # Bonus for getting closer to goal
        old_dist = abs(prev_obs.state["position"][0] - self.goal_pos.x) + \
                   abs(prev_obs.state["position"][1] - self.goal_pos.y)
        new_dist = abs(self._position.x - self.goal_pos.x) + \
                   abs(self._position.y - self.goal_pos.y)
        if new_dist < old_dist:
            reward += 0.05

        return ActionResult(
            action_id=action.id,
            success=True,
            new_observation=new_obs,
            reward=reward,
        )

    def _handle_pickup(self, action: Action, prev_obs: Observation) -> ActionResult:
        """Handle pickup action."""
        if self._position in self.items:
            item = self.items.pop(self._position)
            self._inventory.append(item)
            return ActionResult(
                action_id=action.id,
                success=True,
                new_observation=self.observe(),
                reward=0.1,
            )
        else:
            return ActionResult(
                action_id=action.id,
                success=False,
                new_observation=self.observe(),
                reward=-0.1,
                error="No item to pickup here",
            )

    def _handle_use(self, action: Action, prev_obs: Observation) -> ActionResult:
        """Handle use action."""
        item = action.parameters.get("item", "")
        if item not in self._inventory:
            return ActionResult(
                action_id=action.id,
                success=False,
                new_observation=self.observe(),
                reward=-0.1,
                error=f"Don't have item: {item}",
            )

        # For now, using items doesn't do anything special
        return ActionResult(
            action_id=action.id,
            success=True,
            new_observation=self.observe(),
            reward=0.0,
        )

    def _handle_wait(self, action: Action) -> ActionResult:
        """Handle wait action."""
        return ActionResult(
            action_id=action.id,
            success=True,
            new_observation=self.observe(),
            reward=-0.01,
        )

    def is_terminal(self) -> bool:
        """Check if in terminal state."""
        return self._reached_goal or self._steps_taken >= self.max_steps

    def checkpoint(self) -> EnvironmentState:
        """Save current state."""
        return EnvironmentState(
            data={
                "position": (self._position.x, self._position.y),
                "inventory": self._inventory.copy(),
                "steps": self._steps_taken,
                "reached_goal": self._reached_goal,
                "items": {(p.x, p.y): v for p, v in self.items.items()},
            }
        )

    def restore(self, state: EnvironmentState) -> None:
        """Restore from saved state."""
        data = state.data
        self._position = GridPosition(*data["position"])
        self._inventory = data["inventory"].copy()
        self._steps_taken = data["steps"]
        self._reached_goal = data["reached_goal"]
        self.items = {GridPosition(*k): v for k, v in data["items"].items()}
