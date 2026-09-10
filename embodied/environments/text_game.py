"""
Text adventure game environment.

Provides a simple text-based adventure game environment
for testing natural language action parsing.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import re

from ..environment import ActionSpace, Environment
from ..types import Action, ActionResult, EnvironmentState, Observation


@dataclass
class Room:
    """A room in the text adventure."""
    name: str
    description: str
    exits: dict[str, str]  # direction -> room_name
    items: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)


class TextGameEnvironment(Environment):
    """Text adventure game environment.

    A classic text adventure where players explore rooms,
    collect items, and solve puzzles.

    Attributes:
        rooms: Dict of room name to Room objects
        start_room: Name of starting room
        goal_room: Name of goal room (optional)
        goal_item: Item needed to win (optional)
    """

    def __init__(
        self,
        rooms: Optional[dict[str, Room]] = None,
        start_room: str = "entrance",
        goal_room: Optional[str] = None,
        goal_item: Optional[str] = None,
        max_turns: int = 100,
    ):
        super().__init__()
        self.rooms = rooms or self._default_rooms()
        self.start_room = start_room
        self.goal_room = goal_room
        self.goal_item = goal_item
        self.max_turns = max_turns

        self._current_room: str = start_room
        self._inventory: list[str] = []
        self._turns: int = 0
        self._game_over: bool = False
        self._message: str = ""

    def _default_rooms(self) -> dict[str, Room]:
        """Create a simple default game world."""
        return {
            "entrance": Room(
                name="Entrance Hall",
                description="A dusty entrance hall with a faded carpet.",
                exits={"north": "library", "east": "kitchen"},
                items=["rusty key"],
                features=["old painting", "coat rack"],
            ),
            "library": Room(
                name="Library",
                description="Walls lined with ancient books. Dust motes float in dim light.",
                exits={"south": "entrance", "east": "study"},
                items=["old book"],
                features=["fireplace", "reading chair"],
            ),
            "kitchen": Room(
                name="Kitchen",
                description="A kitchen with copper pots hanging from hooks.",
                exits={"west": "entrance", "north": "study"},
                items=["candle"],
                features=["stove", "sink"],
            ),
            "study": Room(
                name="Study",
                description="A private study with a large desk.",
                exits={"west": "library", "south": "kitchen"},
                items=["treasure"],
                features=["desk", "window"],
            ),
        }

    def _create_action_space(self) -> ActionSpace:
        return ActionSpace(
            action_types=["go", "look", "take", "drop", "examine", "inventory", "use"],
            parameter_specs={
                "go": {
                    "required": ["direction"],
                    "allowed": ["direction"],
                },
                "look": {
                    "required": [],
                    "allowed": ["target"],
                },
                "take": {
                    "required": ["item"],
                    "allowed": ["item"],
                },
                "drop": {
                    "required": ["item"],
                    "allowed": ["item"],
                },
                "examine": {
                    "required": ["target"],
                    "allowed": ["target"],
                },
                "inventory": {
                    "required": [],
                    "allowed": [],
                },
                "use": {
                    "required": ["item"],
                    "allowed": ["item", "target"],
                },
            },
        )

    def reset(self) -> Observation:
        """Reset game to start."""
        self._current_room = self.start_room
        self._inventory = []
        self._turns = 0
        self._game_over = False
        self._message = f"Welcome! You find yourself in the {self.rooms[self._current_room].name}."
        return self.observe()

    def observe(self) -> Observation:
        """Get current observation."""
        room = self.rooms[self._current_room]
        return Observation(
            state={
                "room": room.name,
                "description": room.description,
                "exits": list(room.exits.keys()),
                "items": room.items.copy(),
                "features": room.features.copy(),
                "inventory": self._inventory.copy(),
                "message": self._message,
                "turns": self._turns,
            },
            raw_data={
                "room_id": self._current_room,
                "all_rooms": list(self.rooms.keys()),
            },
        )

    def step(self, action: Action) -> ActionResult:
        """Execute a text adventure action."""
        self._turns += 1
        self._message = ""

        if action.action_type == "go":
            return self._handle_go(action)
        elif action.action_type == "look":
            return self._handle_look(action)
        elif action.action_type == "take":
            return self._handle_take(action)
        elif action.action_type == "drop":
            return self._handle_drop(action)
        elif action.action_type == "examine":
            return self._handle_examine(action)
        elif action.action_type == "inventory":
            return self._handle_inventory(action)
        elif action.action_type == "use":
            return self._handle_use(action)
        else:
            self._message = f"I don't know how to '{action.action_type}'."
            return ActionResult(
                action_id=action.id,
                success=False,
                new_observation=self.observe(),
                reward=-0.1,
                error=self._message,
            )

    def _handle_go(self, action: Action) -> ActionResult:
        """Handle movement."""
        direction = action.parameters.get("direction", "").lower()
        room = self.rooms[self._current_room]

        if direction not in room.exits:
            self._message = f"You can't go {direction} from here."
            return ActionResult(
                action_id=action.id,
                success=False,
                new_observation=self.observe(),
                reward=-0.1,
                error=self._message,
            )

        self._current_room = room.exits[direction]
        new_room = self.rooms[self._current_room]
        self._message = f"You go {direction} to the {new_room.name}. {new_room.description}"

        # Check win condition
        reward = -0.01
        if self.goal_room and self._current_room == self.goal_room:
            if self.goal_item is None or self.goal_item in self._inventory:
                self._game_over = True
                self._message += " Congratulations! You've won!"
                reward = 1.0

        return ActionResult(
            action_id=action.id,
            success=True,
            new_observation=self.observe(),
            reward=reward,
        )

    def _handle_look(self, action: Action) -> ActionResult:
        """Handle looking around."""
        room = self.rooms[self._current_room]
        target = action.parameters.get("target", "").lower()

        if not target or target == "room" or target == "around":
            items_str = ", ".join(room.items) if room.items else "nothing of interest"
            exits_str = ", ".join(room.exits.keys())
            self._message = f"{room.name}: {room.description} You see: {items_str}. Exits: {exits_str}."
        elif target in [i.lower() for i in room.items]:
            self._message = f"You see a {target} here."
        elif target in [f.lower() for f in room.features]:
            self._message = f"You examine the {target}. It looks ordinary."
        else:
            self._message = f"You don't see any {target} here."

        return ActionResult(
            action_id=action.id,
            success=True,
            new_observation=self.observe(),
            reward=0.0,
        )

    def _handle_take(self, action: Action) -> ActionResult:
        """Handle picking up items."""
        item = action.parameters.get("item", "").lower()
        room = self.rooms[self._current_room]

        matching = [i for i in room.items if i.lower() == item]
        if matching:
            taken = matching[0]
            room.items.remove(taken)
            self._inventory.append(taken)
            self._message = f"You take the {taken}."
            return ActionResult(
                action_id=action.id,
                success=True,
                new_observation=self.observe(),
                reward=0.1,
            )
        else:
            self._message = f"There's no {item} here to take."
            return ActionResult(
                action_id=action.id,
                success=False,
                new_observation=self.observe(),
                reward=-0.1,
                error=self._message,
            )

    def _handle_drop(self, action: Action) -> ActionResult:
        """Handle dropping items."""
        item = action.parameters.get("item", "").lower()
        room = self.rooms[self._current_room]

        matching = [i for i in self._inventory if i.lower() == item]
        if matching:
            dropped = matching[0]
            self._inventory.remove(dropped)
            room.items.append(dropped)
            self._message = f"You drop the {dropped}."
            return ActionResult(
                action_id=action.id,
                success=True,
                new_observation=self.observe(),
                reward=0.0,
            )
        else:
            self._message = f"You don't have a {item}."
            return ActionResult(
                action_id=action.id,
                success=False,
                new_observation=self.observe(),
                reward=-0.1,
                error=self._message,
            )

    def _handle_examine(self, action: Action) -> ActionResult:
        """Handle examining things closely."""
        target = action.parameters.get("target", "").lower()
        room = self.rooms[self._current_room]

        if target in [i.lower() for i in self._inventory]:
            self._message = f"You examine the {target} closely. It's a {target}."
        elif target in [i.lower() for i in room.items]:
            self._message = f"You examine the {target}. It sits on the ground."
        elif target in [f.lower() for f in room.features]:
            self._message = f"You examine the {target}. Nothing unusual."
        else:
            self._message = f"You don't see any {target} to examine."

        return ActionResult(
            action_id=action.id,
            success=True,
            new_observation=self.observe(),
            reward=0.0,
        )

    def _handle_inventory(self, action: Action) -> ActionResult:
        """Handle checking inventory."""
        if self._inventory:
            items_str = ", ".join(self._inventory)
            self._message = f"You are carrying: {items_str}."
        else:
            self._message = "You are not carrying anything."

        return ActionResult(
            action_id=action.id,
            success=True,
            new_observation=self.observe(),
            reward=0.0,
        )

    def _handle_use(self, action: Action) -> ActionResult:
        """Handle using items."""
        item = action.parameters.get("item", "").lower()
        target = action.parameters.get("target", "")

        if item not in [i.lower() for i in self._inventory]:
            self._message = f"You don't have a {item}."
            return ActionResult(
                action_id=action.id,
                success=False,
                new_observation=self.observe(),
                reward=-0.1,
                error=self._message,
            )

        self._message = f"You use the {item}." + (f" on the {target}." if target else "")
        return ActionResult(
            action_id=action.id,
            success=True,
            new_observation=self.observe(),
            reward=0.0,
        )

    def is_terminal(self) -> bool:
        """Check if game is over."""
        return self._game_over or self._turns >= self.max_turns

    def checkpoint(self) -> EnvironmentState:
        """Save game state."""
        # Deep copy room items
        room_items = {name: room.items.copy() for name, room in self.rooms.items()}

        return EnvironmentState(
            data={
                "current_room": self._current_room,
                "inventory": self._inventory.copy(),
                "turns": self._turns,
                "game_over": self._game_over,
                "message": self._message,
                "room_items": room_items,
            }
        )

    def restore(self, state: EnvironmentState) -> None:
        """Restore game state."""
        data = state.data
        self._current_room = data["current_room"]
        self._inventory = data["inventory"].copy()
        self._turns = data["turns"]
        self._game_over = data["game_over"]
        self._message = data["message"]

        # Restore room items
        for name, items in data["room_items"].items():
            if name in self.rooms:
                self.rooms[name].items = items.copy()
