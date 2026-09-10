"""
Position assignment for the Argumentative Swarm system.

Handles distributing stances fairly across branches and defining
custom position types.
"""

import uuid
from typing import Optional

from .types import Position, Stance


# Built-in position definitions with descriptions
POSITION_DEFINITIONS: dict[Stance, str] = {
    Stance.FOR: "Argues in favor of the proposition, presenting supporting evidence and reasoning.",
    Stance.AGAINST: "Argues against the proposition, presenting counter-evidence and reasoning.",
    Stance.NEUTRAL: "Takes a balanced view, weighing both sides without committing to either.",
    Stance.STEELMAN_WEAK: "Strengthens the weakest arguments from any side to ensure fair debate.",
    Stance.DEVILS_ADVOCATE: "Deliberately challenges the strongest positions to test their robustness.",
    Stance.SYNTHESIS: "Seeks to find common ground and integrate insights from all positions.",
}


def generate_position_id() -> str:
    """Generate a unique position ID."""
    return f"pos_{uuid.uuid4().hex[:8]}"


def create_position(
    stance: Stance,
    proposition: str,
    description: Optional[str] = None,
    branch_id: Optional[str] = None,
) -> Position:
    """
    Create a new position with the given stance.

    Args:
        stance: The stance type (FOR, AGAINST, etc.)
        proposition: The topic being debated
        description: Custom description (uses default if not provided)
        branch_id: Optional ID of the branch holding this position

    Returns:
        A new Position instance
    """
    return Position(
        id=generate_position_id(),
        stance=stance,
        proposition=proposition,
        description=description or POSITION_DEFINITIONS.get(stance, ""),
        branch_id=branch_id,
    )


def assign_positions(
    proposition: str,
    num_branches: int,
    include_special: bool = False,
    custom_stances: Optional[list[Stance]] = None,
) -> list[Position]:
    """
    Distribute stances fairly across branches.

    The distribution algorithm ensures:
    - At least one FOR and one AGAINST position (for debates)
    - Fair distribution when num_branches > 2
    - Optional special positions (SYNTHESIS, DEVILS_ADVOCATE, etc.)

    Args:
        proposition: The topic being debated
        num_branches: Number of branches/positions to create
        include_special: Whether to include special positions like SYNTHESIS
        custom_stances: Specific stances to use (overrides automatic distribution)

    Returns:
        List of Position instances assigned to each branch
    """
    if num_branches < 1:
        return []

    if custom_stances:
        # Use provided stances
        stances = custom_stances[:num_branches]
        # Pad with NEUTRAL if not enough stances provided
        while len(stances) < num_branches:
            stances.append(Stance.NEUTRAL)
    else:
        stances = _distribute_stances(num_branches, include_special)

    return [
        create_position(stance, proposition, branch_id=f"branch_{i}")
        for i, stance in enumerate(stances)
    ]


def _distribute_stances(num_branches: int, include_special: bool) -> list[Stance]:
    """
    Distribute stances fairly based on number of branches.

    Args:
        num_branches: Number of positions to create
        include_special: Whether to include special positions

    Returns:
        List of Stance values
    """
    if num_branches == 1:
        return [Stance.NEUTRAL]

    if num_branches == 2:
        return [Stance.FOR, Stance.AGAINST]

    # Start with FOR and AGAINST
    stances = [Stance.FOR, Stance.AGAINST]

    if num_branches == 3:
        if include_special:
            stances.append(Stance.SYNTHESIS)
        else:
            stances.append(Stance.NEUTRAL)
    elif num_branches == 4:
        if include_special:
            stances.extend([Stance.SYNTHESIS, Stance.DEVILS_ADVOCATE])
        else:
            stances.extend([Stance.FOR, Stance.AGAINST])  # 2 FOR, 2 AGAINST
    elif num_branches == 5:
        if include_special:
            stances.extend([Stance.SYNTHESIS, Stance.DEVILS_ADVOCATE, Stance.STEELMAN_WEAK])
        else:
            # 2 FOR, 2 AGAINST, 1 NEUTRAL
            stances.extend([Stance.FOR, Stance.AGAINST, Stance.NEUTRAL])
    else:
        # For 6+ branches, alternate and add specials
        remaining = num_branches - 2
        special_stances = [Stance.SYNTHESIS, Stance.DEVILS_ADVOCATE, Stance.STEELMAN_WEAK]

        if include_special:
            # Add up to 3 special stances
            for i, special in enumerate(special_stances):
                if remaining > 0:
                    stances.append(special)
                    remaining -= 1

        # Fill remaining with alternating FOR/AGAINST
        toggle = True
        while remaining > 0:
            stances.append(Stance.FOR if toggle else Stance.AGAINST)
            toggle = not toggle
            remaining -= 1

    return stances


def get_opposing_positions(
    position: Position,
    all_positions: list[Position],
) -> list[Position]:
    """
    Find positions that oppose a given position.

    Args:
        position: The position to find opponents for
        all_positions: All positions in the debate

    Returns:
        List of opposing positions
    """
    if position.stance == Stance.FOR:
        return [p for p in all_positions if p.stance == Stance.AGAINST]
    elif position.stance == Stance.AGAINST:
        return [p for p in all_positions if p.stance == Stance.FOR]
    elif position.stance == Stance.NEUTRAL:
        # Neutral opposes both extremes somewhat
        return [p for p in all_positions if p.stance in (Stance.FOR, Stance.AGAINST)]
    elif position.stance == Stance.DEVILS_ADVOCATE:
        # Devils advocate opposes the strongest, which we approximate as majority stance
        for_count = sum(1 for p in all_positions if p.stance == Stance.FOR)
        against_count = sum(1 for p in all_positions if p.stance == Stance.AGAINST)
        if for_count >= against_count:
            return [p for p in all_positions if p.stance == Stance.FOR]
        else:
            return [p for p in all_positions if p.stance == Stance.AGAINST]
    else:
        # SYNTHESIS, STEELMAN_WEAK don't have direct opponents
        return []


def get_allied_positions(
    position: Position,
    all_positions: list[Position],
) -> list[Position]:
    """
    Find positions that align with a given position.

    Args:
        position: The position to find allies for
        all_positions: All positions in the debate

    Returns:
        List of allied positions (excluding the position itself)
    """
    allies = [
        p for p in all_positions
        if p.id != position.id and p.stance == position.stance
    ]
    return allies


def balance_positions(positions: list[Position]) -> dict[str, int]:
    """
    Calculate the balance of positions in a debate.

    Args:
        positions: List of all positions

    Returns:
        Dictionary mapping stance names to counts
    """
    balance: dict[str, int] = {}
    for position in positions:
        stance_name = position.stance.value
        balance[stance_name] = balance.get(stance_name, 0) + 1
    return balance


def is_balanced(positions: list[Position], tolerance: int = 1) -> bool:
    """
    Check if the debate positions are roughly balanced.

    Args:
        positions: List of all positions
        tolerance: Maximum allowed difference between FOR and AGAINST counts

    Returns:
        True if balanced within tolerance
    """
    for_count = sum(1 for p in positions if p.stance == Stance.FOR)
    against_count = sum(1 for p in positions if p.stance == Stance.AGAINST)
    return abs(for_count - against_count) <= tolerance


class CustomPosition:
    """Builder for custom position types."""

    def __init__(self, name: str, description: str):
        """
        Initialize a custom position type.

        Args:
            name: Name for the custom stance
            description: Description of what this position does
        """
        self.name = name
        self.description = description
        self._opposes: list[Stance] = []
        self._allies: list[Stance] = []

    def opposes(self, *stances: Stance) -> "CustomPosition":
        """Define which stances this position opposes."""
        self._opposes.extend(stances)
        return self

    def allies_with(self, *stances: Stance) -> "CustomPosition":
        """Define which stances this position allies with."""
        self._allies.extend(stances)
        return self

    def create(self, proposition: str, branch_id: Optional[str] = None) -> Position:
        """
        Create a position instance with this custom type.

        Note: Custom positions use NEUTRAL stance internally but with custom description.
        """
        return Position(
            id=generate_position_id(),
            stance=Stance.NEUTRAL,  # Custom positions use NEUTRAL as base
            proposition=proposition,
            description=f"[{self.name}] {self.description}",
            branch_id=branch_id,
        )


# Pre-built custom positions for common debate scenarios
PRAGMATIST = CustomPosition(
    "Pragmatist",
    "Evaluates arguments based on practical outcomes and real-world applicability."
).opposes(Stance.NEUTRAL).allies_with(Stance.SYNTHESIS)

SKEPTIC = CustomPosition(
    "Skeptic",
    "Questions all claims and demands rigorous evidence before accepting arguments."
).opposes(Stance.FOR, Stance.AGAINST)

CONSEQUENTIALIST = CustomPosition(
    "Consequentialist",
    "Judges arguments by their likely outcomes and consequences."
).allies_with(Stance.SYNTHESIS)

PRINCIPLED = CustomPosition(
    "Principled",
    "Evaluates arguments based on adherence to fundamental principles and rights."
).opposes(Stance.SYNTHESIS)
