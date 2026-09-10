"""
Debate mechanics for the Argumentative Swarm system.

Handles debate rounds, cross-examination, refutation generation,
and argument rebuilding.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .arguments import evaluate_strength, find_weaknesses, strengthen_argument
from .types import (
    Argument,
    DebateState,
    Position,
    Refutation,
    Response,
    RoundType,
    Stance,
)


def generate_refutation_id() -> str:
    """Generate a unique refutation ID."""
    return f"ref_{uuid.uuid4().hex[:8]}"


def generate_response_id() -> str:
    """Generate a unique response ID."""
    return f"resp_{uuid.uuid4().hex[:8]}"


@dataclass
class DebateRound:
    """
    A single round of argumentation.

    Attributes:
        round_number: The round number (1-indexed)
        round_type: Type of round (OPENING, REBUTTAL, etc.)
        arguments: Arguments made this round
        refutations: Refutations made this round
        responses: Responses made this round
        active_positions: Positions participating in this round
    """
    round_number: int
    round_type: RoundType
    arguments: list[Argument] = field(default_factory=list)
    refutations: list[Refutation] = field(default_factory=list)
    responses: list[Response] = field(default_factory=list)
    active_positions: list[Position] = field(default_factory=list)


@dataclass
class CrossExaminationResult:
    """
    Result of cross-examination between two arguments.

    Attributes:
        examiner_position_id: Position doing the examining
        examined_argument_id: Argument being examined
        questions: Questions asked during examination
        answers: Responses to questions
        weaknesses_exposed: Weaknesses found during examination
        credibility_impact: Change to examined argument's credibility
    """
    examiner_position_id: str
    examined_argument_id: str
    questions: list[str] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    weaknesses_exposed: list[dict[str, str]] = field(default_factory=list)
    credibility_impact: float = 0.0


def cross_examine(
    examining_position: Position,
    target_argument: Argument,
    question_generator: Optional[Callable[[Argument], list[str]]] = None,
) -> CrossExaminationResult:
    """
    One position critiques another's argument through questioning.

    Generates questions targeting weaknesses in the argument.

    Args:
        examining_position: Position doing the examination
        target_argument: Argument being examined
        question_generator: Optional custom question generator

    Returns:
        CrossExaminationResult with questions and impact assessment
    """
    weaknesses = find_weaknesses(target_argument)

    # Generate questions based on weaknesses
    if question_generator:
        questions = question_generator(target_argument)
    else:
        questions = _generate_default_questions(target_argument, weaknesses)

    # Calculate credibility impact based on weaknesses found
    credibility_impact = -0.1 * len(weaknesses)

    return CrossExaminationResult(
        examiner_position_id=examining_position.id,
        examined_argument_id=target_argument.id,
        questions=questions,
        answers=[],  # Answers populated during debate execution
        weaknesses_exposed=weaknesses,
        credibility_impact=max(credibility_impact, -0.5),  # Cap impact
    )


def _generate_default_questions(
    argument: Argument,
    weaknesses: list[dict[str, str]],
) -> list[str]:
    """Generate default cross-examination questions."""
    questions: list[str] = []

    for weakness in weaknesses:
        weakness_type = weakness.get("type", "")
        if weakness_type == "missing_evidence":
            questions.append(
                f"What specific evidence supports your claim that '{argument.claim[:50]}...'?"
            )
        elif weakness_type == "missing_warrant":
            questions.append(
                "How does your evidence logically lead to your conclusion?"
            )
        elif weakness_type == "unsupported_warrant":
            questions.append(
                "What justifies the reasoning you've used to connect your evidence to your claim?"
            )
        elif weakness_type == "no_exceptions":
            questions.append(
                "Are there any circumstances under which your claim would not hold?"
            )
        elif weakness_type == "overconfident":
            questions.append(
                f"You used '{argument.qualifier}' - are you certain there are no exceptions?"
            )
        elif weakness_type == "thin_evidence":
            questions.append(
                "Can you provide more detailed or additional evidence for your position?"
            )

    # Always add a clarification question
    questions.append(
        f"Can you clarify the core reasoning behind your position?"
    )

    return questions


def refute(
    argument: Argument,
    refuting_position: Position,
    counter_claim: str,
    evidence: str,
    refutation_type: str = "rebut",
    round_number: int = 0,
) -> Refutation:
    """
    Generate a refutation against an argument.

    Refutation types:
    - "undermine": Attack the evidence/data
    - "rebut": Attack the warrant/reasoning
    - "counter": Present opposing evidence
    - "deny": Directly deny the claim

    Args:
        argument: The argument to refute
        refuting_position: Position making the refutation
        counter_claim: The counter-argument claim
        evidence: Supporting evidence for the refutation
        refutation_type: Type of refutation strategy
        round_number: Current round number

    Returns:
        A new Refutation instance
    """
    refutation = Refutation(
        id=generate_refutation_id(),
        target_argument_id=argument.id,
        counter_claim=counter_claim,
        evidence=evidence,
        refutation_type=refutation_type,
        strength=0.5,
        position_id=refuting_position.id,
        round_number=round_number,
    )

    # Calculate refutation strength
    refutation.strength = _evaluate_refutation_strength(refutation, argument)

    return refutation


def _evaluate_refutation_strength(
    refutation: Refutation,
    target_argument: Argument,
) -> float:
    """
    Evaluate how strong a refutation is.

    Considers:
    - Completeness of the refutation
    - Target argument's weaknesses
    - Refutation type appropriateness

    Args:
        refutation: The refutation to evaluate
        target_argument: The argument being refuted

    Returns:
        Strength score from 0.0 to 1.0
    """
    score = 0.0

    # Base score for having counter-claim and evidence
    if refutation.counter_claim:
        score += 0.3

    if refutation.evidence:
        score += 0.3

    # Bonus for targeting weaknesses
    weaknesses = find_weaknesses(target_argument)
    if weaknesses:
        score += 0.1 * min(len(weaknesses), 3)

    # Refutation type bonuses
    type_bonuses = {
        "undermine": 0.1 if not target_argument.evidence else 0.05,
        "rebut": 0.1 if not target_argument.warrant else 0.05,
        "counter": 0.1,
        "deny": 0.05,
    }
    score += type_bonuses.get(refutation.refutation_type, 0.0)

    return min(score, 1.0)


def rebuild(
    argument: Argument,
    refutation: Refutation,
    additional_evidence: str = "",
    stronger_warrant: str = "",
    address_refutation: str = "",
) -> tuple[Argument, Response]:
    """
    Strengthen an argument after it has been attacked.

    Args:
        argument: The original argument
        refutation: The refutation to address
        additional_evidence: New evidence to add
        stronger_warrant: Improved reasoning
        address_refutation: Direct response to the refutation

    Returns:
        Tuple of (strengthened argument, response to refutation)
    """
    # Create strengthened argument
    strengthened = strengthen_argument(
        argument,
        additional_evidence=additional_evidence,
        stronger_warrant=stronger_warrant,
        add_rebuttal=address_refutation or argument.rebuttal,
    )

    # Create response
    response = Response(
        id=generate_response_id(),
        refutation_id=refutation.id,
        argument_id=argument.id,
        response_text=address_refutation,
        strengthened_claim=strengthened.claim if strengthened.claim != argument.claim else "",
        round_number=refutation.round_number,
    )

    return strengthened, response


class RoundManager:
    """
    Manages the progression of debate rounds.

    Handles round transitions, validates moves, and tracks state.
    """

    def __init__(self, debate_state: DebateState, round_types: list[RoundType]):
        """
        Initialize the round manager.

        Args:
            debate_state: Current debate state
            round_types: Sequence of round types for the debate
        """
        self.state = debate_state
        self.round_types = round_types
        self.current_round_index = 0
        self.rounds: list[DebateRound] = []

    def start_round(self) -> DebateRound:
        """
        Start a new debate round.

        Returns:
            The new DebateRound
        """
        if self.current_round_index >= len(self.round_types):
            raise ValueError("All rounds have been completed")

        round_type = self.round_types[self.current_round_index]
        round_number = self.current_round_index + 1

        new_round = DebateRound(
            round_number=round_number,
            round_type=round_type,
            active_positions=list(self.state.active_positions),
        )

        self.rounds.append(new_round)
        self.state.current_round = round_number
        self.state.current_round_type = round_type

        return new_round

    def end_round(self) -> None:
        """End the current round and update state."""
        if not self.rounds:
            return

        current_round = self.rounds[-1]

        # Add round's arguments/refutations/responses to state
        self.state.arguments.extend(current_round.arguments)
        self.state.refutations.extend(current_round.refutations)
        self.state.responses.extend(current_round.responses)

        self.current_round_index += 1

    def add_argument(self, argument: Argument) -> None:
        """Add an argument to the current round."""
        if not self.rounds:
            raise ValueError("No active round")
        self.rounds[-1].arguments.append(argument)

    def add_refutation(self, refutation: Refutation) -> None:
        """Add a refutation to the current round."""
        if not self.rounds:
            raise ValueError("No active round")

        current_type = self.rounds[-1].round_type
        if current_type == RoundType.OPENING:
            raise ValueError("Cannot refute during opening round")

        self.rounds[-1].refutations.append(refutation)

    def add_response(self, response: Response) -> None:
        """Add a response to the current round."""
        if not self.rounds:
            raise ValueError("No active round")
        self.rounds[-1].responses.append(response)

    def is_move_valid(self, move_type: str, position: Position) -> bool:
        """
        Check if a move is valid in the current round.

        Args:
            move_type: Type of move ("argument", "refutation", "response")
            position: Position attempting the move

        Returns:
            True if the move is valid
        """
        if not self.rounds:
            return False

        current_round = self.rounds[-1]

        # Check position is active
        if position.id not in [p.id for p in current_round.active_positions]:
            return False

        # Validate based on round type and move
        round_type = current_round.round_type

        if round_type == RoundType.OPENING:
            return move_type == "argument"
        elif round_type == RoundType.REBUTTAL:
            return move_type in ("argument", "refutation")
        elif round_type == RoundType.CROSS_EXAMINATION:
            return move_type in ("refutation", "response")
        elif round_type == RoundType.CLOSING:
            return move_type in ("argument", "response")

        return True

    def has_more_rounds(self) -> bool:
        """Check if there are more rounds to play."""
        return self.current_round_index < len(self.round_types)

    def get_current_round(self) -> Optional[DebateRound]:
        """Get the current round, if any."""
        return self.rounds[-1] if self.rounds else None


def calculate_position_score(
    position: Position,
    debate_state: DebateState,
) -> float:
    """
    Calculate a position's current score in the debate.

    Considers:
    - Strength of arguments made
    - Success of refutations
    - Damage from received refutations
    - Quality of responses

    Args:
        position: Position to score
        debate_state: Current debate state

    Returns:
        Score from 0.0 to 1.0
    """
    score = 0.5  # Start at neutral

    # Score arguments
    position_args = [a for a in debate_state.arguments if a.position_id == position.id]
    if position_args:
        avg_strength = sum(a.strength for a in position_args) / len(position_args)
        score += 0.2 * avg_strength

    # Score refutations made
    position_refs = [r for r in debate_state.refutations if r.position_id == position.id]
    if position_refs:
        avg_ref_strength = sum(r.strength for r in position_refs) / len(position_refs)
        score += 0.15 * avg_ref_strength

    # Penalty for received refutations
    arg_ids = {a.id for a in position_args}
    received_refs = [r for r in debate_state.refutations if r.target_argument_id in arg_ids]
    if received_refs:
        avg_damage = sum(r.strength for r in received_refs) / len(received_refs)
        score -= 0.15 * avg_damage

    # Bonus for responses
    arg_ids_list = list(arg_ids)
    responses = [r for r in debate_state.responses if r.argument_id in arg_ids_list]
    if responses and received_refs:
        response_rate = len(responses) / len(received_refs)
        score += 0.1 * min(response_rate, 1.0)

    return max(0.0, min(1.0, score))


def should_eliminate_position(
    position: Position,
    debate_state: DebateState,
    threshold: float = 0.3,
) -> bool:
    """
    Determine if a position should be eliminated from the debate.

    Args:
        position: Position to evaluate
        debate_state: Current debate state
        threshold: Score below which position is eliminated

    Returns:
        True if position should be eliminated
    """
    score = calculate_position_score(position, debate_state)
    return score < threshold


def get_position_arguments(
    position: Position,
    debate_state: DebateState,
) -> list[Argument]:
    """Get all arguments made by a position."""
    return [a for a in debate_state.arguments if a.position_id == position.id]


def get_arguments_against_position(
    position: Position,
    debate_state: DebateState,
) -> list[Refutation]:
    """Get all refutations targeting a position's arguments."""
    arg_ids = {a.id for a in debate_state.arguments if a.position_id == position.id}
    return [r for r in debate_state.refutations if r.target_argument_id in arg_ids]


def get_unaddressed_refutations(
    position: Position,
    debate_state: DebateState,
) -> list[Refutation]:
    """Get refutations against a position that haven't been responded to."""
    arg_ids = {a.id for a in debate_state.arguments if a.position_id == position.id}
    received_refs = [r for r in debate_state.refutations if r.target_argument_id in arg_ids]

    responded_ref_ids = {r.refutation_id for r in debate_state.responses}

    return [r for r in received_refs if r.id not in responded_ref_ids]
