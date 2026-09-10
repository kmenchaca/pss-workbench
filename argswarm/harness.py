"""
Main debate harness for the Argumentative Swarm system.

Orchestrates the full debate flow from initialization through
final judgment.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

from .adjudication import adjudicate
from .arguments import (
    ArgumentBuilder,
    construct_argument,
    create_argument_from_llm_response,
    find_weaknesses,
)
from .debate import (
    CrossExaminationResult,
    DebateRound,
    RoundManager,
    calculate_position_score,
    cross_examine,
    rebuild,
    refute,
    should_eliminate_position,
)
from .positions import assign_positions, get_opposing_positions
from .transcript import (
    DebateTranscript,
    add_argument,
    add_note,
    add_refutation,
    add_response,
    create_transcript,
    export_to_text,
    post_to_bulletin_board,
)
from .types import (
    AdjudicationStrategy,
    Argument,
    DebateFormat,
    DebateState,
    Judgment,
    Position,
    Refutation,
    Response,
    RoundType,
    Stance,
)


@dataclass
class DebateConfig:
    """
    Configuration for a debate session.

    Attributes:
        format: The debate format to use
        adjudication_strategy: How to judge the debate
        elimination_enabled: Whether to eliminate weak positions
        elimination_threshold: Score below which positions are eliminated
        require_evidence: Whether arguments must have evidence
        require_warrant: Whether arguments must have warrants
        max_arguments_per_round: Limit arguments per position per round
        llm_generator: Optional function to generate LLM responses
    """
    format: DebateFormat
    adjudication_strategy: AdjudicationStrategy = AdjudicationStrategy.PRAGMATIC
    elimination_enabled: bool = False
    elimination_threshold: float = 0.3
    require_evidence: bool = True
    require_warrant: bool = True
    max_arguments_per_round: int = 3
    llm_generator: Optional[Callable[[str, Position], str]] = None


@dataclass
class GateCheck:
    """
    Result of a gate check during debate.

    Attributes:
        passed: Whether the check passed
        reason: Explanation of the result
        position_id: Position being checked
        check_type: Type of check performed
    """
    passed: bool
    reason: str
    position_id: str
    check_type: str


class DebateHarness:
    """
    Main harness for running structured debates.

    Manages the full debate lifecycle:
    1. Initialize positions from proposition
    2. Run debate rounds
    3. Facilitate cross-examination
    4. Perform gate checks
    5. Handle position elimination
    6. Produce final judgment
    """

    def __init__(
        self,
        proposition: str,
        config: DebateConfig,
        num_positions: int = 2,
        include_special_positions: bool = False,
    ):
        """
        Initialize the debate harness.

        Args:
            proposition: The topic to debate
            config: Debate configuration
            num_positions: Number of positions to create
            include_special_positions: Whether to include SYNTHESIS, etc.
        """
        self.proposition = proposition
        self.config = config
        self.num_positions = num_positions

        # Initialize positions
        self.positions = assign_positions(
            proposition,
            num_positions,
            include_special=include_special_positions,
        )

        # Initialize state
        self.state = DebateState(
            proposition=proposition,
            active_positions=list(self.positions),
        )

        # Initialize transcript
        self.transcript = create_transcript(proposition, self.positions)

        # Initialize round manager
        self.round_manager = RoundManager(
            self.state,
            config.format.round_types,
        )

        # Track debate status
        self.is_started = False
        self.is_finished = False
        self.judgment: Optional[Judgment] = None

    def start_debate(self) -> DebateRound:
        """
        Start the debate with the first round.

        Returns:
            The first DebateRound
        """
        if self.is_started:
            raise ValueError("Debate has already started")

        self.is_started = True
        add_note(self.transcript, f"Debate started: {self.proposition}")

        return self.round_manager.start_round()

    def submit_argument(
        self,
        position: Position,
        claim: str,
        evidence: str,
        warrant: str,
        backing: str = "",
        qualifier: str = "",
        rebuttal: str = "",
    ) -> tuple[Argument, Optional[GateCheck]]:
        """
        Submit an argument from a position.

        Args:
            position: Position making the argument
            claim: The main claim
            evidence: Supporting evidence
            warrant: Reasoning connecting evidence to claim
            backing: Additional warrant support
            qualifier: Certainty qualifier
            rebuttal: Potential exceptions

        Returns:
            Tuple of (created argument, gate check result if any)
        """
        if not self.is_started:
            raise ValueError("Debate has not started")

        if self.is_finished:
            raise ValueError("Debate has finished")

        # Check if move is valid
        if not self.round_manager.is_move_valid("argument", position):
            raise ValueError(f"Position {position.id} cannot make arguments in current round")

        # Construct argument
        argument = construct_argument(
            position=position,
            claim=claim,
            evidence=evidence,
            warrant=warrant,
            backing=backing,
            qualifier=qualifier,
            rebuttal=rebuttal,
            round_number=self.state.current_round,
        )

        # Perform gate check
        gate_check = self._check_argument_gate(argument, position)

        if not gate_check.passed:
            return argument, gate_check

        # Add to round and transcript
        self.round_manager.add_argument(argument)
        current_round = self.round_manager.get_current_round()
        if current_round:
            add_argument(
                self.transcript,
                argument,
                position,
                current_round.round_number,
                current_round.round_type,
            )

        return argument, gate_check

    def submit_argument_from_llm(
        self,
        position: Position,
        llm_response: str,
    ) -> tuple[Argument, Optional[GateCheck]]:
        """
        Submit an argument parsed from LLM response.

        Args:
            position: Position making the argument
            llm_response: Raw LLM output

        Returns:
            Tuple of (created argument, gate check result if any)
        """
        argument = create_argument_from_llm_response(
            llm_response,
            position,
            self.state.current_round,
        )

        # Validate and add
        gate_check = self._check_argument_gate(argument, position)

        if not gate_check.passed:
            return argument, gate_check

        self.round_manager.add_argument(argument)
        current_round = self.round_manager.get_current_round()
        if current_round:
            add_argument(
                self.transcript,
                argument,
                position,
                current_round.round_number,
                current_round.round_type,
            )

        return argument, gate_check

    def submit_refutation(
        self,
        position: Position,
        target_argument: Argument,
        counter_claim: str,
        evidence: str,
        refutation_type: str = "rebut",
    ) -> tuple[Refutation, Optional[GateCheck]]:
        """
        Submit a refutation from a position.

        Args:
            position: Position making the refutation
            target_argument: Argument being refuted
            counter_claim: The counter-argument
            evidence: Supporting evidence
            refutation_type: Type of refutation

        Returns:
            Tuple of (created refutation, gate check result if any)
        """
        if not self.is_started or self.is_finished:
            raise ValueError("Debate not in progress")

        if not self.round_manager.is_move_valid("refutation", position):
            raise ValueError(f"Position {position.id} cannot refute in current round")

        refutation = refute(
            argument=target_argument,
            refuting_position=position,
            counter_claim=counter_claim,
            evidence=evidence,
            refutation_type=refutation_type,
            round_number=self.state.current_round,
        )

        # Gate check: is this refutation substantive?
        gate_check = self._check_refutation_gate(refutation, position)

        if not gate_check.passed:
            return refutation, gate_check

        self.round_manager.add_refutation(refutation)
        current_round = self.round_manager.get_current_round()
        if current_round:
            add_refutation(
                self.transcript,
                refutation,
                position,
                current_round.round_number,
                current_round.round_type,
            )

        return refutation, gate_check

    def submit_response(
        self,
        position: Position,
        argument: Argument,
        target_refutation: Refutation,
        response_text: str,
        additional_evidence: str = "",
        stronger_warrant: str = "",
    ) -> tuple[Argument, Response, Optional[GateCheck]]:
        """
        Submit a response to a refutation (rebuild argument).

        Args:
            position: Position responding
            argument: Original argument being defended
            target_refutation: Refutation being addressed
            response_text: The response content
            additional_evidence: New evidence to strengthen argument
            stronger_warrant: Improved reasoning

        Returns:
            Tuple of (strengthened argument, response, gate check)
        """
        if not self.is_started or self.is_finished:
            raise ValueError("Debate not in progress")

        if not self.round_manager.is_move_valid("response", position):
            raise ValueError(f"Position {position.id} cannot respond in current round")

        strengthened, response = rebuild(
            argument=argument,
            refutation=target_refutation,
            additional_evidence=additional_evidence,
            stronger_warrant=stronger_warrant,
            address_refutation=response_text,
        )

        self.round_manager.add_response(response)
        current_round = self.round_manager.get_current_round()
        if current_round:
            add_response(
                self.transcript,
                response,
                position,
                current_round.round_number,
                current_round.round_type,
            )

        return strengthened, response, None

    def conduct_cross_examination(
        self,
        examining_position: Position,
        target_argument: Argument,
    ) -> CrossExaminationResult:
        """
        Conduct cross-examination of an argument.

        Args:
            examining_position: Position doing the examination
            target_argument: Argument to examine

        Returns:
            CrossExaminationResult with questions and impact
        """
        if self.state.current_round_type != RoundType.CROSS_EXAMINATION:
            add_note(
                self.transcript,
                f"Cross-examination conducted outside CX round by {examining_position.id}",
                self.state.current_round,
            )

        result = cross_examine(examining_position, target_argument)

        # Post questions to bulletin board
        for question in result.questions[:3]:
            post_to_bulletin_board(
                self.transcript,
                f"Question: {question}",
                examining_position,
                message_type="question",
            )

        return result

    def end_round(self) -> Optional[DebateRound]:
        """
        End the current round and start the next if available.

        Returns:
            The next DebateRound, or None if debate is over
        """
        self.round_manager.end_round()

        # Check for eliminations
        if self.config.elimination_enabled:
            self._check_eliminations()

        # Check if debate should end
        if not self.round_manager.has_more_rounds():
            return None

        if len(self.state.active_positions) < 2:
            add_note(
                self.transcript,
                "Debate ended: insufficient active positions",
                self.state.current_round,
            )
            return None

        return self.round_manager.start_round()

    def finish_debate(self) -> Judgment:
        """
        Conclude the debate and produce judgment.

        Returns:
            Final Judgment
        """
        if self.is_finished:
            if self.judgment:
                return self.judgment
            raise ValueError("Debate already finished but no judgment available")

        self.is_finished = True
        self.transcript.end_time = datetime.now().isoformat()

        # Produce judgment
        self.judgment = adjudicate(
            self.transcript,
            self.state,
            self.config.adjudication_strategy,
        )

        add_note(
            self.transcript,
            f"Debate concluded. Winner: {self.judgment.winner or 'No clear winner'}",
            self.state.current_round,
        )

        return self.judgment

    def get_position_score(self, position: Position) -> float:
        """Get current score for a position."""
        return calculate_position_score(position, self.state)

    def get_opposing_arguments(self, position: Position) -> list[Argument]:
        """Get arguments from opposing positions."""
        opponents = get_opposing_positions(position, self.state.active_positions)
        opponent_ids = {p.id for p in opponents}

        return [
            a for a in self.state.arguments
            if a.position_id in opponent_ids
        ]

    def get_transcript_text(self) -> str:
        """Get formatted transcript text."""
        return export_to_text(self.transcript)

    def _check_argument_gate(
        self,
        argument: Argument,
        position: Position,
    ) -> GateCheck:
        """
        Gate check: Is this argument defensible?

        Checks:
        - Has required components
        - Meets minimum strength
        - Doesn't repeat existing arguments
        """
        issues: list[str] = []

        if self.config.require_evidence and not argument.evidence:
            issues.append("Missing evidence")

        if self.config.require_warrant and not argument.warrant:
            issues.append("Missing warrant")

        if argument.strength < 0.3:
            issues.append(f"Argument too weak (strength: {argument.strength:.2f})")

        # Check for repetition
        position_args = [a for a in self.state.arguments if a.position_id == position.id]
        for existing in position_args:
            if _claims_similar(argument.claim, existing.claim):
                issues.append("Similar argument already made")
                break

        if issues:
            return GateCheck(
                passed=False,
                reason=f"Argument gate failed: {'; '.join(issues)}",
                position_id=position.id,
                check_type="argument_defensibility",
            )

        return GateCheck(
            passed=True,
            reason="Argument meets requirements",
            position_id=position.id,
            check_type="argument_defensibility",
        )

    def _check_refutation_gate(
        self,
        refutation: Refutation,
        position: Position,
    ) -> GateCheck:
        """
        Gate check: Is this refutation substantive?
        """
        issues: list[str] = []

        if not refutation.counter_claim:
            issues.append("Missing counter-claim")

        if refutation.strength < 0.2:
            issues.append(f"Refutation too weak (strength: {refutation.strength:.2f})")

        # Check that we're refuting an opponent's argument
        target_arg = None
        for arg in self.state.arguments:
            if arg.id == refutation.target_argument_id:
                target_arg = arg
                break

        if target_arg and target_arg.position_id == position.id:
            issues.append("Cannot refute own argument")

        if issues:
            return GateCheck(
                passed=False,
                reason=f"Refutation gate failed: {'; '.join(issues)}",
                position_id=position.id,
                check_type="refutation_substantive",
            )

        return GateCheck(
            passed=True,
            reason="Refutation is substantive",
            position_id=position.id,
            check_type="refutation_substantive",
        )

    def _check_eliminations(self) -> list[Position]:
        """Check and eliminate weak positions."""
        eliminated: list[Position] = []

        for position in list(self.state.active_positions):
            if should_eliminate_position(
                position,
                self.state,
                self.config.elimination_threshold,
            ):
                self.state.active_positions.remove(position)
                self.state.eliminated_positions.append(position)
                eliminated.append(position)

                add_note(
                    self.transcript,
                    f"Position {position.id} ({position.stance.value}) eliminated",
                    self.state.current_round,
                )

                post_to_bulletin_board(
                    self.transcript,
                    f"Position {position.stance.value} has been eliminated from the debate",
                    position,
                    message_type="info",
                )

        return eliminated


def _claims_similar(claim1: str, claim2: str, threshold: float = 0.5) -> bool:
    """Check if two claims are similar enough to be considered duplicates."""
    words1 = set(claim1.lower().split())
    words2 = set(claim2.lower().split())

    if not words1 or not words2:
        return False

    intersection = words1 & words2
    smaller_set = min(len(words1), len(words2))

    return len(intersection) / smaller_set > threshold if smaller_set > 0 else False


def run_simple_debate(
    proposition: str,
    format: DebateFormat,
    argument_generator: Callable[[Position, int], tuple[str, str, str]],
    refutation_generator: Optional[Callable[[Position, Argument], tuple[str, str]]] = None,
) -> Judgment:
    """
    Run a simple automated debate.

    This is a convenience function for running debates where arguments
    are generated programmatically (e.g., by an LLM).

    Args:
        proposition: Topic to debate
        format: Debate format to use
        argument_generator: Function that generates (claim, evidence, warrant)
        refutation_generator: Optional function that generates (counter_claim, evidence)

    Returns:
        Final Judgment
    """
    config = DebateConfig(format=format)
    harness = DebateHarness(proposition, config)

    # Start debate
    harness.start_debate()

    # Run through rounds
    while harness.round_manager.has_more_rounds():
        current_round = harness.round_manager.get_current_round()
        if not current_round:
            break

        round_type = current_round.round_type

        if round_type == RoundType.OPENING:
            # Each position makes arguments
            for position in harness.state.active_positions:
                claim, evidence, warrant = argument_generator(
                    position, current_round.round_number
                )
                harness.submit_argument(
                    position, claim, evidence, warrant
                )

        elif round_type == RoundType.REBUTTAL:
            # Each position refutes opponents
            if refutation_generator:
                for position in harness.state.active_positions:
                    opposing_args = harness.get_opposing_arguments(position)
                    for arg in opposing_args[:1]:  # Refute strongest
                        counter, evidence = refutation_generator(position, arg)
                        harness.submit_refutation(
                            position, arg, counter, evidence
                        )

        elif round_type == RoundType.CLOSING:
            # Final arguments
            for position in harness.state.active_positions:
                claim, evidence, warrant = argument_generator(
                    position, current_round.round_number
                )
                harness.submit_argument(
                    position, claim, evidence, warrant
                )

        harness.end_round()

    return harness.finish_debate()
