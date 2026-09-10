"""
Type definitions for the Argumentative Swarm system.

Provides dataclasses for positions, arguments, refutations, debate state,
judgments, and debate format configuration.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Stance(Enum):
    """Position stance in a debate."""
    FOR = "for"
    AGAINST = "against"
    NEUTRAL = "neutral"
    STEELMAN_WEAK = "steelman_weak"
    DEVILS_ADVOCATE = "devils_advocate"
    SYNTHESIS = "synthesis"


class RoundType(Enum):
    """Type of debate round."""
    OPENING = "opening"
    REBUTTAL = "rebuttal"
    CROSS_EXAMINATION = "cross_examination"
    CLOSING = "closing"


class AdjudicationStrategy(Enum):
    """Strategy for judging debates."""
    UTILITARIAN = "utilitarian"
    DEONTOLOGICAL = "deontological"
    PRAGMATIC = "pragmatic"
    CONSENSUS = "consensus"


@dataclass
class Position:
    """
    A stance on an issue in a debate.

    Attributes:
        id: Unique identifier for this position
        stance: The type of stance (FOR, AGAINST, etc.)
        proposition: The specific claim being debated
        description: Optional elaboration of the position
        branch_id: Optional ID of the branch holding this position
    """
    id: str
    stance: Stance
    proposition: str
    description: str = ""
    branch_id: Optional[str] = None


@dataclass
class Argument:
    """
    A structured argument following the Toulmin model.

    Attributes:
        id: Unique identifier for this argument
        claim: The main assertion being made
        evidence: Data/facts supporting the claim (Toulmin: data)
        warrant: Reasoning connecting evidence to claim
        position_id: ID of the position this argument supports
        strength: Quality score (0.0 to 1.0)
        backing: Additional support for the warrant
        qualifier: Degree of certainty (e.g., "probably", "certainly")
        rebuttal: Conditions under which the claim wouldn't hold
        round_number: Which debate round this was made in
    """
    id: str
    claim: str
    evidence: str
    warrant: str
    position_id: str
    strength: float = 0.5
    backing: str = ""
    qualifier: str = ""
    rebuttal: str = ""
    round_number: int = 0


@dataclass
class Refutation:
    """
    A counter to an argument.

    Attributes:
        id: Unique identifier for this refutation
        target_argument_id: ID of the argument being refuted
        counter_claim: The opposing assertion
        evidence: Data/facts supporting the counter-claim
        refutation_type: Type of refutation (e.g., "undermine", "rebut", "counter")
        strength: Quality score (0.0 to 1.0)
        position_id: ID of the position making this refutation
        round_number: Which debate round this was made in
    """
    id: str
    target_argument_id: str
    counter_claim: str
    evidence: str
    refutation_type: str = "rebut"
    strength: float = 0.5
    position_id: str = ""
    round_number: int = 0


@dataclass
class Response:
    """
    A response to a refutation (rebuilding an argument).

    Attributes:
        id: Unique identifier for this response
        refutation_id: ID of the refutation being addressed
        argument_id: ID of the original argument
        response_text: The response content
        strengthened_claim: Revised claim if applicable
        round_number: Which debate round this was made in
    """
    id: str
    refutation_id: str
    argument_id: str
    response_text: str
    strengthened_claim: str = ""
    round_number: int = 0


@dataclass
class DebateState:
    """
    Current state of a debate.

    Attributes:
        proposition: The topic being debated
        arguments: All arguments made
        refutations: All refutations made
        responses: All responses to refutations
        active_positions: Positions still in the debate
        eliminated_positions: Positions that have been eliminated
        current_round: Current round number
        current_round_type: Type of the current round
    """
    proposition: str
    arguments: list[Argument] = field(default_factory=list)
    refutations: list[Refutation] = field(default_factory=list)
    responses: list[Response] = field(default_factory=list)
    active_positions: list[Position] = field(default_factory=list)
    eliminated_positions: list[Position] = field(default_factory=list)
    current_round: int = 0
    current_round_type: RoundType = RoundType.OPENING


@dataclass
class StrengthScore:
    """
    Strength score for a position in adjudication.

    Attributes:
        position_id: ID of the position
        score: Overall strength score (0.0 to 1.0)
        argument_quality: Quality of arguments made
        refutation_success: Success at refuting opponents
        defense_success: Success at defending own arguments
        evidence_quality: Quality of evidence presented
    """
    position_id: str
    score: float
    argument_quality: float = 0.0
    refutation_success: float = 0.0
    defense_success: float = 0.0
    evidence_quality: float = 0.0


@dataclass
class Crux:
    """
    A key point of disagreement in a debate.

    Attributes:
        id: Unique identifier
        description: Description of the disagreement
        positions_involved: Position IDs with differing views
        arguments_involved: Argument IDs relevant to this crux
        resolution: How this crux was resolved (if at all)
    """
    id: str
    description: str
    positions_involved: list[str] = field(default_factory=list)
    arguments_involved: list[str] = field(default_factory=list)
    resolution: str = ""


@dataclass
class Judgment:
    """
    Final adjudication of a debate.

    Attributes:
        winner: ID of the winning position (or None for tie/synthesis)
        reasoning: Explanation of the judgment
        key_cruxes: Important points of disagreement
        strength_scores: Scores for each position
        common_ground: Areas where positions agreed
        recommendation: Final recommendation based on debate
        strategy_used: Which adjudication strategy was used
    """
    winner: Optional[str]
    reasoning: str
    key_cruxes: list[Crux] = field(default_factory=list)
    strength_scores: list[StrengthScore] = field(default_factory=list)
    common_ground: list[str] = field(default_factory=list)
    recommendation: str = ""
    strategy_used: AdjudicationStrategy = AdjudicationStrategy.PRAGMATIC


@dataclass
class DebateFormat:
    """
    Configuration for a debate format.

    Attributes:
        name: Name of the format
        rounds: Number of rounds
        time_per_round: Time limit per round in seconds (0 for unlimited)
        position_count: Number of positions in the debate
        round_types: Sequence of round types
        cross_examination_enabled: Whether cross-examination is allowed
        allow_rebuilding: Whether arguments can be strengthened after attack
        elimination_enabled: Whether positions can be eliminated
        elimination_threshold: Score below which positions are eliminated
    """
    name: str
    rounds: int
    time_per_round: int = 0
    position_count: int = 2
    round_types: list[RoundType] = field(default_factory=list)
    cross_examination_enabled: bool = True
    allow_rebuilding: bool = True
    elimination_enabled: bool = False
    elimination_threshold: float = 0.3

    def __post_init__(self):
        """Set default round types if not provided."""
        if not self.round_types:
            self.round_types = [
                RoundType.OPENING,
                RoundType.REBUTTAL,
                RoundType.CROSS_EXAMINATION,
                RoundType.CLOSING,
            ]
