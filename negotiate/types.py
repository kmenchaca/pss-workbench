"""Core dataclasses for the Distributed Negotiation Simulator."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional


class NegotiationStatus(Enum):
    """Status of a negotiation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    AGREED = "agreed"
    DEADLOCKED = "deadlocked"
    WITHDRAWN = "withdrawn"


class OfferStatus(Enum):
    """Status of an offer."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    COUNTERED = "countered"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"


@dataclass
class Stakeholder:
    """A party in the negotiation.

    Attributes:
        id: Unique identifier for the stakeholder.
        name: Human-readable name.
        objectives: What the stakeholder wants to achieve (key -> target value).
        constraints: Hard limits that cannot be violated (key -> (min, max)).
        private_info: Information hidden from other parties.
        utility_function: Optional custom utility function name.
        role: Optional role template (BUYER, SELLER, etc.).
    """

    id: str
    name: str
    objectives: dict[str, float] = field(default_factory=dict)
    constraints: dict[str, tuple[float, float]] = field(default_factory=dict)
    private_info: dict[str, Any] = field(default_factory=dict)
    utility_function: Optional[str] = None
    role: Optional[str] = None


@dataclass
class Offer:
    """A proposal from one party to others.

    Attributes:
        id: Unique identifier for the offer.
        proposer: ID of the stakeholder making the offer.
        terms: The proposed terms (key -> value).
        timestamp: When the offer was made.
        recipients: IDs of stakeholders the offer is directed to.
        status: Current status of the offer.
        expires_at: Optional expiration time.
    """

    id: str
    proposer: str
    terms: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    recipients: list[str] = field(default_factory=list)
    status: OfferStatus = OfferStatus.PENDING
    expires_at: Optional[datetime] = None


@dataclass
class CounterOffer:
    """A response to an existing offer with modifications.

    Attributes:
        id: Unique identifier for the counter-offer.
        original_offer_id: ID of the offer being responded to.
        proposer: ID of the stakeholder making the counter-offer.
        new_terms: The modified terms.
        concessions: What the proposer gave up (key -> amount conceded).
        timestamp: When the counter-offer was made.
        justification: Optional explanation for the counter.
    """

    id: str
    original_offer_id: str
    proposer: str
    new_terms: dict[str, float] = field(default_factory=dict)
    concessions: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    justification: Optional[str] = None


@dataclass
class Agreement:
    """A final deal accepted by all parties.

    Attributes:
        id: Unique identifier for the agreement.
        terms: The agreed-upon terms.
        parties: IDs of stakeholders who agreed.
        utility_scores: Utility score for each party (stakeholder_id -> score).
        timestamp: When agreement was reached.
        offer_id: ID of the final accepted offer.
    """

    id: str
    terms: dict[str, float] = field(default_factory=dict)
    parties: list[str] = field(default_factory=list)
    utility_scores: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    offer_id: Optional[str] = None


@dataclass
class BATNA:
    """Best Alternative To Negotiated Agreement.

    Attributes:
        stakeholder_id: ID of the stakeholder this BATNA belongs to.
        alternative: Description or terms of the alternative.
        utility: Utility score of the alternative.
        is_bluff: Whether this is a bluffed (inflated) BATNA.
        actual_utility: Real utility if is_bluff is True.
    """

    stakeholder_id: str
    alternative: dict[str, Any] = field(default_factory=dict)
    utility: float = 0.0
    is_bluff: bool = False
    actual_utility: Optional[float] = None


@dataclass
class ZOPA:
    """Zone of Possible Agreement.

    Attributes:
        dimension: The term/dimension this ZOPA applies to.
        min_acceptable: Minimum acceptable value across parties.
        max_acceptable: Maximum acceptable value across parties.
        overlap: Whether there is an overlapping zone.
        overlap_range: The (min, max) of the overlap if it exists.
    """

    dimension: str
    min_acceptable: dict[str, float] = field(default_factory=dict)
    max_acceptable: dict[str, float] = field(default_factory=dict)
    overlap: bool = False
    overlap_range: Optional[tuple[float, float]] = None


@dataclass
class NegotiationMove:
    """A single move in the negotiation.

    Attributes:
        move_type: Type of move (offer, counter, accept, reject, withdraw).
        stakeholder_id: ID of the stakeholder making the move.
        timestamp: When the move was made.
        details: Move-specific details.
        round_number: Which round this move occurred in.
    """

    move_type: str
    stakeholder_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: dict[str, Any] = field(default_factory=dict)
    round_number: int = 0


@dataclass
class NegotiationState:
    """Current state of the negotiation.

    Attributes:
        negotiation_id: Unique identifier for this negotiation.
        stakeholders: All participating stakeholders.
        offers: All offers made (offer_id -> Offer).
        counter_offers: All counter-offers (counter_id -> CounterOffer).
        agreements: Any agreements reached.
        deadlocks: Dimensions where deadlock was detected.
        current_round: Current round number.
        status: Overall negotiation status.
        moves: History of all moves.
        batnas: Known BATNAs (stakeholder_id -> BATNA).
        zopas: Calculated ZOPAs (dimension -> ZOPA).
    """

    negotiation_id: str
    stakeholders: dict[str, Stakeholder] = field(default_factory=dict)
    offers: dict[str, Offer] = field(default_factory=dict)
    counter_offers: dict[str, CounterOffer] = field(default_factory=dict)
    agreements: list[Agreement] = field(default_factory=list)
    deadlocks: list[str] = field(default_factory=list)
    current_round: int = 0
    status: NegotiationStatus = NegotiationStatus.PENDING
    moves: list[NegotiationMove] = field(default_factory=list)
    batnas: dict[str, BATNA] = field(default_factory=dict)
    zopas: dict[str, ZOPA] = field(default_factory=dict)


@dataclass
class GameOutcome:
    """An outcome in game-theoretic analysis.

    Attributes:
        strategies: Strategy chosen by each player (stakeholder_id -> strategy).
        payoffs: Payoff for each player (stakeholder_id -> utility).
        is_nash: Whether this is a Nash equilibrium.
        is_pareto: Whether this is Pareto efficient.
    """

    strategies: dict[str, str] = field(default_factory=dict)
    payoffs: dict[str, float] = field(default_factory=dict)
    is_nash: bool = False
    is_pareto: bool = False


@dataclass
class Coalition:
    """A group of stakeholders acting together.

    Attributes:
        id: Unique identifier for the coalition.
        members: IDs of stakeholders in the coalition.
        shared_objectives: Objectives the coalition agrees on.
        internal_split: How to divide coalition gains.
        stability: Measure of coalition stability (0-1).
    """

    id: str
    members: list[str] = field(default_factory=list)
    shared_objectives: dict[str, float] = field(default_factory=dict)
    internal_split: dict[str, float] = field(default_factory=dict)
    stability: float = 1.0
