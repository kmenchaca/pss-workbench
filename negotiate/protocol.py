"""Negotiation protocols defining rules and turn order."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from .types import (
    Agreement,
    NegotiationMove,
    NegotiationState,
    NegotiationStatus,
    Offer,
    OfferStatus,
    Stakeholder,
)
from .offers import make_offer, accept, evaluate_offer
from .utility import evaluate_terms, UtilityFunction


class NegotiationProtocol(ABC):
    """Base class for negotiation protocols.

    Protocols define the rules of engagement: who can make offers when,
    how agreements are reached, and when the negotiation ends.
    """

    @abstractmethod
    def initialize(self, state: NegotiationState) -> None:
        """Initialize the protocol for a new negotiation.

        Args:
            state: The negotiation state to initialize.
        """
        pass

    @abstractmethod
    def get_next_actors(self, state: NegotiationState) -> list[str]:
        """Determine which stakeholders can act next.

        Args:
            state: Current negotiation state.

        Returns:
            List of stakeholder IDs who can act.
        """
        pass

    @abstractmethod
    def is_valid_move(
        self, state: NegotiationState, stakeholder_id: str, move_type: str
    ) -> bool:
        """Check if a move is valid under this protocol.

        Args:
            state: Current negotiation state.
            stakeholder_id: ID of stakeholder attempting the move.
            move_type: Type of move (offer, counter, accept, reject, withdraw).

        Returns:
            True if move is valid, False otherwise.
        """
        pass

    @abstractmethod
    def check_agreement(self, state: NegotiationState) -> Optional[Agreement]:
        """Check if an agreement has been reached.

        Args:
            state: Current negotiation state.

        Returns:
            Agreement if reached, None otherwise.
        """
        pass

    @abstractmethod
    def check_termination(self, state: NegotiationState) -> bool:
        """Check if negotiation should terminate.

        Args:
            state: Current negotiation state.

        Returns:
            True if negotiation should end, False otherwise.
        """
        pass

    def advance_round(self, state: NegotiationState) -> None:
        """Advance to the next round.

        Args:
            state: The negotiation state to advance.
        """
        state.current_round += 1


@dataclass
class AlternatingOffers(NegotiationProtocol):
    """Protocol where stakeholders take turns proposing.

    Classic negotiation format: parties alternate making offers
    until agreement or deadlock.
    """

    turn_order: list[str] = field(default_factory=list)
    max_rounds: int = 100
    current_turn_index: int = 0

    def initialize(self, state: NegotiationState) -> None:
        """Set up alternating offers with stakeholder turn order."""
        if not self.turn_order:
            self.turn_order = list(state.stakeholders.keys())
        self.current_turn_index = 0
        state.current_round = 0

    def get_next_actors(self, state: NegotiationState) -> list[str]:
        """Return the stakeholder whose turn it is."""
        if not self.turn_order:
            return []
        return [self.turn_order[self.current_turn_index % len(self.turn_order)]]

    def is_valid_move(
        self, state: NegotiationState, stakeholder_id: str, move_type: str
    ) -> bool:
        """Check if it's this stakeholder's turn."""
        next_actors = self.get_next_actors(state)
        if stakeholder_id not in next_actors:
            return False
        return move_type in ["offer", "counter", "accept", "reject"]

    def check_agreement(self, state: NegotiationState) -> Optional[Agreement]:
        """Check for accepted offers."""
        for offer in state.offers.values():
            if offer.status == OfferStatus.ACCEPTED:
                # Create agreement from accepted offer
                parties = [offer.proposer]
                if offer.recipients:
                    parties.extend(offer.recipients)
                else:
                    parties.extend(
                        [s for s in state.stakeholders.keys() if s != offer.proposer]
                    )

                utility_scores = {}
                for sid, stakeholder in state.stakeholders.items():
                    utility_scores[sid] = evaluate_terms(stakeholder, offer.terms)

                return Agreement(
                    id=f"agree_{offer.id}",
                    terms=offer.terms.copy(),
                    parties=parties,
                    utility_scores=utility_scores,
                    timestamp=datetime.now(),
                    offer_id=offer.id,
                )
        return None

    def check_termination(self, state: NegotiationState) -> bool:
        """Check if max rounds reached or agreement found."""
        if state.current_round >= self.max_rounds:
            return True
        if self.check_agreement(state):
            return True
        return False

    def advance_round(self, state: NegotiationState) -> None:
        """Move to next turn and possibly next round."""
        self.current_turn_index += 1
        if self.current_turn_index % len(self.turn_order) == 0:
            state.current_round += 1


@dataclass
class SimultaneousBids(NegotiationProtocol):
    """Protocol where all stakeholders propose at once.

    Sealed-bid style: everyone submits simultaneously,
    then bids are revealed and processed.
    """

    max_rounds: int = 10
    current_round_offers: dict[str, Offer] = field(default_factory=dict)

    def initialize(self, state: NegotiationState) -> None:
        """Set up for simultaneous bidding."""
        state.current_round = 0
        self.current_round_offers = {}

    def get_next_actors(self, state: NegotiationState) -> list[str]:
        """All stakeholders who haven't submitted this round."""
        submitted = set(self.current_round_offers.keys())
        all_stakeholders = set(state.stakeholders.keys())
        return list(all_stakeholders - submitted)

    def is_valid_move(
        self, state: NegotiationState, stakeholder_id: str, move_type: str
    ) -> bool:
        """Check if stakeholder can submit a bid."""
        if stakeholder_id in self.current_round_offers:
            return False
        return move_type == "offer"

    def check_agreement(self, state: NegotiationState) -> Optional[Agreement]:
        """Check if all bids are compatible."""
        # Only check when all bids are in
        if len(self.current_round_offers) != len(state.stakeholders):
            return None

        # Find overlapping acceptable terms
        all_terms = [offer.terms for offer in self.current_round_offers.values()]
        if not all_terms:
            return None

        # Simple check: see if any offer is acceptable to all
        for offer in self.current_round_offers.values():
            acceptable_to_all = True
            for sid, stakeholder in state.stakeholders.items():
                eval_result = evaluate_offer(stakeholder, offer)
                if not eval_result["is_acceptable"]:
                    acceptable_to_all = False
                    break

            if acceptable_to_all:
                utility_scores = {}
                for sid, stakeholder in state.stakeholders.items():
                    utility_scores[sid] = evaluate_terms(stakeholder, offer.terms)

                return Agreement(
                    id=f"agree_sim_{state.current_round}",
                    terms=offer.terms.copy(),
                    parties=list(state.stakeholders.keys()),
                    utility_scores=utility_scores,
                    timestamp=datetime.now(),
                    offer_id=offer.id,
                )

        return None

    def check_termination(self, state: NegotiationState) -> bool:
        """Check if max rounds reached or agreement found."""
        if state.current_round >= self.max_rounds:
            return True
        if self.check_agreement(state):
            return True
        return False

    def advance_round(self, state: NegotiationState) -> None:
        """Clear round offers and advance."""
        self.current_round_offers = {}
        state.current_round += 1


@dataclass
class AuctionProtocol(NegotiationProtocol):
    """Auction-style protocol for competitive bidding.

    Supports various auction types: ascending, descending, sealed-bid.
    """

    auction_type: str = "ascending"  # ascending, descending, sealed
    bid_dimension: str = "price"
    min_increment: float = 1.0
    current_high_bid: Optional[Offer] = None
    max_rounds: int = 50

    def initialize(self, state: NegotiationState) -> None:
        """Set up auction parameters."""
        state.current_round = 0
        self.current_high_bid = None

    def get_next_actors(self, state: NegotiationState) -> list[str]:
        """All stakeholders can bid (except current high bidder)."""
        actors = list(state.stakeholders.keys())
        if self.current_high_bid:
            actors = [a for a in actors if a != self.current_high_bid.proposer]
        return actors

    def is_valid_move(
        self, state: NegotiationState, stakeholder_id: str, move_type: str
    ) -> bool:
        """Check if bid is valid (meets increment, correct direction)."""
        if move_type != "offer":
            return move_type == "withdraw"
        return True

    def is_valid_bid(self, bid: Offer) -> bool:
        """Check if a bid meets auction requirements."""
        if self.bid_dimension not in bid.terms:
            return False

        if self.current_high_bid is None:
            return True

        new_value = bid.terms[self.bid_dimension]
        current_value = self.current_high_bid.terms[self.bid_dimension]

        if self.auction_type == "ascending":
            return new_value >= current_value + self.min_increment
        elif self.auction_type == "descending":
            return new_value <= current_value - self.min_increment
        else:  # sealed
            return True

    def process_bid(self, bid: Offer) -> bool:
        """Process a new bid, update high bid if valid."""
        if not self.is_valid_bid(bid):
            return False

        if self.current_high_bid is None:
            self.current_high_bid = bid
            return True

        new_value = bid.terms[self.bid_dimension]
        current_value = self.current_high_bid.terms[self.bid_dimension]

        if self.auction_type == "ascending":
            if new_value > current_value:
                self.current_high_bid = bid
                return True
        elif self.auction_type == "descending":
            if new_value < current_value:
                self.current_high_bid = bid
                return True

        return False

    def check_agreement(self, state: NegotiationState) -> Optional[Agreement]:
        """Check if auction has concluded with a winner."""
        if self.current_high_bid is None:
            return None

        # For sealed bid, check after all bids received
        if self.auction_type == "sealed":
            pending_offers = [
                o for o in state.offers.values() if o.status == OfferStatus.PENDING
            ]
            if len(pending_offers) == len(state.stakeholders):
                # All bids in - select winner
                best_bid = max(
                    pending_offers, key=lambda o: o.terms.get(self.bid_dimension, 0)
                )
                return self._create_agreement(state, best_bid)

        return None

    def _create_agreement(
        self, state: NegotiationState, winning_bid: Offer
    ) -> Agreement:
        """Create agreement from winning bid."""
        utility_scores = {}
        for sid, stakeholder in state.stakeholders.items():
            utility_scores[sid] = evaluate_terms(stakeholder, winning_bid.terms)

        return Agreement(
            id=f"auction_agree_{winning_bid.id}",
            terms=winning_bid.terms.copy(),
            parties=[winning_bid.proposer],
            utility_scores=utility_scores,
            timestamp=datetime.now(),
            offer_id=winning_bid.id,
        )

    def check_termination(self, state: NegotiationState) -> bool:
        """Check if auction should end."""
        if state.current_round >= self.max_rounds:
            return True
        return False


@dataclass
class MediatedNegotiation(NegotiationProtocol):
    """Protocol with a neutral mediator facilitating.

    Mediator can suggest terms, shuttle between parties,
    and help find common ground.
    """

    mediator_id: str = ""
    caucus_mode: bool = False  # If True, parties don't see each other's offers
    max_rounds: int = 50
    phase: str = "opening"  # opening, exploration, bargaining, closing

    def initialize(self, state: NegotiationState) -> None:
        """Set up mediated negotiation."""
        state.current_round = 0
        self.phase = "opening"

        # Ensure mediator exists
        if self.mediator_id not in state.stakeholders:
            from .stakeholders import create_from_template

            mediator = create_from_template("Mediator", "MEDIATOR")
            self.mediator_id = mediator.id
            state.stakeholders[mediator.id] = mediator

    def get_next_actors(self, state: NegotiationState) -> list[str]:
        """Mediator always acts; others based on phase."""
        if self.phase == "opening":
            return [self.mediator_id]
        elif self.phase == "exploration":
            return list(state.stakeholders.keys())
        elif self.phase == "bargaining":
            # Non-mediator stakeholders
            return [s for s in state.stakeholders.keys() if s != self.mediator_id]
        else:  # closing
            return [self.mediator_id]

    def is_valid_move(
        self, state: NegotiationState, stakeholder_id: str, move_type: str
    ) -> bool:
        """Validate moves based on phase and role."""
        if stakeholder_id == self.mediator_id:
            return move_type in ["offer", "suggest"]
        return move_type in ["offer", "counter", "accept", "reject"]

    def advance_phase(self) -> None:
        """Move to next phase."""
        phases = ["opening", "exploration", "bargaining", "closing"]
        current_idx = phases.index(self.phase)
        if current_idx < len(phases) - 1:
            self.phase = phases[current_idx + 1]

    def check_agreement(self, state: NegotiationState) -> Optional[Agreement]:
        """Check for accepted mediator proposals."""
        for offer in state.offers.values():
            if offer.status == OfferStatus.ACCEPTED:
                # In mediation, agreement needs majority acceptance
                acceptances = sum(
                    1
                    for m in state.moves
                    if m.move_type == "accept"
                    and m.details.get("offer_id") == offer.id
                )

                parties = [s for s in state.stakeholders.keys() if s != self.mediator_id]
                if acceptances >= len(parties) / 2:
                    utility_scores = {}
                    for sid, stakeholder in state.stakeholders.items():
                        utility_scores[sid] = evaluate_terms(stakeholder, offer.terms)

                    return Agreement(
                        id=f"mediated_agree_{offer.id}",
                        terms=offer.terms.copy(),
                        parties=parties,
                        utility_scores=utility_scores,
                        timestamp=datetime.now(),
                        offer_id=offer.id,
                    )
        return None

    def check_termination(self, state: NegotiationState) -> bool:
        """Check if mediation should end."""
        if state.current_round >= self.max_rounds:
            return True
        if self.phase == "closing":
            return True
        if self.check_agreement(state):
            return True
        return False

    def suggest_terms(
        self, state: NegotiationState, stakeholder_positions: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        """Mediator suggests compromise terms.

        Args:
            state: Current negotiation state.
            stakeholder_positions: Current position of each stakeholder.

        Returns:
            Suggested compromise terms.
        """
        suggested = {}
        all_keys = set()
        for pos in stakeholder_positions.values():
            all_keys.update(pos.keys())

        for key in all_keys:
            values = [
                pos[key] for pos in stakeholder_positions.values() if key in pos
            ]
            if values:
                # Suggest midpoint
                suggested[key] = sum(values) / len(values)

        return suggested
