"""Negotiation strategies for stakeholders."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from .types import (
    BATNA,
    NegotiationMove,
    NegotiationState,
    Offer,
    Stakeholder,
)
from .utility import evaluate_terms, UtilityFunction


class NegotiationStrategy(ABC):
    """Base class for negotiation strategies.

    Strategies define how a stakeholder behaves during negotiation:
    what offers to make, how to respond to others, when to concede.
    """

    @abstractmethod
    def generate_initial_offer(
        self,
        stakeholder: Stakeholder,
        state: NegotiationState,
    ) -> dict[str, float]:
        """Generate the first offer.

        Args:
            stakeholder: The stakeholder making the offer.
            state: Current negotiation state.

        Returns:
            Terms for the initial offer.
        """
        pass

    @abstractmethod
    def respond_to_offer(
        self,
        stakeholder: Stakeholder,
        offer: Offer,
        state: NegotiationState,
    ) -> tuple[str, Optional[dict[str, float]]]:
        """Decide how to respond to an offer.

        Args:
            stakeholder: The stakeholder responding.
            offer: The offer received.
            state: Current negotiation state.

        Returns:
            Tuple of (action, terms) where action is 'accept', 'reject', or 'counter',
            and terms is the counter-offer terms if action is 'counter'.
        """
        pass

    @abstractmethod
    def should_concede(
        self,
        stakeholder: Stakeholder,
        state: NegotiationState,
    ) -> tuple[bool, float]:
        """Decide whether to make a concession.

        Args:
            stakeholder: The stakeholder considering concession.
            state: Current negotiation state.

        Returns:
            Tuple of (should_concede, concession_rate).
        """
        pass


@dataclass
class CooperativeStrategy(NegotiationStrategy):
    """Strategy focused on maximizing joint value.

    Seeks win-win outcomes, makes fair offers, reciprocates concessions.
    """

    fairness_weight: float = 0.5  # How much to weight other's utility
    initial_generosity: float = 0.3  # How generous the first offer is

    def generate_initial_offer(
        self,
        stakeholder: Stakeholder,
        state: NegotiationState,
    ) -> dict[str, float]:
        """Generate a fair initial offer."""
        terms = {}

        for key, target in stakeholder.objectives.items():
            if target == 0.0:  # Wants to minimize
                # Start somewhat higher than ideal
                if key in stakeholder.constraints:
                    min_val, max_val = stakeholder.constraints[key]
                    terms[key] = min_val + (max_val - min_val) * self.initial_generosity
                else:
                    terms[key] = target * (1 + self.initial_generosity)
            elif target == float("inf"):  # Wants to maximize
                # Start somewhat lower than maximum
                if key in stakeholder.constraints:
                    min_val, max_val = stakeholder.constraints[key]
                    terms[key] = max_val - (max_val - min_val) * self.initial_generosity
                else:
                    terms[key] = 100.0  # Default starting point
            else:
                # Start near target with some room
                terms[key] = target * (1 - self.initial_generosity * 0.5)

        return terms

    def respond_to_offer(
        self,
        stakeholder: Stakeholder,
        offer: Offer,
        state: NegotiationState,
    ) -> tuple[str, Optional[dict[str, float]]]:
        """Respond cooperatively - accept good offers, counter reasonably."""
        utility = evaluate_terms(stakeholder, offer.terms)

        # Check BATNA
        batna_utility = 0.0
        if stakeholder.id in state.batnas:
            batna_utility = state.batnas[stakeholder.id].utility

        if utility >= batna_utility:
            # Offer is better than BATNA - likely accept
            if utility > batna_utility * 1.1:  # 10% better than BATNA
                return ("accept", None)

        # Counter with a fair proposal
        counter_terms = {}
        for key, value in offer.terms.items():
            target = stakeholder.objectives.get(key, value)
            # Move toward middle ground
            counter_terms[key] = (value + target) / 2

        return ("counter", counter_terms)

    def should_concede(
        self,
        stakeholder: Stakeholder,
        state: NegotiationState,
    ) -> tuple[bool, float]:
        """Cooperatively concede based on round progress."""
        rounds_elapsed = state.current_round
        # Increase concession willingness over time
        concession_rate = min(0.3, rounds_elapsed * 0.05)
        return (rounds_elapsed > 2, concession_rate)


@dataclass
class CompetitiveStrategy(NegotiationStrategy):
    """Strategy focused on maximizing own value.

    Hard bargaining: extreme anchors, small concessions, exploits weaknesses.
    """

    aggression: float = 0.8  # How extreme initial offers are
    concession_decay: float = 0.9  # Multiply concession by this each round

    def generate_initial_offer(
        self,
        stakeholder: Stakeholder,
        state: NegotiationState,
    ) -> dict[str, float]:
        """Generate an aggressive initial offer (anchoring)."""
        terms = {}

        for key, target in stakeholder.objectives.items():
            if target == 0.0:  # Wants to minimize
                # Start very low
                if key in stakeholder.constraints:
                    min_val, _ = stakeholder.constraints[key]
                    terms[key] = min_val
                else:
                    terms[key] = 0.0
            elif target == float("inf"):  # Wants to maximize
                # Start very high
                if key in stakeholder.constraints:
                    _, max_val = stakeholder.constraints[key]
                    terms[key] = max_val * (1 + self.aggression)
                else:
                    terms[key] = 1000.0 * self.aggression
            else:
                # Extreme version of target
                terms[key] = target * (1 + self.aggression)

        return terms

    def respond_to_offer(
        self,
        stakeholder: Stakeholder,
        offer: Offer,
        state: NegotiationState,
    ) -> tuple[str, Optional[dict[str, float]]]:
        """Respond competitively - reject unless very favorable."""
        utility = evaluate_terms(stakeholder, offer.terms)

        # Check BATNA
        batna_utility = 0.0
        if stakeholder.id in state.batnas:
            batna_utility = state.batnas[stakeholder.id].utility

        # Only accept if significantly above BATNA
        if utility > batna_utility * 1.3:
            return ("accept", None)

        # Counter with minimal concessions
        counter_terms = {}
        for key, value in offer.terms.items():
            target = stakeholder.objectives.get(key, value)
            # Move only slightly from own position
            counter_terms[key] = target * 0.95 + value * 0.05

        return ("counter", counter_terms)

    def should_concede(
        self,
        stakeholder: Stakeholder,
        state: NegotiationState,
    ) -> tuple[bool, float]:
        """Competitive: minimal concessions, only when necessary."""
        rounds_elapsed = state.current_round
        # Very small concessions that decay
        base_rate = 0.05
        concession_rate = base_rate * (self.concession_decay ** rounds_elapsed)
        # Only concede after several rounds
        return (rounds_elapsed > 5, concession_rate)


@dataclass
class TitForTatStrategy(NegotiationStrategy):
    """Strategy that reciprocates behavior.

    Cooperate initially, then mirror opponent's last move.
    """

    memory_length: int = 3  # How many moves to remember
    forgiveness: float = 0.1  # Probability to forgive defection

    def generate_initial_offer(
        self,
        stakeholder: Stakeholder,
        state: NegotiationState,
    ) -> dict[str, float]:
        """Start cooperatively with a fair offer."""
        terms = {}

        for key, target in stakeholder.objectives.items():
            if target == 0.0:
                if key in stakeholder.constraints:
                    min_val, max_val = stakeholder.constraints[key]
                    terms[key] = (min_val + max_val) / 2
                else:
                    terms[key] = 50.0  # Middle ground
            elif target == float("inf"):
                if key in stakeholder.constraints:
                    min_val, max_val = stakeholder.constraints[key]
                    terms[key] = (min_val + max_val) / 2
                else:
                    terms[key] = 50.0
            else:
                terms[key] = target

        return terms

    def _analyze_opponent_behavior(
        self, state: NegotiationState, opponent_id: str
    ) -> str:
        """Analyze if opponent has been cooperative or competitive."""
        recent_moves = [
            m
            for m in state.moves[-self.memory_length :]
            if m.stakeholder_id == opponent_id
        ]

        if not recent_moves:
            return "neutral"

        concessions = 0
        aggression = 0

        for move in recent_moves:
            if move.move_type == "accept":
                concessions += 1
            elif move.move_type == "reject":
                aggression += 1
            elif move.move_type == "counter":
                # Analyze counter-offer direction
                if "adjustments" in move.details:
                    # If adjustments favor us, count as concession
                    concessions += 0.5

        if concessions > aggression:
            return "cooperative"
        elif aggression > concessions:
            return "competitive"
        return "neutral"

    def respond_to_offer(
        self,
        stakeholder: Stakeholder,
        offer: Offer,
        state: NegotiationState,
    ) -> tuple[str, Optional[dict[str, float]]]:
        """Respond based on opponent's recent behavior."""
        opponent_behavior = self._analyze_opponent_behavior(state, offer.proposer)

        if opponent_behavior == "cooperative":
            # Reciprocate with cooperation
            utility = evaluate_terms(stakeholder, offer.terms)
            batna_utility = state.batnas.get(
                stakeholder.id, BATNA(stakeholder_id="")
            ).utility

            if utility >= batna_utility:
                return ("accept", None)

            # Fair counter
            counter_terms = {}
            for key, value in offer.terms.items():
                target = stakeholder.objectives.get(key, value)
                counter_terms[key] = (value + target) / 2
            return ("counter", counter_terms)

        elif opponent_behavior == "competitive":
            # Mirror aggression (unless forgiving)
            import random

            if random.random() < self.forgiveness:
                # Forgive and cooperate
                counter_terms = {}
                for key, value in offer.terms.items():
                    target = stakeholder.objectives.get(key, value)
                    counter_terms[key] = (value + target) / 2
                return ("counter", counter_terms)

            # Aggressive counter
            counter_terms = {}
            for key, value in offer.terms.items():
                target = stakeholder.objectives.get(key, value)
                counter_terms[key] = target * 0.9 + value * 0.1
            return ("counter", counter_terms)

        # Neutral: standard response
        return ("counter", offer.terms)

    def should_concede(
        self,
        stakeholder: Stakeholder,
        state: NegotiationState,
    ) -> tuple[bool, float]:
        """Concede if opponents have been conceding."""
        # Check if others have made concessions recently
        concession_count = sum(
            1
            for m in state.moves[-5:]
            if m.move_type == "counter" and m.stakeholder_id != stakeholder.id
        )

        if concession_count > 0:
            return (True, 0.1 * concession_count)
        return (False, 0.0)


@dataclass
class BrinksmanshipStrategy(NegotiationStrategy):
    """Strategy that pushes to the edge of breakdown.

    Makes extreme demands, threatens walkaway, creates urgency.
    """

    walkaway_threshold: int = 10  # Rounds before threatening to walk
    threat_credibility: float = 0.7  # How credible threats are

    def generate_initial_offer(
        self,
        stakeholder: Stakeholder,
        state: NegotiationState,
    ) -> dict[str, float]:
        """Generate extreme initial offer."""
        terms = {}

        for key, target in stakeholder.objectives.items():
            if target == 0.0:
                terms[key] = 0.0  # Absolute minimum
            elif target == float("inf"):
                if key in stakeholder.constraints:
                    _, max_val = stakeholder.constraints[key]
                    terms[key] = max_val * 2  # Above maximum
                else:
                    terms[key] = 10000.0  # Extreme high
            else:
                terms[key] = target * 2  # Double the target

        return terms

    def respond_to_offer(
        self,
        stakeholder: Stakeholder,
        offer: Offer,
        state: NegotiationState,
    ) -> tuple[str, Optional[dict[str, float]]]:
        """Reject most offers, accept only under pressure."""
        rounds = state.current_round

        # Near walkaway threshold, become more flexible
        if rounds >= self.walkaway_threshold:
            utility = evaluate_terms(stakeholder, offer.terms)
            batna_utility = state.batnas.get(
                stakeholder.id, BATNA(stakeholder_id="")
            ).utility

            if utility >= batna_utility:
                return ("accept", None)

        # Otherwise, hard reject with minimal counter
        counter_terms = {}
        for key, value in offer.terms.items():
            target = stakeholder.objectives.get(key, value)
            # Barely move from own position
            counter_terms[key] = target * 0.98 + value * 0.02

        return ("counter", counter_terms)

    def should_concede(
        self,
        stakeholder: Stakeholder,
        state: NegotiationState,
    ) -> tuple[bool, float]:
        """Only concede when near breakdown."""
        if state.current_round >= self.walkaway_threshold:
            return (True, 0.2)  # Significant concession at last moment
        return (False, 0.0)


@dataclass
class AnchoringStrategy(NegotiationStrategy):
    """Strategy using extreme first offers to anchor negotiations.

    Sets aggressive anchor, makes small concessions to appear reasonable.
    """

    anchor_multiplier: float = 2.0  # How extreme the anchor is
    concession_per_round: float = 0.03  # Small concessions each round

    def generate_initial_offer(
        self,
        stakeholder: Stakeholder,
        state: NegotiationState,
    ) -> dict[str, float]:
        """Generate extreme anchor offer."""
        terms = {}

        for key, target in stakeholder.objectives.items():
            if target == 0.0:
                terms[key] = 0.0
            elif target == float("inf"):
                if key in stakeholder.constraints:
                    _, max_val = stakeholder.constraints[key]
                    terms[key] = max_val * self.anchor_multiplier
                else:
                    terms[key] = 500.0 * self.anchor_multiplier
            else:
                terms[key] = target * self.anchor_multiplier

        return terms

    def respond_to_offer(
        self,
        stakeholder: Stakeholder,
        offer: Offer,
        state: NegotiationState,
    ) -> tuple[str, Optional[dict[str, float]]]:
        """Counter with small moves from anchor."""
        # Get our last position
        our_last_terms = None
        for move in reversed(state.moves):
            if move.stakeholder_id == stakeholder.id and move.move_type in [
                "offer",
                "counter",
            ]:
                if "terms" in move.details:
                    our_last_terms = move.details["terms"]
                    break
                elif "adjustments" in move.details:
                    our_last_terms = move.details["adjustments"]
                    break

        if our_last_terms is None:
            our_last_terms = self.generate_initial_offer(stakeholder, state)

        # Small concession from our position
        counter_terms = {}
        rounds = state.current_round

        for key, value in offer.terms.items():
            our_value = our_last_terms.get(key, value)
            target = stakeholder.objectives.get(key, value)

            # Move slightly toward their offer
            concession = (value - our_value) * self.concession_per_round * rounds
            counter_terms[key] = our_value + concession

        # Check if we should accept
        utility = evaluate_terms(stakeholder, offer.terms)
        batna_utility = state.batnas.get(
            stakeholder.id, BATNA(stakeholder_id="")
        ).utility

        if utility > batna_utility * 1.1:
            return ("accept", None)

        return ("counter", counter_terms)

    def should_concede(
        self,
        stakeholder: Stakeholder,
        state: NegotiationState,
    ) -> tuple[bool, float]:
        """Make small, regular concessions."""
        return (True, self.concession_per_round)


# Strategy registry for easy lookup
STRATEGIES = {
    "cooperative": CooperativeStrategy,
    "competitive": CompetitiveStrategy,
    "tit_for_tat": TitForTatStrategy,
    "brinksmanship": BrinksmanshipStrategy,
    "anchoring": AnchoringStrategy,
}


def get_strategy(name: str, **kwargs) -> NegotiationStrategy:
    """Get a strategy by name.

    Args:
        name: Strategy name.
        **kwargs: Strategy-specific parameters.

    Returns:
        Initialized strategy instance.

    Raises:
        ValueError: If strategy name is not recognized.
    """
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Valid: {list(STRATEGIES.keys())}")
    return STRATEGIES[name](**kwargs)


def select_strategy_for_stakeholder(
    stakeholder: Stakeholder,
    opponent_analysis: Optional[dict[str, Any]] = None,
) -> NegotiationStrategy:
    """Select an appropriate strategy based on stakeholder role and context.

    Args:
        stakeholder: The stakeholder to select strategy for.
        opponent_analysis: Optional analysis of opponents.

    Returns:
        Appropriate strategy instance.
    """
    role = stakeholder.role

    if role == "BUYER":
        return CompetitiveStrategy(aggression=0.6)
    elif role == "SELLER":
        return AnchoringStrategy(anchor_multiplier=1.5)
    elif role == "MEDIATOR":
        return CooperativeStrategy(fairness_weight=0.9)
    elif role == "REGULATOR":
        return CooperativeStrategy(fairness_weight=0.7)

    # Default based on opponent analysis
    if opponent_analysis:
        if opponent_analysis.get("competitive_opponents", 0) > 0:
            return TitForTatStrategy()

    return CooperativeStrategy()
