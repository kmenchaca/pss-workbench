"""Main negotiation harness for running simulations."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional
from uuid import uuid4

from .types import (
    Agreement,
    BATNA,
    NegotiationMove,
    NegotiationState,
    NegotiationStatus,
    Offer,
    OfferStatus,
    Stakeholder,
)
from .batna import discover_batna, compare_to_batna
from .equilibrium import analyze_game
from .offers import make_offer, evaluate_offer, counter_offer, accept, reject
from .protocol import NegotiationProtocol, AlternatingOffers
from .strategy import NegotiationStrategy, get_strategy, select_strategy_for_stakeholder
from .transcript import NegotiationTranscript, create_transcript, finalize_transcript
from .utility import evaluate_terms, UtilityFunction
from .zopa import find_zopa, no_zopa_detection


@dataclass
class NegotiationConfig:
    """Configuration for a negotiation harness.

    Attributes:
        max_rounds: Maximum rounds before declaring impasse.
        protocol: Negotiation protocol to use.
        auto_discover_batna: Automatically discover BATNAs.
        track_transcript: Record full transcript.
        check_zopa: Check for ZOPA before starting.
        run_game_analysis: Run game-theoretic analysis.
        gate_checks: Custom gate check functions.
        llm_provider: Optional LLM for generating moves.
    """

    max_rounds: int = 50
    protocol: Optional[NegotiationProtocol] = None
    auto_discover_batna: bool = True
    track_transcript: bool = True
    check_zopa: bool = True
    run_game_analysis: bool = False
    gate_checks: list[Callable] = field(default_factory=list)
    llm_provider: Optional[Any] = None


@dataclass
class NegotiationHarness:
    """Main harness for running negotiations.

    Each stakeholder becomes a "branch" in the exploration tree.
    Gate checks evaluate positions at key decision points.
    """

    config: NegotiationConfig = field(default_factory=NegotiationConfig)
    state: Optional[NegotiationState] = None
    strategies: dict[str, NegotiationStrategy] = field(default_factory=dict)
    transcript: Optional[NegotiationTranscript] = None
    utility_registry: dict[str, UtilityFunction] = field(default_factory=dict)

    def initialize(
        self,
        stakeholders: list[Stakeholder],
        initial_state: Optional[NegotiationState] = None,
    ) -> NegotiationState:
        """Initialize a new negotiation.

        Args:
            stakeholders: List of participating stakeholders.
            initial_state: Optional pre-existing state.

        Returns:
            The initialized negotiation state.
        """
        if initial_state:
            self.state = initial_state
        else:
            self.state = NegotiationState(
                negotiation_id=str(uuid4())[:8],
                stakeholders={s.id: s for s in stakeholders},
                status=NegotiationStatus.IN_PROGRESS,
            )

        # Initialize protocol
        if self.config.protocol is None:
            self.config.protocol = AlternatingOffers(max_rounds=self.config.max_rounds)
        self.config.protocol.initialize(self.state)

        # Discover BATNAs
        if self.config.auto_discover_batna:
            for stakeholder in stakeholders:
                batna = discover_batna(stakeholder)
                self.state.batnas[stakeholder.id] = batna

        # Check ZOPA
        if self.config.check_zopa:
            zopa_analysis = no_zopa_detection(stakeholders, state=self.state)
            if zopa_analysis["impossible"]:
                # Store warning but continue - might expand ZOPA
                self.state.deadlocks = zopa_analysis["no_zopa_dimensions"]

        # Assign strategies
        for stakeholder in stakeholders:
            if stakeholder.id not in self.strategies:
                self.strategies[stakeholder.id] = select_strategy_for_stakeholder(
                    stakeholder
                )

        # Initialize transcript
        if self.config.track_transcript:
            self.transcript = create_transcript(self.state)

        return self.state

    def run(self) -> dict[str, Any]:
        """Run the negotiation to completion.

        Returns:
            Results including final state, agreement, and analysis.
        """
        if self.state is None:
            raise ValueError("Harness not initialized. Call initialize() first.")

        protocol = self.config.protocol

        while not protocol.check_termination(self.state):
            # Get actors for this turn
            actors = protocol.get_next_actors(self.state)

            for actor_id in actors:
                stakeholder = self.state.stakeholders[actor_id]
                strategy = self.strategies.get(actor_id)

                if strategy is None:
                    continue

                # Run gate checks
                gate_result = self._run_gate_checks(stakeholder)
                if gate_result.get("should_withdraw"):
                    self._handle_withdrawal(stakeholder)
                    continue

                # Generate and execute move
                move = self._generate_move(stakeholder, strategy)
                if move:
                    self._execute_move(stakeholder, move)

            # Advance round
            protocol.advance_round(self.state)

        # Finalize
        agreement = protocol.check_agreement(self.state)
        if agreement:
            self.state.status = NegotiationStatus.AGREED
            self.state.agreements.append(agreement)
        else:
            self.state.status = NegotiationStatus.DEADLOCKED

        # Finalize transcript
        if self.transcript:
            finalize_transcript(self.transcript, self.state.status, agreement)

        # Build results
        results = {
            "negotiation_id": self.state.negotiation_id,
            "status": self.state.status.value,
            "rounds": self.state.current_round,
            "agreement": agreement,
            "final_state": self.state,
        }

        # Game analysis
        if self.config.run_game_analysis:
            results["game_analysis"] = analyze_game(
                list(self.state.stakeholders.values())
            )

        # ZOPA analysis
        results["zopa"] = find_zopa(
            list(self.state.stakeholders.values()), state=self.state
        )

        return results

    def _run_gate_checks(self, stakeholder: Stakeholder) -> dict[str, Any]:
        """Run gate checks for a stakeholder.

        Gate checks are decision points: "Is my position improving?",
        "Should I concede?", "Should I walk away?"

        Args:
            stakeholder: The stakeholder at the gate.

        Returns:
            Gate check results.
        """
        results = {
            "should_withdraw": False,
            "should_concede": False,
            "concession_rate": 0.0,
        }

        # Check BATNA comparison
        batna = self.state.batnas.get(stakeholder.id)
        if batna:
            # Get best current offer for this stakeholder
            pending_offers = [
                o
                for o in self.state.offers.values()
                if o.status == OfferStatus.PENDING
                and (not o.recipients or stakeholder.id in o.recipients)
            ]

            if pending_offers:
                for offer in pending_offers:
                    comparison = compare_to_batna(
                        stakeholder, offer.terms, batna, self.utility_registry
                    )

                    # If BATNA is significantly better, consider walking away
                    if comparison["difference"] < -20:  # Offer much worse than BATNA
                        results["should_withdraw"] = True
                        results["reason"] = "BATNA significantly better than offers"

        # Run custom gate checks
        for gate_check in self.config.gate_checks:
            gate_result = gate_check(stakeholder, self.state)
            if gate_result.get("should_withdraw"):
                results["should_withdraw"] = True
            if gate_result.get("should_concede"):
                results["should_concede"] = True
                results["concession_rate"] = max(
                    results["concession_rate"],
                    gate_result.get("concession_rate", 0.1),
                )

        # Check strategy's concession recommendation
        strategy = self.strategies.get(stakeholder.id)
        if strategy:
            should_concede, rate = strategy.should_concede(stakeholder, self.state)
            if should_concede:
                results["should_concede"] = True
                results["concession_rate"] = max(results["concession_rate"], rate)

        return results

    def _generate_move(
        self,
        stakeholder: Stakeholder,
        strategy: NegotiationStrategy,
    ) -> Optional[dict[str, Any]]:
        """Generate a move for a stakeholder.

        Args:
            stakeholder: The stakeholder making the move.
            strategy: Their negotiation strategy.

        Returns:
            Move details or None.
        """
        # Check for pending offers to respond to
        pending_offers = [
            o
            for o in self.state.offers.values()
            if o.status == OfferStatus.PENDING
            and (not o.recipients or stakeholder.id in o.recipients)
            and o.proposer != stakeholder.id
        ]

        if pending_offers:
            # Respond to the most recent offer
            offer = max(pending_offers, key=lambda o: o.timestamp)
            action, terms = strategy.respond_to_offer(stakeholder, offer, self.state)

            return {
                "type": action,
                "offer": offer,
                "terms": terms,
            }

        # No pending offers - make initial/new offer
        terms = strategy.generate_initial_offer(stakeholder, self.state)
        return {
            "type": "offer",
            "terms": terms,
        }

    def _execute_move(
        self,
        stakeholder: Stakeholder,
        move: dict[str, Any],
    ) -> None:
        """Execute a negotiation move.

        Args:
            stakeholder: The stakeholder making the move.
            move: The move to execute.
        """
        move_type = move["type"]

        if move_type == "offer":
            make_offer(
                stakeholder,
                move["terms"],
                state=self.state,
            )

        elif move_type == "counter":
            original_offer = move["offer"]
            counter_terms = move["terms"]

            # Calculate adjustments
            adjustments = {
                k: counter_terms.get(k, v) - v
                for k, v in original_offer.terms.items()
                if counter_terms.get(k, v) != v
            }

            counter_offer(
                stakeholder,
                original_offer,
                adjustments,
                state=self.state,
            )

        elif move_type == "accept":
            accept(stakeholder, move["offer"], state=self.state)

        elif move_type == "reject":
            reject(
                stakeholder,
                move["offer"],
                reason=move.get("reason"),
                state=self.state,
            )

    def _handle_withdrawal(self, stakeholder: Stakeholder) -> None:
        """Handle stakeholder withdrawal from negotiation.

        In PSS terms, this is a "dead branch" - failure info propagates
        to remaining stakeholders.

        Args:
            stakeholder: The withdrawing stakeholder.
        """
        # Record withdrawal move
        move = NegotiationMove(
            move_type="withdraw",
            stakeholder_id=stakeholder.id,
            timestamp=datetime.now(),
            details={"reason": "BATNA preferred"},
            round_number=self.state.current_round,
        )
        self.state.moves.append(move)

        # Propagate failure info to other stakeholders
        # (In PSS, this would update the failure registry)
        for sid, other in self.state.stakeholders.items():
            if sid != stakeholder.id:
                # Could update other's private info about this withdrawal
                pass

    def inject_offer(
        self,
        stakeholder_id: str,
        terms: dict[str, float],
    ) -> Offer:
        """Manually inject an offer (for testing/intervention).

        Args:
            stakeholder_id: ID of stakeholder making offer.
            terms: Offer terms.

        Returns:
            The created offer.
        """
        stakeholder = self.state.stakeholders[stakeholder_id]
        return make_offer(stakeholder, terms, state=self.state)

    def get_stakeholder_view(self, stakeholder_id: str) -> dict[str, Any]:
        """Get negotiation state from a stakeholder's perspective.

        Excludes other stakeholders' private info.

        Args:
            stakeholder_id: ID of stakeholder.

        Returns:
            Filtered state view.
        """
        stakeholder = self.state.stakeholders[stakeholder_id]

        # Get known info about others
        others = {}
        for sid, s in self.state.stakeholders.items():
            if sid != stakeholder_id:
                others[sid] = {
                    "id": s.id,
                    "name": s.name,
                    "role": s.role,
                    "objectives": s.objectives,  # Public
                    # private_info excluded
                }

        # Get offers relevant to this stakeholder
        relevant_offers = [
            o
            for o in self.state.offers.values()
            if o.proposer == stakeholder_id
            or not o.recipients
            or stakeholder_id in o.recipients
        ]

        return {
            "stakeholder": stakeholder,
            "others": others,
            "offers": relevant_offers,
            "batna": self.state.batnas.get(stakeholder_id),
            "current_round": self.state.current_round,
            "status": self.state.status.value,
        }


def run_negotiation(
    stakeholders: list[Stakeholder],
    config: Optional[NegotiationConfig] = None,
    strategies: Optional[dict[str, NegotiationStrategy]] = None,
) -> dict[str, Any]:
    """Convenience function to run a complete negotiation.

    Args:
        stakeholders: List of stakeholders.
        config: Optional configuration.
        strategies: Optional strategy assignments.

    Returns:
        Negotiation results.
    """
    harness = NegotiationHarness(
        config=config or NegotiationConfig(),
    )

    if strategies:
        harness.strategies = strategies

    harness.initialize(stakeholders)
    return harness.run()


def create_simple_negotiation(
    buyer_max: float,
    seller_min: float,
    buyer_name: str = "Buyer",
    seller_name: str = "Seller",
) -> tuple[NegotiationHarness, list[Stakeholder]]:
    """Create a simple buyer-seller negotiation.

    Args:
        buyer_max: Maximum buyer will pay.
        seller_min: Minimum seller will accept.
        buyer_name: Name for buyer.
        seller_name: Name for seller.

    Returns:
        Tuple of (harness, stakeholders).
    """
    from .stakeholders import define_stakeholder

    buyer = define_stakeholder(
        name=buyer_name,
        objectives={"price": 0.0},  # Minimize
        constraints={"price": (0.0, buyer_max)},
        role="BUYER",
    )

    seller = define_stakeholder(
        name=seller_name,
        objectives={"price": float("inf")},  # Maximize
        constraints={"price": (seller_min, float("inf"))},
        role="SELLER",
    )

    stakeholders = [buyer, seller]

    harness = NegotiationHarness()
    harness.initialize(stakeholders)

    return harness, stakeholders
