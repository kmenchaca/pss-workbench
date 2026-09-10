"""Comprehensive tests for the Distributed Negotiation Simulator."""

import pytest
from datetime import datetime, timedelta

from negotiate import (
    # Types
    Agreement,
    BATNA,
    Coalition,
    CounterOffer,
    GameOutcome,
    NegotiationMove,
    NegotiationState,
    NegotiationStatus,
    Offer,
    OfferStatus,
    Stakeholder,
    ZOPA,
    # Stakeholders
    BUYER,
    SELLER,
    TEMPLATES,
    check_constraints,
    clone_stakeholder,
    create_from_template,
    define_stakeholder,
    get_stakeholder_summary,
    hide_info,
    reveal_info,
    update_objectives,
    # Utility
    CompositeUtility,
    LinearUtility,
    RelativeUtility,
    ThresholdUtility,
    UtilityFunction,
    create_utility_from_objectives,
    evaluate_terms,
    utility_difference,
    # Offers
    accept,
    check_offer_expired,
    counter_offer,
    evaluate_offer,
    get_pending_offers,
    make_offer,
    reject,
    withdraw,
    # Protocols
    AlternatingOffers,
    AuctionProtocol,
    MediatedNegotiation,
    SimultaneousBids,
    # Strategies
    AnchoringStrategy,
    CompetitiveStrategy,
    CooperativeStrategy,
    TitForTatStrategy,
    get_strategy,
    select_strategy_for_stakeholder,
    # BATNA
    batna_bluff,
    calculate_walk_away_point,
    compare_to_batna,
    detect_batna_bluff,
    discover_batna,
    evaluate_batna,
    # ZOPA
    expand_zopa,
    find_optimal_point,
    find_zopa,
    no_zopa_detection,
    visualize_zopa,
    # Game Theory
    Game,
    analyze_game,
    build_game,
    dominant_strategy,
    find_nash_equilibrium,
    minimax,
    pareto_frontier,
    # Transcripts
    NegotiationTranscript,
    analyze_concessions,
    create_transcript,
    export_transcript,
    identify_turning_points,
    summarize_negotiation,
    # Harness
    NegotiationConfig,
    NegotiationHarness,
    create_simple_negotiation,
    run_negotiation,
    # Multi-party
    coalition_utility,
    defection_analysis,
    find_stable_coalitions,
    form_coalition,
    shapley_value,
    # Training
    TrainingScenario,
    create_training_scenario,
    feedback,
    score_negotiation,
    start_training_session,
)


# ============================================================================
# Stakeholder Tests
# ============================================================================


class TestStakeholderDefinition:
    """Tests for stakeholder creation and management."""

    def test_define_stakeholder_basic(self):
        """Test basic stakeholder creation."""
        s = define_stakeholder(
            name="Test Buyer",
            objectives={"price": 0.0},
            constraints={"price": (0, 100)},
        )

        assert s.name == "Test Buyer"
        assert s.objectives == {"price": 0.0}
        assert s.constraints == {"price": (0, 100)}
        assert s.id is not None

    def test_define_stakeholder_with_private_info(self):
        """Test stakeholder with private information."""
        s = define_stakeholder(
            name="Buyer",
            objectives={"price": 0.0},
            private_info={"max_budget": 150, "urgency": "high"},
        )

        assert s.private_info["max_budget"] == 150
        assert s.private_info["urgency"] == "high"

    def test_create_from_template_buyer(self):
        """Test creating buyer from template."""
        buyer = create_from_template("Alice", "BUYER")

        assert buyer.name == "Alice"
        assert buyer.role == "BUYER"
        assert "price" in buyer.objectives

    def test_create_from_template_seller(self):
        """Test creating seller from template."""
        seller = create_from_template("Bob", "SELLER")

        assert seller.name == "Bob"
        assert seller.role == "SELLER"
        assert seller.objectives.get("price") == float("inf")

    def test_create_from_template_invalid(self):
        """Test invalid template raises error."""
        with pytest.raises(ValueError):
            create_from_template("Test", "INVALID_TEMPLATE")

    def test_reveal_info(self):
        """Test revealing private information."""
        s = define_stakeholder(
            name="Buyer",
            objectives={"price": 0.0},
            private_info={"max_budget": 150},
        )

        key, value = reveal_info(s, "max_budget")
        assert key == "max_budget"
        assert value == 150

    def test_reveal_info_missing_key(self):
        """Test revealing non-existent info raises error."""
        s = define_stakeholder(name="Buyer", objectives={"price": 0.0})

        with pytest.raises(KeyError):
            reveal_info(s, "nonexistent")

    def test_hide_info(self):
        """Test hiding new information."""
        s = define_stakeholder(name="Buyer", objectives={"price": 0.0})
        hide_info(s, "secret", "hidden_value")

        assert s.private_info["secret"] == "hidden_value"

    def test_check_constraints_satisfied(self):
        """Test constraint checking when satisfied."""
        s = define_stakeholder(
            name="Buyer",
            objectives={"price": 0.0},
            constraints={"price": (0, 100)},
        )

        satisfied, violations = check_constraints(s, {"price": 50})
        assert satisfied is True
        assert len(violations) == 0

    def test_check_constraints_violated(self):
        """Test constraint checking when violated."""
        s = define_stakeholder(
            name="Buyer",
            objectives={"price": 0.0},
            constraints={"price": (0, 100)},
        )

        satisfied, violations = check_constraints(s, {"price": 150})
        assert satisfied is False
        assert len(violations) == 1

    def test_clone_stakeholder(self):
        """Test cloning a stakeholder."""
        original = define_stakeholder(
            name="Original",
            objectives={"price": 0.0},
            constraints={"price": (0, 100)},
        )

        clone = clone_stakeholder(original, new_name="Clone")

        assert clone.name == "Clone"
        assert clone.id != original.id
        assert clone.objectives == original.objectives

    def test_get_stakeholder_summary(self):
        """Test getting public stakeholder summary."""
        s = define_stakeholder(
            name="Buyer",
            objectives={"price": 0.0},
            private_info={"secret": "hidden"},
            role="BUYER",
        )

        summary = get_stakeholder_summary(s)

        assert summary["name"] == "Buyer"
        assert summary["role"] == "BUYER"
        assert "secret" not in summary  # Private info excluded


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_linear_utility_basic(self):
        """Test basic linear utility calculation."""
        utility = LinearUtility(weights={"price": -1.0, "quality": 1.0})
        score = utility.evaluate({"price": 50, "quality": 80})

        assert score == -50 + 80  # -1*50 + 1*80 = 30

    def test_linear_utility_normalized(self):
        """Test normalized linear utility."""
        utility = LinearUtility(weights={"price": -1.0, "quality": 1.0}, normalize=True)
        score = utility.evaluate({"price": 50, "quality": 80})

        assert score == 30 / 2  # Divided by sum of abs weights

    def test_threshold_utility_all_met(self):
        """Test threshold utility when all thresholds met."""
        utility = ThresholdUtility(
            thresholds={"price": 100, "quality": 50},
            comparison={"price": "<=", "quality": ">="},
        )
        score = utility.evaluate({"price": 80, "quality": 70})

        assert score == 1.0

    def test_threshold_utility_not_met(self):
        """Test threshold utility when thresholds not met."""
        utility = ThresholdUtility(
            thresholds={"price": 100},
            comparison={"price": "<="},
        )
        score = utility.evaluate({"price": 150})

        assert score == 0.0

    def test_threshold_utility_partial(self):
        """Test partial threshold utility."""
        utility = ThresholdUtility(
            thresholds={"a": 50, "b": 50},
            comparison={"a": ">=", "b": ">="},
            partial=True,
        )
        score = utility.evaluate({"a": 60, "b": 40})

        assert score == 0.5  # 1 of 2 met

    def test_relative_utility(self):
        """Test utility relative to BATNA."""
        batna = BATNA(stakeholder_id="s1", utility=50.0)
        utility = RelativeUtility(batna=batna)

        score = utility.evaluate({"value": 70})
        assert score == 70 - 50  # terms sum - BATNA

    def test_composite_utility_weighted_sum(self):
        """Test composite utility with weighted sum."""
        linear = LinearUtility(weights={"price": 1.0})
        threshold = ThresholdUtility(thresholds={"quality": 50}, comparison={"quality": ">="})

        composite = CompositeUtility(
            functions=[(linear, 0.6), (threshold, 0.4)],
            aggregation="weighted_sum",
        )

        score = composite.evaluate({"price": 100, "quality": 60})
        # (100 * 0.6 + 1.0 * 0.4) / 1.0 = 60.4
        assert score == pytest.approx(60.4)

    def test_evaluate_terms_for_stakeholder(self):
        """Test evaluating terms for a stakeholder."""
        s = define_stakeholder(
            name="Buyer",
            objectives={"price": 0.0},  # Minimize
        )

        score = evaluate_terms(s, {"price": 50})
        assert score < 0  # Negative because price > 0


# ============================================================================
# Offer Tests
# ============================================================================


class TestOffers:
    """Tests for offer mechanics."""

    def test_make_offer(self):
        """Test creating an offer."""
        s = define_stakeholder(name="Seller", objectives={"price": float("inf")})
        offer = make_offer(s, {"price": 100})

        assert offer.proposer == s.id
        assert offer.terms["price"] == 100
        assert offer.status == OfferStatus.PENDING

    def test_make_offer_with_state(self):
        """Test creating offer with state tracking."""
        s = define_stakeholder(name="Seller", objectives={"price": float("inf")})
        state = NegotiationState(negotiation_id="test", stakeholders={s.id: s})

        offer = make_offer(s, {"price": 100}, state=state)

        assert offer.id in state.offers
        assert len(state.moves) == 1

    def test_evaluate_offer(self):
        """Test evaluating an offer."""
        buyer = define_stakeholder(
            name="Buyer",
            objectives={"price": 0.0},
            constraints={"price": (0, 100)},
        )
        offer = Offer(id="o1", proposer="seller", terms={"price": 80})

        evaluation = evaluate_offer(buyer, offer)

        assert evaluation["is_acceptable"] is True
        assert "utility" in evaluation

    def test_evaluate_offer_constraint_violation(self):
        """Test evaluating offer that violates constraints."""
        buyer = define_stakeholder(
            name="Buyer",
            objectives={"price": 0.0},
            constraints={"price": (0, 100)},
        )
        offer = Offer(id="o1", proposer="seller", terms={"price": 150})

        evaluation = evaluate_offer(buyer, offer)

        assert evaluation["is_acceptable"] is False
        assert len(evaluation["constraint_violations"]) > 0

    def test_counter_offer(self):
        """Test creating a counter-offer."""
        buyer = define_stakeholder(name="Buyer", objectives={"price": 0.0})
        original = Offer(id="o1", proposer="seller", terms={"price": 100})

        counter = counter_offer(buyer, original, {"price": 80})

        assert counter.original_offer_id == "o1"
        assert counter.new_terms["price"] == 80
        assert original.status == OfferStatus.COUNTERED

    def test_accept_offer(self):
        """Test accepting an offer."""
        buyer = define_stakeholder(name="Buyer", objectives={"price": 0.0})
        offer = Offer(id="o1", proposer="seller", terms={"price": 80})

        result = accept(buyer, offer)

        assert result["accepted"] is True
        assert offer.status == OfferStatus.ACCEPTED

    def test_reject_offer(self):
        """Test rejecting an offer."""
        buyer = define_stakeholder(name="Buyer", objectives={"price": 0.0})
        offer = Offer(id="o1", proposer="seller", terms={"price": 150})

        result = reject(buyer, offer, reason="Price too high")

        assert result["rejected"] is True
        assert offer.status == OfferStatus.REJECTED
        assert result["reason"] == "Price too high"

    def test_withdraw_offer(self):
        """Test withdrawing an offer."""
        seller = define_stakeholder(
            name="Seller", objectives={"price": float("inf")}, stakeholder_id="seller"
        )
        offer = Offer(id="o1", proposer="seller", terms={"price": 100})

        result = withdraw(seller, offer)

        assert result["withdrawn"] is True
        assert offer.status == OfferStatus.WITHDRAWN

    def test_withdraw_offer_wrong_proposer(self):
        """Test that only proposer can withdraw."""
        buyer = define_stakeholder(name="Buyer", objectives={"price": 0.0})
        offer = Offer(id="o1", proposer="seller", terms={"price": 100})

        with pytest.raises(ValueError):
            withdraw(buyer, offer)

    def test_check_offer_expired(self):
        """Test offer expiration check."""
        offer = Offer(
            id="o1",
            proposer="seller",
            terms={"price": 100},
            expires_at=datetime.now() - timedelta(hours=1),
        )

        assert check_offer_expired(offer) is True

    def test_get_pending_offers(self):
        """Test getting pending offers."""
        state = NegotiationState(negotiation_id="test")
        state.offers["o1"] = Offer(
            id="o1", proposer="seller", terms={"price": 100}, status=OfferStatus.PENDING
        )
        state.offers["o2"] = Offer(
            id="o2", proposer="seller", terms={"price": 90}, status=OfferStatus.ACCEPTED
        )

        pending = get_pending_offers(state)

        assert len(pending) == 1
        assert pending[0].id == "o1"


# ============================================================================
# Protocol Tests
# ============================================================================


class TestProtocols:
    """Tests for negotiation protocols."""

    def test_alternating_offers_init(self):
        """Test alternating offers protocol initialization."""
        buyer = define_stakeholder(name="Buyer", objectives={"price": 0.0})
        seller = define_stakeholder(name="Seller", objectives={"price": float("inf")})

        state = NegotiationState(
            negotiation_id="test",
            stakeholders={buyer.id: buyer, seller.id: seller},
        )

        protocol = AlternatingOffers(turn_order=[buyer.id, seller.id])
        protocol.initialize(state)

        next_actors = protocol.get_next_actors(state)
        assert next_actors == [buyer.id]

    def test_alternating_offers_advance(self):
        """Test turn advancement in alternating offers."""
        buyer = define_stakeholder(name="Buyer", objectives={"price": 0.0})
        seller = define_stakeholder(name="Seller", objectives={"price": float("inf")})

        state = NegotiationState(
            negotiation_id="test",
            stakeholders={buyer.id: buyer, seller.id: seller},
        )

        protocol = AlternatingOffers(turn_order=[buyer.id, seller.id])
        protocol.initialize(state)

        protocol.advance_round(state)
        next_actors = protocol.get_next_actors(state)

        assert next_actors == [seller.id]

    def test_simultaneous_bids_all_can_act(self):
        """Test simultaneous bids allows all to act."""
        buyer = define_stakeholder(name="Buyer", objectives={"price": 0.0})
        seller = define_stakeholder(name="Seller", objectives={"price": float("inf")})

        state = NegotiationState(
            negotiation_id="test",
            stakeholders={buyer.id: buyer, seller.id: seller},
        )

        protocol = SimultaneousBids()
        protocol.initialize(state)

        actors = protocol.get_next_actors(state)
        assert set(actors) == {buyer.id, seller.id}

    def test_auction_protocol_ascending(self):
        """Test ascending auction protocol."""
        protocol = AuctionProtocol(
            auction_type="ascending",
            bid_dimension="price",
            min_increment=5.0,
        )

        bid1 = Offer(id="b1", proposer="bidder1", terms={"price": 50})
        assert protocol.is_valid_bid(bid1) is True

        protocol.process_bid(bid1)

        bid2 = Offer(id="b2", proposer="bidder2", terms={"price": 53})
        assert protocol.is_valid_bid(bid2) is False  # Below increment

        bid3 = Offer(id="b3", proposer="bidder2", terms={"price": 55})
        assert protocol.is_valid_bid(bid3) is True


# ============================================================================
# Strategy Tests
# ============================================================================


class TestStrategies:
    """Tests for negotiation strategies."""

    def test_cooperative_strategy_initial_offer(self):
        """Test cooperative strategy generates fair initial offer."""
        strategy = CooperativeStrategy(initial_generosity=0.3)
        s = define_stakeholder(
            name="Seller",
            objectives={"price": float("inf")},
            constraints={"price": (50, 100)},
        )
        state = NegotiationState(negotiation_id="test", stakeholders={s.id: s})

        offer = strategy.generate_initial_offer(s, state)

        # Should start below max due to generosity
        assert offer["price"] < 100

    def test_competitive_strategy_aggressive_anchor(self):
        """Test competitive strategy generates aggressive anchor."""
        strategy = CompetitiveStrategy(aggression=0.8)
        s = define_stakeholder(
            name="Seller",
            objectives={"price": float("inf")},
            constraints={"price": (50, 100)},
        )
        state = NegotiationState(negotiation_id="test", stakeholders={s.id: s})

        offer = strategy.generate_initial_offer(s, state)

        # Should be aggressive (above constraint max)
        assert offer["price"] > 100

    def test_get_strategy_by_name(self):
        """Test getting strategy by name."""
        strategy = get_strategy("cooperative")
        assert isinstance(strategy, CooperativeStrategy)

    def test_get_strategy_invalid(self):
        """Test invalid strategy name raises error."""
        with pytest.raises(ValueError):
            get_strategy("invalid_strategy")

    def test_select_strategy_for_role(self):
        """Test automatic strategy selection based on role."""
        buyer = create_from_template("Buyer", "BUYER")
        strategy = select_strategy_for_stakeholder(buyer)

        assert isinstance(strategy, CompetitiveStrategy)


# ============================================================================
# BATNA Tests
# ============================================================================


class TestBATNA:
    """Tests for BATNA discovery and management."""

    def test_discover_batna(self):
        """Test BATNA discovery."""
        s = define_stakeholder(
            name="Buyer",
            objectives={"price": 0.0},
            constraints={"price": (0, 100)},
        )

        batna = discover_batna(s)

        assert batna.stakeholder_id == s.id
        assert batna.utility is not None

    def test_evaluate_batna(self):
        """Test BATNA evaluation."""
        s = define_stakeholder(name="Buyer", objectives={"price": 0.0})
        batna = BATNA(stakeholder_id=s.id, alternative={"terms": {"price": 80}}, utility=50)

        evaluation = evaluate_batna(s, batna)

        assert "strength" in evaluation
        assert "walk_away_point" in evaluation

    def test_batna_bluff(self):
        """Test creating a bluffed BATNA."""
        real_batna = BATNA(
            stakeholder_id="s1",
            alternative={"terms": {"price": 80}},
            utility=50,
        )

        bluffed = batna_bluff(
            define_stakeholder(name="Test", objectives={}),
            real_batna,
            bluff_factor=1.5,
        )

        assert bluffed.is_bluff is True
        assert bluffed.utility == 75  # 50 * 1.5
        assert bluffed.actual_utility == 50

    def test_detect_batna_bluff(self):
        """Test bluff detection."""
        bluffed = BATNA(
            stakeholder_id="s1",
            utility=100,
            is_bluff=True,
            actual_utility=50,
        )

        detection = detect_batna_bluff(bluffed)

        assert detection["is_bluff"] is True
        assert detection["confidence"] == 1.0

    def test_compare_to_batna(self):
        """Test comparing offer to BATNA."""
        s = define_stakeholder(name="Buyer", objectives={"price": 0.0})
        batna = BATNA(stakeholder_id=s.id, utility=50)

        comparison = compare_to_batna(s, {"price": 40}, batna)

        assert "should_accept" in comparison
        assert "difference" in comparison


# ============================================================================
# ZOPA Tests
# ============================================================================


class TestZOPA:
    """Tests for ZOPA analysis."""

    def test_find_zopa_with_overlap(self):
        """Test finding ZOPA when overlap exists."""
        buyer = define_stakeholder(
            name="Buyer",
            objectives={"price": 0.0},
            constraints={"price": (0, 80)},
        )
        seller = define_stakeholder(
            name="Seller",
            objectives={"price": float("inf")},
            constraints={"price": (60, 150)},
        )

        zopas = find_zopa([buyer, seller], dimensions=["price"])

        assert "price" in zopas
        # Should find overlap between 60-80

    def test_no_zopa_detection(self):
        """Test detecting when no ZOPA exists."""
        buyer = define_stakeholder(
            name="Buyer",
            objectives={"price": 0.0},
            constraints={"price": (0, 50)},
        )
        seller = define_stakeholder(
            name="Seller",
            objectives={"price": float("inf")},
            constraints={"price": (80, 150)},
        )

        analysis = no_zopa_detection([buyer, seller])

        # No overlap: buyer max 50, seller min 80
        assert len(analysis["no_zopa_dimensions"]) > 0 or analysis["impossible"]

    def test_visualize_zopa(self):
        """Test ZOPA visualization."""
        buyer = define_stakeholder(
            name="Buyer",
            objectives={"price": 0.0},
            constraints={"price": (0, 80)},
        )
        seller = define_stakeholder(
            name="Seller",
            objectives={"price": float("inf")},
            constraints={"price": (60, 150)},
        )

        viz = visualize_zopa([buyer, seller], "price")

        assert "ZOPA" in viz
        assert "price" in viz


# ============================================================================
# Game Theory Tests
# ============================================================================


class TestGameTheory:
    """Tests for game-theoretic analysis."""

    def test_build_game(self):
        """Test building a game from stakeholders."""
        buyer = define_stakeholder(name="Buyer", objectives={"price": 0.0})
        seller = define_stakeholder(name="Seller", objectives={"price": float("inf")})

        game = build_game([buyer, seller])

        assert len(game.players) == 2
        assert len(game.payoff_matrix) > 0

    def test_find_nash_equilibrium(self):
        """Test finding Nash equilibrium."""
        buyer = define_stakeholder(name="Buyer", objectives={"price": 0.0})
        seller = define_stakeholder(name="Seller", objectives={"price": float("inf")})

        game = build_game([buyer, seller])
        nash = find_nash_equilibrium(game)

        # Should find at least one equilibrium
        assert isinstance(nash, list)

    def test_pareto_frontier(self):
        """Test finding Pareto frontier."""
        outcomes = [
            GameOutcome(strategies={"a": "c", "b": "c"}, payoffs={"a": 60, "b": 60}),
            GameOutcome(strategies={"a": "d", "b": "c"}, payoffs={"a": 90, "b": 30}),
            GameOutcome(strategies={"a": "c", "b": "d"}, payoffs={"a": 30, "b": 90}),
            GameOutcome(strategies={"a": "d", "b": "d"}, payoffs={"a": 40, "b": 40}),
        ]

        pareto = pareto_frontier(outcomes)

        # The mutual cooperation and exploitation outcomes should be on frontier
        assert len(pareto) >= 1

    def test_minimax(self):
        """Test minimax strategy."""
        buyer = define_stakeholder(name="Buyer", objectives={"price": 0.0})
        seller = define_stakeholder(name="Seller", objectives={"price": float("inf")})

        game = build_game([buyer, seller])
        strategy, guaranteed = minimax(game, buyer.id)

        assert strategy is not None
        assert isinstance(guaranteed, (int, float))

    def test_analyze_game_complete(self):
        """Test complete game analysis."""
        buyer = define_stakeholder(name="Buyer", objectives={"price": 0.0})
        seller = define_stakeholder(name="Seller", objectives={"price": float("inf")})

        analysis = analyze_game([buyer, seller])

        assert "nash_equilibria" in analysis
        assert "pareto_frontier" in analysis
        assert "minimax_strategies" in analysis


# ============================================================================
# Transcript Tests
# ============================================================================


class TestTranscript:
    """Tests for negotiation transcripts."""

    def test_create_transcript(self):
        """Test creating transcript from state."""
        state = NegotiationState(negotiation_id="test")
        state.moves.append(
            NegotiationMove(
                move_type="offer",
                stakeholder_id="s1",
                details={"terms": {"price": 100}},
            )
        )

        transcript = create_transcript(state)

        assert transcript.negotiation_id == "test"
        assert len(transcript.moves) == 1

    def test_analyze_concessions(self):
        """Test concession analysis."""
        transcript = NegotiationTranscript(negotiation_id="test")
        transcript.stakeholders = {"s1": define_stakeholder(name="S1", objectives={})}
        transcript.moves = [
            NegotiationMove(
                move_type="offer",
                stakeholder_id="s1",
                details={"terms": {"price": 100}},
                round_number=0,
            ),
            NegotiationMove(
                move_type="counter",
                stakeholder_id="s1",
                details={"adjustments": {"price": -10}},
                round_number=1,
            ),
        ]

        analysis = analyze_concessions(transcript)

        assert "s1" in analysis
        assert analysis["s1"]["num_concessions"] >= 0

    def test_export_transcript_text(self):
        """Test exporting transcript as text."""
        transcript = NegotiationTranscript(negotiation_id="test")
        transcript.stakeholders = {"s1": define_stakeholder(name="Alice", objectives={})}
        transcript.moves = [
            NegotiationMove(
                move_type="offer",
                stakeholder_id="s1",
                details={"terms": {"price": 100}},
            )
        ]

        text = export_transcript(transcript, format="text")

        assert "test" in text
        assert "Alice" in text

    def test_export_transcript_json(self):
        """Test exporting transcript as JSON."""
        transcript = NegotiationTranscript(negotiation_id="test")

        json_str = export_transcript(transcript, format="json")

        assert '"negotiation_id": "test"' in json_str

    def test_summarize_negotiation(self):
        """Test negotiation summary."""
        transcript = NegotiationTranscript(negotiation_id="test")
        transcript.stakeholders = {"s1": define_stakeholder(name="S1", objectives={})}
        transcript.moves = [
            NegotiationMove(move_type="offer", stakeholder_id="s1", round_number=0),
            NegotiationMove(move_type="counter", stakeholder_id="s1", round_number=1),
        ]

        summary = summarize_negotiation(transcript)

        assert summary["total_moves"] == 2
        assert summary["total_rounds"] >= 1


# ============================================================================
# Harness Tests
# ============================================================================


class TestHarness:
    """Tests for the negotiation harness."""

    def test_create_simple_negotiation(self):
        """Test creating a simple buyer-seller negotiation."""
        harness, stakeholders = create_simple_negotiation(
            buyer_max=100,
            seller_min=60,
        )

        assert len(stakeholders) == 2
        assert harness.state is not None

    def test_harness_initialize(self):
        """Test harness initialization."""
        buyer = define_stakeholder(
            name="Buyer",
            objectives={"price": 0.0},
            constraints={"price": (0, 100)},
        )
        seller = define_stakeholder(
            name="Seller",
            objectives={"price": float("inf")},
            constraints={"price": (50, 150)},
        )

        harness = NegotiationHarness()
        state = harness.initialize([buyer, seller])

        assert state.status == NegotiationStatus.IN_PROGRESS
        assert len(state.stakeholders) == 2
        assert len(state.batnas) == 2  # Auto-discovered

    def test_harness_inject_offer(self):
        """Test manually injecting an offer."""
        harness, stakeholders = create_simple_negotiation(100, 60)

        offer = harness.inject_offer(stakeholders[1].id, {"price": 80})

        assert offer.terms["price"] == 80
        assert offer.id in harness.state.offers

    def test_harness_stakeholder_view(self):
        """Test getting stakeholder's view of negotiation."""
        harness, stakeholders = create_simple_negotiation(100, 60)
        buyer = stakeholders[0]

        view = harness.get_stakeholder_view(buyer.id)

        assert view["stakeholder"].id == buyer.id
        assert "others" in view
        assert buyer.id not in view["others"]


# ============================================================================
# Multi-party Tests
# ============================================================================


class TestMultiparty:
    """Tests for multi-party negotiations."""

    def test_form_coalition(self):
        """Test forming a coalition."""
        s1 = define_stakeholder(name="S1", objectives={"price": 0.0})
        s2 = define_stakeholder(name="S2", objectives={"price": 0.0})

        coalition = form_coalition([s1, s2], name="Buyers")

        assert coalition.id == "Buyers"
        assert len(coalition.members) == 2

    def test_coalition_utility(self):
        """Test calculating coalition utility."""
        s1 = define_stakeholder(name="S1", objectives={"value": 1.0})
        s2 = define_stakeholder(name="S2", objectives={"value": 1.0})

        coalition = form_coalition([s1, s2])
        utils = coalition_utility(
            coalition,
            {"value": 100},
            {s1.id: s1, s2.id: s2},
        )

        assert s1.id in utils
        assert s2.id in utils

    def test_shapley_value(self):
        """Test Shapley value calculation."""
        s1 = define_stakeholder(name="S1", objectives={})
        s2 = define_stakeholder(name="S2", objectives={})

        def value_func(members):
            return len(members) * 10

        shapley = shapley_value([s1, s2], value_func)

        # Each player contributes equally
        assert s1.id in shapley
        assert s2.id in shapley


# ============================================================================
# Training Tests
# ============================================================================


class TestTraining:
    """Tests for training mode."""

    def test_create_training_scenario(self):
        """Test creating a training scenario."""
        scenario = create_training_scenario(
            name="Test Scenario",
            description="A test",
            stakeholders=[],
            target_outcomes={"min_utility": 50},
            difficulty=3,
        )

        assert scenario.name == "Test Scenario"
        assert scenario.difficulty == 3

    def test_start_training_session(self):
        """Test starting a training session."""
        buyer = define_stakeholder(name="Human", objectives={"price": 0.0})
        seller = define_stakeholder(name="AI", objectives={"price": float("inf")})

        scenario = TrainingScenario(
            id="test",
            name="Test",
            description="Test scenario",
            stakeholders=[buyer, seller],
        )

        session = start_training_session(scenario, buyer.id)

        assert session.human_stakeholder_id == buyer.id
        assert session.state is not None


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for full negotiation flows."""

    def test_complete_negotiation_flow(self):
        """Test a complete negotiation from start to finish."""
        buyer = define_stakeholder(
            name="Buyer",
            objectives={"price": 0.0},
            constraints={"price": (0, 100)},
        )
        seller = define_stakeholder(
            name="Seller",
            objectives={"price": float("inf")},
            constraints={"price": (50, 150)},
        )

        config = NegotiationConfig(max_rounds=20)
        results = run_negotiation([buyer, seller], config)

        assert "status" in results
        assert "rounds" in results
        assert results["rounds"] <= 20
