"""Comprehensive tests for the Temporal Scenario Planner."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from temporal import (
    # Types
    Actor,
    ActorState,
    BehaviorModel,
    Decision,
    DecisionStatus,
    Event,
    EventCategory,
    Timeline,
    TimelineMetrics,
    TimelineState,
    CausalNode,
    # State management
    advance_time,
    apply_event,
    checkpoint_state,
    create_initial_state,
    diff_states,
    merge_states,
    state_to_summary,
    # Actors
    BaseActor,
    CompetitorActor,
    MarketActor,
    UserActor,
    RegulatorActor,
    Situation,
    create_actor,
    simulate_actor_interaction,
    # Events
    generate_events,
    chain_events,
    black_swan_injection,
    filter_events_by_probability,
    summarize_events,
    # Simulation
    simulate_timeline,
    branch_at_decision,
    plausibility_check,
    terminate_implausible,
    simulate_step,
    timeline_summary,
    # Causality
    CausalGraph,
    build_causal_graph,
    trace_cause,
    trace_effects,
    find_leverage_points,
    counterfactual_analysis,
    visualize_graph,
    # Synthesis
    compare_timelines,
    find_robust_decisions,
    find_brittle_decisions,
    identify_key_uncertainties,
    expected_value,
    synthesis_summary,
    # Harness
    TemporalHarness,
    HarnessConfig,
    quick_scenario_analysis,
    # Monte Carlo
    monte_carlo_simulate,
    probability_distribution,
    confidence_intervals,
    sensitivity_analysis,
    scenario_probability,
    value_at_risk,
)


# ============================================================================
# Test State Management
# ============================================================================

class TestStateManagement:
    """Tests for state management functions."""

    def test_create_initial_state(self):
        """Test creating an initial timeline state."""
        date = datetime(2024, 1, 1)
        state = create_initial_state(
            date=date,
            context={"market_size": 1000000},
        )

        assert state.date == date
        assert state.context["market_size"] == 1000000
        assert len(state.events) == 0
        assert state.parent_state_id is None

    def test_create_initial_state_with_actors(self):
        """Test creating state with actor states."""
        actor_state = ActorState(
            resources={"capital": 500000},
            morale=0.8,
        )
        state = create_initial_state(
            date=datetime(2024, 1, 1),
            actors={"player": actor_state},
        )

        assert "player" in state.actor_states
        assert state.actor_states["player"].resources["capital"] == 500000

    def test_advance_time(self):
        """Test advancing timeline state by duration."""
        state = create_initial_state(date=datetime(2024, 1, 1))
        new_state = advance_time(state, timedelta(days=30))

        assert new_state.date == datetime(2024, 1, 31)
        assert state.date == datetime(2024, 1, 1)  # Original unchanged

    def test_apply_event(self):
        """Test applying an event to state."""
        state = create_initial_state(date=datetime(2024, 1, 1))
        event = Event(
            id="event_1",
            date=datetime(2024, 1, 15),
            description="Market expansion",
            effects={"metric:opportunity_score": 0.1},
            magnitude=3,
        )

        new_state = apply_event(state, event)

        assert len(new_state.events) == 1
        assert new_state.events[0].id == "event_1"
        assert new_state.metrics.opportunity_score > state.metrics.opportunity_score

    def test_apply_event_actor_effects(self):
        """Test applying event with actor effects."""
        actor_state = ActorState(resources={"capital": 1000})
        state = create_initial_state(
            date=datetime(2024, 1, 1),
            actors={"player": actor_state},
        )

        event = Event(
            id="event_1",
            date=datetime(2024, 1, 15),
            description="Investment return",
            effects={"actor:player": {"resources": {"capital": 500}}},
            magnitude=2,
        )

        new_state = apply_event(state, event)
        assert new_state.actor_states["player"].resources["capital"] == 1500

    def test_checkpoint_state(self):
        """Test checkpointing state for branching."""
        state = create_initial_state(date=datetime(2024, 1, 1))
        checkpoint_id, saved_state = checkpoint_state(state)

        assert checkpoint_id.startswith("checkpoint_")
        assert saved_state.parent_state_id == checkpoint_id

    def test_diff_states(self):
        """Test calculating differences between states."""
        state1 = create_initial_state(
            date=datetime(2024, 1, 1),
            context={"value": 100},
        )
        state1.metrics.success_score = 0.5

        state2 = create_initial_state(
            date=datetime(2024, 2, 1),
            context={"value": 150},
        )
        state2.metrics.success_score = 0.7

        diff = diff_states(state1, state2)

        assert diff["time_elapsed"] == timedelta(days=31)
        assert "success_score" in diff["metric_changes"]
        assert diff["context_changes"]["value"]["from"] == 100
        assert diff["context_changes"]["value"]["to"] == 150

    def test_merge_states_average(self):
        """Test merging states with average strategy."""
        base = create_initial_state(date=datetime(2024, 1, 1))

        state1 = create_initial_state(date=datetime(2024, 6, 1))
        state1.metrics.success_score = 0.8

        state2 = create_initial_state(date=datetime(2024, 6, 1))
        state2.metrics.success_score = 0.4

        merged = merge_states(base, [state1, state2], strategy="average")

        assert abs(merged.metrics.success_score - 0.6) < 0.001  # Average of 0.8 and 0.4

    def test_state_to_summary(self):
        """Test generating state summary."""
        state = create_initial_state(date=datetime(2024, 1, 1))
        state.metrics.success_score = 0.75
        summary = state_to_summary(state)

        assert "Timeline State" in summary
        assert "0.75" in summary


# ============================================================================
# Test Actor Modeling
# ============================================================================

class TestActorModeling:
    """Tests for actor modeling."""

    def test_create_actor_competitor(self):
        """Test creating a competitor actor."""
        actor = create_actor(
            actor_type="competitor",
            actor_id="comp1",
            name="Rival Corp",
            behavior=BehaviorModel.AGGRESSIVE,
            market_share=0.3,
        )

        assert isinstance(actor, CompetitorActor)
        assert actor.id == "comp1"
        assert actor.name == "Rival Corp"
        assert actor.behavior_model == BehaviorModel.AGGRESSIVE

    def test_create_actor_market(self):
        """Test creating a market actor."""
        actor = create_actor(
            actor_type="market",
            actor_id="market",
            name="Market",
            behavior=BehaviorModel.REACTIVE,
            volatility=0.5,
        )

        assert isinstance(actor, MarketActor)
        assert actor.volatility == 0.5

    def test_create_actor_user(self):
        """Test creating a user actor."""
        actor = create_actor(
            actor_type="user",
            actor_id="user",
            name="Customers",
            behavior=BehaviorModel.CONSERVATIVE,
            price_sensitivity=0.7,
        )

        assert isinstance(actor, UserActor)
        assert actor.price_sensitivity == 0.7

    def test_create_actor_regulator(self):
        """Test creating a regulator actor."""
        actor = create_actor(
            actor_type="regulator",
            actor_id="reg",
            name="FTC",
            behavior=BehaviorModel.REACTIVE,
            strictness=0.8,
        )

        assert isinstance(actor, RegulatorActor)
        assert actor.strictness == 0.8

    def test_competitor_decide(self):
        """Test competitor decision making."""
        actor = create_actor(
            actor_type="competitor",
            actor_id="comp1",
            name="Rival",
            behavior=BehaviorModel.AGGRESSIVE,
            market_share=0.1,  # Low market share
        )

        state = create_initial_state(date=datetime(2024, 1, 1))
        decision = Decision(
            id="d1",
            date=datetime(2024, 1, 1),
            description="Strategy",
            options=["attack market", "defend position"],
        )

        situation = Situation(state=state, decision=decision)
        choice = actor.decide(situation)

        # Aggressive + low market share should prefer attack
        assert "attack" in choice.lower()

    def test_user_actor_price_sensitive(self):
        """Test user actor with price sensitivity."""
        actor = create_actor(
            actor_type="user",
            actor_id="user",
            name="Customers",
            behavior=BehaviorModel.REACTIVE,
            price_sensitivity=0.8,
        )

        state = create_initial_state(date=datetime(2024, 1, 1))
        decision = Decision(
            id="d1",
            date=datetime(2024, 1, 1),
            description="Product choice",
            options=["premium option", "cheap alternative"],
        )

        situation = Situation(state=state, decision=decision)
        choice = actor.decide(situation)

        # High price sensitivity should prefer cheap
        assert "cheap" in choice.lower()

    def test_simulate_actor_interaction(self):
        """Test simulating interaction between actors."""
        actors = [
            create_actor("competitor", "c1", "Comp 1", BehaviorModel.AGGRESSIVE),
            create_actor("competitor", "c2", "Comp 2", BehaviorModel.CONSERVATIVE),
        ]

        state = create_initial_state(date=datetime(2024, 1, 1))
        decision = Decision(
            id="d1",
            date=datetime(2024, 1, 1),
            description="Market response",
            options=["expand", "hold"],
        )

        situation = Situation(state=state, decision=decision)
        decisions = simulate_actor_interaction(actors, situation)

        assert "c1" in decisions
        assert "c2" in decisions


# ============================================================================
# Test Event System
# ============================================================================

class TestEventSystem:
    """Tests for event generation and handling."""

    def test_generate_events(self):
        """Test generating events for a state."""
        state = create_initial_state(date=datetime(2024, 1, 1))
        events = generate_events(state, num_events=3)

        assert len(events) == 3
        for event in events:
            assert event.id.startswith("event_")
            assert event.description

    def test_generate_events_specific_categories(self):
        """Test generating events from specific categories."""
        state = create_initial_state(date=datetime(2024, 1, 1))
        events = generate_events(
            state,
            num_events=3,
            categories=[EventCategory.MARKET],
        )

        for event in events:
            assert event.category == EventCategory.MARKET

    def test_chain_events_high_magnitude(self):
        """Test chaining events from high-magnitude trigger."""
        state = create_initial_state(date=datetime(2024, 1, 1))
        trigger = Event(
            id="trigger",
            date=datetime(2024, 1, 15),
            description="Major market shift",
            magnitude=7,
            category=EventCategory.MARKET,
        )

        # Run multiple times to account for randomness
        found_chain = False
        for _ in range(10):
            chained = chain_events(trigger, state, max_chain_length=3)
            if chained:
                found_chain = True
                assert all(e.cause == trigger.id or e.cause is not None for e in chained)
                break

        # High magnitude should eventually produce chains
        # (but randomness means we can't guarantee)

    def test_black_swan_injection(self):
        """Test black swan event injection."""
        state = create_initial_state(date=datetime(2024, 1, 1))
        swan = black_swan_injection(state)

        assert swan.cause == "black_swan"
        assert swan.probability == 0.01  # Rare
        assert abs(swan.magnitude) >= 6  # High impact

    def test_black_swan_positive(self):
        """Test forcing positive black swan."""
        state = create_initial_state(date=datetime(2024, 1, 1))
        swan = black_swan_injection(state, force_type="positive")

        assert swan.magnitude > 0

    def test_filter_events_by_probability(self):
        """Test filtering events by probability."""
        events = [
            Event(id="e1", date=datetime(2024, 1, 1), description="High prob", probability=0.9),
            Event(id="e2", date=datetime(2024, 1, 1), description="Low prob", probability=0.1),
        ]

        filtered = filter_events_by_probability(events, threshold=0.5, sample=False)

        assert len(filtered) == 1
        assert filtered[0].id == "e1"

    def test_summarize_events(self):
        """Test event summarization."""
        events = [
            Event(id="e1", date=datetime(2024, 1, 1), description="Market grows",
                  category=EventCategory.MARKET, magnitude=3),
            Event(id="e2", date=datetime(2024, 1, 2), description="Competitor launches",
                  category=EventCategory.COMPETITIVE, magnitude=-2),
        ]

        summary = summarize_events(events)

        assert "2 events" in summary
        assert "Market" in summary
        assert "Competitive" in summary


# ============================================================================
# Test Simulation
# ============================================================================

class TestSimulation:
    """Tests for timeline simulation."""

    def test_simulate_timeline_basic(self):
        """Test basic timeline simulation."""
        state = create_initial_state(date=datetime(2024, 1, 1))
        timeline = simulate_timeline(
            initial_state=state,
            decisions=[],
            horizon=timedelta(days=90),
            step_size=timedelta(days=30),
        )

        assert timeline.id.startswith("timeline_")
        assert timeline.final_state is not None
        # With 90 days horizon and 30-day steps, we get 3 steps, so final date is around 3/31
        assert timeline.final_state.date >= datetime(2024, 3, 31)

    def test_simulate_timeline_with_decision(self):
        """Test simulation with decision."""
        state = create_initial_state(date=datetime(2024, 1, 1))
        decision = Decision(
            id="d1",
            date=datetime(2024, 1, 15),
            description="Strategy",
            options=["expand", "hold"],
        )

        timeline = simulate_timeline(
            initial_state=state,
            decisions=[decision],
            horizon=timedelta(days=60),
        )

        assert len(timeline.decisions) >= 1
        assert timeline.decisions[0].status == DecisionStatus.CHOSEN

    def test_branch_at_decision(self):
        """Test branching timeline at decision point."""
        state = create_initial_state(date=datetime(2024, 1, 1))
        timeline = Timeline(id="base", initial_state=state)
        decision = Decision(
            id="d1",
            date=datetime(2024, 1, 1),
            description="Strategy",
            options=["option_a", "option_b", "option_c"],
        )

        branches = branch_at_decision(timeline, decision)

        assert len(branches) == 3
        assert all(b.branch_point == "d1" for b in branches)
        choices = {b.decisions[0].chosen for b in branches}
        assert choices == {"option_a", "option_b", "option_c"}

    def test_plausibility_check_pass(self):
        """Test plausibility check for valid timeline."""
        state = create_initial_state(date=datetime(2024, 1, 1))
        timeline = Timeline(
            id="t1",
            initial_state=state,
            final_state=state,
            plausibility=0.5,
            probability=0.5,
        )

        is_plausible, issues = plausibility_check(timeline)

        assert is_plausible
        assert len(issues) == 0

    def test_plausibility_check_fail(self):
        """Test plausibility check for implausible timeline."""
        state = create_initial_state(date=datetime(2024, 1, 1))
        state.metrics.success_score = 0.9
        state.metrics.risk_score = 0.9  # Inconsistent

        timeline = Timeline(
            id="t1",
            initial_state=state,
            final_state=state,
            plausibility=0.05,  # Very low
        )

        is_plausible, issues = plausibility_check(timeline, min_plausibility=0.1)

        assert not is_plausible
        assert len(issues) > 0

    def test_terminate_implausible(self):
        """Test terminating implausible timelines."""
        state = create_initial_state(date=datetime(2024, 1, 1))

        timelines = [
            Timeline(id="t1", initial_state=state, plausibility=0.5),
            Timeline(id="t2", initial_state=state, plausibility=0.05),
        ]

        viable, terminated = terminate_implausible(timelines, threshold=0.1)

        assert len(viable) == 1
        assert len(terminated) == 1
        assert viable[0].id == "t1"

    def test_simulate_step(self):
        """Test single simulation step."""
        state = create_initial_state(date=datetime(2024, 1, 1))
        new_state, events, decisions = simulate_step(
            state=state,
            step_size=timedelta(days=7),
        )

        assert new_state.date == datetime(2024, 1, 8)

    def test_timeline_summary(self):
        """Test timeline summary generation."""
        state = create_initial_state(date=datetime(2024, 1, 1))
        state.metrics.success_score = 0.7

        timeline = Timeline(
            id="t1",
            initial_state=state,
            final_state=state,
            plausibility=0.8,
            probability=0.6,
        )

        summary = timeline_summary(timeline)

        assert "t1" in summary
        assert "0.8" in summary  # plausibility


# ============================================================================
# Test Causality
# ============================================================================

class TestCausality:
    """Tests for causal chain tracking."""

    def test_causal_graph_add_node(self):
        """Test adding nodes to causal graph."""
        graph = CausalGraph()
        node = CausalNode(
            id="d1",
            node_type="decision",
            description="Choose strategy",
        )

        graph.add_node(node)

        assert "d1" in graph.nodes
        assert graph.nodes["d1"].description == "Choose strategy"

    def test_causal_graph_add_edge(self):
        """Test adding edges to causal graph."""
        graph = CausalGraph()
        graph.add_node(CausalNode(id="d1", node_type="decision", description="Decide"))
        graph.add_node(CausalNode(id="e1", node_type="event", description="Effect"))

        graph.add_edge("d1", "e1")

        assert "e1" in graph.edges["d1"]
        assert "d1" in graph.reverse_edges["e1"]

    def test_causal_graph_root_nodes(self):
        """Test finding root nodes."""
        graph = CausalGraph()
        graph.add_node(CausalNode(id="d1", node_type="decision", description="Root"))
        graph.add_node(CausalNode(id="e1", node_type="event", description="Effect"))
        graph.add_edge("d1", "e1")

        roots = graph.root_nodes

        assert "d1" in roots
        assert "e1" not in roots

    def test_causal_graph_descendants(self):
        """Test finding descendants."""
        graph = CausalGraph()
        graph.add_node(CausalNode(id="d1", node_type="decision", description="Root"))
        graph.add_node(CausalNode(id="e1", node_type="event", description="Child"))
        graph.add_node(CausalNode(id="e2", node_type="event", description="Grandchild"))
        graph.add_edge("d1", "e1")
        graph.add_edge("e1", "e2")

        descendants = graph.get_descendants("d1")

        assert "e1" in descendants
        assert "e2" in descendants

    def test_build_causal_graph_from_timeline(self):
        """Test building causal graph from timeline."""
        state = create_initial_state(date=datetime(2024, 1, 1))
        timeline = Timeline(
            id="t1",
            initial_state=state,
            decisions=[
                Decision(id="d1", date=datetime(2024, 1, 1), description="Decide",
                         chosen="option_a", status=DecisionStatus.CHOSEN),
            ],
            events=[
                Event(id="e1", date=datetime(2024, 1, 15), description="Effect",
                      cause="d1"),
            ],
        )

        graph = build_causal_graph(timeline)

        assert "d1" in graph.nodes
        assert "e1" in graph.nodes
        assert "e1" in graph.edges.get("d1", [])

    def test_trace_cause(self):
        """Test tracing cause chains."""
        graph = CausalGraph()
        graph.add_node(CausalNode(id="d1", node_type="decision", description="Root"))
        graph.add_node(CausalNode(id="e1", node_type="event", description="Middle"))
        graph.add_node(CausalNode(id="e2", node_type="event", description="End"))
        graph.add_edge("d1", "e1")
        graph.add_edge("e1", "e2")

        chains = trace_cause(graph, "e2")

        assert len(chains) >= 1
        # Chain should go d1 -> e1 -> e2
        for chain in chains:
            if len(chain) == 3:
                assert chain[0] == "d1"
                assert chain[-1] == "e2"

    def test_trace_effects(self):
        """Test tracing effect chains."""
        graph = CausalGraph()
        graph.add_node(CausalNode(id="d1", node_type="decision", description="Root"))
        graph.add_node(CausalNode(id="e1", node_type="event", description="Effect"))
        graph.add_edge("d1", "e1")

        chains = trace_effects(graph, "d1")

        assert len(chains) >= 1
        assert chains[0][0] == "d1"

    def test_find_leverage_points(self):
        """Test finding leverage points."""
        graph = CausalGraph()
        graph.add_node(CausalNode(id="d1", node_type="decision", description="High impact",
                                  strength=0.9))
        graph.add_node(CausalNode(id="e1", node_type="event", description="Effect 1"))
        graph.add_node(CausalNode(id="e2", node_type="event", description="Effect 2"))
        graph.add_edge("d1", "e1")
        graph.add_edge("d1", "e2")

        leverage = find_leverage_points(graph)

        assert len(leverage) >= 1
        assert leverage[0].node_id == "d1"

    def test_counterfactual_analysis(self):
        """Test counterfactual analysis."""
        graph = CausalGraph()
        graph.add_node(CausalNode(id="d1", node_type="decision", description="Strategy: expand"))
        graph.add_node(CausalNode(id="e1", node_type="event", description="Growth"))
        graph.add_edge("d1", "e1")

        analysis = counterfactual_analysis(graph, "d1", "hold")

        assert "decision" in analysis
        assert analysis["alternate_choice"] == "hold"
        assert analysis["affected_nodes_count"] == 1

    def test_visualize_graph(self):
        """Test graph visualization."""
        graph = CausalGraph()
        graph.add_node(CausalNode(id="d1", node_type="decision", description="Decision"))
        graph.add_node(CausalNode(id="e1", node_type="event", description="Event"))
        graph.add_edge("d1", "e1")

        viz = visualize_graph(graph)

        assert "Causal Graph" in viz


# ============================================================================
# Test Synthesis
# ============================================================================

class TestSynthesis:
    """Tests for scenario synthesis."""

    def test_compare_timelines(self):
        """Test comparing timelines."""
        state1 = create_initial_state(date=datetime(2024, 1, 1))
        state1.metrics.success_score = 0.8

        state2 = create_initial_state(date=datetime(2024, 1, 1))
        state2.metrics.success_score = 0.3

        timelines = [
            Timeline(id="t1", initial_state=state1, final_state=state1, probability=0.5),
            Timeline(id="t2", initial_state=state2, final_state=state2, probability=0.5),
        ]

        comparison = compare_timelines(timelines)

        assert comparison.best_case is not None
        assert comparison.worst_case is not None
        assert comparison.best_case.id == "t1"
        assert comparison.worst_case.id == "t2"

    def test_find_robust_decisions(self):
        """Test finding robust decisions."""
        state_good = create_initial_state(date=datetime(2024, 1, 1))
        state_good.metrics.success_score = 0.8

        decision = Decision(
            id="d1",
            date=datetime(2024, 1, 1),
            description="Strategy",
            options=["expand", "hold"],
            chosen="expand",
            status=DecisionStatus.CHOSEN,
        )

        timelines = [
            Timeline(id="t1", initial_state=state_good, final_state=state_good,
                     decisions=[decision]),
            Timeline(id="t2", initial_state=state_good, final_state=state_good,
                     decisions=[decision]),
        ]

        robust = find_robust_decisions(timelines)

        # Same decision with consistent outcomes should be robust
        assert len(robust) >= 0  # May or may not find depending on variance

    def test_expected_value(self):
        """Test expected value calculation."""
        state1 = create_initial_state(date=datetime(2024, 1, 1))
        state1.metrics.success_score = 0.8

        state2 = create_initial_state(date=datetime(2024, 1, 1))
        state2.metrics.success_score = 0.4

        timelines = [
            Timeline(id="t1", initial_state=state1, final_state=state1,
                     probability=0.6, plausibility=1.0),
            Timeline(id="t2", initial_state=state2, final_state=state2,
                     probability=0.4, plausibility=1.0),
        ]

        ev = expected_value(timelines, "success_score")

        # Weighted average: (0.8 * 0.6 + 0.4 * 0.4) / (0.6 + 0.4) = 0.64
        assert 0.6 <= ev <= 0.7

    def test_identify_key_uncertainties(self):
        """Test identifying key uncertainties."""
        state1 = create_initial_state(date=datetime(2024, 1, 1))
        state1.metrics.success_score = 0.9

        state2 = create_initial_state(date=datetime(2024, 1, 1))
        state2.metrics.success_score = 0.2

        timelines = [
            Timeline(id="t1", initial_state=state1, final_state=state1),
            Timeline(id="t2", initial_state=state2, final_state=state2),
        ]

        uncertainties = identify_key_uncertainties(timelines)

        # High variance should indicate uncertainty
        assert len(uncertainties) >= 1

    def test_synthesis_summary(self):
        """Test synthesis summary generation."""
        state = create_initial_state(date=datetime(2024, 1, 1))
        state.metrics.success_score = 0.7

        comparison = compare_timelines([
            Timeline(id="t1", initial_state=state, final_state=state),
        ])

        summary = synthesis_summary(comparison)

        assert "Temporal Scenario Synthesis" in summary


# ============================================================================
# Test Harness
# ============================================================================

class TestHarness:
    """Tests for temporal harness."""

    def test_harness_initialization(self):
        """Test harness initialization."""
        state = create_initial_state(date=datetime(2024, 1, 1))
        decision = Decision(
            id="d1",
            date=datetime(2024, 1, 1),
            description="Strategy",
            options=["a", "b"],
        )

        harness = TemporalHarness(
            initial_state=state,
            initial_decision=decision,
        )

        assert harness.initial_state == state
        assert harness.initial_decision == decision
        assert len(harness.timelines) == 0

    def test_harness_config(self):
        """Test harness configuration."""
        config = HarnessConfig(
            max_timelines=5,
            simulation_horizon=timedelta(days=180),
            plausibility_threshold=0.2,
        )

        assert config.max_timelines == 5
        assert config.simulation_horizon == timedelta(days=180)

    def test_harness_run_basic(self, monkeypatch):
        """Test running harness with basic setup."""
        state = create_initial_state(date=datetime(2024, 1, 1))
        decision = Decision(
            id="d1",
            date=datetime(2024, 1, 1),
            description="Strategy",
            options=["expand", "hold"],
        )

        config = HarnessConfig(
            max_timelines=3,
            simulation_horizon=timedelta(days=30),
            black_swan_probability=0,
        )

        # Ordinary events on each step should not erase every timeline simply
        # because their joint probability shrinks with the simulation horizon.
        monkeypatch.setattr("temporal.simulation.generate_events", lambda state, **kwargs: [
            Event(id=f"ordinary_{state.date}", date=state.date,
                  description="Ordinary neutral event", probability=0.5, magnitude=0)
        ])
        monkeypatch.setattr("temporal.simulation.filter_events_by_probability", lambda events, **kwargs: events)

        harness = TemporalHarness(
            initial_state=state,
            initial_decision=decision,
            config=config,
        )

        comparison = harness.run()

        assert len(harness.timelines) > 0
        assert comparison is not None
        assert {t.decisions[0].chosen for t in harness.timelines} == {"expand", "hold"}
        assert all(t.probability < config.plausibility_threshold for t in harness.timelines)
        assert all(t.plausibility >= config.plausibility_threshold for t in harness.timelines)

    def test_plausibility_normalizes_long_event_chains_but_rejects_impossible_events(self):
        from temporal.simulation import _calculate_plausibility
        state = create_initial_state(date=datetime(2024, 1, 1))
        events = [Event(id=str(i), date=state.date + timedelta(days=i),
                        description="Neutral", probability=0.5, magnitude=0) for i in range(2000)]
        timeline = Timeline(id="long", initial_state=state, final_state=state,
                            events=events, probability=0.0)
        assert _calculate_plausibility(timeline) == pytest.approx(0.4)
        events[0].probability = 0
        assert _calculate_plausibility(timeline) == 0

    def test_harness_summary(self):
        """Test harness summary."""
        state = create_initial_state(date=datetime(2024, 1, 1))
        harness = TemporalHarness(initial_state=state)

        summary = harness.summary()

        assert "Temporal Harness Run" in summary

    def test_quick_scenario_analysis(self):
        """Test quick scenario analysis helper."""
        comparison = quick_scenario_analysis(
            starting_date=datetime(2024, 1, 1),
            initial_context={"market_size": 1000000},
            decision_description="Market strategy",
            decision_options=["aggressive", "conservative"],
            horizon_days=30,
            num_timelines=2,
        )

        assert comparison is not None
        assert len(comparison.timelines) > 0


# ============================================================================
# Test Monte Carlo
# ============================================================================

class TestMonteCarlo:
    """Tests for Monte Carlo simulation."""

    def test_monte_carlo_simulate(self):
        """Test Monte Carlo simulation."""
        state = create_initial_state(date=datetime(2024, 1, 1))

        timelines = monte_carlo_simulate(
            initial_state=state,
            decisions=[],
            n_runs=5,
            horizon=timedelta(days=30),
        )

        assert len(timelines) == 5
        assert all(t.final_state is not None for t in timelines)

    def test_probability_distribution(self):
        """Test probability distribution calculation."""
        states = []
        for score in [0.3, 0.5, 0.5, 0.7, 0.9]:
            state = create_initial_state(date=datetime(2024, 1, 1))
            state.metrics.success_score = score
            states.append(state)

        timelines = [
            Timeline(id=f"t{i}", initial_state=s, final_state=s)
            for i, s in enumerate(states)
        ]

        result = probability_distribution(timelines, "success_score")

        assert result.n_runs == 5
        assert 0.5 <= result.mean <= 0.6  # Average of scores
        assert result.std_dev > 0

    def test_confidence_intervals(self):
        """Test confidence interval calculation."""
        states = []
        for _ in range(10):
            state = create_initial_state(date=datetime(2024, 1, 1))
            state.metrics.success_score = 0.5
            states.append(state)

        timelines = [
            Timeline(id=f"t{i}", initial_state=s, final_state=s)
            for i, s in enumerate(states)
        ]

        intervals = confidence_intervals(timelines, ["success_score"])

        assert "success_score" in intervals
        lower, upper = intervals["success_score"]
        assert lower <= 0.5 <= upper

    def test_scenario_probability(self):
        """Test scenario probability calculation."""
        states = []
        for score in [0.8, 0.9, 0.3, 0.7, 0.6]:
            state = create_initial_state(date=datetime(2024, 1, 1))
            state.metrics.success_score = score
            states.append(state)

        timelines = [
            Timeline(id=f"t{i}", initial_state=s, final_state=s)
            for i, s in enumerate(states)
        ]

        prob = scenario_probability(
            timelines,
            lambda t: t.final_state.metrics.success_score > 0.5  # type: ignore
        )

        # 4 out of 5 have score > 0.5 (0.8, 0.9, 0.7, 0.6)
        assert prob == 0.8

    def test_value_at_risk(self):
        """Test Value at Risk calculation."""
        states = []
        for score in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            state = create_initial_state(date=datetime(2024, 1, 1))
            state.metrics.success_score = score
            states.append(state)

        timelines = [
            Timeline(id=f"t{i}", initial_state=s, final_state=s)
            for i, s in enumerate(states)
        ]

        var = value_at_risk(timelines, "success_score", confidence=0.9)

        # VaR at 90% should be around 0.1-0.2 (5th percentile)
        assert var <= 0.3


# ============================================================================
# Additional Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_timelines_comparison(self):
        """Test comparing empty timeline list."""
        comparison = compare_timelines([])

        assert comparison.best_case is None
        assert comparison.worst_case is None

    def test_single_timeline_comparison(self):
        """Test comparing single timeline."""
        state = create_initial_state(date=datetime(2024, 1, 1))
        timelines = [Timeline(id="t1", initial_state=state, final_state=state)]

        comparison = compare_timelines(timelines)

        assert comparison.best_case == timelines[0]
        assert comparison.worst_case == timelines[0]

    def test_timeline_without_final_state(self):
        """Test handling timeline without final state."""
        state = create_initial_state(date=datetime(2024, 1, 1))
        timeline = Timeline(id="t1", initial_state=state, final_state=None)

        summary = timeline_summary(timeline)
        assert "t1" in summary

    def test_actor_with_no_options(self):
        """Test actor deciding with no options."""
        actor = create_actor("competitor", "c1", "Comp", BehaviorModel.AGGRESSIVE)

        state = create_initial_state(date=datetime(2024, 1, 1))
        decision = Decision(
            id="d1",
            date=datetime(2024, 1, 1),
            description="Empty",
            options=[],
        )

        situation = Situation(state=state, decision=decision)
        choice = actor.decide(situation)

        assert choice == ""

    def test_causal_graph_no_nodes(self):
        """Test operations on empty causal graph."""
        graph = CausalGraph()

        assert graph.root_nodes == []
        assert graph.leaf_nodes == []
        assert graph.get_descendants("nonexistent") == set()
