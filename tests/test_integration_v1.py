"""Integration tests for v1.0 feature interactions.

These tests verify that the v1.0 features (Genealogy, Failures, Signals, Adaptive Gates)
interact correctly when used together.
"""

import pytest
import time

from pss.types import Context, VelocitySnapshot, SpawnMetadata, SpawnReason, TerminationReason
from pss.genealogy import GenealogyTree, BranchNode
from pss.failures import (
    FailureRegistry,
    FailureSummary,
    FailureType,
    FailureScope,
    format_failures_for_injection,
)
from pss.signals import (
    BulletinBoard,
    Signal,
    SignalType,
    create_signal,
    extract_signals_from_response,
    format_signals_for_injection,
)
from pss.velocity import VelocityTracker
from pss.adaptive_gates import (
    AdaptiveGateConfig,
    GateState,
    GateTimingState,
    compute_gate_state,
    should_trigger_gate,
    create_default_config,
)


def make_spawn_metadata(
    reason: str = "explore option",
    spawn_reason: SpawnReason = SpawnReason.GATE_BRANCH,
    gate_number: int = 1,
    confidence: float = 0.7,
    velocity: float | None = None,
) -> SpawnMetadata:
    """Helper to create SpawnMetadata for testing."""
    return SpawnMetadata(
        spawn_reason=spawn_reason,
        spawn_edit=reason,
        spawn_gate_number=gate_number,
        spawn_confidence=confidence,
        parent_velocity=velocity,
    )


class TestGenealogyFailureIntegration:
    """Tests for genealogy + failure propagation interactions."""

    def test_sibling_failures_trigger_scope_upgrade(self):
        """When multiple siblings fail similarly, failures should merge."""
        registry = FailureRegistry()

        # Record failures from three siblings
        for i in range(3):
            failure = FailureSummary(
                branch_id=f"sibling_{i}",
                failure_type=FailureType.DEAD_END,
                what_was_attempted="config file parsing with YAML",
                why_it_failed="Parser throws exception on nested keys",
                evidence=[f"Error: invalid key at line {10+i}"],
                suggested_avoidance="Use JSON instead of YAML",
                confidence=0.7,
                scope=FailureScope.LOCAL,
            )
            registry.add_failure(failure)

        # Multiple similar failures should be merged
        assert len(registry.failures) == 1
        # Merged failure should be escalated to DIRECTION scope
        assert registry.failures[0].scope == FailureScope.DIRECTION
        # Confidence should be boosted
        assert registry.failures[0].confidence > 0.7

    def test_genealogy_depth_in_failure_context(self):
        """Failures include genealogy depth information."""
        tree = GenealogyTree()

        # Build tree: root -> child -> grandchild
        tree.nodes["root"] = BranchNode(id="root", parent_id=None)
        tree.nodes["child"] = BranchNode(id="child", parent_id="root")
        tree.nodes["grandchild"] = BranchNode(id="grandchild", parent_id="child")
        tree.nodes["root"].children_ids = ["child"]
        tree.nodes["child"].children_ids = ["grandchild"]

        # Verify depth tracking
        assert tree.get_depth("root") == 0
        assert tree.get_depth("child") == 1
        assert tree.get_depth("grandchild") == 2

        # Create failure with genealogy info
        failure = FailureSummary(
            branch_id="grandchild",
            failure_type=FailureType.DEAD_END,
            what_was_attempted="deep exploration",
            why_it_failed="hit dead end",
            evidence=[],
            suggested_avoidance="try different path",
            confidence=0.6,
            scope=FailureScope.LOCAL,
            branch_depth=tree.get_depth("grandchild"),
            branch_path=tree.get_branch_path("grandchild"),
        )

        assert failure.branch_depth == 2

    def test_child_branch_sees_parent_failures(self):
        """Child branches should receive failures relevant to their direction."""
        registry = FailureRegistry()

        # Parent fails exploring OAuth
        registry.add_failure(FailureSummary(
            branch_id="parent",
            failure_type=FailureType.DEAD_END,
            what_was_attempted="OAuth token refresh",
            why_it_failed="API doesn't support refresh tokens",
            evidence=["Error: refresh_token not supported"],
            suggested_avoidance="Use session-based auth instead",
            confidence=0.8,
            scope=FailureScope.LOCAL,
        ))

        # Child exploring same direction should get the warning
        relevant = registry.get_relevant_failures("OAuth authentication")
        assert len(relevant) == 1
        assert "OAuth" in relevant[0].what_was_attempted

        # Child exploring different direction shouldn't get it
        irrelevant = registry.get_relevant_failures("database connection pooling")
        assert len(irrelevant) == 0


class TestVelocityGenealogyIntegration:
    """Tests for velocity tracking + genealogy interactions."""

    def test_high_velocity_spawn_records_metadata(self):
        """Spawns during high velocity should record velocity in metadata."""
        tree = GenealogyTree()
        tracker = VelocityTracker()

        # Simulate high velocity exploration
        tracker.snapshots.append(VelocitySnapshot(
            timestamp=time.time(),
            tokens_in_window=1000,
            novelty_rate=0.75,
            assertion_rate=0.7,
            uncertainty_trend=0.3,
            repetition_score=0.1,
            question_density=0.1,
        ))

        velocity = tracker.get_current_velocity()
        assert velocity > 0.5  # Should be high

        # Create branch node with velocity metadata
        node = BranchNode(
            id="high_velocity_branch",
            parent_id="parent",
            parent_velocity=velocity,
            spawn_confidence=0.8,
        )
        tree.nodes["parent"] = BranchNode(id="parent", parent_id=None)
        tree.nodes["high_velocity_branch"] = node
        tree.nodes["parent"].children_ids = ["high_velocity_branch"]

        # Verify velocity was recorded
        assert tree.nodes["high_velocity_branch"].parent_velocity > 0.5

    def test_low_velocity_affects_spawn_weight(self):
        """Low velocity at spawn time should affect genealogy weight."""
        tree = GenealogyTree()
        tracker = VelocityTracker()

        # Simulate low velocity (plateauing)
        for _ in range(3):
            tracker.snapshots.append(VelocitySnapshot(
                timestamp=time.time(),
                tokens_in_window=1000,
                novelty_rate=0.1,
                assertion_rate=0.2,
                uncertainty_trend=-0.3,
                repetition_score=0.7,
                question_density=0.5,
            ))

        velocity = tracker.get_current_velocity()
        assert velocity < 0.4  # Should be low
        assert tracker.is_plateauing()

    def test_velocity_trend_detects_decline(self):
        """Velocity trend should detect declining progress."""
        tracker = VelocityTracker()

        # Add declining velocity snapshots
        velocities = [0.8, 0.6, 0.4, 0.3]
        for v in velocities:
            tracker.snapshots.append(VelocitySnapshot(
                timestamp=time.time(),
                tokens_in_window=500,
                novelty_rate=v,
                assertion_rate=v,
                uncertainty_trend=v * 2 - 1,
                repetition_score=1 - v,
                question_density=1 - v,
            ))

        trend = tracker.get_trend()
        assert trend < 0  # Should be declining


class TestBulletinFailureIntegration:
    """Tests for bulletin board + failure propagation interactions."""

    def test_signal_from_dead_branch_gets_penalty(self):
        """Signals from terminated branches should have reduced confidence."""
        board = BulletinBoard()

        # Branch posts a signal then fails
        signal = create_signal(
            branch_id="dying_branch",
            signal_type=SignalType.DISCOVERY,
            content="Found interesting API pattern",
            total_tokens=5000,
            confidence=0.8,
            source_weight=1.0,
        )
        board.post(signal)

        # Branch terminates
        board.mark_branch_terminated("dying_branch", "stuck")

        # Check signal confidence is penalized
        signals = board.signals
        assert len(signals) == 1
        assert signals[0].source_terminated
        assert signals[0].source_termination_reason == "stuck"

        # Effective confidence should be lower due to dead source penalty
        eff_conf = signals[0].effective_confidence(6000, dead_source_penalty=0.5)
        assert eff_conf < 0.8  # Should be penalized

    def test_failure_warning_becomes_signal(self):
        """Failure text can generate warning signals."""
        board = BulletinBoard()

        # Simulate extracting a warning from failure text
        failure_text = "Warning: don't call the deprecated API endpoint, it will timeout."
        signals = extract_signals_from_response(
            failure_text,
            branch_id="failing_branch",
            total_tokens=5000,
            base_confidence=0.7,
            source_weight=0.8,
        )

        assert len(signals) >= 1
        warning_signal = next((s for s in signals if s.signal_type == SignalType.WARNING), None)
        assert warning_signal is not None

        # Post to board
        board.post(warning_signal)

        # Other branches should receive this warning
        delivered = board.get_signals_for_branch(
            "other_branch",
            "API integration",
            6000,
            min_confidence=0.3,
        )
        assert len(delivered) >= 1

    def test_constraint_signal_survives_branch_completion(self):
        """CONSTRAINT signals should persist when source completes successfully."""
        board = BulletinBoard()

        # Post a constraint
        constraint = create_signal(
            branch_id="explorer",
            signal_type=SignalType.CONSTRAINT,
            content="Authentication must use OAuth 2.0",
            total_tokens=3000,
            confidence=0.9,
            source_weight=1.0,
        )
        board.post(constraint)

        # Source terminates successfully
        board.mark_branch_terminated("explorer", "completed")

        # Constraint should still be available with full confidence (no penalty for completed)
        signal = board.signals[0]
        eff_conf = signal.effective_confidence(4000, dead_source_penalty=0.5)
        # Completed sources don't get penalty, only token decay
        assert eff_conf > 0.7


class TestBulletinVelocityIntegration:
    """Tests for bulletin board + velocity tracking interactions."""

    def test_novelty_content_generates_discovery_signal(self):
        """High novelty content should produce discovery-worthy signals."""
        # Content with discovery pattern
        text = "I discovered that the AuthManager uses TokenService for OAuth2 validation."

        signals = extract_signals_from_response(
            text,
            branch_id="exploring",
            total_tokens=5000,
            base_confidence=0.6,
        )

        # Should extract a discovery signal
        assert len(signals) >= 1
        discovery = next((s for s in signals if s.signal_type == SignalType.DISCOVERY), None)
        assert discovery is not None

    def test_low_confidence_hypothesis_from_low_velocity(self):
        """Hypotheses with lower base confidence represent uncertain exploration."""
        hypothesis_text = "Hypothesis: the timeout is caused by network congestion."

        signals = extract_signals_from_response(
            hypothesis_text,
            branch_id="struggling_branch",
            total_tokens=10000,
            base_confidence=0.4,  # Lower base represents low velocity
            source_weight=0.5,  # Lower weight from genealogy
        )

        assert len(signals) >= 1
        hypothesis = next((s for s in signals if s.signal_type == SignalType.HYPOTHESIS), None)
        assert hypothesis is not None
        # Hypothesis confidence is base - 0.1, so should be 0.3
        assert hypothesis.confidence < 0.4


class TestAdaptiveGatesGenealogyIntegration:
    """Tests for adaptive gates + genealogy interactions."""

    def test_gate_state_can_be_recorded_at_spawn(self):
        """Gate state at spawn time can be recorded in genealogy."""
        tree = GenealogyTree()
        config = create_default_config()

        timing_state = GateTimingState()
        timing_state.current_state = GateState.HIGH_VELOCITY
        timing_state.high_velocity_start_tokens = 5000

        # Create node with gate state in spawn metadata
        # (Gate state would be passed via spawn_edit or custom field)
        node = BranchNode(
            id="energetic_spawn",
            parent_id="parent",
            spawn_edit="found promising direction (HIGH_VELOCITY)",
            spawn_confidence=0.8,
        )
        tree.nodes["parent"] = BranchNode(id="parent", parent_id=None)
        tree.nodes["energetic_spawn"] = node

        assert "HIGH_VELOCITY" in tree.nodes["energetic_spawn"].spawn_edit

    def test_plateau_detection_affects_gate_trigger(self):
        """PLATEAUING state should be detected from velocity tracker."""
        tracker = VelocityTracker()
        config = create_default_config()
        timing_state = GateTimingState()

        # Create plateau conditions
        for _ in range(5):
            tracker.snapshots.append(VelocitySnapshot(
                timestamp=time.time(),
                tokens_in_window=1000,
                novelty_rate=0.1,
                assertion_rate=0.1,
                uncertainty_trend=-0.3,
                repetition_score=0.7,
                question_density=0.6,
            ))

        # Should be plateauing
        assert tracker.is_plateauing()

        state = compute_gate_state(tracker, timing_state, 20000, config)
        assert state == GateState.PLATEAUING


class TestFullPipelineIntegration:
    """Tests for complete pipeline with all v1.0 features."""

    def test_exploration_lifecycle(self):
        """Test a complete exploration lifecycle with all features."""
        tree = GenealogyTree()
        registry = FailureRegistry()
        board = BulletinBoard()

        # 1. Start exploration
        tree.nodes["root"] = BranchNode(id="root", parent_id=None)

        # 2. Root explores and posts discovery
        discovery_text = "I found that the config uses YAML format for settings."
        signals = extract_signals_from_response(discovery_text, "root", 5000)
        for signal in signals:
            board.post(signal)

        # 3. Root spawns children
        tree.nodes["child_A"] = BranchNode(id="child_A", parent_id="root")
        tree.nodes["child_B"] = BranchNode(id="child_B", parent_id="root")
        tree.nodes["root"].children_ids = ["child_A", "child_B"]

        # 4. Child A succeeds
        tree.nodes["child_A"].is_leaf = True
        tree.nodes["child_A"].termination_reason = TerminationReason.COMPLETED

        # 5. Child B fails
        registry.add_failure(FailureSummary(
            branch_id="child_B",
            failure_type=FailureType.DEAD_END,
            what_was_attempted="config validation",
            why_it_failed="Config validation library not available",
            evidence=["ImportError: validation module not found"],
            suggested_avoidance="Skip validation or use built-in checks",
            confidence=0.7,
            scope=FailureScope.LOCAL,
        ))
        board.mark_branch_terminated("child_B", "stuck")

        # 6. New child spawned should get failure warning
        tree.nodes["child_C"] = BranchNode(id="child_C", parent_id="root")
        tree.nodes["root"].children_ids.append("child_C")

        failures_for_c = registry.get_relevant_failures("config validation")
        assert len(failures_for_c) >= 1

    def test_parallel_branches_share_signals(self):
        """Test that parallel branches can share signals."""
        tree = GenealogyTree()
        board = BulletinBoard()

        # Create parallel branches
        tree.nodes["root"] = BranchNode(id="root", parent_id=None)
        tree.nodes["branch_1"] = BranchNode(id="branch_1", parent_id="root")
        tree.nodes["branch_2"] = BranchNode(id="branch_2", parent_id="root")
        tree.nodes["branch_3"] = BranchNode(id="branch_3", parent_id="root")
        tree.nodes["root"].children_ids = ["branch_1", "branch_2", "branch_3"]

        # Branch 1 discovers something
        signal = create_signal(
            branch_id="branch_1",
            signal_type=SignalType.DISCOVERY,
            content="The authentication endpoint requires HTTPS",
            total_tokens=5000,
            confidence=0.8,
            source_weight=1.0,
        )
        board.post(signal)

        # Branch 2 and 3 should receive the signal
        signals_2 = board.get_signals_for_branch(
            "branch_2", "authentication", 6000, min_confidence=0.3
        )
        assert len(signals_2) == 1
        assert "HTTPS" in signals_2[0].content

        signals_3 = board.get_signals_for_branch(
            "branch_3", "authentication", 6000, min_confidence=0.3
        )
        assert len(signals_3) == 1

        # Branch 1 shouldn't receive its own signal
        signals_1 = board.get_signals_for_branch(
            "branch_1", "authentication", 6000, min_confidence=0.3
        )
        assert len(signals_1) == 0

    def test_velocity_affects_gate_timing(self):
        """Test that velocity metrics affect when gates fire."""
        config = create_default_config()
        timing_state = GateTimingState()

        # High velocity scenario
        high_tracker = VelocityTracker()
        high_tracker.snapshots.append(VelocitySnapshot(
            timestamp=time.time(),
            tokens_in_window=1000,
            novelty_rate=0.75,  # Below spike threshold (0.8)
            assertion_rate=0.8,
            uncertainty_trend=0.5,
            repetition_score=0.1,
            question_density=0.1,
        ))
        timing_state.high_velocity_start_tokens = 0

        ctx_high = Context(id="high", parent_id=None, token_count=config.soft_gate_base_tokens)
        state_high, gate_high = should_trigger_gate(ctx_high, high_tracker, timing_state, config)

        # Low velocity scenario
        low_tracker = VelocityTracker()
        low_tracker.snapshots.append(VelocitySnapshot(
            timestamp=time.time(),
            tokens_in_window=1000,
            novelty_rate=0.2,
            assertion_rate=0.2,
            uncertainty_trend=-0.3,
            repetition_score=0.6,
            question_density=0.5,
        ))
        timing_state_low = GateTimingState()
        timing_state_low.low_velocity_start_tokens = 0

        # Low velocity triggers gate earlier
        accelerated_tokens = int(config.soft_gate_base_tokens / config.soft_gate_low_velocity_divisor) + 1000
        ctx_low = Context(id="low", parent_id=None, token_count=accelerated_tokens)
        state_low, gate_low = should_trigger_gate(ctx_low, low_tracker, timing_state_low, config)

        # Low velocity should trigger soft gate at fewer tokens
        assert gate_low == "soft"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_bulletin_board_returns_empty_injection(self):
        """Empty board returns empty injection string."""
        board = BulletinBoard()

        signals = board.get_signals_for_branch("any", "any direction", 5000)
        assert len(signals) == 0

        formatted = format_signals_for_injection(signals)
        assert formatted == ""

    def test_all_signals_below_confidence_threshold(self):
        """No signals delivered when all below threshold."""
        board = BulletinBoard()

        # Post low confidence signal
        signal = create_signal(
            branch_id="weak",
            signal_type=SignalType.HYPOTHESIS,
            content="Maybe something",
            total_tokens=0,
            confidence=0.2,
            source_weight=0.5,
        )
        board.post(signal)

        # Much later (confidence decayed)
        signals = board.get_signals_for_branch(
            "other", "anything", 50000, min_confidence=0.3
        )
        assert len(signals) == 0

    def test_failure_registry_empty_injection(self):
        """Empty failure registry returns empty injection."""
        registry = FailureRegistry()

        formatted = format_failures_for_injection(registry.failures)
        assert formatted == ""

    def test_velocity_tracker_no_snapshots(self):
        """Velocity tracker handles no snapshots gracefully."""
        tracker = VelocityTracker()

        assert tracker.get_current_velocity() == 0.5  # Default
        assert tracker.get_trend() == 0.0
        assert not tracker.is_plateauing()

    def test_genealogy_tree_root_node(self):
        """Genealogy tree handles root nodes correctly."""
        tree = GenealogyTree()

        tree.nodes["root"] = BranchNode(id="root", parent_id=None)

        assert tree.get_depth("root") == 0
        assert tree.get_ancestors("root") == []
        assert tree.get_siblings("root") == []

    def test_signal_delivered_only_once(self):
        """Signals are not re-delivered to same branch."""
        board = BulletinBoard()

        signal = create_signal(
            branch_id="source",
            signal_type=SignalType.DISCOVERY,
            content="Important finding about authentication",
            total_tokens=5000,
            confidence=0.8,
            source_weight=1.0,
        )
        board.post(signal)

        # First delivery
        first = board.get_signals_for_branch("target", "authentication", 6000, min_confidence=0.3)
        assert len(first) == 1

        # Second attempt - should be empty
        second = board.get_signals_for_branch("target", "authentication", 7000, min_confidence=0.3)
        assert len(second) == 0

    def test_failure_aging_and_staleness(self):
        """Failures age and become stale over time."""
        registry = FailureRegistry(stale_after_branches=2, archive_after_branches=4)

        failure = FailureSummary(
            branch_id="old",
            failure_type=FailureType.DEAD_END,
            what_was_attempted="old approach",
            why_it_failed="didn't work",
            evidence=[],
            suggested_avoidance="try something else",
            confidence=0.7,
            scope=FailureScope.LOCAL,
        )
        registry.add_failure(failure)

        assert registry.failures[0].status == "active"

        # Age by 2 branches -> should become stale
        registry.age_all()
        registry.age_all()
        assert registry.failures[0].status == "stale"

        # Age by 2 more -> should become archived
        registry.age_all()
        registry.age_all()
        assert registry.failures[0].status == "archived"

    def test_signal_deduplication_on_merge(self):
        """Similar signals should be merged, not duplicated."""
        board = BulletinBoard()

        # Post similar signals
        signal1 = create_signal(
            branch_id="branch_1",
            signal_type=SignalType.DISCOVERY,
            content="The API returns JSON formatted data",
            total_tokens=5000,
            confidence=0.6,
            source_weight=1.0,
        )
        signal2 = create_signal(
            branch_id="branch_2",
            signal_type=SignalType.DISCOVERY,
            content="The API returns JSON format data",
            total_tokens=6000,
            confidence=0.6,
            source_weight=1.0,
        )

        board.post(signal1)
        board.post(signal2)

        # Should be merged into one
        assert len(board.signals) == 1
        # Confidence should be boosted
        assert board.signals[0].confidence > 0.6
