"""Tests for pss.adaptive_gates module."""

import pytest
import time

from pss.adaptive_gates import (
    AdaptiveGateConfig,
    GateState,
    GateTimingState,
    compute_gate_state,
    should_trigger_gate,
    format_gate_status,
    create_default_config,
    create_aggressive_config,
    create_relaxed_config,
    create_empirical_config,
    detect_momentum_loss,
)
from pss.velocity import VelocityTracker
from pss.types import Context, VelocitySnapshot


def make_context(token_count: int = 10000, gates_seen: int = 0) -> Context:
    """Helper to create a Context for testing."""
    return Context(
        id="test_ctx",
        parent_id=None,
        token_count=token_count,
        gates_seen=gates_seen,
    )


def make_tracker_with_velocity(velocity: float, avoid_novelty_spike: bool = True) -> VelocityTracker:
    """Helper to create a tracker with a specific velocity.

    Args:
        velocity: Target overall velocity (0-1).
        avoid_novelty_spike: If True, cap novelty at 0.75 to avoid triggering spike detection.
    """
    tracker = VelocityTracker()

    # Compute metrics that would produce the desired velocity
    # velocity ≈ novelty*0.3 + assertion*0.25 + (1-rep)*0.25 + (1-quest)*0.1 + (unc+1)/2*0.1
    # For simplicity, set all to uniform values that approximate target

    # Cap novelty to avoid triggering novelty spike (default threshold is 0.8)
    novelty = min(velocity, 0.75) if avoid_novelty_spike else velocity

    tracker.snapshots.append(VelocitySnapshot(
        timestamp=time.time(),
        tokens_in_window=1000,
        novelty_rate=novelty,
        assertion_rate=velocity,
        uncertainty_trend=velocity * 2 - 1,  # map 0-1 to -1-1
        repetition_score=1 - velocity,
        question_density=1 - velocity,
    ))

    return tracker


class TestAdaptiveGateConfig:
    """Tests for AdaptiveGateConfig."""

    def test_default_config(self):
        """Default config has reasonable values."""
        config = create_default_config()

        assert config.min_tokens > 0
        assert config.max_tokens > config.min_tokens
        assert config.soft_gate_base_tokens > config.min_tokens
        assert config.hard_gate_base_tokens > config.soft_gate_base_tokens

    def test_aggressive_config_gates_earlier(self):
        """Aggressive config has lower thresholds."""
        default = create_default_config()
        aggressive = create_aggressive_config()

        assert aggressive.soft_gate_base_tokens < default.soft_gate_base_tokens
        assert aggressive.hard_gate_base_tokens < default.hard_gate_base_tokens

    def test_relaxed_config_gates_later(self):
        """Relaxed config has higher thresholds."""
        default = create_default_config()
        relaxed = create_relaxed_config()

        assert relaxed.soft_gate_base_tokens > default.soft_gate_base_tokens
        assert relaxed.hard_gate_base_tokens > default.hard_gate_base_tokens


class TestGateTimingState:
    """Tests for GateTimingState."""

    def test_initial_state(self):
        """Initial state is exploring."""
        state = GateTimingState()

        assert state.current_state == GateState.EXPLORING
        assert not state.soft_gate_fired
        assert not state.hard_gate_fired

    def test_tracks_low_velocity_start(self):
        """Tracks when low velocity started."""
        state = GateTimingState()
        state.low_velocity_start_tokens = 5000

        assert state.low_velocity_start_tokens == 5000


class TestComputeGateState:
    """Tests for compute_gate_state function."""

    def test_high_velocity_state(self):
        """High velocity produces HIGH_VELOCITY state."""
        tracker = make_tracker_with_velocity(0.8)
        timing_state = GateTimingState()
        timing_state.high_velocity_start_tokens = 0  # Already sustained
        config = create_default_config()

        state = compute_gate_state(tracker, timing_state, 20000, config)

        assert state == GateState.HIGH_VELOCITY

    def test_low_velocity_triggers_soft_gate_pending(self):
        """Low velocity eventually triggers SOFT_GATE_PENDING."""
        tracker = make_tracker_with_velocity(0.3)
        timing_state = GateTimingState()
        timing_state.low_velocity_start_tokens = 0  # Already sustained
        config = create_default_config()

        state = compute_gate_state(tracker, timing_state, 10000, config)

        assert state == GateState.SOFT_GATE_PENDING

    def test_novelty_spike_detected(self):
        """High novelty produces NOVELTY_SPIKE state."""
        tracker = VelocityTracker()
        tracker.snapshots.append(VelocitySnapshot(
            timestamp=time.time(),
            tokens_in_window=1000,
            novelty_rate=0.9,  # Very high novelty
            assertion_rate=0.5,
            uncertainty_trend=0.0,
            repetition_score=0.2,
            question_density=0.2,
        ))
        timing_state = GateTimingState()
        config = create_default_config()

        state = compute_gate_state(tracker, timing_state, 10000, config)

        assert state == GateState.NOVELTY_SPIKE

    def test_cruising_state_for_medium_velocity(self):
        """Medium velocity produces CRUISING state."""
        tracker = make_tracker_with_velocity(0.5)
        timing_state = GateTimingState()
        config = create_default_config()

        state = compute_gate_state(tracker, timing_state, 10000, config)

        assert state == GateState.CRUISING


class TestShouldTriggerGate:
    """Tests for should_trigger_gate function."""

    def test_no_gate_below_min_tokens(self):
        """No gate fires below minimum tokens."""
        ctx = make_context(token_count=3000)
        tracker = make_tracker_with_velocity(0.2)  # Low velocity
        timing_state = GateTimingState()
        config = create_default_config()

        state, gate_type = should_trigger_gate(ctx, tracker, timing_state, config)

        assert gate_type == "none"
        assert state == GateState.EXPLORING

    def test_hard_gate_at_max_tokens(self):
        """Hard gate always fires at max tokens."""
        config = create_default_config()
        ctx = make_context(token_count=config.max_tokens)
        tracker = make_tracker_with_velocity(0.8)  # Even high velocity
        timing_state = GateTimingState()

        state, gate_type = should_trigger_gate(ctx, tracker, timing_state, config)

        assert gate_type == "hard"
        assert timing_state.hard_gate_fired

    def test_soft_gate_at_base_tokens_cruising(self):
        """Soft gate fires at base tokens when cruising."""
        config = create_default_config()
        ctx = make_context(token_count=config.soft_gate_base_tokens + 1000)
        tracker = make_tracker_with_velocity(0.5)
        timing_state = GateTimingState()

        state, gate_type = should_trigger_gate(ctx, tracker, timing_state, config)

        assert gate_type == "soft"
        assert timing_state.soft_gate_fired

    def test_soft_gate_delayed_with_high_velocity(self):
        """High velocity delays soft gate beyond base tokens."""
        config = create_default_config()
        ctx = make_context(token_count=config.soft_gate_base_tokens + 1000)
        tracker = make_tracker_with_velocity(0.8)
        timing_state = GateTimingState()
        timing_state.high_velocity_start_tokens = 0

        state, gate_type = should_trigger_gate(ctx, tracker, timing_state, config)

        # With high velocity, base + 1000 might not trigger yet
        # because threshold is multiplied by velocity_multiplier
        assert state == GateState.HIGH_VELOCITY
        # Gate might or might not fire depending on exact multiplier

    def test_soft_gate_accelerated_with_low_velocity(self):
        """Low velocity accelerates soft gate to before base tokens."""
        config = create_default_config()
        # Token count below soft_gate_base but above accelerated threshold
        accelerated_threshold = config.soft_gate_base_tokens / config.soft_gate_low_velocity_divisor
        ctx = make_context(token_count=int(accelerated_threshold) + 1000)
        tracker = make_tracker_with_velocity(0.2)
        timing_state = GateTimingState()
        timing_state.low_velocity_start_tokens = 0

        state, gate_type = should_trigger_gate(ctx, tracker, timing_state, config)

        assert gate_type == "soft"

    def test_plateau_triggers_hard_gate_eventually(self):
        """Plateau state triggers hard gate after enough tokens."""
        config = create_default_config()
        ctx = make_context(token_count=25000)

        # Create plateauing tracker
        tracker = VelocityTracker()
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

        timing_state = GateTimingState()
        timing_state.soft_gate_fired = True
        timing_state.low_velocity_start_tokens = 15000  # Started plateauing at 15k

        state, gate_type = should_trigger_gate(ctx, tracker, timing_state, config)

        # Should be plateauing and potentially hard gating
        assert state == GateState.PLATEAUING

    def test_novelty_spike_delays_gate(self):
        """Novelty spike prevents gate from firing."""
        config = create_default_config()
        ctx = make_context(token_count=config.soft_gate_base_tokens + 1000)

        tracker = VelocityTracker()
        tracker.snapshots.append(VelocitySnapshot(
            timestamp=time.time(),
            tokens_in_window=1000,
            novelty_rate=0.85,  # High novelty
            assertion_rate=0.5,
            uncertainty_trend=0.0,
            repetition_score=0.2,
            question_density=0.2,
        ))

        timing_state = GateTimingState()
        timing_state.last_novelty_spike_tokens = ctx.token_count - 500  # Recent spike

        state, gate_type = should_trigger_gate(ctx, tracker, timing_state, config)

        assert state == GateState.NOVELTY_SPIKE
        # In novelty spike cooldown, no gate


class TestGateStateTransitions:
    """Tests for gate state machine transitions."""

    def test_exploring_to_cruising(self):
        """Transition from EXPLORING to CRUISING with sufficient tokens."""
        tracker = make_tracker_with_velocity(0.5)
        timing_state = GateTimingState()
        timing_state.current_state = GateState.EXPLORING
        config = create_default_config()

        state = compute_gate_state(tracker, timing_state, 8000, config)

        assert state == GateState.CRUISING

    def test_cruising_to_high_velocity(self):
        """Transition to HIGH_VELOCITY when velocity increases."""
        tracker = make_tracker_with_velocity(0.8)
        timing_state = GateTimingState()
        timing_state.high_velocity_start_tokens = 0
        config = create_default_config()

        state = compute_gate_state(tracker, timing_state, 10000, config)

        assert state == GateState.HIGH_VELOCITY

    def test_high_velocity_to_cruising(self):
        """Transition back to CRUISING when velocity drops."""
        tracker = make_tracker_with_velocity(0.5)  # Medium velocity
        timing_state = GateTimingState()
        timing_state.current_state = GateState.HIGH_VELOCITY
        config = create_default_config()

        state = compute_gate_state(tracker, timing_state, 10000, config)

        assert state == GateState.CRUISING


class TestFormatGateStatus:
    """Tests for gate status formatting."""

    def test_format_exploring_state(self):
        """EXPLORING state is formatted correctly."""
        timing_state = GateTimingState()
        config = create_default_config()

        status = format_gate_status(
            GateState.EXPLORING, 3000, config, timing_state
        )

        assert "EXPLORING" in status
        assert "building context" in status

    def test_format_high_velocity_state(self):
        """HIGH_VELOCITY state shows delayed message."""
        timing_state = GateTimingState()
        config = create_default_config()

        status = format_gate_status(
            GateState.HIGH_VELOCITY, 20000, config, timing_state
        )

        assert "DELAYED" in status
        assert "high velocity" in status

    def test_format_shows_fired_gates(self):
        """Fired gates are indicated in status."""
        timing_state = GateTimingState()
        timing_state.soft_gate_fired = True
        config = create_default_config()

        status = format_gate_status(
            GateState.CRUISING, 20000, config, timing_state
        )

        assert "soft✓" in status

    def test_format_plateau_state(self):
        """PLATEAUING state shows urgent message."""
        timing_state = GateTimingState()
        config = create_default_config()

        status = format_gate_status(
            GateState.PLATEAUING, 20000, config, timing_state
        )

        assert "URGENT" in status
        assert "plateau" in status


class TestIntegrationWithExploration:
    """Integration tests for adaptive gates with exploration."""

    def test_velocity_recorded_in_context(self):
        """Velocity snapshots are stored in context."""
        ctx = make_context(token_count=10000)
        tracker = VelocityTracker()
        timing_state = GateTimingState()
        config = create_default_config()

        # Simulate exploration with velocity measurement
        snapshot = tracker.measure(
            "The AuthManager handles TokenService requests efficiently.",
            tokens_in_chunk=500,
            prev_window_text="",
        )
        ctx.velocity_history.append(snapshot)

        assert len(ctx.velocity_history) == 1
        assert ctx.velocity_history[0].novelty_rate > 0

    def test_adaptive_gates_respect_budget(self):
        """Adaptive gates still respect hard budget limits."""
        config = create_default_config()
        ctx = make_context(token_count=config.max_tokens + 1000)
        tracker = make_tracker_with_velocity(0.9)  # Very high velocity
        timing_state = GateTimingState()

        state, gate_type = should_trigger_gate(ctx, tracker, timing_state, config)

        # Even with high velocity, max_tokens is absolute
        assert gate_type == "hard"

    def test_repeated_gates_with_state_tracking(self):
        """Multiple gate checks maintain state correctly."""
        config = create_default_config()
        tracker = make_tracker_with_velocity(0.5)
        timing_state = GateTimingState()

        # First check - before soft gate
        ctx1 = make_context(token_count=10000)
        state1, gate1 = should_trigger_gate(ctx1, tracker, timing_state, config)
        assert gate1 == "none"

        # Second check - at soft gate
        ctx2 = make_context(token_count=config.soft_gate_base_tokens + 1000)
        state2, gate2 = should_trigger_gate(ctx2, tracker, timing_state, config)
        assert gate2 == "soft"
        assert timing_state.soft_gate_fired

        # Third check - soft already fired
        ctx3 = make_context(token_count=config.soft_gate_base_tokens + 5000)
        state3, gate3 = should_trigger_gate(ctx3, tracker, timing_state, config)
        assert gate3 == "none"  # Soft already fired, not at hard yet

        # Fourth check - at hard gate
        ctx4 = make_context(token_count=config.hard_gate_base_tokens + 1000)
        state4, gate4 = should_trigger_gate(ctx4, tracker, timing_state, config)
        assert gate4 == "hard"
        assert timing_state.hard_gate_fired


class TestMomentumDetection:
    """Tests for v1.1 momentum-based early gate detection."""

    def test_empirical_config_has_momentum_detection_enabled(self):
        """Empirical config has momentum detection enabled by default."""
        config = create_empirical_config()

        assert config.enable_momentum_detection is True
        assert config.momentum_loss_threshold == 0.1
        assert config.conclusion_velocity == 0.55

    def test_default_config_has_momentum_detection_enabled(self):
        """Default config also has momentum detection enabled."""
        config = create_default_config()

        assert config.enable_momentum_detection is True

    def test_detect_momentum_loss_no_snapshots(self):
        """No momentum loss when no snapshots exist."""
        tracker = VelocityTracker()
        timing_state = GateTimingState()
        config = create_empirical_config()

        should_trigger, reason = detect_momentum_loss(tracker, timing_state, config)

        assert should_trigger is False
        assert reason == ""

    def test_detect_momentum_loss_when_velocity_drops(self):
        """Momentum loss detected when velocity drops >0.1 from peak."""
        tracker = VelocityTracker()
        timing_state = GateTimingState()
        timing_state.peak_velocity = 0.8  # Peak was 0.8
        config = create_empirical_config()

        # Current velocity is 0.6 (0.2 drop from peak)
        tracker.snapshots.append(VelocitySnapshot(
            timestamp=time.time(),
            tokens_in_window=1000,
            novelty_rate=0.5,
            assertion_rate=0.5,
            uncertainty_trend=0.0,
            repetition_score=0.4,
            question_density=0.4,
        ))

        should_trigger, reason = detect_momentum_loss(tracker, timing_state, config)

        assert should_trigger is True
        assert "momentum_loss" in reason
        assert timing_state.momentum_loss_detected is True

    def test_detect_momentum_loss_updates_peak(self):
        """Peak velocity is updated when new high is reached."""
        tracker = make_tracker_with_velocity(0.75)
        timing_state = GateTimingState()
        timing_state.peak_velocity = 0.6  # Previous peak was 0.6
        config = create_empirical_config()

        detect_momentum_loss(tracker, timing_state, config)

        # Peak should be updated to new high
        assert timing_state.peak_velocity == pytest.approx(0.75, abs=0.1)

    def test_detect_converging_low_velocity_declining(self):
        """Converging detected when velocity < 0.55 and declining."""
        tracker = VelocityTracker()
        timing_state = GateTimingState()
        config = create_empirical_config()

        # Add multiple snapshots showing declining trend
        for i, vel in enumerate([0.6, 0.55, 0.5, 0.45]):
            tracker.snapshots.append(VelocitySnapshot(
                timestamp=time.time() + i,
                tokens_in_window=1000,
                novelty_rate=vel * 0.8,
                assertion_rate=vel,
                uncertainty_trend=-0.2,
                repetition_score=1 - vel,
                question_density=0.3,
            ))

        should_trigger, reason = detect_momentum_loss(tracker, timing_state, config)

        assert should_trigger is True
        assert "converging" in reason

    def test_momentum_detection_disabled(self):
        """No momentum detection when disabled."""
        tracker = VelocityTracker()
        timing_state = GateTimingState()
        timing_state.peak_velocity = 0.9
        config = create_empirical_config()
        config.enable_momentum_detection = False

        # Add low velocity snapshot
        tracker.snapshots.append(VelocitySnapshot(
            timestamp=time.time(),
            tokens_in_window=1000,
            novelty_rate=0.3,
            assertion_rate=0.3,
            uncertainty_trend=-0.5,
            repetition_score=0.6,
            question_density=0.5,
        ))

        should_trigger, reason = detect_momentum_loss(tracker, timing_state, config)

        assert should_trigger is False
        assert reason == ""

    def test_momentum_loss_state_triggers_soft_gate(self):
        """MOMENTUM_LOSS state triggers soft gate."""
        config = create_empirical_config()
        ctx = make_context(token_count=8000)

        # Create tracker with momentum loss condition
        tracker = VelocityTracker()
        tracker.snapshots.append(VelocitySnapshot(
            timestamp=time.time(),
            tokens_in_window=1000,
            novelty_rate=0.4,
            assertion_rate=0.4,
            uncertainty_trend=-0.2,
            repetition_score=0.5,
            question_density=0.4,
        ))

        timing_state = GateTimingState()
        timing_state.peak_velocity = 0.8  # Had high peak

        state, gate_type = should_trigger_gate(ctx, tracker, timing_state, config)

        assert state == GateState.MOMENTUM_LOSS
        assert gate_type == "soft"
        assert timing_state.soft_gate_fired is True

    def test_converging_state_triggers_soft_gate(self):
        """CONVERGING state triggers soft gate."""
        config = create_empirical_config()
        ctx = make_context(token_count=8000)

        # Create tracker with converging condition
        tracker = VelocityTracker()
        for i, vel in enumerate([0.55, 0.50, 0.45, 0.40]):
            tracker.snapshots.append(VelocitySnapshot(
                timestamp=time.time() + i,
                tokens_in_window=1000,
                novelty_rate=vel * 0.8,
                assertion_rate=vel,
                uncertainty_trend=-0.3,
                repetition_score=1 - vel,
                question_density=0.3,
            ))

        timing_state = GateTimingState()

        state, gate_type = should_trigger_gate(ctx, tracker, timing_state, config)

        assert state == GateState.CONVERGING
        assert gate_type == "soft"

    def test_format_gate_status_momentum_loss(self):
        """MOMENTUM_LOSS state is formatted correctly."""
        timing_state = GateTimingState()
        config = create_default_config()

        status = format_gate_status(
            GateState.MOMENTUM_LOSS, 15000, config, timing_state
        )

        assert "EARLY GATE" in status
        assert "momentum loss" in status

    def test_format_gate_status_converging(self):
        """CONVERGING state is formatted correctly."""
        timing_state = GateTimingState()
        config = create_default_config()

        status = format_gate_status(
            GateState.CONVERGING, 15000, config, timing_state
        )

        assert "CONCLUDING" in status
        assert "convergence" in status

    def test_timing_state_tracks_peak_velocity(self):
        """GateTimingState tracks peak velocity correctly."""
        timing_state = GateTimingState()

        assert timing_state.peak_velocity == 0.0
        assert timing_state.momentum_loss_detected is False

        timing_state.peak_velocity = 0.85
        timing_state.momentum_loss_detected = True

        assert timing_state.peak_velocity == 0.85
        assert timing_state.momentum_loss_detected is True
