"""Adaptive gate timing for PSS v1.0.

Instead of fixed token thresholds, adaptive gates use velocity metrics to decide
when to interrupt exploration. High-velocity branches (making progress) get more
runway. Low-velocity branches (repeating, uncertain) get gates sooner.

Key behaviors:
- High velocity (>0.7): Delay gates, let exploration continue
- Medium velocity (0.4-0.7): Normal gate timing
- Low velocity (<0.4): Trigger gates earlier
- Plateau detection: If velocity is low AND stable, hard gate
- Novelty spikes: Brief cooldown after discovering something new

v1.1 additions (empirically derived from trace analysis):
- Momentum loss: velocity drop >0.1 from peak → early gate
- Conclusion threshold: velocity <0.55 + declining → branch is converging
- These thresholds reduced 89% of budget-exhausted branches in testing
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pss.types import Context
from pss.velocity import VelocityTracker


class GateState(Enum):
    """State machine for adaptive gate timing."""

    EXPLORING = "exploring"  # Initial state, building context
    HIGH_VELOCITY = "high_velocity"  # Making rapid progress, delay gates
    CRUISING = "cruising"  # Normal progress, standard timing
    PLATEAUING = "plateauing"  # Low velocity, stable - gate soon
    NOVELTY_SPIKE = "novelty_spike"  # Just found something new, brief cooldown
    SOFT_GATE_PENDING = "soft_gate_pending"  # About to trigger soft gate
    HARD_GATE_PENDING = "hard_gate_pending"  # About to trigger hard gate
    # v1.1: Empirically-derived states
    MOMENTUM_LOSS = "momentum_loss"  # Lost >0.1 velocity from peak
    CONVERGING = "converging"  # Velocity <0.55 + declining - branch concluding


@dataclass
class AdaptiveGateConfig:
    """Configuration for adaptive gate timing."""

    # Token bounds - gates always fire within these limits
    min_tokens: int = 5000  # Never gate before this
    max_tokens: int = 50000  # Always gate by this (hard cap)

    # Velocity thresholds
    high_velocity_threshold: float = 0.7  # Above this = delay gates
    low_velocity_threshold: float = 0.4  # Below this = accelerate gates
    plateau_velocity_threshold: float = 0.3  # Below this + stable = hard gate

    # Soft gate triggers
    soft_gate_base_tokens: int = 15000  # Default soft gate position
    soft_gate_velocity_multiplier: float = 1.5  # High velocity delays by this factor
    soft_gate_low_velocity_divisor: float = 1.5  # Low velocity accelerates by this

    # Hard gate triggers
    hard_gate_base_tokens: int = 35000  # Default hard gate position
    hard_gate_plateau_tokens: int = 5000  # After plateau detected, hard gate in N tokens

    # Sustain requirements
    low_velocity_sustain_tokens: int = 1500  # Must sustain low velocity for N tokens
    high_velocity_sustain_tokens: int = 1000  # Must sustain high velocity for N tokens

    # Novelty spike handling
    novelty_spike_threshold: float = 0.8  # Novelty above this triggers spike
    novelty_spike_cooldown_tokens: int = 2000  # Cooldown after spike

    # Measurement
    velocity_window_tokens: int = 2000  # Window for velocity measurement
    measurement_interval_tokens: int = 500  # How often to measure

    # v1.1: Empirically-derived momentum thresholds (from trace analysis)
    # These reduced 89% of budget-exhausted branches in testing
    momentum_loss_threshold: float = 0.1  # Velocity drop from peak to trigger early gate
    conclusion_velocity: float = 0.55  # Completed branches converge to this value
    enable_momentum_detection: bool = True  # Enable v1.1 momentum-based early gates


@dataclass
class GateTimingState:
    """Tracks state for adaptive gate timing within a single context."""

    current_state: GateState = GateState.EXPLORING
    tokens_at_state_change: int = 0
    tokens_in_current_state: int = 0

    # Tracking for sustained velocity
    low_velocity_start_tokens: int | None = None
    high_velocity_start_tokens: int | None = None

    # Novelty spike tracking
    last_novelty_spike_tokens: int | None = None

    # Whether gates have fired
    soft_gate_fired: bool = False
    hard_gate_fired: bool = False

    # v1.1: Momentum tracking
    peak_velocity: float = 0.0  # Highest velocity seen in this context
    momentum_loss_detected: bool = False  # True if we've lost >0.1 from peak


def detect_momentum_loss(
    tracker: VelocityTracker,
    timing_state: GateTimingState,
    config: AdaptiveGateConfig,
) -> tuple[bool, str]:
    """Detect if the branch has lost momentum.

    Based on empirical analysis of traces (v1.1):
    - 89% of declining branches hit budget exhaustion
    - Branches that complete naturally converge to ~0.55 velocity

    Returns:
        Tuple of (should_trigger, reason)
    """
    if not config.enable_momentum_detection:
        return (False, "")

    if not tracker.snapshots:
        return (False, "")

    velocity = tracker.get_current_velocity()
    trend = tracker.get_trend()

    # Update peak velocity tracking
    if velocity > timing_state.peak_velocity:
        timing_state.peak_velocity = velocity

    # Check 1: Lost >0.1 from peak velocity
    velocity_drop = timing_state.peak_velocity - velocity
    if velocity_drop >= config.momentum_loss_threshold:
        timing_state.momentum_loss_detected = True
        return (True, f"momentum_loss (peak={timing_state.peak_velocity:.2f}, now={velocity:.2f})")

    # Check 2: Below conclusion threshold AND declining
    if velocity < config.conclusion_velocity and trend < -0.05:
        return (True, f"converging (velocity={velocity:.2f}, trend={trend:+.2f})")

    return (False, "")


def compute_gate_state(
    tracker: VelocityTracker,
    timing_state: GateTimingState,
    current_tokens: int,
    config: AdaptiveGateConfig,
) -> GateState:
    """Compute the current gate state based on velocity.

    This is a state machine that transitions based on velocity metrics.
    """
    velocity = tracker.get_current_velocity()
    trend = tracker.get_trend()

    # v1.1: Check for momentum loss (empirically-derived)
    momentum_lost, reason = detect_momentum_loss(tracker, timing_state, config)
    if momentum_lost:
        if "converging" in reason:
            return GateState.CONVERGING
        return GateState.MOMENTUM_LOSS

    # Check for novelty spike
    if tracker.snapshots:
        latest = tracker.snapshots[-1]
        if latest.novelty_rate >= config.novelty_spike_threshold:
            timing_state.last_novelty_spike_tokens = current_tokens
            return GateState.NOVELTY_SPIKE

    # In novelty spike cooldown?
    if timing_state.last_novelty_spike_tokens is not None:
        tokens_since_spike = current_tokens - timing_state.last_novelty_spike_tokens
        if tokens_since_spike < config.novelty_spike_cooldown_tokens:
            return GateState.NOVELTY_SPIKE

    # Check for plateau (low velocity + stable)
    if tracker.is_plateauing(threshold=0.1, window=3):
        if velocity < config.plateau_velocity_threshold:
            return GateState.PLATEAUING

    # Check for sustained high velocity
    if velocity >= config.high_velocity_threshold:
        if timing_state.high_velocity_start_tokens is None:
            timing_state.high_velocity_start_tokens = current_tokens
        elif current_tokens - timing_state.high_velocity_start_tokens >= config.high_velocity_sustain_tokens:
            timing_state.low_velocity_start_tokens = None  # Reset low counter
            return GateState.HIGH_VELOCITY
    else:
        timing_state.high_velocity_start_tokens = None

    # Check for sustained low velocity
    if velocity < config.low_velocity_threshold:
        if timing_state.low_velocity_start_tokens is None:
            timing_state.low_velocity_start_tokens = current_tokens
        elif current_tokens - timing_state.low_velocity_start_tokens >= config.low_velocity_sustain_tokens:
            timing_state.high_velocity_start_tokens = None  # Reset high counter
            # Declining trend makes it worse
            if trend < -0.2:
                return GateState.PLATEAUING
            return GateState.SOFT_GATE_PENDING
    else:
        timing_state.low_velocity_start_tokens = None

    # Default: cruising
    return GateState.CRUISING


def should_trigger_gate(
    ctx: Context,
    tracker: VelocityTracker,
    timing_state: GateTimingState,
    config: AdaptiveGateConfig,
    token_count: int | None = None,
) -> tuple[GateState, Literal["soft", "hard", "none"]]:
    """Determine if a gate should fire based on adaptive timing.

    Args:
        ctx: The current context with token count.
        tracker: VelocityTracker with recent snapshots.
        timing_state: GateTimingState tracking this context's timing.
        config: AdaptiveGateConfig with thresholds.
        token_count: Optional explicit token count (overrides ctx.token_count).
                     Use this when ctx.token_count hasn't been updated yet.

    Returns:
        Tuple of (current_state, gate_type) where gate_type is "soft", "hard", or "none".
    """
    tokens = token_count if token_count is not None else ctx.token_count

    # Hard limits - always respect
    if tokens < config.min_tokens:
        return (GateState.EXPLORING, "none")

    if tokens >= config.max_tokens:
        timing_state.hard_gate_fired = True
        return (GateState.HARD_GATE_PENDING, "hard")

    # Compute current state
    state = compute_gate_state(tracker, timing_state, tokens, config)
    timing_state.current_state = state

    # Determine gate based on state
    gate_type: Literal["soft", "hard", "none"] = "none"

    # v1.1: Handle momentum-based states first (empirically-derived)
    if state == GateState.MOMENTUM_LOSS:
        # Lost momentum - trigger soft gate to check if branch should terminate
        if not timing_state.soft_gate_fired:
            timing_state.soft_gate_fired = True
            gate_type = "soft"

    elif state == GateState.CONVERGING:
        # Branch is naturally concluding - soft gate to let it finish gracefully
        if not timing_state.soft_gate_fired:
            timing_state.soft_gate_fired = True
            gate_type = "soft"

    elif state == GateState.PLATEAUING:
        # Plateau = hard gate soon
        if not timing_state.hard_gate_fired:
            # Check if we've been plateauing long enough
            plateau_tokens = tokens - (timing_state.low_velocity_start_tokens or tokens)
            if plateau_tokens >= config.hard_gate_plateau_tokens:
                timing_state.hard_gate_fired = True
                gate_type = "hard"
            elif not timing_state.soft_gate_fired:
                timing_state.soft_gate_fired = True
                gate_type = "soft"

    elif state == GateState.SOFT_GATE_PENDING:
        # Low velocity but not plateau - soft gate
        if not timing_state.soft_gate_fired:
            effective_soft_threshold = config.soft_gate_base_tokens / config.soft_gate_low_velocity_divisor
            if tokens >= effective_soft_threshold:
                timing_state.soft_gate_fired = True
                gate_type = "soft"

    elif state == GateState.HIGH_VELOCITY:
        # High velocity - delay gates
        effective_soft_threshold = config.soft_gate_base_tokens * config.soft_gate_velocity_multiplier
        effective_hard_threshold = config.hard_gate_base_tokens * config.soft_gate_velocity_multiplier

        if not timing_state.soft_gate_fired and tokens >= effective_soft_threshold:
            timing_state.soft_gate_fired = True
            gate_type = "soft"
        elif timing_state.soft_gate_fired and not timing_state.hard_gate_fired and tokens >= effective_hard_threshold:
            timing_state.hard_gate_fired = True
            gate_type = "hard"

    elif state == GateState.NOVELTY_SPIKE:
        # Novelty spike - no gate during cooldown (already handled by min_tokens)
        pass

    elif state == GateState.CRUISING:
        # Normal velocity - standard timing
        if not timing_state.soft_gate_fired and tokens >= config.soft_gate_base_tokens:
            timing_state.soft_gate_fired = True
            gate_type = "soft"
        elif timing_state.soft_gate_fired and not timing_state.hard_gate_fired and tokens >= config.hard_gate_base_tokens:
            timing_state.hard_gate_fired = True
            gate_type = "hard"

    return (state, gate_type)


def format_gate_status(
    state: GateState,
    tokens: int,
    config: AdaptiveGateConfig,
    timing_state: GateTimingState,
) -> str:
    """Format gate status for display."""
    state_messages = {
        GateState.EXPLORING: "EXPLORING (building context)",
        GateState.HIGH_VELOCITY: "DELAYED (high velocity)",
        GateState.CRUISING: "NORMAL (standard timing)",
        GateState.PLATEAUING: "URGENT (plateau detected)",
        GateState.NOVELTY_SPIKE: "COOLDOWN (novelty spike)",
        GateState.SOFT_GATE_PENDING: "SOFT GATE PENDING",
        GateState.HARD_GATE_PENDING: "HARD GATE PENDING",
        # v1.1 states
        GateState.MOMENTUM_LOSS: "EARLY GATE (momentum loss)",
        GateState.CONVERGING: "CONCLUDING (natural convergence)",
    }

    status = state_messages.get(state, f"UNKNOWN ({state})")

    # Add gate fired indicators
    gates = []
    if timing_state.soft_gate_fired:
        gates.append("soft✓")
    if timing_state.hard_gate_fired:
        gates.append("hard✓")

    if gates:
        status += f" [{', '.join(gates)}]"

    return f"Gate status: {status}"


def create_default_config() -> AdaptiveGateConfig:
    """Create a default adaptive gate configuration."""
    return AdaptiveGateConfig()


def create_aggressive_config() -> AdaptiveGateConfig:
    """Create a more aggressive config that gates earlier."""
    return AdaptiveGateConfig(
        min_tokens=3000,
        soft_gate_base_tokens=10000,
        hard_gate_base_tokens=25000,
        low_velocity_sustain_tokens=1000,
        high_velocity_sustain_tokens=1500,
    )


def create_relaxed_config() -> AdaptiveGateConfig:
    """Create a relaxed config that allows longer exploration."""
    return AdaptiveGateConfig(
        min_tokens=8000,
        max_tokens=80000,
        soft_gate_base_tokens=25000,
        hard_gate_base_tokens=50000,
        soft_gate_velocity_multiplier=2.0,
        low_velocity_sustain_tokens=2000,
    )


def create_empirical_config() -> AdaptiveGateConfig:
    """Create a config tuned from trace analysis (v1.1).

    Based on analysis of 17 branches across 3 traces:
    - 89% of declining velocity branches hit budget exhaustion
    - Completed branches converge to 0.55 velocity
    - Average decline before budget exhaustion: -0.143

    This config triggers early gates when momentum is lost, reducing
    wasted tokens on stuck branches.
    """
    return AdaptiveGateConfig(
        # Enable momentum detection with empirically-derived thresholds
        enable_momentum_detection=True,
        momentum_loss_threshold=0.1,  # From: avg decline = -0.143
        conclusion_velocity=0.55,  # From: completed branches avg = 0.550

        # Lower min_tokens to allow early gates to fire
        min_tokens=2000,

        # Keep other defaults reasonable
        soft_gate_base_tokens=10000,
        hard_gate_base_tokens=30000,
    )
