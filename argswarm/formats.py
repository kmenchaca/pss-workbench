"""
Debate format templates for the Argumentative Swarm system.

Provides pre-built formats like Oxford-style, Lincoln-Douglas, and Socratic,
plus a builder for creating custom formats.
"""

from dataclasses import dataclass, field
from typing import Optional

from .types import DebateFormat, RoundType


# =============================================================================
# Pre-built Debate Formats
# =============================================================================


def OxfordFormat() -> DebateFormat:
    """
    Oxford-style formal debate format.

    Traditional format with:
    - Two opposing teams
    - Opening statements, rebuttals, and closing
    - Cross-examination between teams
    - Formal structure with time limits

    Typically used for academic and competitive debates.
    """
    return DebateFormat(
        name="Oxford",
        rounds=4,
        time_per_round=300,  # 5 minutes per round
        position_count=2,
        round_types=[
            RoundType.OPENING,
            RoundType.REBUTTAL,
            RoundType.CROSS_EXAMINATION,
            RoundType.CLOSING,
        ],
        cross_examination_enabled=True,
        allow_rebuilding=True,
        elimination_enabled=False,
    )


def LincolnDouglasFormat() -> DebateFormat:
    """
    Lincoln-Douglas debate format.

    One-on-one format focused on values and philosophical reasoning:
    - Affirmative vs Negative
    - Constructive speeches followed by rebuttals
    - Cross-examination periods
    - Emphasis on value premises and criteria

    Named after the famous Lincoln-Douglas debates of 1858.
    """
    return DebateFormat(
        name="Lincoln-Douglas",
        rounds=5,
        time_per_round=360,  # 6 minutes average
        position_count=2,
        round_types=[
            RoundType.OPENING,       # Affirmative constructive
            RoundType.CROSS_EXAMINATION,
            RoundType.OPENING,       # Negative constructive
            RoundType.REBUTTAL,      # Rebuttals
            RoundType.CLOSING,       # Final focus
        ],
        cross_examination_enabled=True,
        allow_rebuilding=True,
        elimination_enabled=False,
    )


def SocraticFormat() -> DebateFormat:
    """
    Socratic questioning-based format.

    Dialogue-style format emphasizing questions over assertions:
    - One position leads with claims
    - Others probe through questions
    - Truth emerges through dialectic
    - Less adversarial, more exploratory

    Based on the Socratic method of philosophical inquiry.
    """
    return DebateFormat(
        name="Socratic",
        rounds=6,
        time_per_round=180,  # 3 minutes - shorter, more exchanges
        position_count=3,  # Includes a questioner role
        round_types=[
            RoundType.OPENING,
            RoundType.CROSS_EXAMINATION,
            RoundType.CROSS_EXAMINATION,
            RoundType.REBUTTAL,
            RoundType.CROSS_EXAMINATION,
            RoundType.CLOSING,
        ],
        cross_examination_enabled=True,
        allow_rebuilding=True,
        elimination_enabled=False,
    )


def RoundRobinFormat() -> DebateFormat:
    """
    Round-robin format where everyone debates everyone.

    Multi-position format:
    - Multiple positions (typically 4+)
    - Each position argues against all others
    - Points accumulated across matchups
    - Good for exploring multiple perspectives

    Best for complex issues with many valid viewpoints.
    """
    return DebateFormat(
        name="RoundRobin",
        rounds=8,
        time_per_round=240,  # 4 minutes
        position_count=4,
        round_types=[
            RoundType.OPENING,
            RoundType.REBUTTAL,
            RoundType.CROSS_EXAMINATION,
            RoundType.REBUTTAL,
            RoundType.CROSS_EXAMINATION,
            RoundType.REBUTTAL,
            RoundType.CROSS_EXAMINATION,
            RoundType.CLOSING,
        ],
        cross_examination_enabled=True,
        allow_rebuilding=True,
        elimination_enabled=True,  # Weak positions can be eliminated
        elimination_threshold=0.25,
    )


def PolicyFormat() -> DebateFormat:
    """
    Policy debate format.

    Evidence-heavy format for policy proposals:
    - Affirmative presents plan
    - Negative offers counter-plan or criticisms
    - Heavy emphasis on evidence cards
    - Technical and fast-paced

    Common in high school and college competitive debate.
    """
    return DebateFormat(
        name="Policy",
        rounds=8,
        time_per_round=480,  # 8 minutes - longer for evidence
        position_count=2,
        round_types=[
            RoundType.OPENING,       # 1AC - First Affirmative Constructive
            RoundType.CROSS_EXAMINATION,
            RoundType.OPENING,       # 1NC - First Negative Constructive
            RoundType.CROSS_EXAMINATION,
            RoundType.REBUTTAL,      # 2AC
            RoundType.CROSS_EXAMINATION,
            RoundType.REBUTTAL,      # 2NC
            RoundType.CLOSING,       # Final rebuttals
        ],
        cross_examination_enabled=True,
        allow_rebuilding=True,
        elimination_enabled=False,
    )


def ParliamentaryFormat() -> DebateFormat:
    """
    Parliamentary debate format.

    British Parliament style:
    - Government vs Opposition
    - Points of Information allowed
    - More rhetorical, less evidence-focused
    - Emphasis on wit and argumentation

    Modeled after House of Commons debates.
    """
    return DebateFormat(
        name="Parliamentary",
        rounds=4,
        time_per_round=420,  # 7 minutes
        position_count=4,  # PM, LO, DPM, DLO
        round_types=[
            RoundType.OPENING,
            RoundType.REBUTTAL,
            RoundType.REBUTTAL,
            RoundType.CLOSING,
        ],
        cross_examination_enabled=False,  # POIs instead
        allow_rebuilding=True,
        elimination_enabled=False,
    )


def QuickFormat() -> DebateFormat:
    """
    Quick debate format for rapid exploration.

    Simplified format for fast debates:
    - Two positions
    - Minimal rounds
    - No cross-examination
    - Good for quick topic exploration
    """
    return DebateFormat(
        name="Quick",
        rounds=2,
        time_per_round=120,  # 2 minutes
        position_count=2,
        round_types=[
            RoundType.OPENING,
            RoundType.CLOSING,
        ],
        cross_examination_enabled=False,
        allow_rebuilding=False,
        elimination_enabled=False,
    )


def ExtendedFormat() -> DebateFormat:
    """
    Extended debate format for deep exploration.

    Thorough format for complex topics:
    - Multiple rounds of each type
    - Extended time limits
    - Full cross-examination
    - Elimination of weak positions
    """
    return DebateFormat(
        name="Extended",
        rounds=10,
        time_per_round=600,  # 10 minutes
        position_count=4,
        round_types=[
            RoundType.OPENING,
            RoundType.OPENING,
            RoundType.CROSS_EXAMINATION,
            RoundType.REBUTTAL,
            RoundType.CROSS_EXAMINATION,
            RoundType.REBUTTAL,
            RoundType.CROSS_EXAMINATION,
            RoundType.REBUTTAL,
            RoundType.CROSS_EXAMINATION,
            RoundType.CLOSING,
        ],
        cross_examination_enabled=True,
        allow_rebuilding=True,
        elimination_enabled=True,
        elimination_threshold=0.3,
    )


# =============================================================================
# Custom Format Builder
# =============================================================================


@dataclass
class FormatBuilder:
    """
    Builder for creating custom debate formats.

    Example:
        format = (FormatBuilder("MyFormat")
            .with_positions(3)
            .add_round(RoundType.OPENING)
            .add_round(RoundType.REBUTTAL)
            .add_round(RoundType.CLOSING)
            .with_time_limit(300)
            .enable_cross_examination()
            .enable_elimination(threshold=0.25)
            .build())
    """

    name: str
    _position_count: int = 2
    _round_types: list[RoundType] = field(default_factory=list)
    _time_per_round: int = 0
    _cross_examination: bool = False
    _allow_rebuilding: bool = True
    _elimination: bool = False
    _elimination_threshold: float = 0.3

    def with_positions(self, count: int) -> "FormatBuilder":
        """Set the number of positions."""
        self._position_count = count
        return self

    def add_round(self, round_type: RoundType) -> "FormatBuilder":
        """Add a round to the format."""
        self._round_types.append(round_type)
        return self

    def add_rounds(self, *round_types: RoundType) -> "FormatBuilder":
        """Add multiple rounds to the format."""
        self._round_types.extend(round_types)
        return self

    def with_time_limit(self, seconds: int) -> "FormatBuilder":
        """Set time limit per round in seconds."""
        self._time_per_round = seconds
        return self

    def enable_cross_examination(self) -> "FormatBuilder":
        """Enable cross-examination."""
        self._cross_examination = True
        return self

    def disable_cross_examination(self) -> "FormatBuilder":
        """Disable cross-examination."""
        self._cross_examination = False
        return self

    def enable_rebuilding(self) -> "FormatBuilder":
        """Allow arguments to be strengthened after attack."""
        self._allow_rebuilding = True
        return self

    def disable_rebuilding(self) -> "FormatBuilder":
        """Prevent argument rebuilding."""
        self._allow_rebuilding = False
        return self

    def enable_elimination(self, threshold: float = 0.3) -> "FormatBuilder":
        """Enable position elimination."""
        self._elimination = True
        self._elimination_threshold = threshold
        return self

    def disable_elimination(self) -> "FormatBuilder":
        """Disable position elimination."""
        self._elimination = False
        return self

    def build(self) -> DebateFormat:
        """Build and return the debate format."""
        if not self._round_types:
            # Default to basic format
            self._round_types = [
                RoundType.OPENING,
                RoundType.REBUTTAL,
                RoundType.CLOSING,
            ]

        return DebateFormat(
            name=self.name,
            rounds=len(self._round_types),
            time_per_round=self._time_per_round,
            position_count=self._position_count,
            round_types=self._round_types,
            cross_examination_enabled=self._cross_examination,
            allow_rebuilding=self._allow_rebuilding,
            elimination_enabled=self._elimination,
            elimination_threshold=self._elimination_threshold,
        )


# =============================================================================
# Format Utilities
# =============================================================================


def get_format_by_name(name: str) -> Optional[DebateFormat]:
    """
    Get a pre-built format by name.

    Args:
        name: Name of the format (case-insensitive)

    Returns:
        DebateFormat or None if not found
    """
    formats = {
        "oxford": OxfordFormat,
        "lincoln-douglas": LincolnDouglasFormat,
        "ld": LincolnDouglasFormat,
        "socratic": SocraticFormat,
        "roundrobin": RoundRobinFormat,
        "round-robin": RoundRobinFormat,
        "policy": PolicyFormat,
        "parliamentary": ParliamentaryFormat,
        "parli": ParliamentaryFormat,
        "quick": QuickFormat,
        "extended": ExtendedFormat,
    }

    factory = formats.get(name.lower())
    return factory() if factory else None


def list_formats() -> list[str]:
    """List all available pre-built format names."""
    return [
        "Oxford",
        "Lincoln-Douglas",
        "Socratic",
        "RoundRobin",
        "Policy",
        "Parliamentary",
        "Quick",
        "Extended",
    ]


def describe_format(format: DebateFormat) -> str:
    """
    Get a human-readable description of a format.

    Args:
        format: The format to describe

    Returns:
        Formatted description string
    """
    lines = [
        f"Format: {format.name}",
        f"Positions: {format.position_count}",
        f"Rounds: {format.rounds}",
    ]

    if format.time_per_round:
        minutes = format.time_per_round // 60
        seconds = format.time_per_round % 60
        time_str = f"{minutes}m{seconds}s" if seconds else f"{minutes}m"
        lines.append(f"Time per round: {time_str}")

    lines.append(f"Round sequence: {' -> '.join(r.value for r in format.round_types)}")

    features = []
    if format.cross_examination_enabled:
        features.append("cross-examination")
    if format.allow_rebuilding:
        features.append("argument rebuilding")
    if format.elimination_enabled:
        features.append(f"elimination (threshold: {format.elimination_threshold})")

    if features:
        lines.append(f"Features: {', '.join(features)}")

    return "\n".join(lines)


def estimate_debate_duration(format: DebateFormat) -> int:
    """
    Estimate total debate duration in seconds.

    Args:
        format: The debate format

    Returns:
        Estimated duration in seconds
    """
    if format.time_per_round:
        return format.rounds * format.time_per_round
    else:
        # Estimate based on typical times
        round_estimates = {
            RoundType.OPENING: 300,
            RoundType.REBUTTAL: 240,
            RoundType.CROSS_EXAMINATION: 180,
            RoundType.CLOSING: 180,
        }

        total = 0
        for round_type in format.round_types:
            total += round_estimates.get(round_type, 240)

        return total


def validate_format(format: DebateFormat) -> tuple[bool, list[str]]:
    """
    Validate that a format is well-formed.

    Args:
        format: The format to validate

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues: list[str] = []

    if format.rounds < 1:
        issues.append("Format must have at least 1 round")

    if format.position_count < 1:
        issues.append("Format must have at least 1 position")

    if format.position_count < 2 and format.cross_examination_enabled:
        issues.append("Cross-examination requires at least 2 positions")

    if len(format.round_types) != format.rounds:
        issues.append(f"Round count ({format.rounds}) doesn't match round_types length ({len(format.round_types)})")

    if format.elimination_enabled and format.elimination_threshold <= 0:
        issues.append("Elimination threshold must be positive")

    if format.elimination_enabled and format.elimination_threshold >= 1:
        issues.append("Elimination threshold must be less than 1")

    if format.time_per_round < 0:
        issues.append("Time per round cannot be negative")

    # Check round type sequence makes sense
    if format.round_types:
        if format.round_types[0] == RoundType.CLOSING:
            issues.append("Format should not start with CLOSING round")
        if format.round_types[-1] == RoundType.OPENING:
            issues.append("Format should not end with OPENING round")

    return len(issues) == 0, issues
