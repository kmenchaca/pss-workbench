"""
Argumentative Swarm System

A structured debate system where branches adopt positions and argue against each other.
Synthesis adjudicates the final outcome.

Example usage:

    from argswarm import DebateHarness, DebateConfig, OxfordFormat

    # Create a debate
    config = DebateConfig(format=OxfordFormat())
    harness = DebateHarness(
        proposition="Renewable energy should replace fossil fuels",
        config=config,
    )

    # Run the debate
    harness.start_debate()
    # ... submit arguments, refutations, responses ...
    judgment = harness.finish_debate()

    print(f"Winner: {judgment.winner}")
    print(f"Reasoning: {judgment.reasoning}")
"""

# Types
from .types import (
    AdjudicationStrategy,
    Argument,
    Crux,
    DebateFormat,
    DebateState,
    Judgment,
    Position,
    Refutation,
    Response,
    RoundType,
    Stance,
    StrengthScore,
)

# Position management
from .positions import (
    POSITION_DEFINITIONS,
    CustomPosition,
    assign_positions,
    balance_positions,
    create_position,
    get_allied_positions,
    get_opposing_positions,
    is_balanced,
    CONSEQUENTIALIST,
    PRAGMATIST,
    PRINCIPLED,
    SKEPTIC,
)

# Argument construction
from .arguments import (
    ArgumentBuilder,
    compare_arguments,
    construct_argument,
    create_argument_from_llm_response,
    evaluate_strength,
    find_weaknesses,
    parse_argument,
    strengthen_argument,
    validate_argument,
)

# Debate mechanics
from .debate import (
    CrossExaminationResult,
    DebateRound,
    RoundManager,
    calculate_position_score,
    cross_examine,
    get_arguments_against_position,
    get_position_arguments,
    get_unaddressed_refutations,
    rebuild,
    refute,
    should_eliminate_position,
)

# Transcript management
from .transcript import (
    DebateTranscript,
    TranscriptEntry,
    add_argument,
    add_note,
    add_refutation,
    add_response,
    create_transcript,
    export_to_json,
    export_to_text,
    get_addressed_points,
    get_bulletin_board_for_position,
    get_round_summary,
    get_unaddressed_arguments,
    import_from_json,
    post_to_bulletin_board,
)

# Adjudication
from .adjudication import (
    AdjudicationContext,
    adjudicate,
    find_common_ground,
    identify_cruxes,
    weigh_arguments,
)

# Main harness
from .harness import (
    DebateConfig,
    DebateHarness,
    GateCheck,
    run_simple_debate,
)

# Formats
from .formats import (
    ExtendedFormat,
    FormatBuilder,
    LincolnDouglasFormat,
    OxfordFormat,
    ParliamentaryFormat,
    PolicyFormat,
    QuickFormat,
    RoundRobinFormat,
    SocraticFormat,
    describe_format,
    estimate_debate_duration,
    get_format_by_name,
    list_formats,
    validate_format,
)

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Types
    "AdjudicationStrategy",
    "Argument",
    "Crux",
    "DebateFormat",
    "DebateState",
    "Judgment",
    "Position",
    "Refutation",
    "Response",
    "RoundType",
    "Stance",
    "StrengthScore",
    # Positions
    "POSITION_DEFINITIONS",
    "CustomPosition",
    "assign_positions",
    "balance_positions",
    "create_position",
    "get_allied_positions",
    "get_opposing_positions",
    "is_balanced",
    "CONSEQUENTIALIST",
    "PRAGMATIST",
    "PRINCIPLED",
    "SKEPTIC",
    # Arguments
    "ArgumentBuilder",
    "compare_arguments",
    "construct_argument",
    "create_argument_from_llm_response",
    "evaluate_strength",
    "find_weaknesses",
    "parse_argument",
    "strengthen_argument",
    "validate_argument",
    # Debate
    "CrossExaminationResult",
    "DebateRound",
    "RoundManager",
    "calculate_position_score",
    "cross_examine",
    "get_arguments_against_position",
    "get_position_arguments",
    "get_unaddressed_refutations",
    "rebuild",
    "refute",
    "should_eliminate_position",
    # Transcript
    "DebateTranscript",
    "TranscriptEntry",
    "add_argument",
    "add_note",
    "add_refutation",
    "add_response",
    "create_transcript",
    "export_to_json",
    "export_to_text",
    "get_addressed_points",
    "get_bulletin_board_for_position",
    "get_round_summary",
    "get_unaddressed_arguments",
    "import_from_json",
    "post_to_bulletin_board",
    # Adjudication
    "AdjudicationContext",
    "adjudicate",
    "find_common_ground",
    "identify_cruxes",
    "weigh_arguments",
    # Harness
    "DebateConfig",
    "DebateHarness",
    "GateCheck",
    "run_simple_debate",
    # Formats
    "ExtendedFormat",
    "FormatBuilder",
    "LincolnDouglasFormat",
    "OxfordFormat",
    "ParliamentaryFormat",
    "PolicyFormat",
    "QuickFormat",
    "RoundRobinFormat",
    "SocraticFormat",
    "describe_format",
    "estimate_debate_duration",
    "get_format_by_name",
    "list_formats",
    "validate_format",
]
