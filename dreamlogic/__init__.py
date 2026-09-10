"""
Dream Logic Generator

An experimental system that embraces incoherence and associative leaps.
Gates check interestingness not correctness. Synthesis amplifies the weird.

Key principles:
1. Incoherence as feature, not bug
2. Interestingness over correctness
3. Amplification over reconciliation
4. Human curation for guidance
"""

# Types
from .types import (
    DreamFragment,
    AssociativeLeap,
    InterestingnessScore,
    DreamCollage,
    Constraint,
    DreamStyle,
    DreamBranch,
    DreamTree,
    LeapType,
    ConstraintType,
)

# Associations
from .associations import (
    AssociationDB,
    Association,
    get_associations,
    add_association,
    surprising_association,
    chain_associations,
)

# Leaps
from .leaps import (
    generate_leap,
    score_leap,
    enforce_leap,
    random_constraint,
    chain_constraints as chain_leap_constraints,
    LeapContext,
    leap_types,
    get_leap_description,
)

# Interestingness
from .interestingness import (
    InterestingnessModel,
    score_interestingness,
    novelty_score,
    surprise_score,
    aesthetic_score,
    coherence_penalty,
    score_fragment,
    is_interesting_enough,
    get_default_model,
)

# Constraints
from .constraints import (
    random_constraint as get_random_constraint,
    chain_constraints,
    apply_constraint,
    constraint_to_prompt,
    constraints_to_prompt,
    check_constraint_satisfied,
    constraint_difficulty_level,
    get_random_by_difficulty,
    all_constraint_types,
    describe_constraint_type,
)

# Styles
from .styles import (
    get_style_description,
    get_style_keywords,
    get_style_phrases,
    apply_style,
    style_prompt,
    random_style,
    blend_styles,
    style_fragment,
    detect_style,
    style_intensity,
    all_styles,
)

# Boring detection
from .boring import (
    BoringDetector,
    BoringDetection,
    propagate_boring,
    avoid_patterns,
    boring_score_for_fragment,
    push_toward_strange,
    is_too_boring,
)

# Synthesis
from .synthesis import (
    amplify,
    find_resonances,
    collage,
    emergent_themes,
    embrace_contradictions,
    synthesis_prompt,
    create_collage,
)

# Harness
from .harness import (
    DreamHarness,
    DreamConfig,
    DreamState,
    GateResult,
    create_harness,
    dream,
)

# Prompts
from .prompts import (
    what_does_this_remind_you_of,
    what_is_the_opposite,
    if_this_were_a_myth,
    zoom_to_scale,
    time_shift,
    personify,
    style_specific_prompt,
    random_dream_prompt,
    chain_prompts,
    meta_prompt,
    constraint_prompt,
    all_prompt_types,
)

# Curation
from .curation import (
    CurationSession,
    CurationRating,
    CurationChoice,
    InteractiveCurator,
    create_curation_session,
    rate_fragment,
    select_favorites,
    learn_from_curation,
)

# Outputs
from .outputs import (
    to_story,
    to_image_prompt,
    to_game_concept,
    to_poem,
    to_raw,
    to_json_serializable,
    all_output_formats,
)


__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Types
    "DreamFragment",
    "AssociativeLeap",
    "InterestingnessScore",
    "DreamCollage",
    "Constraint",
    "DreamStyle",
    "DreamBranch",
    "DreamTree",
    "LeapType",
    "ConstraintType",
    # Associations
    "AssociationDB",
    "Association",
    "get_associations",
    "add_association",
    "surprising_association",
    "chain_associations",
    # Leaps
    "generate_leap",
    "score_leap",
    "enforce_leap",
    "random_constraint",
    "chain_leap_constraints",
    "LeapContext",
    "leap_types",
    "get_leap_description",
    # Interestingness
    "InterestingnessModel",
    "score_interestingness",
    "novelty_score",
    "surprise_score",
    "aesthetic_score",
    "coherence_penalty",
    "score_fragment",
    "is_interesting_enough",
    "get_default_model",
    # Constraints
    "get_random_constraint",
    "chain_constraints",
    "apply_constraint",
    "constraint_to_prompt",
    "constraints_to_prompt",
    "check_constraint_satisfied",
    "constraint_difficulty_level",
    "get_random_by_difficulty",
    "all_constraint_types",
    "describe_constraint_type",
    # Styles
    "get_style_description",
    "get_style_keywords",
    "get_style_phrases",
    "apply_style",
    "style_prompt",
    "random_style",
    "blend_styles",
    "style_fragment",
    "detect_style",
    "style_intensity",
    "all_styles",
    # Boring
    "BoringDetector",
    "BoringDetection",
    "propagate_boring",
    "avoid_patterns",
    "boring_score_for_fragment",
    "push_toward_strange",
    "is_too_boring",
    # Synthesis
    "amplify",
    "find_resonances",
    "collage",
    "emergent_themes",
    "embrace_contradictions",
    "synthesis_prompt",
    "create_collage",
    # Harness
    "DreamHarness",
    "DreamConfig",
    "DreamState",
    "GateResult",
    "create_harness",
    "dream",
    # Prompts
    "what_does_this_remind_you_of",
    "what_is_the_opposite",
    "if_this_were_a_myth",
    "zoom_to_scale",
    "time_shift",
    "personify",
    "style_specific_prompt",
    "random_dream_prompt",
    "chain_prompts",
    "meta_prompt",
    "constraint_prompt",
    "all_prompt_types",
    # Curation
    "CurationSession",
    "CurationRating",
    "CurationChoice",
    "InteractiveCurator",
    "create_curation_session",
    "rate_fragment",
    "select_favorites",
    "learn_from_curation",
    # Outputs
    "to_story",
    "to_image_prompt",
    "to_game_concept",
    "to_poem",
    "to_raw",
    "to_json_serializable",
    "all_output_formats",
]
