"""Distributed Negotiation Simulator.

A multi-party negotiation modeling system that combines game theory,
strategy simulation, and transcription analysis.

Key concepts:
- Stakeholders: Parties with objectives, constraints, and private info
- Offers: Proposals between parties
- BATNA: Best Alternative To Negotiated Agreement
- ZOPA: Zone of Possible Agreement
- Protocols: Rules governing negotiation flow
- Strategies: How stakeholders make decisions

Example usage:
    from negotiate import (
        define_stakeholder,
        NegotiationHarness,
        NegotiationConfig,
        run_negotiation,
    )

    # Create stakeholders
    buyer = define_stakeholder(
        name="Buyer",
        objectives={"price": 0.0},  # Minimize price
        constraints={"price": (0, 100)},
    )

    seller = define_stakeholder(
        name="Seller",
        objectives={"price": float("inf")},  # Maximize price
        constraints={"price": (50, 150)},
    )

    # Run negotiation
    results = run_negotiation([buyer, seller])
    print(results["agreement"])
"""

# Types
from .types import (
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
)

# Stakeholders
from .stakeholders import (
    BUYER,
    MEDIATOR,
    REGULATOR,
    SELLER,
    TEMPLATES,
    check_constraints,
    clone_stakeholder,
    create_from_template,
    define_stakeholder,
    get_stakeholder_summary,
    hide_info,
    reveal_info,
    update_constraints,
    update_objectives,
)

# Utility functions
from .utility import (
    CompositeUtility,
    LinearUtility,
    RelativeUtility,
    RiskAdjustedUtility,
    ThresholdUtility,
    UtilityFunction,
    create_utility_from_objectives,
    evaluate_terms,
    utility_difference,
)

# Offers
from .offers import (
    accept,
    check_offer_expired,
    counter_offer,
    evaluate_offer,
    get_best_offer,
    get_pending_offers,
    improve_offer,
    make_offer,
    reject,
    withdraw,
)

# Protocols
from .protocol import (
    AlternatingOffers,
    AuctionProtocol,
    MediatedNegotiation,
    NegotiationProtocol,
    SimultaneousBids,
)

# Strategies
from .strategy import (
    AnchoringStrategy,
    BrinksmanshipStrategy,
    CompetitiveStrategy,
    CooperativeStrategy,
    NegotiationStrategy,
    STRATEGIES,
    TitForTatStrategy,
    get_strategy,
    select_strategy_for_stakeholder,
)

# BATNA
from .batna import (
    batna_bluff,
    calculate_walk_away_point,
    compare_to_batna,
    detect_batna_bluff,
    discover_batna,
    evaluate_batna,
    improve_batna,
)

# ZOPA
from .zopa import (
    expand_zopa,
    find_optimal_point,
    find_zopa,
    no_zopa_detection,
    visualize_zopa,
)

# Game Theory
from .equilibrium import (
    Game,
    analyze_game,
    build_game,
    compute_bargaining_solution,
    dominant_strategy,
    find_nash_equilibrium,
    minimax,
    pareto_frontier,
)

# Transcripts
from .transcript import (
    NegotiationTranscript,
    add_move,
    analyze_concessions,
    create_transcript,
    export_transcript,
    finalize_transcript,
    identify_turning_points,
    summarize_negotiation,
)

# Harness
from .harness import (
    NegotiationConfig,
    NegotiationHarness,
    create_simple_negotiation,
    run_negotiation,
)

# Multi-party
from .multiparty import (
    coalition_utility,
    core_membership,
    defection_analysis,
    find_stable_coalitions,
    form_coalition,
    multiparty_mediation,
    negotiate_internal_split,
    shapley_value,
)

# Training
from .training import (
    BASIC_SCENARIOS,
    TrainingScenario,
    TrainingSession,
    create_training_scenario,
    feedback,
    score_negotiation,
    start_training_session,
    submit_human_move,
    suggest_improvement,
)

__version__ = "1.0.0"

__all__ = [
    # Types
    "Agreement",
    "BATNA",
    "Coalition",
    "CounterOffer",
    "GameOutcome",
    "NegotiationMove",
    "NegotiationState",
    "NegotiationStatus",
    "Offer",
    "OfferStatus",
    "Stakeholder",
    "ZOPA",
    # Stakeholders
    "BUYER",
    "MEDIATOR",
    "REGULATOR",
    "SELLER",
    "TEMPLATES",
    "check_constraints",
    "clone_stakeholder",
    "create_from_template",
    "define_stakeholder",
    "get_stakeholder_summary",
    "hide_info",
    "reveal_info",
    "update_constraints",
    "update_objectives",
    # Utility
    "CompositeUtility",
    "LinearUtility",
    "RelativeUtility",
    "RiskAdjustedUtility",
    "ThresholdUtility",
    "UtilityFunction",
    "create_utility_from_objectives",
    "evaluate_terms",
    "utility_difference",
    # Offers
    "accept",
    "check_offer_expired",
    "counter_offer",
    "evaluate_offer",
    "get_best_offer",
    "get_pending_offers",
    "improve_offer",
    "make_offer",
    "reject",
    "withdraw",
    # Protocols
    "AlternatingOffers",
    "AuctionProtocol",
    "MediatedNegotiation",
    "NegotiationProtocol",
    "SimultaneousBids",
    # Strategies
    "AnchoringStrategy",
    "BrinksmanshipStrategy",
    "CompetitiveStrategy",
    "CooperativeStrategy",
    "NegotiationStrategy",
    "STRATEGIES",
    "TitForTatStrategy",
    "get_strategy",
    "select_strategy_for_stakeholder",
    # BATNA
    "batna_bluff",
    "calculate_walk_away_point",
    "compare_to_batna",
    "detect_batna_bluff",
    "discover_batna",
    "evaluate_batna",
    "improve_batna",
    # ZOPA
    "expand_zopa",
    "find_optimal_point",
    "find_zopa",
    "no_zopa_detection",
    "visualize_zopa",
    # Game Theory
    "Game",
    "analyze_game",
    "build_game",
    "compute_bargaining_solution",
    "dominant_strategy",
    "find_nash_equilibrium",
    "minimax",
    "pareto_frontier",
    # Transcripts
    "NegotiationTranscript",
    "add_move",
    "analyze_concessions",
    "create_transcript",
    "export_transcript",
    "finalize_transcript",
    "identify_turning_points",
    "summarize_negotiation",
    # Harness
    "NegotiationConfig",
    "NegotiationHarness",
    "create_simple_negotiation",
    "run_negotiation",
    # Multi-party
    "coalition_utility",
    "core_membership",
    "defection_analysis",
    "find_stable_coalitions",
    "form_coalition",
    "multiparty_mediation",
    "negotiate_internal_split",
    "shapley_value",
    # Training
    "BASIC_SCENARIOS",
    "TrainingScenario",
    "TrainingSession",
    "create_training_scenario",
    "feedback",
    "score_negotiation",
    "start_training_session",
    "submit_human_move",
    "suggest_improvement",
]
