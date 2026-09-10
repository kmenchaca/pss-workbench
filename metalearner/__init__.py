"""Self-Improving Meta-Learner for exploration strategy optimization.

This package provides a meta-learning system that learns which exploration
strategies work best for different types of problems and continuously
improves through experience.

Main Components:
    - types: Core data structures (Strategy, StrategyResult, etc.)
    - strategies: Built-in strategy definitions
    - classification: Problem classification
    - mutation: Strategy mutation operators
    - evaluation: Strategy evaluation and comparison
    - database: Performance tracking database
    - recommendation: Strategy recommendation engine
    - learning: Learning loop for continuous improvement
    - harness: Main orchestration harness
    - bootstrap: Self-improvement capabilities
    - antipatterns: Anti-pattern tracking

Quick Start:
    >>> from metalearner import MetaHarness, recommend_strategy
    >>>
    >>> # Get a strategy recommendation
    >>> result = recommend_strategy("How do I implement a binary search?")
    >>> print(f"Recommended: {result.strategy.name}")
    >>>
    >>> # Use the full harness
    >>> harness = MetaHarness()
    >>> harness.set_executor(my_llm_function)
    >>> result = await harness.solve("What is the capital of France?")
"""

# Types
from .types import (
    ClassificationResult,
    EvaluationResult,
    LearningState,
    MetaConfig,
    MutationType,
    PerformanceRecord,
    ProblemCategory,
    ProblemType,
    RecommendationResult,
    Strategy,
    StrategyMutation,
    StrategyResult,
)

# Strategies
from .strategies import (
    ADVERSARIAL,
    ANALOGICAL,
    BUILTIN_STRATEGIES,
    CHAIN_OF_THOUGHT,
    DECOMPOSITION,
    EXHAUSTIVE,
    MINIMAL,
    TOOL_HEAVY,
    StrategyApplicator,
    apply_strategy,
    compose_strategies,
    create_custom_strategy,
    get_builtin_strategy,
    list_builtin_strategies,
    load_strategies,
    load_strategy,
    save_strategies,
    save_strategy,
)

# Classification
from .classification import (
    ProblemClassifier,
    classify_problem,
    classify_with_llm,
    extract_features,
    get_problem_type,
    list_problem_types,
    register_pattern,
    register_problem_type,
)

# Mutation
from .mutation import (
    StrategyMutator,
    crossover_strategies,
    llm_guided_mutation,
    mutate_strategy,
    random_strategy,
)

# Evaluation
from .evaluation import (
    ABTest,
    StrategyEvaluator,
    calculate_pass_at_k,
    calculate_quality_adjusted_score,
    calculate_token_efficiency,
    compare_strategies,
)

# Database
from .database import (
    PerformanceDB,
    best_strategy_for,
    get_db,
    query_performance,
    record_result,
    strategy_rankings,
)

# Recommendation
from .recommendation import (
    StrategyRecommender,
    ThompsonSamplingRecommender,
    UCBRecommender,
    explain_recommendation,
    fallback_strategy,
    recommend_strategy,
)

# Learning
from .learning import (
    IncrementalLearner,
    LearningLoop,
    discover_strategy,
    meta_learn,
    retire_strategy,
    update_recommendations,
)

# Harness
from .harness import (
    BatchResult,
    MetaHarness,
    SimplifiedHarness,
    SolveResult,
    create_harness,
    run_harness_sync,
)

# Bootstrap
from .bootstrap import (
    BootstrapOptimizer,
    BootstrapResult,
    MetaMetaOptimizer,
    MetaMetaResult,
    bootstrap_optimization,
    optimize_meta_strategy,
)

# Anti-patterns
from .antipatterns import (
    AntiPattern,
    AntiPatternDB,
    get_antipattern_db,
    get_strategy_blacklist,
    is_antipattern,
    record_antipattern,
)


__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Types
    "Strategy",
    "StrategyResult",
    "ProblemType",
    "ProblemCategory",
    "PerformanceRecord",
    "StrategyMutation",
    "MutationType",
    "MetaConfig",
    "ClassificationResult",
    "RecommendationResult",
    "EvaluationResult",
    "LearningState",
    # Strategies
    "CHAIN_OF_THOUGHT",
    "DECOMPOSITION",
    "TOOL_HEAVY",
    "ANALOGICAL",
    "ADVERSARIAL",
    "MINIMAL",
    "EXHAUSTIVE",
    "BUILTIN_STRATEGIES",
    "StrategyApplicator",
    "apply_strategy",
    "get_builtin_strategy",
    "list_builtin_strategies",
    "save_strategy",
    "load_strategy",
    "save_strategies",
    "load_strategies",
    "create_custom_strategy",
    "compose_strategies",
    # Classification
    "ProblemClassifier",
    "classify_problem",
    "classify_with_llm",
    "extract_features",
    "get_problem_type",
    "list_problem_types",
    "register_problem_type",
    "register_pattern",
    # Mutation
    "StrategyMutator",
    "mutate_strategy",
    "crossover_strategies",
    "random_strategy",
    "llm_guided_mutation",
    # Evaluation
    "StrategyEvaluator",
    "compare_strategies",
    "ABTest",
    "calculate_pass_at_k",
    "calculate_token_efficiency",
    "calculate_quality_adjusted_score",
    # Database
    "PerformanceDB",
    "get_db",
    "record_result",
    "query_performance",
    "best_strategy_for",
    "strategy_rankings",
    # Recommendation
    "StrategyRecommender",
    "UCBRecommender",
    "ThompsonSamplingRecommender",
    "recommend_strategy",
    "explain_recommendation",
    "fallback_strategy",
    # Learning
    "LearningLoop",
    "IncrementalLearner",
    "meta_learn",
    "update_recommendations",
    "discover_strategy",
    "retire_strategy",
    # Harness
    "MetaHarness",
    "SimplifiedHarness",
    "SolveResult",
    "BatchResult",
    "create_harness",
    "run_harness_sync",
    # Bootstrap
    "BootstrapOptimizer",
    "MetaMetaOptimizer",
    "BootstrapResult",
    "MetaMetaResult",
    "bootstrap_optimization",
    "optimize_meta_strategy",
    # Anti-patterns
    "AntiPattern",
    "AntiPatternDB",
    "get_antipattern_db",
    "record_antipattern",
    "is_antipattern",
    "get_strategy_blacklist",
]
