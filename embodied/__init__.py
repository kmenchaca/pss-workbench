"""
Embodied Action Planner.

MCTS-style branching exploration for embodied agents.
Branches act in environments and receive feedback.
Failures are shared across branches to avoid repeated mistakes.

Usage:
    from embodied import EmbodiedHarness, MockEnvironment, Goal

    env = MockEnvironment(start_pos=(0, 0), goal_pos=(4, 4))
    goal = Goal("Reach the goal position")

    harness = EmbodiedHarness(env, goal)
    result = harness.run()

    print(f"Success: {result.success}")
    print(f"Steps: {result.total_steps}")
    print(f"Best sequence: {result.best_sequence}")
"""

# Types
from .types import (
    Action,
    ActionResult,
    ActionSequence,
    EnvironmentState,
    ExplorationConfig,
    Goal,
    Observation,
)

# Environment
from .environment import (
    ActionSpace,
    Environment,
    EnvironmentWrapper,
    LoggingEnvironment,
    SafetyEnvironment,
)

# Concrete environments
from .environments import (
    MockEnvironment,
    TextGameEnvironment,
    BrowserEnvironment,
    APIEnvironment,
)

# Observation processing
from .observation import (
    process_observation,
    observation_to_text,
    diff_observations,
    extract_state_features,
    observation_hash,
    states_equivalent,
    summarize_observation,
)

# Planning
from .planning import (
    plan_action,
    action_from_text,
    explain_action,
    replan_on_failure,
)

# Execution
from .execution import (
    execute_action,
    retry_with_backoff,
    timeout_action,
    safe_execute,
    ActionExecutor,
    ExecutionConstraints,
    ExecutionStats,
    rollback_on_failure,
)

# Exploration
from .exploration import (
    ExplorationBranch,
    BranchStatus,
    StateSpaceCoverage,
    ExplorationTree,
    branch_at_choice,
    evaluate_branch,
    detect_loop,
    prune_stuck_branches,
)

# Failure handling
from .failure import (
    FailureRecord,
    FailureMemory,
    propagate_failure,
    FailureAwareActionFilter,
)

# Synthesis
from .synthesis import (
    ActionPattern,
    synthesize_plan,
    merge_sequences,
    verify_sequence,
    extract_reusable_patterns,
    SequenceSynthesizer,
)

# Rewards
from .rewards import (
    RewardFunction,
    GoalReward,
    ProgressReward,
    ExplorationReward,
    CompositeReward,
    ActionCostReward,
    SuccessReward,
    DistanceReward,
    TimeReward,
    create_goal_reward,
    create_exploration_reward,
)

# Main harness
from .harness import (
    EmbodiedHarness,
    HarnessConfig,
    ExplorationResult,
    GateType,
    GateResult,
    run_embodied_async,
)

__version__ = "0.1.0"

__all__ = [
    # Types
    "Action",
    "ActionResult",
    "ActionSequence",
    "EnvironmentState",
    "ExplorationConfig",
    "Goal",
    "Observation",
    # Environment
    "ActionSpace",
    "Environment",
    "EnvironmentWrapper",
    "LoggingEnvironment",
    "SafetyEnvironment",
    # Concrete environments
    "MockEnvironment",
    "TextGameEnvironment",
    "BrowserEnvironment",
    "APIEnvironment",
    # Observation
    "process_observation",
    "observation_to_text",
    "diff_observations",
    "extract_state_features",
    "observation_hash",
    "states_equivalent",
    "summarize_observation",
    # Planning
    "plan_action",
    "action_from_text",
    "explain_action",
    "replan_on_failure",
    # Execution
    "execute_action",
    "retry_with_backoff",
    "timeout_action",
    "safe_execute",
    "ActionExecutor",
    "ExecutionConstraints",
    "ExecutionStats",
    "rollback_on_failure",
    # Exploration
    "ExplorationBranch",
    "BranchStatus",
    "StateSpaceCoverage",
    "ExplorationTree",
    "branch_at_choice",
    "evaluate_branch",
    "detect_loop",
    "prune_stuck_branches",
    # Failure
    "FailureRecord",
    "FailureMemory",
    "propagate_failure",
    "FailureAwareActionFilter",
    # Synthesis
    "ActionPattern",
    "synthesize_plan",
    "merge_sequences",
    "verify_sequence",
    "extract_reusable_patterns",
    "SequenceSynthesizer",
    # Rewards
    "RewardFunction",
    "GoalReward",
    "ProgressReward",
    "ExplorationReward",
    "CompositeReward",
    "ActionCostReward",
    "SuccessReward",
    "DistanceReward",
    "TimeReward",
    "create_goal_reward",
    "create_exploration_reward",
    # Harness
    "EmbodiedHarness",
    "HarnessConfig",
    "ExplorationResult",
    "GateType",
    "GateResult",
    "run_embodied_async",
]
