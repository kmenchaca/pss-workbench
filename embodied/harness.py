"""
Main embodied harness for MCTS-style exploration.

Coordinates environment interaction, branching exploration,
failure sharing, and action synthesis.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional
from enum import Enum
import asyncio

from .environment import Environment
from .types import (
    Action,
    ActionResult,
    ActionSequence,
    ExplorationConfig,
    Goal,
    Observation,
)
from .exploration import (
    ExplorationBranch,
    ExplorationTree,
    BranchStatus,
    evaluate_branch,
)
from .failure import FailureMemory, propagate_failure
from .planning import plan_action, replan_on_failure
from .execution import (
    ActionExecutor,
    ExecutionConstraints,
    safe_execute,
)
from .synthesis import (
    synthesize_plan,
    verify_sequence,
    SequenceSynthesizer,
)
from .rewards import RewardFunction, CompositeReward
from .observation import observation_to_text, diff_observations


class GateType(Enum):
    """Types of exploration gates."""
    PROGRESS = "progress"  # Check if making progress
    LOOP = "loop"  # Check for loops
    STUCK = "stuck"  # Check if stuck
    BUDGET = "budget"  # Token/step budget


@dataclass
class GateResult:
    """Result of a gate check."""
    gate_type: GateType
    passed: bool
    message: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class HarnessConfig:
    """Configuration for the embodied harness.

    Attributes:
        max_branches: Maximum parallel branches
        max_depth: Maximum steps per branch
        branch_on_uncertainty: Whether to branch when uncertain
        failure_propagation: Whether to share failures across branches
        gate_interval: Steps between gate checks
        budget_steps: Total step budget
        budget_reward: Minimum cumulative reward before termination
    """
    max_branches: int = 4
    max_depth: int = 50
    branch_on_uncertainty: bool = True
    failure_propagation: bool = True
    gate_interval: int = 5
    budget_steps: int = 200
    budget_reward: float = -10.0


@dataclass
class ExplorationResult:
    """Result of embodied exploration.

    Attributes:
        success: Whether goal was achieved
        best_sequence: Best action sequence found
        total_branches: Number of branches explored
        total_steps: Total steps taken
        total_reward: Cumulative reward
        goal_achieved: Whether goal was reached
        metadata: Additional result metadata
    """
    success: bool
    best_sequence: ActionSequence
    total_branches: int
    total_steps: int
    total_reward: float
    goal_achieved: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class EmbodiedHarness:
    """Main harness for embodied MCTS exploration.

    Coordinates:
    - Environment interaction
    - Branch spawning and management
    - Failure sharing across branches
    - Gate checks for progress/stuck detection
    - Action synthesis from successful branches

    Usage:
        env = MockEnvironment()
        goal = Goal("Reach position (4, 4)")
        harness = EmbodiedHarness(env, goal)
        result = harness.run()
    """

    def __init__(
        self,
        env: Environment,
        goal: Goal,
        config: Optional[HarnessConfig] = None,
        reward_fn: Optional[RewardFunction] = None,
        llm_call: Optional[Callable[[str, str], str]] = None,
    ):
        self.env = env
        self.goal = goal
        self.config = config or HarnessConfig()
        self.reward_fn = reward_fn
        self.llm_call = llm_call

        # Initialize components
        self.tree = ExplorationTree(ExplorationConfig(
            max_branches=self.config.max_branches,
            max_depth=self.config.max_depth,
            branch_on_uncertainty=self.config.branch_on_uncertainty,
        ))
        self.failure_memory = FailureMemory()
        self.synthesizer = SequenceSynthesizer()

        # Per-branch failure memories for propagation
        self._branch_memories: dict[str, FailureMemory] = {}

        # Execution tracking
        self._total_steps = 0
        self._gate_checks = 0

    def run(self) -> ExplorationResult:
        """Run embodied exploration.

        Main exploration loop that:
        1. Spawns initial branch
        2. Explores with branching on uncertainty
        3. Applies gate checks
        4. Propagates failures
        5. Synthesizes best action sequence

        Returns:
            ExplorationResult with best sequence and metrics
        """
        # Reset environment and create root branch
        self.env.reset()
        root = self.tree.create_root(self.env)
        self._branch_memories[root.id] = FailureMemory()

        # Initial observation
        initial_obs = self.env.observe()

        # Main exploration loop
        active_branches = [root]
        steps_since_gate = 0

        while active_branches and self._total_steps < self.config.budget_steps:
            # Process each active branch
            next_active = []

            for branch in active_branches:
                if not branch.is_active:
                    continue

                # Restore branch state
                self.env.restore(branch.checkpoint)

                # Get current observation
                obs = self.env.observe()

                # Check goal
                if self.goal.is_achieved(obs):
                    branch.status = BranchStatus.SUCCESS
                    continue

                # Plan next action
                history = list(zip(branch.actions, branch.results))
                action = plan_action(
                    obs,
                    self.goal,
                    history,
                    self.env.action_space,
                    self.llm_call,
                )

                # Check failure memory
                branch_memory = self._branch_memories.get(branch.id, self.failure_memory)
                should_avoid, reason = branch_memory.should_avoid(action, obs)

                if should_avoid:
                    # Try to replan
                    action = replan_on_failure(
                        ActionResult(action.id, False, obs, -0.1, reason),
                        history,
                        obs,
                        self.goal,
                        self.env.action_space,
                        self.llm_call,
                    )

                # Execute action
                result = safe_execute(
                    self.env,
                    action,
                    ExecutionConstraints(max_retries=1),
                )

                self._total_steps += 1
                steps_since_gate += 1

                # Get new observation
                new_obs = self.env.observe()

                # Apply custom reward if provided
                if self.reward_fn:
                    custom_reward = self.reward_fn(obs, action, new_obs)
                    result = ActionResult(
                        result.action_id,
                        result.success,
                        result.new_observation,
                        result.reward + custom_reward,
                        result.error,
                    )

                # Record step
                new_checkpoint = self.env.checkpoint()
                branch.checkpoint = new_checkpoint
                self.tree.record_step(branch, action, result, new_obs)

                # Handle failure
                if not result.success:
                    branch_memory.record_failure(action, obs, result.error or "Unknown")

                    # Propagate failures
                    if self.config.failure_propagation:
                        other_memories = [
                            m for bid, m in self._branch_memories.items()
                            if bid != branch.id
                        ]
                        propagate_failure(branch_memory, other_memories)

                # Check for branching opportunity
                if self.config.branch_on_uncertainty and len(active_branches) < self.config.max_branches:
                    if self._should_branch(branch, new_obs):
                        new_branches = self._spawn_branches(branch, new_obs)
                        for nb in new_branches:
                            self._branch_memories[nb.id] = FailureMemory()
                        next_active.extend(new_branches)

                # Gate check
                if steps_since_gate >= self.config.gate_interval:
                    gate_result = self._check_gates(branch)
                    self._gate_checks += 1
                    steps_since_gate = 0

                    if not gate_result.passed:
                        if gate_result.gate_type == GateType.STUCK:
                            branch.status = BranchStatus.STUCK
                            continue
                        elif gate_result.gate_type == GateType.LOOP:
                            branch.status = BranchStatus.STUCK
                            branch.metadata["stuck_reason"] = "loop"
                            continue

                # Check depth limit
                if branch.length >= self.config.max_depth:
                    branch.status = BranchStatus.TERMINATED
                    continue

                # Branch still active
                if branch.is_active:
                    next_active.append(branch)

            active_branches = next_active

            # Prune stuck branches
            self.tree.prune()

        # Synthesize best sequence
        all_branches = list(self.tree.branches.values())
        self.synthesizer.learn_from_branches(all_branches)
        best_sequence = self.synthesizer.synthesize(all_branches, self.goal)

        # Verify sequence
        success, verified_sequence = verify_sequence(self.env, best_sequence, self.goal)

        # Find best branch
        best_branch = self.tree.get_best_branch(self.goal)
        goal_achieved = any(
            b.status == BranchStatus.SUCCESS
            for b in self.tree.branches.values()
        )

        return ExplorationResult(
            success=success or goal_achieved,
            best_sequence=verified_sequence if success else best_sequence,
            total_branches=len(self.tree.branches),
            total_steps=self._total_steps,
            total_reward=best_branch.total_reward if best_branch else 0.0,
            goal_achieved=goal_achieved,
            metadata={
                "gate_checks": self._gate_checks,
                "states_visited": self.tree.coverage.coverage,
                "patterns_learned": len(self.synthesizer.patterns),
                "failures_recorded": self.failure_memory.total_failures,
            },
        )

    def _should_branch(self, branch: ExplorationBranch, obs: Observation) -> bool:
        """Decide whether to branch at current state.

        Args:
            branch: Current branch
            obs: Current observation

        Returns:
            True if should branch
        """
        # Don't branch too often
        if branch.length < 3:
            return False

        # Branch at unexplored states
        visit_count = self.tree.coverage.get_visit_count(obs)
        if visit_count == 1:
            return True

        # Branch after failures
        if branch.results and not branch.results[-1].success:
            return True

        # Branch periodically
        if branch.length % 10 == 0:
            return True

        return False

    def _spawn_branches(
        self,
        parent: ExplorationBranch,
        obs: Observation,
    ) -> list[ExplorationBranch]:
        """Create new branches from current state.

        Args:
            parent: Parent branch
            obs: Current observation

        Returns:
            List of new branches
        """
        # Generate action options
        action_types = self.env.get_available_actions()
        options = []

        for action_type in action_types[:self.config.max_branches]:
            # Generate actions with different parameters
            if action_type == "move":
                for direction in ["north", "south", "east", "west"]:
                    options.append(Action(
                        action_type=action_type,
                        parameters={"direction": direction},
                    ))
            elif action_type == "go":
                exits = obs.state.get("exits", [])
                for direction in exits:
                    options.append(Action(
                        action_type=action_type,
                        parameters={"direction": direction},
                    ))
            else:
                options.append(Action(action_type=action_type))

        # Limit options
        options = options[:self.config.max_branches]

        if not options:
            return []

        return self.tree.branch(self.env, parent, options)

    def _check_gates(self, branch: ExplorationBranch) -> GateResult:
        """Check exploration gates for a branch.

        Args:
            branch: Branch to check

        Returns:
            GateResult indicating pass/fail
        """
        # Progress gate: is reward improving?
        if len(branch.results) >= 5:
            recent_rewards = [r.reward for r in branch.results[-5:]]
            older_rewards = [r.reward for r in branch.results[-10:-5]] if len(branch.results) >= 10 else []

            if older_rewards:
                recent_avg = sum(recent_rewards) / len(recent_rewards)
                older_avg = sum(older_rewards) / len(older_rewards)

                if recent_avg < older_avg - 0.1:
                    return GateResult(
                        gate_type=GateType.PROGRESS,
                        passed=False,
                        message="Not making progress (reward declining)",
                        metrics={"recent_avg": recent_avg, "older_avg": older_avg},
                    )

        # Loop gate: check for repeated states
        if len(branch.observations) >= 6:
            from .exploration import detect_loop
            if detect_loop(branch, window=3):
                return GateResult(
                    gate_type=GateType.LOOP,
                    passed=False,
                    message="Loop detected",
                )

        # Stuck gate: consecutive failures
        if len(branch.results) >= 3:
            recent_failures = sum(1 for r in branch.results[-3:] if not r.success)
            if recent_failures >= 3:
                return GateResult(
                    gate_type=GateType.STUCK,
                    passed=False,
                    message="Stuck (consecutive failures)",
                    metrics={"failures": recent_failures},
                )

        # Budget gate: check cumulative reward
        if branch.total_reward < self.config.budget_reward:
            return GateResult(
                gate_type=GateType.BUDGET,
                passed=False,
                message="Below reward budget",
                metrics={"total_reward": branch.total_reward},
            )

        return GateResult(
            gate_type=GateType.PROGRESS,
            passed=True,
            message="All gates passed",
        )

    def get_exploration_summary(self) -> dict[str, Any]:
        """Get summary of exploration state.

        Returns:
            Dict with exploration metrics
        """
        return {
            "total_branches": len(self.tree.branches),
            "active_branches": len(self.tree.get_active_branches()),
            "total_steps": self._total_steps,
            "states_visited": self.tree.coverage.coverage,
            "failures_recorded": self.failure_memory.total_failures,
            "patterns_learned": len(self.synthesizer.patterns),
        }


async def run_embodied_async(
    env: Environment,
    goal: Goal,
    config: Optional[HarnessConfig] = None,
    llm_call: Optional[Callable[[str, str], str]] = None,
) -> ExplorationResult:
    """Async wrapper for embodied exploration.

    Args:
        env: Environment to explore
        goal: Goal to achieve
        config: Harness configuration
        llm_call: LLM function for planning

    Returns:
        ExplorationResult
    """
    harness = EmbodiedHarness(env, goal, config, llm_call=llm_call)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, harness.run)
    return result
