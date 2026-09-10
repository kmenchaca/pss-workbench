"""
Branching exploration for embodied agents.

MCTS-style exploration with state branching and parallel evaluation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional
import uuid
from enum import Enum

from .environment import Environment
from .types import (
    Action,
    ActionResult,
    ActionSequence,
    EnvironmentState,
    ExplorationConfig,
    Goal,
    Observation,
)
from .observation import observation_hash, states_equivalent


class BranchStatus(Enum):
    """Status of an exploration branch."""
    ACTIVE = "active"
    STUCK = "stuck"
    TERMINATED = "terminated"
    SUCCESS = "success"


@dataclass
class ExplorationBranch:
    """One line of exploration.

    Tracks a sequence of actions and their outcomes,
    along with the branch state and metrics.

    Attributes:
        id: Unique branch identifier
        parent_id: ID of parent branch (if any)
        checkpoint: Environment state at branch point
        actions: Sequence of actions taken
        observations: Observations at each step
        results: Results of each action
        total_reward: Cumulative reward
        status: Current branch status
        depth: Depth from root
        created_at: When branch was created
    """
    checkpoint: EnvironmentState
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: Optional[str] = None
    actions: list[Action] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    results: list[ActionResult] = field(default_factory=list)
    total_reward: float = 0.0
    status: BranchStatus = BranchStatus.ACTIVE
    depth: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(
        self,
        action: Action,
        result: ActionResult,
        observation: Observation,
    ) -> None:
        """Record a step in this branch."""
        self.actions.append(action)
        self.results.append(result)
        self.observations.append(observation)
        self.total_reward += result.reward

    def to_sequence(self) -> ActionSequence:
        """Convert branch to action sequence."""
        return ActionSequence(
            actions=self.actions.copy(),
            total_reward=self.total_reward,
            terminal=self.status in [BranchStatus.SUCCESS, BranchStatus.TERMINATED],
        )

    @property
    def is_active(self) -> bool:
        """Check if branch is still active."""
        return self.status == BranchStatus.ACTIVE

    @property
    def length(self) -> int:
        """Number of actions in branch."""
        return len(self.actions)


@dataclass
class StateSpaceCoverage:
    """Tracks coverage of state space during exploration.

    Attributes:
        visited_states: Set of state hashes visited
        state_visits: Count of visits per state
        transitions: Map of (state, action) -> next_state
    """
    visited_states: set[str] = field(default_factory=set)
    state_visits: dict[str, int] = field(default_factory=dict)
    transitions: dict[tuple[str, str], str] = field(default_factory=dict)

    def record_visit(self, obs: Observation) -> bool:
        """Record a state visit.

        Args:
            obs: Observation to record

        Returns:
            True if this is a new state
        """
        state_hash = observation_hash(obs)
        is_new = state_hash not in self.visited_states

        self.visited_states.add(state_hash)
        self.state_visits[state_hash] = self.state_visits.get(state_hash, 0) + 1

        return is_new

    def record_transition(
        self,
        from_obs: Observation,
        action: Action,
        to_obs: Observation,
    ) -> None:
        """Record a state transition."""
        from_hash = observation_hash(from_obs)
        to_hash = observation_hash(to_obs)
        action_key = f"{action.action_type}:{str(action.parameters)}"

        self.transitions[(from_hash, action_key)] = to_hash

    @property
    def coverage(self) -> int:
        """Number of unique states visited."""
        return len(self.visited_states)

    def get_visit_count(self, obs: Observation) -> int:
        """Get number of times a state was visited."""
        state_hash = observation_hash(obs)
        return self.state_visits.get(state_hash, 0)


def branch_at_choice(
    env: Environment,
    options: list[Action],
    current_branch: ExplorationBranch,
    max_branches: int = 4,
) -> list[ExplorationBranch]:
    """Create parallel branches at a decision point.

    Args:
        env: Environment to checkpoint
        options: Action options to branch on
        current_branch: Branch to spawn from
        max_branches: Maximum branches to create

    Returns:
        List of new branches (one per option, up to max_branches)
    """
    checkpoint = env.checkpoint()
    branches = []

    for i, action in enumerate(options[:max_branches]):
        branch = ExplorationBranch(
            checkpoint=checkpoint,
            parent_id=current_branch.id,
            actions=current_branch.actions.copy(),
            observations=current_branch.observations.copy(),
            results=current_branch.results.copy(),
            total_reward=current_branch.total_reward,
            depth=current_branch.depth + 1,
            metadata={
                "branch_action": str(action),
                "option_index": i,
            },
        )
        branches.append(branch)

    return branches


def evaluate_branch(
    branch: ExplorationBranch,
    goal: Optional[Goal] = None,
    coverage: Optional[StateSpaceCoverage] = None,
) -> float:
    """Score a branch based on progress and exploration.

    Args:
        branch: Branch to evaluate
        goal: Optional goal for goal-oriented scoring
        coverage: Optional coverage tracker for exploration bonus

    Returns:
        Branch score
    """
    score = 0.0

    # Base reward
    score += branch.total_reward

    # Length penalty (prefer shorter solutions)
    score -= branch.length * 0.01

    # Success bonus
    if branch.status == BranchStatus.SUCCESS:
        score += 10.0

    # Stuck penalty
    if branch.status == BranchStatus.STUCK:
        score -= 5.0

    # Goal progress
    if goal and branch.observations:
        last_obs = branch.observations[-1]
        if goal.is_achieved(last_obs):
            score += 20.0

    # Exploration bonus (UCB-style)
    if coverage and branch.observations:
        for obs in branch.observations:
            visit_count = coverage.get_visit_count(obs)
            if visit_count == 1:
                score += 0.5  # Bonus for new states
            elif visit_count > 3:
                score -= 0.1  # Penalty for revisiting

    return score


def detect_loop(
    branch: ExplorationBranch,
    window: int = 5,
) -> bool:
    """Check if branch is stuck in a loop.

    Args:
        branch: Branch to check
        window: Number of recent states to check

    Returns:
        True if loop detected
    """
    if len(branch.observations) < window * 2:
        return False

    recent = branch.observations[-window:]
    recent_hashes = [observation_hash(obs) for obs in recent]

    # Check for repeated state sequence
    for i in range(len(branch.observations) - window * 2, len(branch.observations) - window):
        older = branch.observations[i:i + window]
        older_hashes = [observation_hash(obs) for obs in older]

        if recent_hashes == older_hashes:
            return True

    # Check for single state repeat
    if len(set(recent_hashes)) == 1:
        return True

    return False


def prune_stuck_branches(
    branches: list[ExplorationBranch],
    loop_window: int = 5,
    failure_threshold: int = 3,
) -> list[ExplorationBranch]:
    """Remove branches that are stuck or in loops.

    Args:
        branches: Branches to prune
        loop_window: Window size for loop detection
        failure_threshold: Number of consecutive failures to mark as stuck

    Returns:
        Filtered list of active branches
    """
    active = []

    for branch in branches:
        if branch.status != BranchStatus.ACTIVE:
            # Keep non-active branches but don't process
            active.append(branch)
            continue

        # Check for loops
        if detect_loop(branch, loop_window):
            branch.status = BranchStatus.STUCK
            branch.metadata["stuck_reason"] = "loop_detected"
            active.append(branch)
            continue

        # Check for consecutive failures
        if branch.results:
            recent_failures = 0
            for result in reversed(branch.results):
                if not result.success:
                    recent_failures += 1
                else:
                    break

            if recent_failures >= failure_threshold:
                branch.status = BranchStatus.STUCK
                branch.metadata["stuck_reason"] = "consecutive_failures"
                active.append(branch)
                continue

        active.append(branch)

    return active


class ExplorationTree:
    """Manages a tree of exploration branches.

    Provides methods for creating, pruning, and selecting branches.
    """

    def __init__(self, config: Optional[ExplorationConfig] = None):
        self.config = config or ExplorationConfig()
        self.branches: dict[str, ExplorationBranch] = {}
        self.root_id: Optional[str] = None
        self.coverage = StateSpaceCoverage()

    def create_root(self, env: Environment) -> ExplorationBranch:
        """Create root branch from current environment state.

        Args:
            env: Environment to checkpoint

        Returns:
            Root branch
        """
        checkpoint = env.checkpoint()
        initial_obs = env.observe()

        root = ExplorationBranch(
            checkpoint=checkpoint,
            observations=[initial_obs],
        )

        self.branches[root.id] = root
        self.root_id = root.id
        self.coverage.record_visit(initial_obs)

        return root

    def branch(
        self,
        env: Environment,
        parent: ExplorationBranch,
        options: list[Action],
    ) -> list[ExplorationBranch]:
        """Create branches at a decision point.

        Args:
            env: Environment to checkpoint
            parent: Parent branch
            options: Action options

        Returns:
            New branches
        """
        new_branches = branch_at_choice(
            env,
            options,
            parent,
            max_branches=self.config.max_branches,
        )

        for branch in new_branches:
            self.branches[branch.id] = branch

        return new_branches

    def record_step(
        self,
        branch: ExplorationBranch,
        action: Action,
        result: ActionResult,
        observation: Observation,
    ) -> None:
        """Record a step in a branch.

        Args:
            branch: Branch to update
            action: Action taken
            result: Result of action
            observation: New observation
        """
        # Record state coverage
        prev_obs = branch.observations[-1] if branch.observations else None
        is_new_state = self.coverage.record_visit(observation)

        if prev_obs:
            self.coverage.record_transition(prev_obs, action, observation)

        # Update branch
        branch.add_step(action, result, observation)

        # Add exploration metadata
        if is_new_state:
            branch.metadata["new_states_found"] = \
                branch.metadata.get("new_states_found", 0) + 1

    def get_active_branches(self) -> list[ExplorationBranch]:
        """Get all active branches."""
        return [b for b in self.branches.values() if b.is_active]

    def get_best_branch(self, goal: Optional[Goal] = None) -> Optional[ExplorationBranch]:
        """Get the highest-scoring branch.

        Args:
            goal: Optional goal for scoring

        Returns:
            Best branch or None
        """
        if not self.branches:
            return None

        return max(
            self.branches.values(),
            key=lambda b: evaluate_branch(b, goal, self.coverage),
        )

    def prune(self) -> int:
        """Prune stuck branches.

        Returns:
            Number of branches pruned
        """
        before = len([b for b in self.branches.values() if b.is_active])

        for branch in self.branches.values():
            if branch.is_active:
                if detect_loop(branch, self.config.loop_detection_window):
                    branch.status = BranchStatus.STUCK

                # Check failure threshold
                if branch.results:
                    failures = sum(1 for r in branch.results[-self.config.failure_threshold:]
                                   if not r.success)
                    if failures >= self.config.failure_threshold:
                        branch.status = BranchStatus.STUCK

        after = len([b for b in self.branches.values() if b.is_active])
        return before - after

    def get_lineage(self, branch: ExplorationBranch) -> list[ExplorationBranch]:
        """Get ancestry of a branch back to root.

        Args:
            branch: Branch to get lineage for

        Returns:
            List from root to branch
        """
        lineage = [branch]
        current = branch

        while current.parent_id:
            parent = self.branches.get(current.parent_id)
            if parent:
                lineage.insert(0, parent)
                current = parent
            else:
                break

        return lineage

    def get_children(self, branch: ExplorationBranch) -> list[ExplorationBranch]:
        """Get direct children of a branch.

        Args:
            branch: Parent branch

        Returns:
            List of child branches
        """
        return [b for b in self.branches.values() if b.parent_id == branch.id]
