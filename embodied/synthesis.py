"""
Action sequence synthesis from exploration branches.

Combines successful subsequences from multiple branches
to create optimized action plans.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from collections import defaultdict

from .environment import Environment
from .types import Action, ActionResult, ActionSequence, Goal, Observation
from .exploration import ExplorationBranch, BranchStatus, ExplorationTree
from .observation import observation_hash


@dataclass
class ActionPattern:
    """A reusable pattern of actions.

    Attributes:
        actions: Sequence of actions in the pattern
        start_state_features: Key features of starting state
        end_state_features: Key features of ending state
        success_rate: How often this pattern succeeds
        avg_reward: Average reward from this pattern
        occurrences: Number of times pattern was observed
    """
    actions: list[Action]
    start_state_features: dict[str, Any] = field(default_factory=dict)
    end_state_features: dict[str, Any] = field(default_factory=dict)
    success_rate: float = 1.0
    avg_reward: float = 0.0
    occurrences: int = 1

    def __len__(self) -> int:
        return len(self.actions)

    def matches_start(self, obs: Observation, tolerance: float = 0.8) -> bool:
        """Check if observation matches pattern's starting state.

        Args:
            obs: Observation to check
            tolerance: How strict the match should be (0-1)

        Returns:
            True if observation matches
        """
        if not self.start_state_features:
            return True

        matches = 0
        total = len(self.start_state_features)

        for key, value in self.start_state_features.items():
            if key in obs.state:
                if obs.state[key] == value:
                    matches += 1
                elif isinstance(value, (int, float)) and isinstance(obs.state[key], (int, float)):
                    # Allow numeric tolerance
                    if abs(obs.state[key] - value) / max(abs(value), 1) < 0.1:
                        matches += 1

        return (matches / total if total > 0 else 1.0) >= tolerance


def synthesize_plan(
    branches: list[ExplorationBranch],
    goal: Optional[Goal] = None,
) -> ActionSequence:
    """Combine successful subsequences from branches.

    Args:
        branches: Branches to synthesize from
        goal: Optional goal for evaluating success

    Returns:
        Synthesized action sequence
    """
    if not branches:
        return ActionSequence()

    # Find the best complete path
    best_branch = None
    best_score = float('-inf')

    for branch in branches:
        score = _score_branch(branch, goal)
        if score > best_score:
            best_score = score
            best_branch = branch

    if best_branch is None:
        return ActionSequence()

    # Start with best branch's sequence
    base_sequence = best_branch.to_sequence()

    # Try to improve with subsequences from other branches
    improved = _improve_with_subsequences(base_sequence, branches, goal)

    return improved


def _score_branch(branch: ExplorationBranch, goal: Optional[Goal] = None) -> float:
    """Score a branch for synthesis selection."""
    score = branch.total_reward

    # Bonus for successful branches
    if branch.status == BranchStatus.SUCCESS:
        score += 10.0

    # Penalty for stuck branches
    if branch.status == BranchStatus.STUCK:
        score -= 5.0

    # Goal completion bonus
    if goal and branch.observations:
        last_obs = branch.observations[-1]
        if goal.is_achieved(last_obs):
            score += 20.0

    # Length efficiency (prefer shorter)
    if branch.actions:
        efficiency = branch.total_reward / len(branch.actions)
        score += efficiency * 2

    return score


def _improve_with_subsequences(
    base: ActionSequence,
    branches: list[ExplorationBranch],
    goal: Optional[Goal] = None,
) -> ActionSequence:
    """Try to improve base sequence with subsequences from other branches.

    Args:
        base: Base sequence to improve
        branches: Branches to extract subsequences from
        goal: Optional goal for evaluation

    Returns:
        Improved sequence (or original if no improvement found)
    """
    # Extract all successful subsequences
    subsequences = []

    for branch in branches:
        if branch.status not in [BranchStatus.STUCK, BranchStatus.TERMINATED]:
            subseqs = _extract_successful_subsequences(branch)
            subsequences.extend(subseqs)

    if not subsequences:
        return base

    # Try inserting subsequences at various points
    best_sequence = base
    best_reward = base.total_reward

    for subseq in subsequences:
        # Try replacing sections of base with this subsequence
        for i in range(len(base.actions)):
            for j in range(i + 1, len(base.actions) + 1):
                candidate = ActionSequence(
                    actions=base.actions[:i] + subseq.actions + base.actions[j:],
                    total_reward=_estimate_reward(
                        base.actions[:i] + subseq.actions + base.actions[j:]
                    ),
                )

                if candidate.total_reward > best_reward:
                    best_reward = candidate.total_reward
                    best_sequence = candidate

    return best_sequence


def _extract_successful_subsequences(
    branch: ExplorationBranch,
    min_length: int = 2,
    max_length: int = 5,
) -> list[ActionSequence]:
    """Extract successful action subsequences from a branch.

    Args:
        branch: Branch to extract from
        min_length: Minimum subsequence length
        max_length: Maximum subsequence length

    Returns:
        List of successful subsequences
    """
    subsequences = []

    for length in range(min_length, min(max_length + 1, len(branch.actions) + 1)):
        for start in range(len(branch.actions) - length + 1):
            end = start + length

            # Check if all actions in this range succeeded
            all_success = all(
                result.success
                for result in branch.results[start:end]
            )

            if all_success:
                subseq_reward = sum(
                    result.reward
                    for result in branch.results[start:end]
                )

                subsequences.append(ActionSequence(
                    actions=branch.actions[start:end],
                    total_reward=subseq_reward,
                ))

    return subsequences


def _estimate_reward(actions: list[Action]) -> float:
    """Estimate reward for an action sequence.

    Simple heuristic - actual reward would require execution.
    """
    return len(actions) * -0.01  # Small negative for length


def merge_sequences(
    s1: ActionSequence,
    s2: ActionSequence,
    strategy: str = "interleave",
) -> ActionSequence:
    """Merge two action sequences.

    Args:
        s1: First sequence
        s2: Second sequence
        strategy: How to merge ("interleave", "concat", "best_first")

    Returns:
        Merged sequence
    """
    if strategy == "concat":
        return ActionSequence(
            actions=s1.actions + s2.actions,
            total_reward=s1.total_reward + s2.total_reward,
        )

    elif strategy == "interleave":
        merged = []
        i, j = 0, 0

        while i < len(s1.actions) and j < len(s2.actions):
            merged.append(s1.actions[i])
            i += 1
            if j < len(s2.actions):
                merged.append(s2.actions[j])
                j += 1

        # Add remaining
        merged.extend(s1.actions[i:])
        merged.extend(s2.actions[j:])

        return ActionSequence(
            actions=merged,
            total_reward=(s1.total_reward + s2.total_reward) / 2,
        )

    elif strategy == "best_first":
        if s1.total_reward >= s2.total_reward:
            return ActionSequence(
                actions=s1.actions + s2.actions,
                total_reward=s1.total_reward + s2.total_reward * 0.5,
            )
        else:
            return ActionSequence(
                actions=s2.actions + s1.actions,
                total_reward=s2.total_reward + s1.total_reward * 0.5,
            )

    else:
        raise ValueError(f"Unknown merge strategy: {strategy}")


def verify_sequence(
    env: Environment,
    sequence: ActionSequence,
    goal: Optional[Goal] = None,
) -> tuple[bool, ActionSequence]:
    """Test a synthesized sequence in the environment.

    Args:
        env: Environment to test in
        sequence: Sequence to verify
        goal: Optional goal to check

    Returns:
        Tuple of (success, executed_sequence)
    """
    # Reset and checkpoint
    env.reset()
    initial_state = env.checkpoint()

    executed_actions = []
    total_reward = 0.0
    success = True

    for action in sequence.actions:
        result = env.step(action)
        executed_actions.append(action)
        total_reward += result.reward

        if not result.success:
            success = False
            break

        # Check goal
        if goal and goal.is_achieved(result.new_observation):
            break

        # Check terminal
        if env.is_terminal():
            break

    # Create verified sequence
    verified = ActionSequence(
        actions=executed_actions,
        total_reward=total_reward,
        terminal=env.is_terminal(),
    )

    # Check goal achievement
    if goal:
        final_obs = env.observe()
        success = success and goal.is_achieved(final_obs)

    # Restore original state
    env.restore(initial_state)

    return success, verified


def extract_reusable_patterns(
    branches: list[ExplorationBranch],
    min_occurrences: int = 2,
    min_length: int = 2,
    max_length: int = 5,
) -> list[ActionPattern]:
    """Extract common successful action patterns from branches.

    Args:
        branches: Branches to analyze
        min_occurrences: Minimum times pattern must occur
        min_length: Minimum pattern length
        max_length: Maximum pattern length

    Returns:
        List of reusable patterns
    """
    # Count action type sequences
    sequence_counts: dict[tuple, list[ExplorationBranch]] = defaultdict(list)

    for branch in branches:
        if not branch.actions:
            continue

        for length in range(min_length, min(max_length + 1, len(branch.actions) + 1)):
            for start in range(len(branch.actions) - length + 1):
                end = start + length

                # Create sequence key (action types only)
                seq_key = tuple(
                    (a.action_type, tuple(sorted(a.parameters.items())))
                    for a in branch.actions[start:end]
                )

                sequence_counts[seq_key].append(branch)

    # Convert frequent sequences to patterns
    patterns = []

    for seq_key, occurrences in sequence_counts.items():
        if len(occurrences) >= min_occurrences:
            # Get representative actions
            representative = occurrences[0]
            start_idx = 0  # Would need better tracking

            # Calculate success rate
            success_count = sum(
                1 for b in occurrences
                if b.status == BranchStatus.SUCCESS
            )
            success_rate = success_count / len(occurrences)

            # Calculate average reward
            avg_reward = sum(b.total_reward for b in occurrences) / len(occurrences)

            # Extract state features if available
            start_features = {}
            end_features = {}

            if representative.observations:
                start_obs = representative.observations[0]
                start_features = {k: v for k, v in start_obs.state.items()
                                  if not isinstance(v, (list, dict))}

            if representative.observations and len(representative.observations) > 1:
                end_obs = representative.observations[-1]
                end_features = {k: v for k, v in end_obs.state.items()
                                if not isinstance(v, (list, dict))}

            # Reconstruct actions
            actions = [
                Action(action_type=action_type, parameters=dict(params))
                for action_type, params in seq_key
            ]

            pattern = ActionPattern(
                actions=actions,
                start_state_features=start_features,
                end_state_features=end_features,
                success_rate=success_rate,
                avg_reward=avg_reward,
                occurrences=len(occurrences),
            )

            patterns.append(pattern)

    # Sort by usefulness (success_rate * occurrences)
    patterns.sort(key=lambda p: p.success_rate * p.occurrences, reverse=True)

    return patterns


class SequenceSynthesizer:
    """Stateful synthesizer for action sequences.

    Maintains a library of patterns and provides synthesis methods.
    """

    def __init__(self):
        self.patterns: list[ActionPattern] = []
        self.synthesis_history: list[ActionSequence] = []

    def learn_from_branches(
        self,
        branches: list[ExplorationBranch],
    ) -> int:
        """Extract and store patterns from branches.

        Args:
            branches: Branches to learn from

        Returns:
            Number of new patterns learned
        """
        new_patterns = extract_reusable_patterns(branches)

        # Add new patterns, update existing
        existing_keys = {
            tuple((a.action_type, tuple(sorted(a.parameters.items())))
                  for a in p.actions)
            for p in self.patterns
        }

        added = 0
        for pattern in new_patterns:
            key = tuple(
                (a.action_type, tuple(sorted(a.parameters.items())))
                for a in pattern.actions
            )
            if key not in existing_keys:
                self.patterns.append(pattern)
                added += 1

        return added

    def synthesize(
        self,
        branches: list[ExplorationBranch],
        goal: Optional[Goal] = None,
    ) -> ActionSequence:
        """Synthesize best action sequence.

        Args:
            branches: Current branches
            goal: Optional goal

        Returns:
            Synthesized sequence
        """
        sequence = synthesize_plan(branches, goal)
        self.synthesis_history.append(sequence)
        return sequence

    def suggest_next_action(
        self,
        current_obs: Observation,
    ) -> Optional[Action]:
        """Suggest next action based on patterns.

        Args:
            current_obs: Current observation

        Returns:
            Suggested action or None
        """
        for pattern in self.patterns:
            if pattern.matches_start(current_obs) and pattern.actions:
                return pattern.actions[0]

        return None

    def get_applicable_patterns(
        self,
        obs: Observation,
    ) -> list[ActionPattern]:
        """Get patterns that might apply in current state.

        Args:
            obs: Current observation

        Returns:
            List of applicable patterns
        """
        return [p for p in self.patterns if p.matches_start(obs)]
