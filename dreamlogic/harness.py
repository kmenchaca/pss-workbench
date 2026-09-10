"""
Dream Logic Generator - Main Dream Harness

The central controller for dream exploration. Manages branches,
injects styles and constraints, checks interestingness at gates,
enforces leap distances, propagates boring, and synthesizes collages.
"""

import asyncio
import random
from dataclasses import dataclass, field
from typing import Callable, Optional, Any

from .types import (
    DreamFragment,
    DreamCollage,
    DreamBranch,
    DreamTree,
    DreamStyle,
    Constraint,
)
from .associations import AssociationDB
from .leaps import generate_leap, LeapContext, score_leap, random_constraint
from .interestingness import (
    InterestingnessModel,
    score_interestingness,
    is_interesting_enough,
)
from .constraints import chain_constraints, constraints_to_prompt
from .styles import apply_style, random_style, style_prompt
from .boring import BoringDetector, propagate_boring, push_toward_strange
from .synthesis import amplify, find_resonances, emergent_themes


@dataclass
class DreamConfig:
    """Configuration for dream harness."""
    # Branching
    max_branches: int = 5
    branch_on_boring: bool = True
    min_leap_distance: float = 0.3

    # Gates
    interestingness_threshold: float = 0.4
    boring_threshold: float = 0.6
    gate_interval_tokens: int = 500

    # Constraints
    inject_constraints: bool = True
    constraints_per_branch: int = 2
    escalate_constraint_difficulty: bool = True

    # Style
    style: Optional[DreamStyle] = None
    random_style_per_branch: bool = False

    # Synthesis
    never_converge: bool = True
    amplify_contradictions: bool = True


@dataclass
class GateResult:
    """Result of a gate check."""
    passed: bool
    interestingness: float
    boring_score: float
    should_branch: bool
    should_terminate: bool
    leap_required: bool
    suggestions: list[str] = field(default_factory=list)


@dataclass
class DreamState:
    """Current state of dream exploration."""
    tree: DreamTree = field(default_factory=DreamTree)
    total_tokens: int = 0
    gates_triggered: int = 0
    branches_terminated: int = 0
    branches_spawned: int = 0
    collage: Optional[DreamCollage] = None


class DreamHarness:
    """
    Main harness for dream logic exploration.

    Key principles:
    - Gate checks ask "Is this INTERESTING?" not "Is this correct?"
    - Enforce minimum associative leap distance
    - Propagate "boring" to push exploration toward strange territory
    - Synthesize into collage, never converge to single answer
    """

    def __init__(
        self,
        config: Optional[DreamConfig] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
        async_llm_fn: Optional[Callable[[str], Any]] = None,
    ):
        """
        Initialize the dream harness.

        Args:
            config: Dream configuration.
            llm_fn: Synchronous LLM function (prompt -> response).
            async_llm_fn: Async LLM function (prompt -> response).
        """
        self.config = config or DreamConfig()
        self.llm_fn = llm_fn
        self.async_llm_fn = async_llm_fn

        # Internal state
        self.state = DreamState()
        self.association_db = AssociationDB()
        self.interestingness_model = InterestingnessModel()
        self.boring_detector = BoringDetector()

        # Pending constraints by branch
        self.branch_constraints: dict[str, list[Constraint]] = {}

    def create_branch(
        self,
        parent_id: Optional[str] = None,
        style: Optional[DreamStyle] = None,
    ) -> DreamBranch:
        """
        Create a new dream branch.

        Args:
            parent_id: Optional parent branch ID.
            style: Optional style for this branch.

        Returns:
            The new branch.
        """
        # Determine style
        if style is None:
            if self.config.random_style_per_branch:
                style = random_style()
            else:
                style = self.config.style

        branch = DreamBranch(parent_id=parent_id, style=style)

        # Generate constraints if enabled
        if self.config.inject_constraints:
            constraints = chain_constraints(
                self.config.constraints_per_branch,
                escalate_difficulty=self.config.escalate_constraint_difficulty,
            )
            branch.constraints = constraints
            self.branch_constraints[branch.id] = constraints

        self.state.tree.add_branch(branch)
        self.state.branches_spawned += 1

        return branch

    def check_gate(
        self,
        branch: DreamBranch,
        content: str,
        context: Optional[str] = None,
    ) -> GateResult:
        """
        Check if content passes the interestingness gate.

        Args:
            branch: The branch being checked.
            content: Content to evaluate.
            context: Optional preceding context.

        Returns:
            GateResult with evaluation and recommendations.
        """
        self.state.gates_triggered += 1

        # Score interestingness
        score = score_interestingness(content, context, self.interestingness_model)
        interestingness = score.total

        # Check boring
        boring = self.boring_detector.score_predictability(content, context)

        # Determine outcomes
        passed = interestingness >= self.config.interestingness_threshold
        is_boring = boring >= self.config.boring_threshold

        # Should we branch?
        should_branch = (
            is_boring and
            self.config.branch_on_boring and
            len(self.state.tree.get_alive_branches()) < self.config.max_branches
        )

        # Should we terminate?
        should_terminate = (
            boring > 0.8 and
            interestingness < 0.2
        )

        # Check if leap is required
        leap_required = False
        if branch.fragments:
            # Check if we've made sufficient leaps
            recent_fragment = branch.fragments[-1]
            leap_required = recent_fragment.surprise_score < self.config.min_leap_distance

        # Gather suggestions
        suggestions = []
        if not passed:
            suggestions.append("Content needs more surprise/novelty")
        if is_boring:
            push_prompt = push_toward_strange(content, self.boring_detector)
            suggestions.append(push_prompt)
        if leap_required:
            suggestions.append(f"Make a bigger conceptual leap (min distance: {self.config.min_leap_distance})")

        return GateResult(
            passed=passed,
            interestingness=interestingness,
            boring_score=boring,
            should_branch=should_branch,
            should_terminate=should_terminate,
            leap_required=leap_required,
            suggestions=suggestions,
        )

    def add_fragment(
        self,
        branch: DreamBranch,
        content: str,
    ) -> DreamFragment:
        """
        Add a fragment to a branch.

        Args:
            branch: Branch to add to.
            content: Content of the fragment.

        Returns:
            The created fragment.
        """
        # Score interestingness
        score = score_interestingness(content)

        # Create fragment
        fragment = DreamFragment(
            content=content,
            associations=[],
            surprise_score=score.surprise,
            branch_id=branch.id,
            style=branch.style,
        )

        # Extract and add associations
        from .associations import chain_associations
        leaps = chain_associations(content[:50], depth=2, db=self.association_db)
        fragment.associations = [l.target for l in leaps]

        branch.fragments.append(fragment)

        # Update interestingness model history
        self.interestingness_model.add_to_history(content)

        # Check for boring and possibly mark
        if score.coherence_penalty > 0.5:
            self.boring_detector.mark_boring(content)

        return fragment

    def terminate_branch(self, branch: DreamBranch) -> None:
        """Terminate a branch."""
        branch.alive = False
        self.state.branches_terminated += 1

    def propagate_boring(self) -> dict[str, float]:
        """
        Propagate boring scores across all alive branches.

        Returns:
            Dictionary of branch_id -> boring_score.
        """
        alive = self.state.tree.get_alive_branches()
        return propagate_boring(alive, self.boring_detector)

    def build_prompt(self, branch: DreamBranch, base_prompt: str) -> str:
        """
        Build a full prompt for a branch including style and constraints.

        Args:
            branch: The branch to build prompt for.
            base_prompt: The base prompt/question.

        Returns:
            Full prompt with style and constraints.
        """
        parts = []

        # Add style instruction
        if branch.style:
            parts.append(style_prompt(branch.style))
            parts.append("")

        # Add constraints
        if branch.constraints:
            parts.append(constraints_to_prompt(branch.constraints))
            parts.append("")

        # Add base prompt
        parts.append(base_prompt)

        # Add leap requirement if needed
        if branch.fragments and branch.fragments[-1].surprise_score < self.config.min_leap_distance:
            parts.append("")
            parts.append(f"[REQUIRED: Make a conceptual leap. Be stranger than expected.]")

        return "\n".join(parts)

    async def explore_branch(
        self,
        branch: DreamBranch,
        prompt: str,
        max_iterations: int = 5,
    ) -> list[DreamFragment]:
        """
        Explore a single branch asynchronously.

        Args:
            branch: Branch to explore.
            prompt: Initial prompt.
            max_iterations: Maximum exploration iterations.

        Returns:
            List of fragments generated.
        """
        if self.async_llm_fn is None:
            raise ValueError("async_llm_fn required for async exploration")

        fragments = []

        for i in range(max_iterations):
            if not branch.alive:
                break

            # Build prompt
            full_prompt = self.build_prompt(branch, prompt)

            # Get LLM response
            response = await self.async_llm_fn(full_prompt)

            # Add fragment
            fragment = self.add_fragment(branch, response)
            fragments.append(fragment)

            # Check gate
            context = branch.fragments[-2].content if len(branch.fragments) > 1 else None
            gate = self.check_gate(branch, response, context)

            # Handle gate result
            if gate.should_terminate:
                self.terminate_branch(branch)
                break

            if gate.should_branch:
                # Create child branch
                child = self.create_branch(parent_id=branch.id)
                # Child gets its own exploration (handled by caller)

            if gate.leap_required:
                # Force a leap
                ctx = LeapContext(
                    current_concept=response[:50],
                    history=[f.content for f in branch.fragments],
                    style_hints=[],
                    minimum_distance=self.config.min_leap_distance,
                )
                leap = generate_leap(ctx, self.association_db)
                # Modify next prompt to incorporate leap
                prompt = f"{prompt}\n\n[LEAP: {leap.target}]"

            # Update prompt for next iteration
            prompt = response[:200]  # Continue from response

        return fragments

    def explore_branch_sync(
        self,
        branch: DreamBranch,
        prompt: str,
        max_iterations: int = 5,
    ) -> list[DreamFragment]:
        """
        Explore a single branch synchronously.

        Args:
            branch: Branch to explore.
            prompt: Initial prompt.
            max_iterations: Maximum exploration iterations.

        Returns:
            List of fragments generated.
        """
        if self.llm_fn is None:
            raise ValueError("llm_fn required for sync exploration")

        fragments = []

        for i in range(max_iterations):
            if not branch.alive:
                break

            # Build prompt
            full_prompt = self.build_prompt(branch, prompt)

            # Get LLM response
            response = self.llm_fn(full_prompt)

            # Add fragment
            fragment = self.add_fragment(branch, response)
            fragments.append(fragment)

            # Check gate
            context = branch.fragments[-2].content if len(branch.fragments) > 1 else None
            gate = self.check_gate(branch, response, context)

            # Handle gate result
            if gate.should_terminate:
                self.terminate_branch(branch)
                break

            if gate.should_branch:
                # Create child branch
                self.create_branch(parent_id=branch.id)

            # Update prompt for next iteration
            prompt = response[:200]

        return fragments

    def synthesize(self) -> DreamCollage:
        """
        Synthesize all branches into a dream collage.

        Never converges to a single answer - amplifies and collages.

        Returns:
            DreamCollage combining all exploration.
        """
        # Gather all fragments from all branches
        all_fragments = []
        for branch in self.state.tree.branches.values():
            all_fragments.extend(branch.fragments)

        if not all_fragments:
            return DreamCollage()

        # Amplify the weird
        collage = amplify(all_fragments)

        # Find resonances
        connections = find_resonances(all_fragments)
        for conn in connections:
            collage.add_connection(conn)

        # Extract emergent themes
        themes = emergent_themes(all_fragments)
        for theme in themes:
            collage.add_theme(theme)

        # Set overall style
        if self.config.style:
            collage.style = self.config.style

        self.state.collage = collage
        return collage

    def get_stats(self) -> dict[str, Any]:
        """Get exploration statistics."""
        return {
            "total_branches": len(self.state.tree.branches),
            "alive_branches": len(self.state.tree.get_alive_branches()),
            "terminated_branches": self.state.branches_terminated,
            "gates_triggered": self.state.gates_triggered,
            "total_fragments": sum(
                len(b.fragments) for b in self.state.tree.branches.values()
            ),
        }


def create_harness(
    style: Optional[DreamStyle] = None,
    max_branches: int = 5,
    llm_fn: Optional[Callable[[str], str]] = None,
) -> DreamHarness:
    """
    Convenience function to create a dream harness.

    Args:
        style: Dream style to use.
        max_branches: Maximum number of branches.
        llm_fn: LLM function for generation.

    Returns:
        Configured DreamHarness.
    """
    config = DreamConfig(
        style=style,
        max_branches=max_branches,
    )
    return DreamHarness(config=config, llm_fn=llm_fn)


async def dream(
    prompt: str,
    harness: Optional[DreamHarness] = None,
    style: Optional[DreamStyle] = None,
    max_branches: int = 3,
    async_llm_fn: Optional[Callable[[str], Any]] = None,
) -> DreamCollage:
    """
    Run a complete dream exploration.

    Args:
        prompt: Initial prompt to explore.
        harness: Optional existing harness.
        style: Dream style.
        max_branches: Maximum branches.
        async_llm_fn: Async LLM function.

    Returns:
        Final DreamCollage.
    """
    if harness is None:
        config = DreamConfig(style=style, max_branches=max_branches)
        harness = DreamHarness(config=config, async_llm_fn=async_llm_fn)

    # Create initial branch
    root = harness.create_branch()

    # Explore
    await harness.explore_branch(root, prompt)

    # Explore any spawned branches
    while True:
        alive = harness.state.tree.get_alive_branches()
        unexplored = [b for b in alive if not b.fragments]
        if not unexplored:
            break

        tasks = [
            harness.explore_branch(b, prompt)
            for b in unexplored[:max_branches]
        ]
        await asyncio.gather(*tasks)

    # Synthesize
    return harness.synthesize()
