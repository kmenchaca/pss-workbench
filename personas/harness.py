"""Persona Harness - Main ensemble orchestration.

Coordinates persona assignment, parallel exploration, consistency checking,
and synthesis into a complete ensemble run.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Optional

from .assignment import assign_personas, detect_missing_perspective, dynamic_spawn
from .consistency import check_consistency, create_reinforcement_injection
from .library import PersonaLibrary
from .synthesis import build_ensemble_result, multi_perspective_brief
from .tracking import PersonaTracker
from .types import (
    AssignmentResult,
    ConsistencyScore,
    EnsembleResult,
    Persona,
    PersonaResponse,
)


# ============================================================================
# Harness Configuration
# ============================================================================


@dataclass
class HarnessConfig:
    """Configuration for the persona harness.

    Attributes:
        num_branches: Number of parallel branches to run
        check_consistency: Whether to check persona consistency at gates
        consistency_threshold: Score below which to reinforce persona
        allow_dynamic_spawn: Whether to spawn branches for missing perspectives
        max_dynamic_spawns: Maximum additional branches to spawn
        parallel: Whether to run branches in parallel
        track_performance: Whether to track persona performance
    """

    num_branches: int = 4
    check_consistency: bool = True
    consistency_threshold: float = 0.6
    allow_dynamic_spawn: bool = True
    max_dynamic_spawns: int = 2
    parallel: bool = True
    track_performance: bool = True


# ============================================================================
# Branch Context
# ============================================================================


@dataclass
class PersonaBranch:
    """A single branch with its assigned persona.

    Attributes:
        branch_id: Unique identifier for this branch
        persona: The persona assigned to this branch
        messages: Conversation messages for this branch
        response: Final response from this branch (if completed)
        consistency_scores: Consistency checks performed
        is_dynamic: Whether this branch was dynamically spawned
    """

    branch_id: str
    persona: Persona
    messages: list[dict] = field(default_factory=list)
    response: Optional[PersonaResponse] = None
    consistency_scores: list[ConsistencyScore] = field(default_factory=list)
    is_dynamic: bool = False


# ============================================================================
# LLM Provider Protocol
# ============================================================================


# Type alias for LLM call function
LLMCallFunc = Callable[[list[dict], str], str]


async def default_llm_call(messages: list[dict], system_prompt: str) -> str:
    """Default placeholder for LLM calls.

    In real usage, this would be replaced with actual API calls.

    Args:
        messages: Conversation messages
        system_prompt: System prompt for the persona

    Returns:
        Response string
    """
    # Placeholder - returns empty string
    return ""


# ============================================================================
# Persona Harness
# ============================================================================


class PersonaHarness:
    """Main harness for running persona ensembles.

    Coordinates:
    - Persona assignment to branches
    - Parallel branch execution
    - Consistency checking at gates
    - Dynamic spawning for missing perspectives
    - Performance tracking
    - Final synthesis
    """

    def __init__(
        self,
        config: Optional[HarnessConfig] = None,
        library: Optional[PersonaLibrary] = None,
        llm_call: Optional[LLMCallFunc] = None,
    ):
        """Initialize the harness.

        Args:
            config: Harness configuration
            library: Persona library to use
            llm_call: Function to call LLM (async)
        """
        self.config = config or HarnessConfig()
        self.library = library or PersonaLibrary()
        self.llm_call = llm_call or default_llm_call
        self.tracker = PersonaTracker() if self.config.track_performance else None

        self.branches: list[PersonaBranch] = []
        self.assignment_result: Optional[AssignmentResult] = None
        self.result: Optional[EnsembleResult] = None

    def assign(
        self,
        problem: str,
        required_personas: Optional[list[str]] = None,
        excluded_personas: Optional[list[str]] = None,
    ) -> AssignmentResult:
        """Assign personas to branches based on the problem.

        Args:
            problem: The problem to analyze
            required_personas: Persona IDs that must be included
            excluded_personas: Persona IDs to exclude

        Returns:
            AssignmentResult with branch assignments
        """
        self.assignment_result = assign_personas(
            problem=problem,
            num_branches=self.config.num_branches,
            library=self.library,
            required_personas=required_personas,
            excluded_personas=excluded_personas,
        )

        # Create branch objects
        self.branches = []
        for branch_id, persona in self.assignment_result.assignments.items():
            self.branches.append(
                PersonaBranch(
                    branch_id=branch_id,
                    persona=persona,
                )
            )

        return self.assignment_result

    async def run_branch(
        self,
        branch: PersonaBranch,
        problem: str,
    ) -> PersonaResponse:
        """Run exploration for a single branch.

        Args:
            branch: The branch to run
            problem: The problem to analyze

        Returns:
            PersonaResponse from this branch
        """
        # Initialize messages with the problem
        messages = [{"role": "user", "content": problem}]
        branch.messages = messages.copy()

        # Get response from LLM
        response_content = await self.llm_call(messages, branch.persona.system_prompt)

        # Check consistency
        if self.config.check_consistency:
            temp_response = PersonaResponse(
                persona_id=branch.persona.id,
                content=response_content,
            )
            consistency = check_consistency(branch.persona, temp_response)
            branch.consistency_scores.append(consistency)

            # Inject reinforcement if needed
            if consistency.score < self.config.consistency_threshold:
                reinforcement = create_reinforcement_injection(branch.persona, consistency)
                if reinforcement:
                    # Add reinforcement and get another response
                    messages.append({"role": "assistant", "content": response_content})
                    messages.append({"role": "user", "content": reinforcement})
                    response_content = await self.llm_call(messages, branch.persona.system_prompt)
                    branch.messages = messages.copy()

        # Create final response
        response = PersonaResponse(
            persona_id=branch.persona.id,
            content=response_content,
            raw_response=response_content,
        )
        branch.response = response

        return response

    async def run_all(
        self,
        problem: str,
        required_personas: Optional[list[str]] = None,
        excluded_personas: Optional[list[str]] = None,
    ) -> EnsembleResult:
        """Run the complete ensemble analysis.

        Args:
            problem: The problem to analyze
            required_personas: Persona IDs that must be included
            excluded_personas: Persona IDs to exclude

        Returns:
            EnsembleResult with all responses and synthesis
        """
        # Assign personas if not already done
        if not self.branches:
            self.assign(problem, required_personas, excluded_personas)

        # Run branches
        if self.config.parallel:
            # Run all branches in parallel
            responses = await asyncio.gather(
                *[self.run_branch(branch, problem) for branch in self.branches]
            )
        else:
            # Run sequentially
            responses = []
            for branch in self.branches:
                response = await self.run_branch(branch, problem)
                responses.append(response)

        # Check for missing perspectives and dynamically spawn if needed
        if self.config.allow_dynamic_spawn:
            spawned = 0
            while spawned < self.config.max_dynamic_spawns:
                missing = detect_missing_perspective(responses, self.library)
                if not missing:
                    break

                # Spawn new branch for missing perspective
                new_persona = dynamic_spawn(
                    missing,
                    self.library,
                    excluded_personas=[r.persona_id for r in responses],
                )

                if not new_persona:
                    break

                # Create and run new branch
                new_branch = PersonaBranch(
                    branch_id=f"branch_dynamic_{spawned}",
                    persona=new_persona,
                    is_dynamic=True,
                )
                self.branches.append(new_branch)

                new_response = await self.run_branch(new_branch, problem)
                responses.append(new_response)
                spawned += 1

        # Build result
        self.result = build_ensemble_result(responses, self.library)

        # Track performance if enabled
        if self.tracker:
            for branch in self.branches:
                if branch.response:
                    # Simple heuristic: useful if provided unique insights
                    unique_insights = self.result.attributed_insights.get(
                        branch.persona.id, []
                    )
                    was_useful = len(unique_insights) > 0
                    self.tracker.record_use(
                        branch.persona.id,
                        success=was_useful,
                        unique_insights=len(unique_insights),
                    )

        return self.result

    def get_brief(self, max_length: int = 2000) -> str:
        """Get a multi-perspective brief from the results.

        Args:
            max_length: Maximum length of brief

        Returns:
            Formatted brief string
        """
        if not self.result:
            return "No results available. Run the ensemble first."

        return multi_perspective_brief(self.result.responses, self.library, max_length)

    def get_consistency_report(self) -> str:
        """Get a report on persona consistency across branches.

        Returns:
            Consistency report string
        """
        lines = ["## Persona Consistency Report\n"]

        for branch in self.branches:
            lines.append(f"\n### {branch.persona.name} ({branch.branch_id})")
            if branch.is_dynamic:
                lines.append(" [DYNAMICALLY SPAWNED]")
            lines.append("\n")

            if not branch.consistency_scores:
                lines.append("  No consistency checks performed.\n")
                continue

            avg_score = sum(s.score for s in branch.consistency_scores) / len(
                branch.consistency_scores
            )
            lines.append(f"  Average consistency: {avg_score:.0%}\n")

            # Show any issues
            all_issues = []
            for score in branch.consistency_scores:
                all_issues.extend(score.inconsistent_elements)

            if all_issues:
                lines.append("  Issues detected:\n")
                for issue in all_issues[:3]:
                    lines.append(f"    - {issue}\n")

        return "".join(lines)

    def get_performance_recommendations(self) -> list[str]:
        """Get recommendations for future persona selection.

        Returns:
            List of recommendation strings
        """
        if not self.tracker:
            return ["Performance tracking not enabled."]

        recommendations = []

        # Get top performers
        top = self.tracker.get_top_performers(3)
        if top:
            recommendations.append(
                f"Top performers: {', '.join(p[0] for p in top)}"
            )

        # Get underperformers
        under = self.tracker.get_underperformers(0.3)
        if under:
            recommendations.append(
                f"Consider replacing: {', '.join(under)} (low unique insight rate)"
            )

        # Suggest for diversity
        if self.result and self.result.diversity_score < 0.5:
            recommendations.append(
                "Diversity score is low. Consider adding more contrasting personas."
            )

        return recommendations


# ============================================================================
# Convenience Functions
# ============================================================================


async def run_ensemble(
    problem: str,
    num_branches: int = 4,
    llm_call: Optional[LLMCallFunc] = None,
    library: Optional[PersonaLibrary] = None,
) -> EnsembleResult:
    """Convenience function to run a quick ensemble analysis.

    Args:
        problem: The problem to analyze
        num_branches: Number of branches to run
        llm_call: LLM call function
        library: Persona library

    Returns:
        EnsembleResult
    """
    config = HarnessConfig(num_branches=num_branches)
    harness = PersonaHarness(config, library, llm_call)
    return await harness.run_all(problem)


def create_persona_prompt(persona: Persona, problem: str) -> str:
    """Create a complete prompt for a persona branch.

    Combines the persona's system prompt with the problem.

    Args:
        persona: The persona to create prompt for
        problem: The problem to analyze

    Returns:
        Complete prompt string
    """
    return f"""{persona.system_prompt}

---

Problem to analyze:
{problem}

Provide your analysis from your unique perspective. Be specific and stay in character."""


def inject_persona_into_context(
    persona: Persona,
    messages: list[dict],
) -> list[dict]:
    """Inject persona identity into a message context.

    Adds persona reminders to help maintain character consistency.

    Args:
        persona: The persona to inject
        messages: Existing messages

    Returns:
        Messages with persona injection
    """
    # Add persona reminder at the start
    reminder = {
        "role": "system",
        "content": f"You are {persona.name}. {persona.description}. "
        f"Think in a {persona.thinking_style.value} way. "
        f"Stay in character throughout your response.",
    }

    # Create new list with reminder first
    return [reminder] + messages
