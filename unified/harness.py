"""Unified Harness - Main orchestration for all experimental systems.

Provides a single interface to run any of the 10 experimental directions
with consistent configuration and output handling.
"""

import asyncio
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pss.providers import Provider, create_provider

from .config import (
    SystemConfig,
    SystemType,
    UnifiedConfig,
    get_info,
    get_system_type,
    list_all_systems,
)
from .router import RouteDecision, SystemRouter


@dataclass
class SystemResult:
    """Result from running a system.

    Attributes:
        system_type: Which system was run
        prompt: Original prompt
        output: Main output from the system
        metadata: System-specific metadata
        elapsed_seconds: Runtime
        tokens_used: Approximate token usage
        success: Whether the run completed successfully
        error: Error message if failed
    """

    system_type: SystemType
    prompt: str
    output: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    tokens_used: int = 0
    success: bool = True
    error: Optional[str] = None


class UnifiedHarness:
    """Unified harness for running experimental systems.

    Provides a consistent interface to all 10 systems with:
    - Automatic provider/model configuration
    - Optional auto-routing to best system
    - Parallel execution support
    - Unified result format
    """

    def __init__(
        self,
        config: Optional[UnifiedConfig] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize the unified harness.

        Args:
            config: Full configuration object
            provider: Override default provider
            model: Override default model
        """
        self.config = config or UnifiedConfig()

        # Apply overrides
        if provider:
            self.config.default_provider = provider
        if model:
            self.config.default_model = model

        self.router = SystemRouter()
        self._provider_cache: dict[str, Provider] = {}

    def _get_provider(self, provider_name: str, model: str) -> Provider:
        """Get or create a provider instance.

        Args:
            provider_name: Provider name
            model: Model identifier

        Returns:
            Provider instance
        """
        key = f"{provider_name}:{model}"
        if key not in self._provider_cache:
            self._provider_cache[key] = create_provider(provider_name, model)
        return self._provider_cache[key]

    async def run(
        self,
        system: str,
        prompt: str,
        config: Optional[SystemConfig] = None,
    ) -> SystemResult:
        """Run a specific system.

        Args:
            system: System name (e.g., "argswarm", "personas")
            prompt: The prompt to run
            config: Optional system-specific configuration

        Returns:
            SystemResult with output and metadata
        """
        # Parse system type
        sys_type = get_system_type(system)
        if sys_type is None:
            return SystemResult(
                system_type=SystemType.PERSONAS,  # fallback
                prompt=prompt,
                output=None,
                success=False,
                error=f"Unknown system: {system}. Use list_systems() to see available systems.",
            )

        # Create config if not provided
        if config is None:
            config = SystemConfig(
                system_type=sys_type,
                provider=self.config.default_provider,
                model=self.config.default_model,
            )

        # Get provider
        provider = self._get_provider(config.provider, config.model)

        # Route to appropriate runner
        start_time = time.time()
        try:
            result = await self._run_system(sys_type, prompt, provider, config)
            elapsed = time.time() - start_time

            return SystemResult(
                system_type=sys_type,
                prompt=prompt,
                output=result.get("output"),
                metadata=result.get("metadata", {}),
                elapsed_seconds=elapsed,
                tokens_used=result.get("tokens_used", 0),
                success=True,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            return SystemResult(
                system_type=sys_type,
                prompt=prompt,
                output=None,
                elapsed_seconds=elapsed,
                success=False,
                error=str(e),
            )

    async def auto_run(self, prompt: str) -> SystemResult:
        """Automatically route and run the best system for a prompt.

        Args:
            prompt: The prompt to run

        Returns:
            SystemResult from the selected system
        """
        decision = self.router.route(prompt)

        if self.config.verbose:
            print(f"[Auto-routing] {decision.reasoning}")

        return await self.run(decision.system_type.value, prompt)

    async def run_multiple(
        self,
        systems: list[str],
        prompt: str,
        parallel: bool = True,
    ) -> list[SystemResult]:
        """Run multiple systems on the same prompt.

        Args:
            systems: List of system names
            prompt: The prompt to run
            parallel: Whether to run in parallel

        Returns:
            List of SystemResults
        """
        if parallel:
            tasks = [self.run(sys, prompt) for sys in systems]
            return await asyncio.gather(*tasks)
        else:
            results = []
            for sys in systems:
                result = await self.run(sys, prompt)
                results.append(result)
            return results

    async def _run_system(
        self,
        sys_type: SystemType,
        prompt: str,
        provider: Provider,
        config: SystemConfig,
    ) -> dict[str, Any]:
        """Internal dispatch to system-specific runners.

        Args:
            sys_type: Which system to run
            prompt: The prompt
            provider: LLM provider
            config: System configuration

        Returns:
            Dictionary with output and metadata
        """
        runners = {
            SystemType.ARGSWARM: self._run_argswarm,
            SystemType.PERSONAS: self._run_personas,
            SystemType.REDTEAM: self._run_redteam,
            SystemType.EVOLUTION: self._run_evolution,
            SystemType.TEMPORAL: self._run_temporal,
            SystemType.KNOWLEDGE: self._run_knowledge,
            SystemType.EMBODIED: self._run_embodied,
            SystemType.METALEARNER: self._run_metalearner,
            SystemType.DREAMLOGIC: self._run_dreamlogic,
            SystemType.NEGOTIATE: self._run_negotiate,
        }

        runner = runners.get(sys_type)
        if runner:
            return await runner(prompt, provider, config)
        else:
            raise ValueError(f"No runner for system: {sys_type}")

    # =========================================================================
    # System Runners
    # =========================================================================

    async def _run_argswarm(
        self,
        prompt: str,
        provider: Provider,
        config: SystemConfig,
    ) -> dict[str, Any]:
        """Run Argumentative Swarm."""
        from argswarm import Position, Stance, Argument, Refutation, Judgment, assign_positions

        # Assign positions
        positions = assign_positions(prompt, 2)
        all_arguments: list[Argument] = []
        all_refutations: list[Refutation] = []

        # Generate opening arguments
        for pos in positions:
            stance_desc = "FOR" if pos.stance == Stance.FOR else "AGAINST"
            messages = [
                {"role": "system", "content": f"You are arguing {stance_desc} the proposition."},
                {"role": "user", "content": f"Proposition: {prompt}\n\nMake your argument."},
            ]
            response, _, _, _ = await provider.chat_async(messages)

            arg = Argument(
                id=f"arg_{pos.id}",
                position_id=pos.id,
                claim=response[:200],
                evidence=response,
                warrant="",
                strength=0.7,
                round_number=1,
            )
            all_arguments.append(arg)

        # Generate rebuttals
        for pos in positions:
            opponent_args = [a for a in all_arguments if a.position_id != pos.id]
            if opponent_args:
                target = opponent_args[0]
                stance_desc = "FOR" if pos.stance == Stance.FOR else "AGAINST"
                messages = [
                    {"role": "system", "content": f"You are arguing {stance_desc}. Rebut the opponent."},
                    {"role": "user", "content": f"Opponent's argument:\n{target.evidence[:500]}\n\nRebut this."},
                ]
                response, _, _, _ = await provider.chat_async(messages)

                ref = Refutation(
                    id=f"ref_{pos.id}",
                    target_argument_id=target.id,
                    counter_claim=response[:200],
                    evidence=response,
                    refutation_type="rebut",
                    strength=0.7,
                    position_id=pos.id,
                    round_number=2,
                )
                all_refutations.append(ref)

        # Adjudicate
        debate_summary = "\n".join(
            f"[{('FOR' if p.stance == Stance.FOR else 'AGAINST')}]: {next((a.evidence[:300] for a in all_arguments if a.position_id == p.id), '')}"
            for p in positions
        )

        messages = [
            {"role": "system", "content": "You are a debate judge. Declare a winner and explain why."},
            {"role": "user", "content": f"Proposition: {prompt}\n\nDebate:\n{debate_summary}\n\nWho wins?"},
        ]
        judgment_text, _, _, _ = await provider.chat_async(messages)

        return {
            "output": {
                "proposition": prompt,
                "arguments": [{"position": "FOR" if a.position_id == positions[0].id else "AGAINST", "text": a.evidence} for a in all_arguments],
                "refutations": [{"position": "FOR" if r.position_id == positions[0].id else "AGAINST", "text": r.evidence} for r in all_refutations],
                "judgment": judgment_text,
            },
            "metadata": {
                "num_arguments": len(all_arguments),
                "num_refutations": len(all_refutations),
            },
        }

    async def _run_personas(
        self,
        prompt: str,
        provider: Provider,
        config: SystemConfig,
    ) -> dict[str, Any]:
        """Run Persona Ensemble."""
        from personas import PersonaLibrary, assign_personas, synthesize_perspectives

        library = PersonaLibrary()

        # Assign personas
        assignment = assign_personas(
            problem=prompt,
            num_branches=config.num_branches,
            library=library,
        )

        personas = list(assignment.assignments.values())
        responses = []

        # Get responses from each persona
        for persona in personas:
            messages = [
                {"role": "system", "content": persona.system_prompt},
                {"role": "user", "content": prompt},
            ]
            response, _, _, _ = await provider.chat_async(messages)
            responses.append({
                "persona": persona.name,
                "response": response,
            })

        # Synthesize
        synthesis_prompt = f"Synthesize these perspectives on: {prompt}\n\n"
        for r in responses:
            synthesis_prompt += f"[{r['persona']}]: {r['response'][:500]}...\n\n"

        messages = [
            {"role": "system", "content": "Synthesize multiple perspectives into a balanced summary."},
            {"role": "user", "content": synthesis_prompt},
        ]
        synthesis, _, _, _ = await provider.chat_async(messages)

        return {
            "output": {
                "problem": prompt,
                "perspectives": responses,
                "synthesis": synthesis,
            },
            "metadata": {
                "num_personas": len(personas),
                "persona_names": [p.name for p in personas],
            },
        }

    async def _run_redteam(
        self,
        prompt: str,
        provider: Provider,
        config: SystemConfig,
    ) -> dict[str, Any]:
        """Run Red Team Engine."""
        from redteam import AttackVector, VulnerabilityReport

        # Generate attack vectors
        messages = [
            {"role": "system", "content": "You are a security researcher. Identify potential vulnerabilities."},
            {"role": "user", "content": f"Analyze for security issues:\n{prompt}"},
        ]
        vulnerabilities, _, _, _ = await provider.chat_async(messages)

        # Generate recommendations
        messages = [
            {"role": "system", "content": "You are a security consultant. Provide fix recommendations."},
            {"role": "user", "content": f"Vulnerabilities found:\n{vulnerabilities}\n\nProvide fixes."},
        ]
        recommendations, _, _, _ = await provider.chat_async(messages)

        return {
            "output": {
                "target": prompt,
                "vulnerabilities": vulnerabilities,
                "recommendations": recommendations,
            },
            "metadata": {"analysis_type": "red_team"},
        }

    async def _run_evolution(
        self,
        prompt: str,
        provider: Provider,
        config: SystemConfig,
    ) -> dict[str, Any]:
        """Run Evolution Engine."""
        generations = config.extra.get("generations", 3)
        population = []

        # Initial generation
        messages = [
            {"role": "system", "content": "Generate code to solve this problem."},
            {"role": "user", "content": prompt},
        ]
        initial, _, _, _ = await provider.chat_async(messages)
        population.append({"gen": 0, "code": initial, "fitness": 0.5})

        # Evolve
        for gen in range(1, generations + 1):
            best = max(population, key=lambda x: x["fitness"])
            messages = [
                {"role": "system", "content": "Improve this code. Make it better."},
                {"role": "user", "content": f"Original:\n{best['code']}\n\nImprove it."},
            ]
            evolved, _, _, _ = await provider.chat_async(messages)
            population.append({"gen": gen, "code": evolved, "fitness": 0.5 + gen * 0.1})

        best_solution = max(population, key=lambda x: x["fitness"])

        return {
            "output": {
                "specification": prompt,
                "best_solution": best_solution["code"],
                "generations": generations,
            },
            "metadata": {
                "population_size": len(population),
                "final_fitness": best_solution["fitness"],
            },
        }

    async def _run_temporal(
        self,
        prompt: str,
        provider: Provider,
        config: SystemConfig,
    ) -> dict[str, Any]:
        """Run Temporal Planner."""
        horizons = config.extra.get("horizons", ["short", "medium", "long"])
        scenarios = {}

        for horizon in horizons:
            messages = [
                {"role": "system", "content": f"Project {horizon}-term scenarios for this situation."},
                {"role": "user", "content": prompt},
            ]
            response, _, _, _ = await provider.chat_async(messages)
            scenarios[horizon] = response

        return {
            "output": {
                "initial_conditions": prompt,
                "scenarios": scenarios,
            },
            "metadata": {"horizons_analyzed": horizons},
        }

    async def _run_knowledge(
        self,
        prompt: str,
        provider: Provider,
        config: SystemConfig,
    ) -> dict[str, Any]:
        """Run Knowledge Graph builder."""
        # Extract concepts
        messages = [
            {"role": "system", "content": "Extract key concepts from this text. List them."},
            {"role": "user", "content": prompt},
        ]
        concepts, _, _, _ = await provider.chat_async(messages)

        # Find relationships
        messages = [
            {"role": "system", "content": "Identify relationships between these concepts."},
            {"role": "user", "content": f"Concepts:\n{concepts}\n\nDescribe their relationships."},
        ]
        relationships, _, _, _ = await provider.chat_async(messages)

        return {
            "output": {
                "domain": prompt,
                "concepts": concepts,
                "relationships": relationships,
            },
            "metadata": {"analysis_type": "knowledge_graph"},
        }

    async def _run_embodied(
        self,
        prompt: str,
        provider: Provider,
        config: SystemConfig,
    ) -> dict[str, Any]:
        """Run Embodied Planner."""
        messages = [
            {"role": "system", "content": "Plan physical actions to achieve this goal. Consider constraints, physics, and safety."},
            {"role": "user", "content": prompt},
        ]
        plan, _, _, _ = await provider.chat_async(messages)

        # Verify feasibility
        messages = [
            {"role": "system", "content": "Check if this action plan is feasible. Identify issues."},
            {"role": "user", "content": f"Plan:\n{plan}\n\nIs this feasible?"},
        ]
        verification, _, _, _ = await provider.chat_async(messages)

        return {
            "output": {
                "goal": prompt,
                "action_plan": plan,
                "feasibility": verification,
            },
            "metadata": {"analysis_type": "embodied_planning"},
        }

    async def _run_metalearner(
        self,
        prompt: str,
        provider: Provider,
        config: SystemConfig,
    ) -> dict[str, Any]:
        """Run Meta-Learner."""
        # Analyze task
        messages = [
            {"role": "system", "content": "Analyze this task. What strategies would work best?"},
            {"role": "user", "content": prompt},
        ]
        analysis, _, _, _ = await provider.chat_async(messages)

        # Generate strategy
        messages = [
            {"role": "system", "content": "Based on your analysis, create an optimal strategy."},
            {"role": "user", "content": f"Analysis:\n{analysis}\n\nCreate strategy."},
        ]
        strategy, _, _, _ = await provider.chat_async(messages)

        return {
            "output": {
                "task": prompt,
                "analysis": analysis,
                "strategy": strategy,
            },
            "metadata": {"analysis_type": "meta_learning"},
        }

    async def _run_dreamlogic(
        self,
        prompt: str,
        provider: Provider,
        config: SystemConfig,
    ) -> dict[str, Any]:
        """Run Dream Logic Generator."""
        messages = [
            {"role": "system", "content": "Think creatively and unconventionally. Make unexpected associations. Be surreal and imaginative."},
            {"role": "user", "content": f"Starting concept: {prompt}\n\nExplore this with dream logic."},
        ]
        associations, _, _, _ = await provider.chat_async(messages)

        # Extract insights
        messages = [
            {"role": "system", "content": "Find practical insights hidden in these creative associations."},
            {"role": "user", "content": f"Creative exploration:\n{associations}\n\nWhat insights emerge?"},
        ]
        insights, _, _, _ = await provider.chat_async(messages)

        return {
            "output": {
                "seed": prompt,
                "associations": associations,
                "insights": insights,
            },
            "metadata": {"analysis_type": "dream_logic"},
        }

    async def _run_negotiate(
        self,
        prompt: str,
        provider: Provider,
        config: SystemConfig,
    ) -> dict[str, Any]:
        """Run Negotiation Simulator."""
        # Identify parties
        messages = [
            {"role": "system", "content": "Identify the parties and their interests in this negotiation."},
            {"role": "user", "content": prompt},
        ]
        parties, _, _, _ = await provider.chat_async(messages)

        # Simulate negotiation
        messages = [
            {"role": "system", "content": "Simulate a negotiation between these parties. Show offers and counteroffers."},
            {"role": "user", "content": f"Parties:\n{parties}\n\nSimulate negotiation."},
        ]
        simulation, _, _, _ = await provider.chat_async(messages)

        # Find resolution
        messages = [
            {"role": "system", "content": "Propose a fair resolution that balances all interests."},
            {"role": "user", "content": f"Negotiation:\n{simulation}\n\nPropose resolution."},
        ]
        resolution, _, _, _ = await provider.chat_async(messages)

        return {
            "output": {
                "situation": prompt,
                "parties": parties,
                "negotiation": simulation,
                "resolution": resolution,
            },
            "metadata": {"analysis_type": "negotiation"},
        }


# =============================================================================
# Convenience Functions
# =============================================================================


async def run_system(
    system: str,
    prompt: str,
    provider: str = "openrouter",
    model: str = "meta-llama/llama-3.1-8b-instruct",
) -> SystemResult:
    """Convenience function to run a system.

    Args:
        system: System name
        prompt: The prompt
        provider: LLM provider
        model: Model name

    Returns:
        SystemResult
    """
    harness = UnifiedHarness(provider=provider, model=model)
    return await harness.run(system, prompt)


def list_systems() -> list[dict[str, Any]]:
    """List all available systems.

    Returns:
        List of system info dictionaries
    """
    return list_all_systems()


def get_system_info(system: str) -> Optional[dict[str, Any]]:
    """Get info about a specific system.

    Args:
        system: System name

    Returns:
        System info dictionary or None
    """
    sys_type = get_system_type(system)
    if sys_type:
        return get_info(sys_type)
    return None
