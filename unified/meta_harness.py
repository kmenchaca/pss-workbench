"""Meta-Harness - Higher-level orchestration for complex multi-system tasks.

The meta-harness provides intelligent routing and orchestration across
all 10 experimental systems. It can:

1. Auto-route to the best system for a task
2. Combine multiple systems for complex problems
3. Learn from results to improve future routing
4. Synthesize outputs from parallel system runs

Example usage:
    from unified import MetaHarness

    harness = MetaHarness()
    result = await harness.smart_run("Should we migrate to microservices?")
    # Automatically selects systems, runs them, and synthesizes
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import SystemType, UnifiedConfig
from .harness import UnifiedHarness, SystemResult
from .router import SystemRouter, RouteDecision


@dataclass
class MetaResult:
    """Result from a meta-harness run.

    Attributes:
        primary_result: Main result from primary system
        supporting_results: Results from supporting systems
        routing_decision: How routing was decided
        synthesis: Combined synthesis across systems
        confidence: Overall confidence score
        recommendations: Suggested follow-up actions
    """

    primary_result: SystemResult
    supporting_results: list[SystemResult] = field(default_factory=list)
    routing_decision: Optional[RouteDecision] = None
    synthesis: str = ""
    confidence: float = 0.0
    recommendations: list[str] = field(default_factory=list)


@dataclass
class TaskAnalysis:
    """Analysis of a task for routing.

    Attributes:
        complexity: Estimated complexity (low, medium, high)
        recommended_systems: Systems to use
        parallel_capable: Whether systems can run in parallel
        requires_synthesis: Whether final synthesis is needed
    """

    complexity: str
    recommended_systems: list[SystemType]
    parallel_capable: bool
    requires_synthesis: bool


class MetaHarness:
    """Meta-level orchestration across all experimental systems.

    Provides intelligent routing and multi-system orchestration
    for complex tasks that may benefit from multiple perspectives.
    """

    def __init__(
        self,
        config: Optional[UnifiedConfig] = None,
        provider: str = "openrouter",
        model: str = "meta-llama/llama-3.1-8b-instruct",
    ):
        """Initialize the meta-harness.

        Args:
            config: Unified configuration
            provider: Default LLM provider
            model: Default model
        """
        self.config = config or UnifiedConfig(
            default_provider=provider,
            default_model=model,
        )
        self.unified = UnifiedHarness(config=self.config)
        self.router = SystemRouter()

        # Track routing history for learning
        self.history: list[dict] = []

    def analyze_task(self, prompt: str) -> TaskAnalysis:
        """Analyze a task to determine optimal routing.

        Args:
            prompt: The task prompt

        Returns:
            TaskAnalysis with routing recommendations
        """
        # Get primary routing decision
        decision = self.router.route(prompt)

        # Determine complexity based on prompt characteristics
        complexity = self._estimate_complexity(prompt)

        # Get supporting systems based on task type
        systems = [decision.system_type]

        if complexity in ["medium", "high"]:
            # Add complementary systems
            systems.extend(self._get_complementary_systems(decision.system_type))

        return TaskAnalysis(
            complexity=complexity,
            recommended_systems=systems[:3],  # Limit to 3 max
            parallel_capable=len(systems) > 1,
            requires_synthesis=len(systems) > 1,
        )

    async def smart_run(
        self,
        prompt: str,
        max_systems: int = 3,
    ) -> MetaResult:
        """Run with automatic system selection and orchestration.

        This is the main entry point for intelligent multi-system runs.

        Args:
            prompt: The task prompt
            max_systems: Maximum number of systems to run

        Returns:
            MetaResult with synthesized output
        """
        # Analyze the task
        analysis = self.analyze_task(prompt)

        if self.config.verbose:
            print(f"[MetaHarness] Task complexity: {analysis.complexity}")
            print(f"[MetaHarness] Systems: {[s.value for s in analysis.recommended_systems]}")

        # Get routing decision
        decision = self.router.route(prompt)

        # Run systems
        systems_to_run = analysis.recommended_systems[:max_systems]

        if len(systems_to_run) == 1:
            # Single system run
            primary = await self.unified.run(systems_to_run[0].value, prompt)
            return MetaResult(
                primary_result=primary,
                routing_decision=decision,
                synthesis=self._extract_output_text(primary),
                confidence=decision.confidence,
            )

        # Multi-system run
        results = await self.unified.run_multiple(
            [s.value for s in systems_to_run],
            prompt,
            parallel=analysis.parallel_capable,
        )

        # Separate primary from supporting
        primary = results[0]
        supporting = results[1:]

        # Synthesize if multiple systems
        synthesis = await self._synthesize_results(prompt, results)

        # Generate recommendations
        recommendations = self._generate_recommendations(analysis, results)

        # Record in history
        self.history.append({
            "prompt": prompt,
            "systems": [s.value for s in systems_to_run],
            "success": all(r.success for r in results),
        })

        return MetaResult(
            primary_result=primary,
            supporting_results=supporting,
            routing_decision=decision,
            synthesis=synthesis,
            confidence=sum(r.success for r in results) / len(results),
            recommendations=recommendations,
        )

    async def run_pipeline(
        self,
        prompt: str,
        pipeline: list[str],
    ) -> list[SystemResult]:
        """Run a specific pipeline of systems sequentially.

        Each system's output is passed to the next as context.

        Args:
            prompt: Initial prompt
            pipeline: List of system names to run in order

        Returns:
            List of results from each system
        """
        results = []
        current_prompt = prompt

        for system in pipeline:
            result = await self.unified.run(system, current_prompt)
            results.append(result)

            if not result.success:
                break

            # Build context for next system
            output_text = self._extract_output_text(result)
            current_prompt = f"Previous analysis:\n{output_text}\n\nOriginal task: {prompt}\n\nContinue the analysis."

        return results

    async def run_comparison(
        self,
        prompt: str,
        systems: list[str],
    ) -> dict[str, SystemResult]:
        """Run multiple systems for comparison.

        Args:
            prompt: The prompt to run
            systems: Systems to compare

        Returns:
            Dict mapping system name to result
        """
        results = await self.unified.run_multiple(systems, prompt, parallel=True)

        return {
            systems[i]: results[i]
            for i in range(len(systems))
        }

    async def _synthesize_results(
        self,
        prompt: str,
        results: list[SystemResult],
    ) -> str:
        """Synthesize results from multiple systems.

        Args:
            prompt: Original prompt
            results: Results to synthesize

        Returns:
            Synthesized text
        """
        # Build synthesis prompt
        synthesis_parts = [f"Original task: {prompt}\n\nAnalyses from different systems:\n"]

        for result in results:
            if result.success:
                output_text = self._extract_output_text(result)
                synthesis_parts.append(f"\n[{result.system_type.value.upper()}]:\n{output_text[:1000]}\n")

        synthesis_prompt = "".join(synthesis_parts)
        synthesis_prompt += "\n\nSynthesize these perspectives into a unified, actionable summary."

        # Run synthesis through the unified harness
        from pss.providers import create_provider

        provider = create_provider(
            self.config.default_provider,
            self.config.default_model,
        )

        messages = [
            {"role": "system", "content": "You synthesize multiple analytical perspectives into clear, actionable summaries."},
            {"role": "user", "content": synthesis_prompt},
        ]

        synthesis, _, _, _ = await provider.chat_async(messages)
        return synthesis

    def _estimate_complexity(self, prompt: str) -> str:
        """Estimate task complexity from prompt.

        Args:
            prompt: The prompt to analyze

        Returns:
            Complexity level: "low", "medium", or "high"
        """
        # Simple heuristics
        word_count = len(prompt.split())

        # Check for complexity indicators
        complex_indicators = [
            "complex", "comprehensive", "thorough", "detailed",
            "analyze", "compare", "evaluate", "synthesize",
            "multiple", "various", "different aspects",
        ]

        indicator_count = sum(
            1 for ind in complex_indicators
            if ind.lower() in prompt.lower()
        )

        if word_count > 100 or indicator_count >= 3:
            return "high"
        elif word_count > 30 or indicator_count >= 1:
            return "medium"
        else:
            return "low"

    def _get_complementary_systems(
        self,
        primary: SystemType,
    ) -> list[SystemType]:
        """Get systems that complement the primary system.

        Args:
            primary: Primary system type

        Returns:
            List of complementary system types
        """
        complements = {
            SystemType.ARGSWARM: [SystemType.PERSONAS, SystemType.REDTEAM],
            SystemType.PERSONAS: [SystemType.ARGSWARM, SystemType.TEMPORAL],
            SystemType.REDTEAM: [SystemType.PERSONAS, SystemType.ARGSWARM],
            SystemType.EVOLUTION: [SystemType.REDTEAM, SystemType.METALEARNER],
            SystemType.TEMPORAL: [SystemType.PERSONAS, SystemType.KNOWLEDGE],
            SystemType.KNOWLEDGE: [SystemType.TEMPORAL, SystemType.DREAMLOGIC],
            SystemType.EMBODIED: [SystemType.TEMPORAL, SystemType.REDTEAM],
            SystemType.METALEARNER: [SystemType.EVOLUTION, SystemType.KNOWLEDGE],
            SystemType.DREAMLOGIC: [SystemType.PERSONAS, SystemType.KNOWLEDGE],
            SystemType.NEGOTIATE: [SystemType.PERSONAS, SystemType.ARGSWARM],
        }

        return complements.get(primary, [SystemType.PERSONAS])

    def _extract_output_text(self, result: SystemResult) -> str:
        """Extract text from system result output.

        Args:
            result: System result

        Returns:
            Text representation of output
        """
        if not result.success:
            return f"Error: {result.error}"

        output = result.output
        if isinstance(output, str):
            return output
        elif isinstance(output, dict):
            # Extract key content
            parts = []
            for key, value in output.items():
                if isinstance(value, str):
                    parts.append(f"{key}: {value[:500]}")
                elif isinstance(value, list):
                    parts.append(f"{key}: {len(value)} items")
            return "\n".join(parts)
        else:
            return str(output)

    def _generate_recommendations(
        self,
        analysis: TaskAnalysis,
        results: list[SystemResult],
    ) -> list[str]:
        """Generate recommendations based on results.

        Args:
            analysis: Task analysis
            results: System results

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Check for failures
        failures = [r for r in results if not r.success]
        if failures:
            recommendations.append(
                f"Consider re-running {len(failures)} failed system(s)."
            )

        # Suggest follow-up based on task type
        if analysis.complexity == "high":
            recommendations.append(
                "Consider running additional systems for deeper analysis."
            )

        # Check diversity
        if len(results) == 1:
            recommendations.append(
                "Consider running personas or argswarm for additional perspectives."
            )

        return recommendations

    def get_history_stats(self) -> dict[str, Any]:
        """Get statistics from routing history.

        Returns:
            Dictionary with history statistics
        """
        if not self.history:
            return {"total_runs": 0}

        total = len(self.history)
        successes = sum(1 for h in self.history if h["success"])

        # Count system usage
        system_counts: dict[str, int] = {}
        for h in self.history:
            for sys in h["systems"]:
                system_counts[sys] = system_counts.get(sys, 0) + 1

        return {
            "total_runs": total,
            "success_rate": successes / total,
            "system_usage": system_counts,
        }


# =============================================================================
# Convenience Functions
# =============================================================================


async def smart_run(
    prompt: str,
    provider: str = "openrouter",
    model: str = "meta-llama/llama-3.1-8b-instruct",
) -> MetaResult:
    """Convenience function for smart runs.

    Args:
        prompt: The prompt to process
        provider: LLM provider
        model: Model name

    Returns:
        MetaResult
    """
    harness = MetaHarness(provider=provider, model=model)
    return await harness.smart_run(prompt)


def analyze_task(prompt: str) -> TaskAnalysis:
    """Convenience function to analyze a task.

    Args:
        prompt: The prompt to analyze

    Returns:
        TaskAnalysis
    """
    harness = MetaHarness()
    return harness.analyze_task(prompt)
