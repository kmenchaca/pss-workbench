"""Fitness functions framework for evolutionary code synthesis.

Provides multi-objective fitness evaluation with:
- TestFitness: Do tests pass?
- PerformanceFitness: Benchmark speed/memory
- QualityFitness: Code quality metrics
- NoveltyFitness: Uniqueness vs population
- CompositeFitness: Weighted multi-objective
"""

import ast
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .genome import genome_distance
from .types import FitnessResult, Genome, Population


class FitnessFunction(ABC):
    """Base class for fitness functions.

    Subclasses implement specific fitness dimensions like
    test passing, performance, or code quality.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this fitness function."""
        ...

    @abstractmethod
    def evaluate(self, genome: Genome, **kwargs: Any) -> float:
        """Evaluate fitness of a genome.

        Args:
            genome: The genome to evaluate
            **kwargs: Additional context (tests, population, etc.)

        Returns:
            Fitness score (higher is better, typically 0-1)
        """
        ...

    def batch_evaluate(
        self, genomes: list[Genome], **kwargs: Any
    ) -> dict[str, float]:
        """Evaluate multiple genomes.

        Default implementation calls evaluate() for each.
        Override for batch optimizations.

        Args:
            genomes: Genomes to evaluate
            **kwargs: Additional context

        Returns:
            Dict mapping genome ID to fitness score
        """
        return {g.id: self.evaluate(g, **kwargs) for g in genomes}


class TestCaseFitness(FitnessFunction):
    """Fitness based on test case passing.

    Evaluates what fraction of provided tests pass.
    """

    @property
    def name(self) -> str:
        return "tests_passed"

    def __init__(self, tests: list[dict[str, Any]] | None = None):
        """Initialize with test cases.

        Args:
            tests: List of test cases, each with:
                - 'input': Input arguments
                - 'expected': Expected output
                - 'function': Function name to test (optional)
        """
        self.tests = tests or []

    def evaluate(self, genome: Genome, **kwargs: Any) -> float:
        """Run tests and return fraction passing."""
        tests = kwargs.get("tests", self.tests)
        if not tests:
            return 1.0  # No tests = assume passing

        passed = 0
        total = len(tests)

        for test in tests:
            try:
                if self._run_test(genome, test):
                    passed += 1
            except Exception:
                pass  # Test failed

        return passed / total if total > 0 else 1.0

    def _run_test(self, genome: Genome, test: dict[str, Any]) -> bool:
        """Run a single test case.

        Args:
            genome: Genome containing code to test
            test: Test specification

        Returns:
            True if test passes
        """
        # Create isolated execution environment
        namespace: dict[str, Any] = {}

        try:
            exec(genome.code, namespace)
        except Exception:
            return False

        # Get function to test
        func_name = test.get("function")
        if func_name is None:
            # Find first function defined
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith("_"):
                    func_name = name
                    break

        if func_name is None or func_name not in namespace:
            return False

        func = namespace[func_name]

        # Run test
        try:
            input_args = test.get("input", [])
            input_kwargs = test.get("input_kwargs", {})

            if isinstance(input_args, (list, tuple)):
                result = func(*input_args, **input_kwargs)
            else:
                result = func(input_args, **input_kwargs)

            expected = test.get("expected")
            return result == expected

        except Exception:
            return False


class PerformanceFitness(FitnessFunction):
    """Fitness based on execution speed and memory.

    Benchmarks the code and normalizes to a score.
    """

    @property
    def name(self) -> str:
        return "performance"

    def __init__(
        self,
        benchmark_inputs: list[Any] | None = None,
        timeout: float = 5.0,
        target_time: float = 0.1,
    ):
        """Initialize performance fitness.

        Args:
            benchmark_inputs: Inputs to use for benchmarking
            timeout: Max execution time in seconds
            target_time: Target execution time for perfect score
        """
        self.benchmark_inputs = benchmark_inputs or []
        self.timeout = timeout
        self.target_time = target_time

    def evaluate(self, genome: Genome, **kwargs: Any) -> float:
        """Benchmark code and return performance score."""
        inputs = kwargs.get("benchmark_inputs", self.benchmark_inputs)

        if not inputs:
            return 0.5  # Neutral score if no benchmarks

        namespace: dict[str, Any] = {}
        try:
            exec(genome.code, namespace)
        except Exception:
            return 0.0

        # Find function to benchmark
        func = None
        for name, obj in namespace.items():
            if callable(obj) and not name.startswith("_"):
                func = obj
                break

        if func is None:
            return 0.0

        # Benchmark
        total_time = 0.0
        num_runs = 0

        for inp in inputs:
            try:
                start = time.perf_counter()

                if isinstance(inp, (list, tuple)):
                    func(*inp)
                else:
                    func(inp)

                elapsed = time.perf_counter() - start

                if elapsed > self.timeout:
                    return 0.0  # Timeout penalty

                total_time += elapsed
                num_runs += 1

            except Exception:
                return 0.0

        if num_runs == 0:
            return 0.0

        avg_time = total_time / num_runs

        # Convert time to score (lower time = higher score)
        # Score of 1.0 at target_time, asymptotically approaching 0 for slow code
        score = self.target_time / (avg_time + self.target_time)

        return min(1.0, score)


class QualityFitness(FitnessFunction):
    """Fitness based on code quality metrics.

    Evaluates:
    - Cyclomatic complexity
    - Lines of code
    - Naming conventions
    - Documentation
    """

    @property
    def name(self) -> str:
        return "quality"

    def __init__(
        self,
        max_complexity: int = 20,
        max_lines: int = 200,
        require_docstrings: bool = False,
    ):
        """Initialize quality fitness.

        Args:
            max_complexity: Maximum acceptable cyclomatic complexity
            max_lines: Maximum lines of code
            require_docstrings: Whether to penalize missing docstrings
        """
        self.max_complexity = max_complexity
        self.max_lines = max_lines
        self.require_docstrings = require_docstrings

    def evaluate(self, genome: Genome, **kwargs: Any) -> float:
        """Calculate code quality score."""
        tree = ast.parse(genome.code) if genome.code else None
        if tree is None:
            return 0.0

        scores = []

        # Complexity score
        complexity = self._calculate_complexity(tree)
        complexity_score = max(0, 1 - complexity / self.max_complexity)
        scores.append(complexity_score)

        # Length score
        lines = len(genome.code.split("\n"))
        length_score = max(0, 1 - lines / self.max_lines)
        scores.append(length_score)

        # Naming score
        naming_score = self._evaluate_naming(tree)
        scores.append(naming_score)

        # Docstring score
        if self.require_docstrings:
            docstring_score = self._evaluate_docstrings(tree)
            scores.append(docstring_score)

        return sum(scores) / len(scores)

    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            # Branches add complexity
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1

        return complexity

    def _evaluate_naming(self, tree: ast.AST) -> float:
        """Evaluate naming conventions."""
        good_names = 0
        total_names = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_names += 1
                # snake_case for functions
                if node.name.islower() or "_" in node.name:
                    good_names += 1

            elif isinstance(node, ast.ClassDef):
                total_names += 1
                # PascalCase for classes
                if node.name[0].isupper():
                    good_names += 1

            elif isinstance(node, ast.Name):
                total_names += 1
                name = node.id
                # Allow single letters, snake_case, UPPER_CASE
                if (
                    len(name) <= 2
                    or name.islower()
                    or name.isupper()
                    or "_" in name
                ):
                    good_names += 1

        return good_names / total_names if total_names > 0 else 1.0

    def _evaluate_docstrings(self, tree: ast.AST) -> float:
        """Evaluate docstring coverage."""
        has_docstring = 0
        total = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                total += 1
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                ):
                    has_docstring += 1

        return has_docstring / total if total > 0 else 1.0


class NoveltyFitness(FitnessFunction):
    """Fitness based on uniqueness vs population.

    Rewards genomes that are structurally different from
    the rest of the population.
    """

    @property
    def name(self) -> str:
        return "novelty"

    def __init__(self, k_nearest: int = 5):
        """Initialize novelty fitness.

        Args:
            k_nearest: Number of nearest neighbors to consider
        """
        self.k_nearest = k_nearest

    def evaluate(self, genome: Genome, **kwargs: Any) -> float:
        """Calculate novelty based on population distance."""
        population = kwargs.get("population")
        if population is None or not population.genomes:
            return 1.0  # Novel if no population

        others = [g for g in population.genomes if g.id != genome.id]
        if not others:
            return 1.0

        # Calculate distances to all others
        distances = [genome_distance(genome, other) for other in others]

        # Average distance to k nearest
        k = min(self.k_nearest, len(distances))
        distances.sort()
        avg_distance = sum(distances[:k]) / k

        return avg_distance  # Distance is already 0-1


class CompositeFitness(FitnessFunction):
    """Weighted multi-objective fitness.

    Combines multiple fitness functions with configurable weights.
    """

    @property
    def name(self) -> str:
        return "composite"

    def __init__(
        self,
        functions: list[tuple[FitnessFunction, float]] | None = None,
    ):
        """Initialize composite fitness.

        Args:
            functions: List of (fitness_function, weight) tuples
        """
        self.functions = functions or []

    def add_function(self, func: FitnessFunction, weight: float) -> None:
        """Add a fitness function with weight."""
        self.functions.append((func, weight))

    def evaluate(self, genome: Genome, **kwargs: Any) -> float:
        """Calculate weighted composite fitness."""
        if not self.functions:
            return 0.5

        total_weight = sum(w for _, w in self.functions)
        if total_weight == 0:
            return 0.5

        weighted_sum = 0.0
        for func, weight in self.functions:
            score = func.evaluate(genome, **kwargs)
            weighted_sum += score * weight

        return weighted_sum / total_weight


def evaluate_genome(
    genome: Genome,
    fitness_functions: list[FitnessFunction],
    **kwargs: Any,
) -> FitnessResult:
    """Evaluate a genome with multiple fitness functions.

    Args:
        genome: Genome to evaluate
        fitness_functions: List of fitness functions to apply
        **kwargs: Context passed to fitness functions

    Returns:
        FitnessResult with all scores
    """
    result = FitnessResult()

    for func in fitness_functions:
        score = func.evaluate(genome, **kwargs)
        result.raw_scores[func.name] = score

        # Map to standard fields if applicable
        if func.name == "tests_passed":
            result.tests_passed = score
        elif func.name == "performance":
            result.performance = score
        elif func.name == "quality":
            result.quality = score
        elif func.name == "novelty":
            result.novelty = score

    return result


def pareto_rank(genomes: list[Genome]) -> dict[str, int]:
    """Calculate Pareto rank for each genome.

    Rank 0 = Pareto optimal (not dominated by any other)
    Rank 1 = Dominated only by rank 0
    etc.

    Args:
        genomes: Genomes to rank

    Returns:
        Dict mapping genome ID to Pareto rank
    """
    evaluated = [g for g in genomes if g.fitness_scores is not None]
    if not evaluated:
        return {}

    ranks: dict[str, int] = {}
    remaining = set(g.id for g in evaluated)
    current_rank = 0

    while remaining:
        # Find non-dominated in remaining
        non_dominated = []

        for gid in remaining:
            genome = next(g for g in evaluated if g.id == gid)
            is_dominated = False

            for other_id in remaining:
                if other_id == gid:
                    continue
                other = next(g for g in evaluated if g.id == other_id)

                if other.fitness_scores and genome.fitness_scores:
                    if other.fitness_scores.dominates(genome.fitness_scores):
                        is_dominated = True
                        break

            if not is_dominated:
                non_dominated.append(gid)

        # Assign rank to non-dominated
        for gid in non_dominated:
            ranks[gid] = current_rank
            remaining.remove(gid)

        current_rank += 1

    return ranks


def create_default_fitness_suite(
    tests: list[dict[str, Any]] | None = None,
    benchmark_inputs: list[Any] | None = None,
) -> list[FitnessFunction]:
    """Create a standard fitness evaluation suite.

    Args:
        tests: Test cases for TestFitness
        benchmark_inputs: Inputs for PerformanceFitness

    Returns:
        List of fitness functions
    """
    return [
        TestFitness(tests),
        PerformanceFitness(benchmark_inputs),
        QualityFitness(),
        NoveltyFitness(),
    ]
