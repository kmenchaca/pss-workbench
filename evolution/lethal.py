"""Lethal gene tracking for evolutionary code synthesis.

Tracks code patterns that cause test failures (lethal genes) to
prevent reintroducing them in future mutations. Learns from failures
across generations.
"""

import ast
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from .genome import extract_genes, parse_to_ast
from .types import Gene, GeneCategory, Genome


@dataclass
class LethalPattern:
    """A code pattern known to cause failures.

    Attributes:
        id: Unique identifier
        pattern: The problematic code pattern
        pattern_hash: Hash for quick comparison
        failure_count: How many times this caused failures
        genome_origins: IDs of genomes where this was found
        error_types: Types of errors caused
        description: Human-readable description
    """

    id: str
    pattern: str
    pattern_hash: str
    failure_count: int = 1
    genome_origins: list[str] = field(default_factory=list)
    error_types: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class FailureRecord:
    """Record of a genome failure.

    Attributes:
        genome_id: ID of the failed genome
        code: The code that failed
        error_type: Type of error
        error_message: Error message
        test_results: Which tests failed
        patterns_extracted: Lethal patterns found
    """

    genome_id: str
    code: str
    error_type: str = ""
    error_message: str = ""
    test_results: dict[str, bool] = field(default_factory=dict)
    patterns_extracted: list[str] = field(default_factory=list)


class LethalGeneDB:
    """Database of lethal genes learned from failures.

    Tracks patterns that cause test failures and filters mutations
    that would reintroduce them.
    """

    def __init__(self, min_failures: int = 2):
        """Initialize the lethal gene database.

        Args:
            min_failures: Minimum failures before marking as lethal
        """
        self.min_failures = min_failures
        self.patterns: dict[str, LethalPattern] = {}
        self.failure_history: list[FailureRecord] = []
        self.pattern_hashes: set[str] = set()

    def record_failure(self, genome: Genome, error_info: dict[str, Any] | None = None) -> None:
        """Record a genome failure and extract lethal patterns.

        Args:
            genome: The failed genome
            error_info: Optional error details
        """
        error_info = error_info or {}

        record = FailureRecord(
            genome_id=genome.id,
            code=genome.code,
            error_type=error_info.get("type", "unknown"),
            error_message=error_info.get("message", ""),
            test_results=error_info.get("test_results", {}),
        )

        # Extract potentially lethal patterns
        patterns = self._extract_patterns(genome)

        for pattern, pattern_hash in patterns:
            record.patterns_extracted.append(pattern_hash)

            if pattern_hash in self.patterns:
                # Increment existing pattern
                self.patterns[pattern_hash].failure_count += 1
                if genome.id not in self.patterns[pattern_hash].genome_origins:
                    self.patterns[pattern_hash].genome_origins.append(genome.id)
                if record.error_type and record.error_type not in self.patterns[pattern_hash].error_types:
                    self.patterns[pattern_hash].error_types.append(record.error_type)
            else:
                # New pattern
                lethal = LethalPattern(
                    id=pattern_hash[:8],
                    pattern=pattern,
                    pattern_hash=pattern_hash,
                    failure_count=1,
                    genome_origins=[genome.id],
                    error_types=[record.error_type] if record.error_type else [],
                )
                self.patterns[pattern_hash] = lethal

            # Mark as lethal if threshold reached
            if self.patterns[pattern_hash].failure_count >= self.min_failures:
                self.pattern_hashes.add(pattern_hash)

        self.failure_history.append(record)

    def _extract_patterns(self, genome: Genome) -> list[tuple[str, str]]:
        """Extract potentially lethal patterns from genome.

        Returns list of (pattern_text, pattern_hash) tuples.
        """
        patterns = []

        tree = parse_to_ast(genome.code)
        if tree is None:
            # If code doesn't parse, the whole thing is a lethal pattern
            code_hash = hashlib.sha256(genome.code.encode()).hexdigest()
            return [(genome.code, code_hash)]

        # Extract structural patterns
        for node in ast.walk(tree):
            pattern = self._node_to_pattern(node)
            if pattern:
                pattern_hash = hashlib.sha256(pattern.encode()).hexdigest()
                patterns.append((pattern, pattern_hash))

        return patterns

    def _node_to_pattern(self, node: ast.AST) -> str | None:
        """Convert AST node to a pattern string if it's significant."""
        # Focus on patterns that commonly cause issues

        if isinstance(node, ast.While):
            # Infinite loop risk
            try:
                return f"while_{ast.unparse(node.test)}"
            except Exception:
                return None

        if isinstance(node, ast.For):
            # Iteration patterns
            try:
                return f"for_{ast.unparse(node.target)}_{ast.unparse(node.iter)}"
            except Exception:
                return None

        if isinstance(node, ast.Call):
            # Recursive calls or dangerous operations
            try:
                if isinstance(node.func, ast.Name):
                    return f"call_{node.func.id}_{len(node.args)}"
            except Exception:
                return None

        if isinstance(node, ast.BinOp):
            # Division by zero risk
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                try:
                    return f"div_{ast.unparse(node.left)}_{ast.unparse(node.right)}"
                except Exception:
                    return None

        if isinstance(node, ast.Subscript):
            # Index out of bounds risk
            try:
                return f"subscript_{ast.unparse(node.value)}_{ast.unparse(node.slice)}"
            except Exception:
                return None

        return None

    def contains_lethal(self, code: str) -> bool:
        """Check if code contains known lethal patterns.

        Args:
            code: Code to check

        Returns:
            True if code contains lethal patterns
        """
        if not self.pattern_hashes:
            return False

        tree = parse_to_ast(code)
        if tree is None:
            # Unparseable code is suspect
            code_hash = hashlib.sha256(code.encode()).hexdigest()
            return code_hash in self.pattern_hashes

        # Check for lethal patterns
        for node in ast.walk(tree):
            pattern = self._node_to_pattern(node)
            if pattern:
                pattern_hash = hashlib.sha256(pattern.encode()).hexdigest()
                if pattern_hash in self.pattern_hashes:
                    return True

        return False

    def filter_mutations(self, genomes: list[Genome]) -> list[Genome]:
        """Filter out genomes containing lethal patterns.

        Args:
            genomes: Genomes to filter

        Returns:
            Genomes without lethal patterns
        """
        return [g for g in genomes if not self.contains_lethal(g.code)]

    def get_lethal_patterns(self) -> list[LethalPattern]:
        """Get all confirmed lethal patterns.

        Returns:
            List of lethal patterns above threshold
        """
        return [
            p for p in self.patterns.values()
            if p.failure_count >= self.min_failures
        ]

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about lethal gene tracking.

        Returns:
            Dictionary with stats
        """
        lethal = self.get_lethal_patterns()
        return {
            "total_patterns": len(self.patterns),
            "lethal_patterns": len(lethal),
            "total_failures": len(self.failure_history),
            "most_common_errors": self._get_common_errors(),
            "most_lethal_patterns": sorted(
                lethal,
                key=lambda p: p.failure_count,
                reverse=True,
            )[:5],
        }

    def _get_common_errors(self) -> dict[str, int]:
        """Get count of error types."""
        error_counts: dict[str, int] = {}
        for record in self.failure_history:
            if record.error_type:
                error_counts[record.error_type] = error_counts.get(record.error_type, 0) + 1
        return error_counts

    def clear(self) -> None:
        """Clear all recorded patterns and history."""
        self.patterns.clear()
        self.failure_history.clear()
        self.pattern_hashes.clear()

    def save_to_dict(self) -> dict[str, Any]:
        """Serialize database to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "min_failures": self.min_failures,
            "patterns": {
                k: {
                    "id": v.id,
                    "pattern": v.pattern,
                    "pattern_hash": v.pattern_hash,
                    "failure_count": v.failure_count,
                    "genome_origins": v.genome_origins,
                    "error_types": v.error_types,
                    "description": v.description,
                }
                for k, v in self.patterns.items()
            },
            "pattern_hashes": list(self.pattern_hashes),
        }

    def load_from_dict(self, data: dict[str, Any]) -> None:
        """Load database from dictionary.

        Args:
            data: Dictionary representation
        """
        self.min_failures = data.get("min_failures", 2)
        self.pattern_hashes = set(data.get("pattern_hashes", []))

        for k, v in data.get("patterns", {}).items():
            self.patterns[k] = LethalPattern(
                id=v["id"],
                pattern=v["pattern"],
                pattern_hash=v["pattern_hash"],
                failure_count=v["failure_count"],
                genome_origins=v.get("genome_origins", []),
                error_types=v.get("error_types", []),
                description=v.get("description", ""),
            )


def identify_failure_cause(genome: Genome, error: Exception) -> dict[str, Any]:
    """Analyze an exception to identify failure cause.

    Args:
        genome: The genome that caused the error
        error: The exception raised

    Returns:
        Dictionary with error analysis
    """
    error_type = type(error).__name__
    error_message = str(error)

    analysis: dict[str, Any] = {
        "type": error_type,
        "message": error_message,
        "likely_cause": "unknown",
    }

    # Analyze common error types
    if error_type == "IndexError":
        analysis["likely_cause"] = "out_of_bounds"
    elif error_type == "KeyError":
        analysis["likely_cause"] = "missing_key"
    elif error_type == "ZeroDivisionError":
        analysis["likely_cause"] = "division_by_zero"
    elif error_type == "RecursionError":
        analysis["likely_cause"] = "infinite_recursion"
    elif error_type == "TypeError":
        analysis["likely_cause"] = "type_mismatch"
    elif error_type == "AttributeError":
        analysis["likely_cause"] = "missing_attribute"
    elif error_type == "NameError":
        analysis["likely_cause"] = "undefined_variable"
    elif error_type == "SyntaxError":
        analysis["likely_cause"] = "syntax_error"

    return analysis


def extract_error_patterns(code: str, error: Exception) -> list[str]:
    """Extract code patterns related to an error.

    Args:
        code: Code that caused the error
        error: The error that occurred

    Returns:
        List of suspicious patterns
    """
    patterns = []
    error_type = type(error).__name__

    tree = parse_to_ast(code)
    if tree is None:
        return [code[:100]]  # Return truncated code as pattern

    for node in ast.walk(tree):
        # Look for patterns related to the error type
        if error_type == "ZeroDivisionError":
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                try:
                    patterns.append(ast.unparse(node))
                except Exception:
                    pass

        elif error_type == "IndexError":
            if isinstance(node, ast.Subscript):
                try:
                    patterns.append(ast.unparse(node))
                except Exception:
                    pass

        elif error_type == "RecursionError":
            if isinstance(node, ast.FunctionDef):
                # Check for self-calls
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name) and child.func.id == node.name:
                            try:
                                patterns.append(ast.unparse(node))
                            except Exception:
                                pass
                            break

    return patterns
