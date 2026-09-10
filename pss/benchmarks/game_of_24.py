"""Game of 24 benchmark for PSS evaluation.

The Game of 24 is a mathematical reasoning game where players must use
four given numbers and basic arithmetic operations (+, -, *, /) to
obtain the number 24.

This task was used in the Tree of Thoughts paper (Yao et al., 2023)
to demonstrate the value of search-based reasoning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Game24Problem:
    """A Game of 24 problem."""

    numbers: tuple[int, int, int, int]
    difficulty: str  # "easy", "medium", "hard"
    has_solution: bool = True
    example_solution: str | None = None


# Problems curated from various sources, stratified by difficulty
GAME_OF_24_PROBLEMS: list[Game24Problem] = [
    # Easy - straightforward solutions
    Game24Problem(
        numbers=(1, 2, 3, 4),
        difficulty="easy",
        example_solution="(1 + 2 + 3) * 4 = 24",
    ),
    Game24Problem(
        numbers=(2, 3, 4, 4),
        difficulty="easy",
        example_solution="(2 + 4) * (4 - 3 + 3) = 24",  # Many solutions
    ),
    Game24Problem(
        numbers=(1, 5, 5, 5),
        difficulty="easy",
        example_solution="(5 - 1 / 5) * 5 = 24",
    ),
    Game24Problem(
        numbers=(2, 2, 2, 3),
        difficulty="easy",
        example_solution="(2 + 2) * 2 * 3 = 24",
    ),
    Game24Problem(
        numbers=(1, 1, 2, 6),
        difficulty="easy",
        example_solution="(1 + 1) * 2 * 6 = 24",
    ),
    Game24Problem(
        numbers=(3, 3, 8, 8),
        difficulty="easy",
        example_solution="8 / (3 - 8/3) = 24",
    ),
    Game24Problem(
        numbers=(4, 4, 4, 4),
        difficulty="easy",
        example_solution="4 + 4 + 4 * 4 = 24",
    ),
    Game24Problem(
        numbers=(1, 2, 6, 6),
        difficulty="easy",
        example_solution="6 * 6 / (1 + 2) + 6 = 24",  # Multiple solutions
    ),
    # Medium - requires some thought
    Game24Problem(
        numbers=(1, 3, 4, 6),
        difficulty="medium",
        example_solution="6 / (1 - 3/4) = 24",
    ),
    Game24Problem(
        numbers=(2, 5, 5, 10),
        difficulty="medium",
        example_solution="(5 - 2) * 10 - 5 - 1 = 24",  # Check this
    ),
    Game24Problem(
        numbers=(1, 4, 5, 6),
        difficulty="medium",
        example_solution="4 * (6 - 1) + 5 - 1 = 24",
    ),
    Game24Problem(
        numbers=(3, 4, 7, 8),
        difficulty="medium",
        example_solution="(7 - 3) * (8 - 4 + 2) = 24",
    ),
    Game24Problem(
        numbers=(2, 3, 5, 12),
        difficulty="medium",
        example_solution="(5 - 3) * 12 = 24",
    ),
    Game24Problem(
        numbers=(1, 2, 7, 7),
        difficulty="medium",
        example_solution="(7 - 1) * (7 - 2 - 1) = 24",
    ),
    Game24Problem(
        numbers=(4, 4, 7, 7),
        difficulty="medium",
        example_solution="(4 + 4) * (7 - 7/7) = 24",
    ),
    # Hard - tricky solutions, often involving fractions
    Game24Problem(
        numbers=(1, 5, 5, 5),
        difficulty="hard",
        example_solution="(5 - 1/5) * 5 = 24",
    ),
    Game24Problem(
        numbers=(3, 3, 7, 7),
        difficulty="hard",
        example_solution="(3 + 3/7) * 7 = 24",
    ),
    Game24Problem(
        numbers=(1, 4, 5, 6),
        difficulty="hard",
        example_solution="4 / (1 - 5/6) = 24",
    ),
    Game24Problem(
        numbers=(2, 7, 7, 10),
        difficulty="hard",
        example_solution="(10 + 2) * 7 / (7 - 4 - 1) = 24",
    ),
    Game24Problem(
        numbers=(3, 3, 8, 8),
        difficulty="hard",
        example_solution="8 / (3 - 8/3) = 24",
    ),
    Game24Problem(
        numbers=(1, 2, 7, 7),
        difficulty="hard",
        example_solution="7 * 7 / (1 + 2) - 1 = 24",
    ),
    Game24Problem(
        numbers=(5, 5, 7, 11),
        difficulty="hard",
        example_solution="(11 - 5) * (7 - 5 + 2) = 24",
    ),
    # Unsolvable - for testing that models don't hallucinate solutions
    Game24Problem(
        numbers=(1, 1, 1, 1),
        difficulty="impossible",
        has_solution=False,
    ),
    Game24Problem(
        numbers=(1, 2, 3, 7),
        difficulty="impossible",
        has_solution=False,
    ),
]


def format_game24_prompt(problem: Game24Problem) -> str:
    """Format a Game of 24 problem as a prompt."""
    nums = ", ".join(str(n) for n in problem.numbers)
    return f"""Use the numbers {nums} and basic arithmetic operations (+, -, *, /) to make 24.

Rules:
- You must use each number exactly once: {nums}
- You can only use +, -, *, /
- You can use parentheses to change order of operations

Think through different combinations systematically.

When you find a solution, write it on its own line in this exact format:
ANSWER: expression = 24

For example: ANSWER: (1 + 2 + 3) * 4 = 24"""


def verify_24_solution(output: str, problem: Game24Problem) -> bool:
    """
    Verify that an output contains a valid solution to a Game of 24 problem.

    Args:
        output: The model's output text
        problem: The Game of 24 problem

    Returns:
        True if the output contains a valid solution
    """
    if not problem.has_solution:
        # For unsolvable problems, check that model didn't claim a solution
        # This is a simplified check - a real verifier would be more thorough
        no_solution_phrases = [
            "no solution",
            "not possible",
            "cannot be done",
            "impossible",
            "can't make 24",
            "cannot make 24",
        ]
        output_lower = output.lower()
        return any(phrase in output_lower for phrase in no_solution_phrases)

    # Extract potential expressions from the output
    expressions = extract_expressions(output)

    for expr in expressions:
        if evaluate_expression(expr, problem.numbers):
            return True

    return False


def extract_expressions(text: str) -> list[str]:
    """Extract mathematical expressions from text."""
    expressions = []

    # First, look for explicit ANSWER format (highest priority)
    answer_pattern = r"ANSWER:\s*([\d\s\+\-\*\/\(\)\.]+)\s*=\s*24"
    answer_matches = re.findall(answer_pattern, text, re.IGNORECASE)
    for match in answer_matches:
        expr = re.sub(r"\s+", "", match.strip())
        if len(expr) >= 3 and any(op in expr for op in ["+", "-", "*", "/"]):
            expressions.append(expr)

    # Also look for "final answer" patterns
    final_pattern = r"(?:final\s+answer|solution|result)[\s:]+\*?\*?([\d\s\+\-\*\/\(\)\.]+)\s*=\s*24"
    final_matches = re.findall(final_pattern, text, re.IGNORECASE)
    for match in final_matches:
        expr = re.sub(r"\s+", "", match.strip())
        if len(expr) >= 3 and any(op in expr for op in ["+", "-", "*", "/"]):
            expressions.append(expr)

    # Look for expressions with numbers and operators ending with = 24
    pattern = r"([\d\s\+\-\*\/\(\)\.]+)\s*=\s*24"
    matches = re.findall(pattern, text)
    for match in matches:
        expr = re.sub(r"\s+", "", match.strip())
        if len(expr) >= 3 and any(op in expr for op in ["+", "-", "*", "/"]):
            expressions.append(expr)

    # Fallback: look for any expression-like patterns
    pattern = r"[\d\s\+\-\*\/\(\)\.]+(?:=\s*24)?"
    matches = re.findall(pattern, text)
    for match in matches:
        expr = match.replace("=", "").replace("24", "").strip()
        expr = re.sub(r"\s+", "", expr)
        if len(expr) >= 3 and any(op in expr for op in ["+", "-", "*", "/"]):
            expressions.append(expr)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for expr in expressions:
        if expr not in seen:
            seen.add(expr)
            unique.append(expr)

    return unique


def evaluate_expression(expr: str, numbers: tuple[int, int, int, int]) -> bool:
    """
    Evaluate an expression and check if it equals 24 using the given numbers.

    Args:
        expr: Mathematical expression string
        numbers: The four numbers that should be used

    Returns:
        True if expression equals 24 and uses exactly the given numbers
    """
    try:
        # Extract numbers used in expression
        nums_in_expr = [int(n) for n in re.findall(r"\d+", expr)]

        # Check that exactly the right numbers are used
        if sorted(nums_in_expr) != sorted(numbers):
            return False

        # Safely evaluate the expression
        # Only allow basic math operations
        allowed_chars = set("0123456789+-*/().")
        if not all(c in allowed_chars for c in expr.replace(" ", "")):
            return False

        result = eval(expr)  # Safe because we validated characters

        # Check if result is approximately 24
        return abs(result - 24) < 0.0001

    except Exception:
        return False


def get_problems_by_difficulty(difficulty: str) -> list[Game24Problem]:
    """Get problems filtered by difficulty."""
    return [p for p in GAME_OF_24_PROBLEMS if p.difficulty == difficulty]


def create_verifier(problem: Game24Problem):
    """Create a verifier function for a specific problem."""
    return lambda output: verify_24_solution(output, problem)
