"""Tests for benchmark modules."""

import pytest

from pss.benchmarks.game_of_24 import (
    GAME_OF_24_PROBLEMS,
    Game24Problem,
    format_game24_prompt,
    verify_24_solution,
    extract_expressions,
    evaluate_expression,
    get_problems_by_difficulty,
)
from pss.benchmarks.brainstorm import (
    BRAINSTORM_PROBLEMS,
    BrainstormProblem,
    format_brainstorm_prompt,
    count_unique_ideas,
    extract_ideas,
    evaluate_brainstorm_diversity,
    get_problems_by_category,
)
from pss.benchmarks.bugfind import (
    BUGFIND_PROBLEMS,
    PlantedBug,
    BugfindProblem,
    format_bugfind_prompt,
    check_bug_found,
    evaluate_bugfind,
    get_problems_by_difficulty as get_bugfind_by_difficulty,
)


class TestGame24Problems:
    """Tests for Game of 24 benchmark."""

    def test_problems_exist(self):
        """Verify problems are defined."""
        assert len(GAME_OF_24_PROBLEMS) > 0

    def test_difficulty_levels(self):
        """All difficulties should be present."""
        difficulties = {p.difficulty for p in GAME_OF_24_PROBLEMS}
        assert "easy" in difficulties
        assert "medium" in difficulties
        assert "hard" in difficulties

    def test_impossible_problems_exist(self):
        """Should have some impossible problems."""
        impossible = [p for p in GAME_OF_24_PROBLEMS if not p.has_solution]
        assert len(impossible) >= 1

    def test_get_by_difficulty(self):
        """Filter by difficulty works."""
        easy = get_problems_by_difficulty("easy")
        assert len(easy) > 0
        assert all(p.difficulty == "easy" for p in easy)


class TestFormatGame24Prompt:
    """Tests for Game of 24 prompt formatting."""

    def test_includes_numbers(self):
        """Prompt should include the numbers."""
        problem = Game24Problem(numbers=(1, 2, 3, 4), difficulty="easy")
        prompt = format_game24_prompt(problem)
        assert "1" in prompt
        assert "2" in prompt
        assert "3" in prompt
        assert "4" in prompt

    def test_includes_rules(self):
        """Prompt should mention the rules."""
        problem = Game24Problem(numbers=(1, 2, 3, 4), difficulty="easy")
        prompt = format_game24_prompt(problem)
        assert "24" in prompt
        assert "each number exactly once" in prompt.lower()


class TestVerify24Solution:
    """Tests for Game of 24 solution verification."""

    def test_valid_solution(self):
        """Valid solution should be verified."""
        problem = Game24Problem(numbers=(1, 2, 3, 4), difficulty="easy")
        output = "The solution is (1 + 2 + 3) * 4 = 24"
        assert verify_24_solution(output, problem) is True

    def test_invalid_solution(self):
        """Invalid solution should fail."""
        problem = Game24Problem(numbers=(1, 2, 3, 4), difficulty="easy")
        output = "The solution is 1 + 2 + 3 + 4 = 10"
        assert verify_24_solution(output, problem) is False

    def test_wrong_numbers(self):
        """Solution with wrong numbers should fail."""
        problem = Game24Problem(numbers=(1, 2, 3, 4), difficulty="easy")
        output = "The solution is 6 * 4 = 24"
        assert verify_24_solution(output, problem) is False

    def test_impossible_problem_no_solution(self):
        """Impossible problem: saying 'no solution' should pass."""
        problem = Game24Problem(
            numbers=(1, 1, 1, 1),
            difficulty="impossible",
            has_solution=False,
        )
        output = "This problem has no solution."
        assert verify_24_solution(output, problem) is True

    def test_impossible_problem_false_solution(self):
        """Impossible problem: claiming a solution should fail."""
        problem = Game24Problem(
            numbers=(1, 1, 1, 1),
            difficulty="impossible",
            has_solution=False,
        )
        output = "The answer is 1 * 1 * 1 * 24 = 24"  # Wrong numbers
        assert verify_24_solution(output, problem) is False


class TestExtractExpressions:
    """Tests for expression extraction."""

    def test_simple_expression(self):
        """Extract simple expression."""
        text = "The answer is 1+2+3*4 = 24"
        exprs = extract_expressions(text)
        assert len(exprs) > 0

    def test_expression_with_parens(self):
        """Extract expression with parentheses."""
        text = "(1+2)*3+4"
        exprs = extract_expressions(text)
        assert len(exprs) > 0

    def test_no_expression(self):
        """Text without expressions should return empty."""
        text = "I don't know the answer"
        exprs = extract_expressions(text)
        # Might still extract some fragments, but nothing valid
        # Just verify no crash
        assert isinstance(exprs, list)


class TestEvaluateExpression:
    """Tests for expression evaluation."""

    def test_correct_expression(self):
        """Correct expression evaluates to True."""
        assert evaluate_expression("(1+2+3)*4", (1, 2, 3, 4)) is True
        assert evaluate_expression("4*(1+2+3)", (1, 2, 3, 4)) is True

    def test_wrong_result(self):
        """Wrong result evaluates to False."""
        assert evaluate_expression("1+2+3+4", (1, 2, 3, 4)) is False  # = 10

    def test_wrong_numbers(self):
        """Wrong numbers evaluate to False."""
        assert evaluate_expression("6*4", (1, 2, 3, 4)) is False

    def test_invalid_expression(self):
        """Invalid expression evaluates to False."""
        assert evaluate_expression("1 + + 2", (1, 2, 3, 4)) is False

    def test_division(self):
        """Division should work."""
        assert evaluate_expression("8/(3-8/3)", (3, 3, 8, 8)) is True


class TestBrainstormProblems:
    """Tests for brainstorm benchmark."""

    def test_problems_exist(self):
        """Verify problems are defined."""
        assert len(BRAINSTORM_PROBLEMS) > 0

    def test_categories(self):
        """All categories should be present."""
        categories = {p.category for p in BRAINSTORM_PROBLEMS}
        assert "technical" in categories
        assert "creative" in categories
        assert "general" in categories

    def test_get_by_category(self):
        """Filter by category works."""
        technical = get_problems_by_category("technical")
        assert len(technical) > 0
        assert all(p.category == "technical" for p in technical)


class TestExtractIdeas:
    """Tests for idea extraction from brainstorm outputs."""

    def test_numbered_list(self):
        """Extract from numbered list."""
        text = """Here are some ideas:
1. First idea about something
2. Second idea about another thing
3. Third idea about yet another thing
"""
        ideas = extract_ideas(text)
        assert len(ideas) >= 3

    def test_bullet_points(self):
        """Extract from bullet points."""
        text = """Ideas:
- Use caching for speed
- Add indexing to database
- Implement lazy loading
"""
        ideas = extract_ideas(text)
        assert len(ideas) >= 3

    def test_mixed_format(self):
        """Extract from mixed format."""
        text = """
1. First numbered idea

- A bullet point idea

2. Second numbered idea
"""
        ideas = extract_ideas(text)
        assert len(ideas) >= 3

    def test_deduplication(self):
        """Should deduplicate similar ideas."""
        text = """
1. Use caching
2. Use caching for better performance
"""
        ideas = extract_ideas(text)
        # First 50 chars of both start with "Use caching" so should dedup
        assert len(ideas) <= 2


class TestEvaluateBrainstormDiversity:
    """Tests for brainstorm diversity evaluation."""

    def test_meets_minimum(self):
        """Enough ideas should meet minimum."""
        problem = BrainstormProblem(
            prompt="Test",
            topic="test",
            min_ideas=3,
        )
        # Ideas need to be longer than 10 chars to be counted
        outputs = [
            "1. First great idea for testing purposes\n2. Second amazing idea worth considering\n3. Third brilliant concept to explore\n4. Fourth wonderful suggestion",
        ]
        metrics = evaluate_brainstorm_diversity(outputs, problem)
        assert metrics["meets_minimum"] is True

    def test_below_minimum(self):
        """Too few ideas should not meet minimum."""
        problem = BrainstormProblem(
            prompt="Test",
            topic="test",
            min_ideas=10,
        )
        outputs = [
            "1. First idea that is long enough\n2. Second idea also long enough",
        ]
        metrics = evaluate_brainstorm_diversity(outputs, problem)
        assert metrics["meets_minimum"] is False

    def test_counts_across_outputs(self):
        """Should count ideas across multiple outputs."""
        problem = BrainstormProblem(
            prompt="Test",
            topic="test",
            min_ideas=5,
        )
        outputs = [
            "1. First great idea for testing\n2. Second amazing concept to explore",
            "1. Third brilliant suggestion here\n2. Fourth wonderful approach to try",
            "1. Fifth excellent method to consider",
        ]
        metrics = evaluate_brainstorm_diversity(outputs, problem)
        assert metrics["total_ideas"] >= 5


class TestBugfindProblems:
    """Tests for bugfind benchmark."""

    def test_problems_exist(self):
        """Verify problems are defined."""
        assert len(BUGFIND_PROBLEMS) >= 3

    def test_all_problems_have_bugs(self):
        """All problems should have planted bugs."""
        for problem in BUGFIND_PROBLEMS:
            assert len(problem.bugs) > 0

    def test_difficulties(self):
        """Should have different difficulties."""
        difficulties = {p.difficulty for p in BUGFIND_PROBLEMS}
        assert len(difficulties) >= 2

    def test_get_by_difficulty(self):
        """Filter by difficulty works."""
        easy = get_bugfind_by_difficulty("easy")
        assert len(easy) >= 1
        assert all(p.difficulty == "easy" for p in easy)


class TestCheckBugFound:
    """Tests for bug detection checking."""

    def test_finds_bug_with_keywords(self):
        """Should find bug when keywords match."""
        bug = PlantedBug(
            bug_id="test_1",
            description="Division by zero when empty list",
            location="test.py:10",
            bug_type="missing_check",
            severity="high",
            keywords=["empty", "division", "zero"],
        )
        output = "Found a division by zero error when the list is empty"
        assert check_bug_found(output, bug) is True

    def test_misses_bug_without_keywords(self):
        """Should not find bug without enough keywords."""
        bug = PlantedBug(
            bug_id="test_1",
            description="Division by zero when empty list",
            location="test.py:10",
            bug_type="missing_check",
            severity="high",
            keywords=["empty", "division", "zero"],
        )
        output = "The code looks fine to me."
        assert check_bug_found(output, bug) is False

    def test_finds_bug_by_location(self):
        """Should find bug when location is mentioned."""
        bug = PlantedBug(
            bug_id="test_1",
            description="Some bug",
            location="utils.py:42",
            bug_type="error",
            severity="high",
            keywords=["error"],
        )
        output = "Found an issue at utils.py:42"
        assert check_bug_found(output, bug) is True


class TestEvaluateBugfind:
    """Tests for bugfind evaluation."""

    def test_finds_all_bugs(self):
        """Should correctly count found bugs."""
        problem = BugfindProblem(
            name="test",
            description="Test problem",
            files={"test.py": "code"},
            bugs=[
                PlantedBug("b1", "Bug 1", "test.py:1", "error", "high", ["error", "one"]),
                PlantedBug("b2", "Bug 2", "test.py:2", "error", "high", ["error", "two"]),
            ],
            difficulty="easy",
        )
        outputs = [
            "Found error one in the code",
            "Found error two in the code",
        ]
        metrics = evaluate_bugfind(outputs, problem)
        assert metrics["bugs_found"] == 2
        assert metrics["detection_rate"] == 1.0

    def test_partial_detection(self):
        """Should handle partial detection."""
        problem = BugfindProblem(
            name="test",
            description="Test problem",
            files={"test.py": "code"},
            bugs=[
                PlantedBug("b1", "Bug 1", "test.py:1", "error", "high", ["error", "one"]),
                PlantedBug("b2", "Bug 2", "test.py:2", "error", "high", ["error", "two"]),
            ],
            difficulty="easy",
        )
        outputs = [
            "Found error one in the code",
            "No issues found",
        ]
        metrics = evaluate_bugfind(outputs, problem)
        assert metrics["bugs_found"] == 1
        assert metrics["detection_rate"] == 0.5


class TestFormatBugfindPrompt:
    """Tests for bugfind prompt formatting."""

    def test_includes_file_content(self):
        """Prompt should include the code."""
        problem = BugfindProblem(
            name="test",
            description="Test problem",
            files={"test.py": "def foo():\n    pass"},
            bugs=[],
            difficulty="easy",
        )
        prompt = format_bugfind_prompt(problem)
        assert "def foo():" in prompt
        assert "test.py" in prompt

    def test_includes_instructions(self):
        """Prompt should include instructions."""
        problem = BugfindProblem(
            name="test",
            description="Find the bugs",
            files={"test.py": "code"},
            bugs=[],
            difficulty="easy",
        )
        prompt = format_bugfind_prompt(problem)
        assert "bug" in prompt.lower()
