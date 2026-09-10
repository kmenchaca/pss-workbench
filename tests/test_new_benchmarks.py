"""Tests for GSM8K and HumanEval benchmarks."""

import pytest


class TestGSM8K:
    """Tests for GSM8K benchmark."""

    def test_problems_exist(self):
        """Test that GSM8K problems are defined."""
        from pss.benchmarks.gsm8k import GSM8K_PROBLEMS

        assert len(GSM8K_PROBLEMS) >= 10
        for p in GSM8K_PROBLEMS:
            assert p.question
            assert p.answer is not None
            assert p.difficulty in ("easy", "medium", "hard")

    def test_format_prompt(self):
        """Test prompt formatting."""
        from pss.benchmarks.gsm8k import GSM8K_PROBLEMS, format_gsm8k_prompt

        problem = GSM8K_PROBLEMS[0]
        prompt = format_gsm8k_prompt(problem)

        assert problem.question in prompt
        assert "ANSWER:" in prompt

    def test_extract_answer_explicit(self):
        """Test extracting answer from explicit format."""
        from pss.benchmarks.gsm8k import extract_answer

        text = "The calculation shows that we get ANSWER: 42"
        assert extract_answer(text) == 42

        text = "Working through step by step... ANSWER: 1000"
        assert extract_answer(text) == 1000

        text = "The total is ANSWER: $125"
        assert extract_answer(text) == 125

    def test_extract_answer_final_answer(self):
        """Test extracting from 'final answer' format."""
        from pss.benchmarks.gsm8k import extract_answer

        text = "Therefore, the final answer is 24"
        assert extract_answer(text) == 24

        text = "Thus, the answer is 100"
        assert extract_answer(text) == 100

    def test_verify_solution_correct(self):
        """Test verification of correct solution."""
        from pss.benchmarks.gsm8k import GSM8K_PROBLEMS, verify_gsm8k_solution

        # Problem with answer 18
        problem = GSM8K_PROBLEMS[0]  # Janet's ducks

        output = "Janet sells 9 eggs for $2 each. ANSWER: 18"
        assert verify_gsm8k_solution(output, problem)

    def test_verify_solution_incorrect(self):
        """Test verification of incorrect solution."""
        from pss.benchmarks.gsm8k import GSM8K_PROBLEMS, verify_gsm8k_solution

        problem = GSM8K_PROBLEMS[0]  # Answer is 18

        output = "ANSWER: 100"
        assert not verify_gsm8k_solution(output, problem)

    def test_get_problems_by_difficulty(self):
        """Test filtering by difficulty."""
        from pss.benchmarks.gsm8k import get_problems_by_difficulty

        easy = get_problems_by_difficulty("easy")
        assert all(p.difficulty == "easy" for p in easy)
        assert len(easy) >= 3

        hard = get_problems_by_difficulty("hard")
        assert all(p.difficulty == "hard" for p in hard)

    def test_cost_estimation(self):
        """Test cost estimation."""
        from pss.benchmarks.gsm8k import estimate_cost

        cost = estimate_cost(10, branches_per_problem=5)
        assert cost["total_cost"] > 0
        assert cost["total_tokens"] > 0
        assert cost["exploration_cost"] > 0


class TestHumanEval:
    """Tests for HumanEval benchmark."""

    def test_problems_exist(self):
        """Test that HumanEval problems are defined."""
        from pss.benchmarks.humaneval import HUMANEVAL_PROBLEMS

        assert len(HUMANEVAL_PROBLEMS) >= 10
        for p in HUMANEVAL_PROBLEMS:
            assert p.task_id
            assert p.prompt
            assert p.canonical_solution
            assert p.test_cases
            assert p.entry_point
            assert p.difficulty in ("easy", "medium", "hard")

    def test_format_prompt(self):
        """Test prompt formatting."""
        from pss.benchmarks.humaneval import HUMANEVAL_PROBLEMS, format_humaneval_prompt

        problem = HUMANEVAL_PROBLEMS[0]
        prompt = format_humaneval_prompt(problem)

        assert problem.prompt in prompt
        assert "```python" in prompt

    def test_extract_code_from_markdown(self):
        """Test extracting code from markdown."""
        from pss.benchmarks.humaneval import extract_code, HUMANEVAL_PROBLEMS

        problem = HUMANEVAL_PROBLEMS[0]  # has_close_elements

        output = '''Here's the solution:
```python
    for i, n1 in enumerate(numbers):
        for n2 in numbers[i+1:]:
            if abs(n1 - n2) < threshold:
                return True
    return False
```'''

        code = extract_code(output, problem)
        assert code is not None
        assert "for i, n1" in code

    def test_verify_solution_correct(self):
        """Test verification of correct solution."""
        from pss.benchmarks.humaneval import HUMANEVAL_PROBLEMS, verify_humaneval_solution

        # truncate_number is simple
        problem = HUMANEVAL_PROBLEMS[2]  # truncate_number

        output = '''```python
    return number % 1.0
```'''

        assert verify_humaneval_solution(output, problem)

    def test_verify_solution_incorrect(self):
        """Test verification of incorrect solution."""
        from pss.benchmarks.humaneval import HUMANEVAL_PROBLEMS, verify_humaneval_solution

        problem = HUMANEVAL_PROBLEMS[2]  # truncate_number

        output = '''```python
    return number * 2  # wrong!
```'''

        assert not verify_humaneval_solution(output, problem)

    def test_get_problems_by_difficulty(self):
        """Test filtering by difficulty."""
        from pss.benchmarks.humaneval import get_problems_by_difficulty

        easy = get_problems_by_difficulty("easy")
        assert all(p.difficulty == "easy" for p in easy)
        assert len(easy) >= 3

    def test_cost_estimation(self):
        """Test cost estimation."""
        from pss.benchmarks.humaneval import estimate_cost

        cost = estimate_cost(10, branches_per_problem=5)
        assert cost["total_cost"] > 0
        assert cost["total_tokens"] > 0


class TestEvalTasks:
    """Test that eval tasks are properly configured."""

    def test_eval_tasks_include_new_benchmarks(self):
        """Test that EVAL_TASKS includes the new benchmarks."""
        from pss.__main__ import EVAL_TASKS

        assert "gsm8k" in EVAL_TASKS
        assert "humaneval" in EVAL_TASKS
        assert "all" in EVAL_TASKS
