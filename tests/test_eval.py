"""Tests for the evaluation module."""

import pytest

from pss.eval import (
    pass_at_k,
    pass_at_k_from_results,
    self_bleu,
    distinct_n,
    majority_vote,
    weighted_vote,
    extract_number,
    extract_yes_no,
    extract_multiple_choice,
    evaluate_outputs,
    evaluate_with_answer_key,
    EvalResult,
    format_eval_result,
    compare_results,
)


class TestPassAtK:
    """Tests for the pass@k metric."""

    def test_all_correct(self):
        """All samples correct should give 1.0."""
        assert pass_at_k(10, 10, 1) == 1.0
        assert pass_at_k(10, 10, 5) == 1.0
        assert pass_at_k(10, 10, 10) == 1.0

    def test_none_correct(self):
        """No samples correct should give 0.0."""
        assert pass_at_k(10, 0, 1) == 0.0
        assert pass_at_k(10, 0, 5) == 0.0
        assert pass_at_k(10, 0, 10) == 0.0

    def test_one_correct_k_equals_n(self):
        """One correct out of n, with k=n, should give 1.0."""
        assert pass_at_k(10, 1, 10) == 1.0
        assert pass_at_k(5, 1, 5) == 1.0

    def test_half_correct(self):
        """Half correct should have pass@1 = 0.5."""
        result = pass_at_k(10, 5, 1)
        assert result == 0.5

    def test_k_greater_than_n(self):
        """k > n should behave correctly."""
        assert pass_at_k(3, 1, 5) == 1.0
        assert pass_at_k(3, 0, 5) == 0.0

    def test_pass_at_k_increases_with_k(self):
        """pass@k should increase as k increases."""
        p1 = pass_at_k(10, 3, 1)
        p5 = pass_at_k(10, 3, 5)
        p10 = pass_at_k(10, 3, 10)
        assert p1 < p5 < p10

    def test_known_values(self):
        """Test against known correct values from the paper."""
        # With n=10, c=3, k=5:
        # pass@5 = 1 - C(7,5)/C(10,5) = 1 - 21/252 = 0.9167
        result = pass_at_k(10, 3, 5)
        assert abs(result - 0.9167) < 0.01


class TestPassAtKFromResults:
    """Tests for pass_at_k_from_results helper."""

    def test_all_true(self):
        """All True should give (1.0, 1.0)."""
        results = [True, True, True, True]
        p1, pk = pass_at_k_from_results(results)
        assert p1 == 1.0
        assert pk == 1.0

    def test_all_false(self):
        """All False should give (0.0, 0.0)."""
        results = [False, False, False, False]
        p1, pk = pass_at_k_from_results(results)
        assert p1 == 0.0
        assert pk == 0.0

    def test_mixed_results(self):
        """Mixed results should calculate correctly."""
        results = [True, False, True, False]
        p1, pk = pass_at_k_from_results(results)
        assert p1 == 0.5
        assert pk == 1.0  # k=4, at least one True exists

    def test_custom_k(self):
        """Custom k value should work."""
        results = [True, False, False, False, False]
        p1, p2 = pass_at_k_from_results(results, k=2)
        assert p1 == pytest.approx(0.2, abs=0.01)
        assert p2 == pytest.approx(0.4, abs=0.01)


class TestSelfBleu:
    """Tests for Self-BLEU diversity metric."""

    def test_identical_outputs_high_bleu(self):
        """Identical outputs should have high Self-BLEU."""
        outputs = [
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog",
        ]
        score = self_bleu(outputs)
        assert score > 0.9

    def test_different_outputs_lower_bleu(self):
        """Different outputs should have lower Self-BLEU."""
        outputs = [
            "The quick brown fox jumps over the lazy dog",
            "A slow green turtle walks under the busy cat",
            "One tiny red bird flies through the open window",
        ]
        score = self_bleu(outputs)
        assert score < 0.5

    def test_single_output_returns_zero(self):
        """Single output should return 0."""
        outputs = ["Just one output here"]
        score = self_bleu(outputs)
        assert score == 0.0

    def test_empty_outputs(self):
        """Empty list should return 0."""
        score = self_bleu([])
        assert score == 0.0


class TestDistinctN:
    """Tests for Distinct-n metric."""

    def test_all_unique_unigrams(self):
        """All unique unigrams should give 1.0."""
        outputs = ["a b c d", "e f g h"]
        score = distinct_n(outputs, n=1)
        assert score == 1.0

    def test_all_same_unigrams(self):
        """Repeated unigrams should give low score."""
        outputs = ["a a a a", "a a a a"]
        score = distinct_n(outputs, n=1)
        assert score == pytest.approx(1/8, abs=0.01)

    def test_bigrams(self):
        """Bigram diversity."""
        outputs = ["a b c d", "a b c d"]
        score_1 = distinct_n(outputs, n=1)
        score_2 = distinct_n(outputs, n=2)
        # Same text repeated - unigrams and bigrams should be similar
        assert score_1 == pytest.approx(0.5, abs=0.01)
        assert score_2 == pytest.approx(0.5, abs=0.01)

    def test_empty_outputs(self):
        """Empty outputs should return 0."""
        assert distinct_n([], n=1) == 0.0

    def test_empty_text(self):
        """Empty text in outputs should return 0."""
        assert distinct_n(["", ""], n=1) == 0.0


class TestMajorityVote:
    """Tests for majority vote."""

    def test_clear_majority(self):
        """Clear majority should be found."""
        outputs = ["answer: 42", "answer: 42", "answer: 42", "answer: 7"]
        answer, count = majority_vote(outputs, extract_number)
        assert answer == "42"
        assert count == 3

    def test_tie(self):
        """Tie should return one of them."""
        outputs = ["answer: 42", "answer: 42", "answer: 7", "answer: 7"]
        answer, count = majority_vote(outputs, extract_number)
        assert answer in ("42", "7")
        assert count == 2

    def test_no_extractable_answers(self):
        """No extractable answers should return (None, 0)."""
        outputs = ["no numbers here", "or here"]
        answer, count = majority_vote(outputs, extract_number)
        assert answer is None
        assert count == 0


class TestWeightedVote:
    """Tests for weighted vote."""

    def test_weighted_majority(self):
        """Higher confidence should win."""
        outputs = ["answer: 42", "answer: 7"]
        confidences = [0.9, 0.1]
        answer, weight = weighted_vote(outputs, confidences, extract_number)
        assert answer == "42"
        assert weight == 0.9

    def test_accumulates_weights(self):
        """Same answer from multiple sources should accumulate."""
        outputs = ["answer: 42", "answer: 42", "answer: 7"]
        confidences = [0.3, 0.4, 0.5]
        answer, weight = weighted_vote(outputs, confidences, extract_number)
        assert answer == "42"
        assert weight == pytest.approx(0.7, abs=0.01)

    def test_mismatched_lengths(self):
        """Mismatched lengths should raise."""
        with pytest.raises(ValueError):
            weighted_vote(["a", "b"], [0.5], extract_number)


class TestExtractNumber:
    """Tests for number extraction."""

    def test_boxed_answer(self):
        """Extract from \\boxed{}."""
        text = "The answer is \\boxed{42}."
        assert extract_number(text) == "42"

    def test_answer_is_pattern(self):
        """Extract from 'answer is X'."""
        text = "The answer is 42."
        assert extract_number(text) == "42"

    def test_equals_pattern(self):
        """Extract from '= X'."""
        text = "1 + 1 = 2"
        assert extract_number(text) == "2"

    def test_trailing_number(self):
        """Extract trailing number."""
        text = "Some calculation yields 42"
        assert extract_number(text) == "42"

    def test_decimal(self):
        """Extract decimal numbers."""
        text = "The result is 3.14"
        assert extract_number(text) == "3.14"

    def test_negative(self):
        """Extract negative numbers."""
        text = "The answer is -5"
        assert extract_number(text) == "-5"

    def test_no_number(self):
        """No number should return None."""
        text = "No numbers in this text"
        assert extract_number(text) is None


class TestExtractYesNo:
    """Tests for yes/no extraction."""

    def test_yes(self):
        """Extract yes."""
        assert extract_yes_no("Yes, that is correct") == "yes"
        assert extract_yes_no("That's true") == "yes"
        assert extract_yes_no("Correct!") == "yes"

    def test_no(self):
        """Extract no."""
        assert extract_yes_no("No, that is wrong") == "no"
        assert extract_yes_no("That's false") == "no"
        assert extract_yes_no("Incorrect") == "no"

    def test_ambiguous(self):
        """Ambiguous should return None."""
        text = "Yes and no, it depends"
        assert extract_yes_no(text) is None

    def test_neither(self):
        """Neither yes nor no should return None."""
        text = "I'm not sure about that"
        assert extract_yes_no(text) is None


class TestExtractMultipleChoice:
    """Tests for multiple choice extraction."""

    def test_answer_is_pattern(self):
        """Extract from 'answer is X'."""
        assert extract_multiple_choice("The answer is A") == "A"
        assert extract_multiple_choice("answer: B") == "B"

    def test_option_pattern(self):
        """Extract from 'option X'."""
        assert extract_multiple_choice("I choose option C") == "C"

    def test_trailing_letter(self):
        """Extract trailing letter."""
        assert extract_multiple_choice("Based on the analysis, D") == "D"

    def test_lowercase(self):
        """Lowercase should be converted."""
        assert extract_multiple_choice("The answer is a") == "A"

    def test_with_parentheses(self):
        """Parentheses should work."""
        assert extract_multiple_choice("The answer is (B)") == "B"

    def test_no_answer(self):
        """No answer should return None."""
        assert extract_multiple_choice("I don't know") is None


class TestEvaluateOutputs:
    """Tests for evaluate_outputs function."""

    def test_basic_evaluation(self):
        """Basic evaluation with simple verifier."""
        outputs = ["correct", "wrong", "correct", "wrong"]
        verifier = lambda x: "correct" in x

        result = evaluate_outputs(
            outputs, verifier,
            task="test_task",
            method="test_method",
            compute_diversity=False,
        )

        assert result.task == "test_task"
        assert result.method == "test_method"
        assert result.num_samples == 4
        assert result.num_correct == 2
        assert result.pass_at_1 == 0.5
        assert result.pass_at_k == 1.0

    def test_with_cost(self):
        """Evaluation with cost tracking."""
        outputs = ["output1", "output2"]
        verifier = lambda x: True

        result = evaluate_outputs(
            outputs, verifier,
            total_cost=0.01,
            total_tokens=1000,
            compute_diversity=False,
        )

        assert result.total_cost == 0.01
        assert result.total_tokens == 1000


class TestEvaluateWithAnswerKey:
    """Tests for evaluate_with_answer_key function."""

    def test_correct_extraction(self):
        """Correct answer extraction and comparison."""
        outputs = [
            "The answer is 42",
            "Therefore, the result is 42",
            "Maybe 24?",
            "Final answer: 42",
        ]

        result = evaluate_with_answer_key(
            outputs,
            correct_answer="42",
            extract_answer=extract_number,
            task="math",
            method="pss",
        )

        assert result.num_correct == 3
        assert result.majority_vote_correct is True


class TestFormatEvalResult:
    """Tests for result formatting."""

    def test_basic_format(self):
        """Basic formatting includes key metrics."""
        result = EvalResult(
            task="test",
            method="pss",
            num_samples=10,
            num_correct=7,
            pass_at_1=0.7,
            pass_at_k=0.95,
        )

        formatted = format_eval_result(result)
        assert "test" in formatted
        assert "pss" in formatted
        assert "70.00%" in formatted or "0.70" in formatted


class TestCompareResults:
    """Tests for comparing multiple results."""

    def test_comparison_table(self):
        """Comparison should produce readable table."""
        results = [
            EvalResult(
                task="math", method="pss", num_samples=10,
                num_correct=8, pass_at_1=0.8, pass_at_k=1.0,
            ),
            EvalResult(
                task="math", method="single_shot", num_samples=10,
                num_correct=5, pass_at_1=0.5, pass_at_k=0.9,
            ),
        ]

        comparison = compare_results(results)
        assert "pss" in comparison
        assert "single_shot" in comparison
        assert "Comparison" in comparison

    def test_empty_results(self):
        """Empty results should handle gracefully."""
        comparison = compare_results([])
        assert "No results" in comparison
