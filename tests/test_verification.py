"""Tests for verification strategies."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from pss.config import PSSConfig
from pss.verification import (
    VerificationResult,
    checkbox_verify,
    combined_verify,
    create_eval_fn,
    keyword_verify,
    length_verify,
    programmatic_verify,
)


class TestCheckboxVerify:
    """Tests for checkbox_verify."""

    def test_all_complete(self):
        output = """
        - [x] Task 1
        - [x] Task 2
        - [x] Task 3
        """
        result = checkbox_verify(output)
        assert result.passed is True
        assert result.score == 1.0

    def test_none_complete(self):
        output = """
        - [ ] Task 1
        - [ ] Task 2
        - [ ] Task 3
        """
        result = checkbox_verify(output)
        assert result.passed is False
        assert result.score == 0.0

    def test_partial_complete(self):
        output = """
        - [x] Task 1
        - [ ] Task 2
        - [x] Task 3
        - [ ] Task 4
        """
        result = checkbox_verify(output)
        assert result.passed is False
        assert result.score == 0.5
        assert "2/4" in result.reason

    def test_no_checkboxes(self):
        output = "This text has no checkboxes."
        result = checkbox_verify(output)
        assert result.passed is True
        assert result.score == 1.0

    def test_uppercase_x(self):
        output = "- [X] Done task"
        result = checkbox_verify(output)
        assert result.passed is True
        assert result.score == 1.0


class TestKeywordVerify:
    """Tests for keyword_verify."""

    def test_all_required_present(self):
        output = "The function returns a string value."
        result = keyword_verify(
            output, required_keywords=["function", "returns", "string"]
        )
        assert result.passed is True
        assert result.score == 1.0

    def test_some_required_missing(self):
        output = "The function returns a value."
        result = keyword_verify(
            output, required_keywords=["function", "returns", "string"]
        )
        assert result.score == pytest.approx(2 / 3)

    def test_forbidden_present(self):
        output = "TODO: implement this function"
        result = keyword_verify(
            output,
            required_keywords=["function"],
            forbidden_keywords=["TODO", "FIXME"],
        )
        assert result.passed is False
        assert result.score < 1.0

    def test_case_insensitive(self):
        output = "The FUNCTION returns a STRING."
        result = keyword_verify(output, required_keywords=["function", "string"])
        assert result.passed is True
        assert result.score == 1.0

    def test_empty_required(self):
        output = "Any text here."
        result = keyword_verify(output, required_keywords=[])
        assert result.passed is True
        assert result.score == 1.0


class TestLengthVerify:
    """Tests for length_verify."""

    def test_within_bounds(self):
        output = "This is a test output with enough words."
        result = length_verify(output, min_words=5, max_words=20)
        assert result.passed is True
        assert result.score == 1.0

    def test_too_short(self):
        output = "Short"
        result = length_verify(output, min_words=10)
        assert result.passed is False
        assert result.score == 0.0
        assert "Too short" in result.reason

    def test_too_long(self):
        output = " ".join(["word"] * 100)
        result = length_verify(output, max_words=50)
        assert result.passed is False
        assert result.score == 0.0
        assert "Too long" in result.reason

    def test_char_limits(self):
        output = "abc"
        result = length_verify(output, min_chars=10)
        assert result.passed is False

        result = length_verify(output, max_chars=5)
        assert result.passed is True


class TestProgrammaticVerify:
    """Tests for programmatic_verify."""

    @patch("subprocess.run")
    def test_tests_pass(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        result = programmatic_verify("output", "pytest")
        assert result.passed is True
        assert result.score == 1.0

    @patch("subprocess.run")
    def test_tests_fail(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="AssertionError")
        result = programmatic_verify("output", "pytest")
        assert result.passed is False
        assert result.score == 0.0

    @patch("subprocess.run")
    def test_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired("pytest", 60)
        result = programmatic_verify("output", "pytest", timeout=60)
        assert result.passed is False
        assert result.score == 0.0
        assert "timed out" in result.reason

    @patch("subprocess.run")
    def test_execution_error(self, mock_run):
        mock_run.side_effect = OSError("Command not found")
        result = programmatic_verify("output", "nonexistent_command")
        assert result.passed is False
        assert result.score == 0.0


class TestCombinedVerify:
    """Tests for combined_verify."""

    def test_weighted_average(self):
        def always_pass(output):
            return VerificationResult(passed=True, score=1.0, reason="pass")

        def always_fail(output):
            return VerificationResult(passed=False, score=0.0, reason="fail")

        # 50% weight to pass, 50% to fail = 0.5 average
        result = combined_verify(
            "test",
            [(always_pass, 1.0), (always_fail, 1.0)],
        )
        assert result.score == 0.5

    def test_unequal_weights(self):
        def score_75(output):
            return VerificationResult(passed=True, score=0.75, reason="ok")

        def score_25(output):
            return VerificationResult(passed=False, score=0.25, reason="meh")

        # Weight 3:1 toward score_75
        result = combined_verify(
            "test",
            [(score_75, 3.0), (score_25, 1.0)],
        )
        # (0.75 * 3 + 0.25 * 1) / 4 = 2.5 / 4 = 0.625
        assert result.score == pytest.approx(0.625)

    def test_empty_verifiers(self):
        result = combined_verify("test", [])
        assert result.passed is True
        assert result.score == 1.0


class TestCreateEvalFn:
    """Tests for create_eval_fn factory."""

    def test_none_strategy(self):
        config = PSSConfig(verification_strategy="none")
        eval_fn = create_eval_fn(config)
        assert eval_fn is None

    def test_checkbox_strategy(self):
        config = PSSConfig(verification_strategy="checkbox")
        eval_fn = create_eval_fn(config)
        assert eval_fn is not None

        # Test it works
        score = eval_fn("- [x] Done\n- [ ] Todo")
        assert score == 0.5

    @patch("subprocess.run")
    def test_programmatic_strategy(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)

        config = PSSConfig(verification_strategy="programmatic")
        eval_fn = create_eval_fn(config)
        assert eval_fn is not None

        score = eval_fn("any output")
        assert score == 1.0

    def test_reviewer_strategy_placeholder(self):
        config = PSSConfig(verification_strategy="reviewer")
        eval_fn = create_eval_fn(config)
        assert eval_fn is not None

        # Placeholder returns 0.5
        score = eval_fn("any output")
        assert score == 0.5
