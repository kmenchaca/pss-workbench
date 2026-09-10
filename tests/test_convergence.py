"""Tests for convergent task detection (v1.2)."""

import pytest

from pss.convergence import (
    ConvergenceConfig,
    TaskType,
    detect_task_type,
    format_convergence_result,
    should_branch,
)


class TestTaskTypeDetection:
    """Tests for detect_task_type()."""

    def test_factual_questions_are_convergent(self):
        """Factual questions with single answers should be convergent."""
        prompts = [
            "What is the capital of France?",
            "Who wrote Hamlet?",
            "Who invented the telephone?",
            "When was the Eiffel Tower built?",
            "How many legs does a spider have?",
            "Where is the Great Wall located?",
        ]
        for prompt in prompts:
            task_type, confidence, reason = detect_task_type(prompt)
            assert task_type == TaskType.CONVERGENT, f"Expected CONVERGENT for '{prompt}', got {task_type}"
            assert confidence >= 0.7, f"Expected confidence >= 0.7 for '{prompt}', got {confidence}"

    def test_calculation_questions_are_convergent(self):
        """Math and calculation questions should be convergent."""
        prompts = [
            "What is 5 plus 3?",
            "Calculate 15 times 7",
            "Convert 100 Celsius to Fahrenheit",
            "What is 10 divided by 2?",
            "5 + 3",
            "100 - 25",
        ]
        for prompt in prompts:
            task_type, confidence, reason = detect_task_type(prompt)
            assert task_type == TaskType.CONVERGENT, f"Expected CONVERGENT for '{prompt}', got {task_type}"

    def test_definition_questions_are_convergent(self):
        """Definitional questions should be convergent."""
        prompts = [
            "Define photosynthesis",
            "What does 'ubiquitous' mean?",
            "What is the definition of entropy?",
            "What is the meaning of life?",  # Should still match pattern
        ]
        for prompt in prompts:
            task_type, confidence, reason = detect_task_type(prompt)
            assert task_type == TaskType.CONVERGENT, f"Expected CONVERGENT for '{prompt}', got {task_type}"

    def test_creative_tasks_are_divergent(self):
        """Creative writing tasks should be divergent."""
        prompts = [
            "Write a story about a dragon",
            "Write a poem about love",
            "Compose a song about summer",
            "Create a new character for a video game",
            "Design a logo for a coffee shop",
        ]
        for prompt in prompts:
            task_type, confidence, reason = detect_task_type(prompt)
            assert task_type == TaskType.DIVERGENT, f"Expected DIVERGENT for '{prompt}', got {task_type}"
            assert confidence >= 0.7, f"Expected confidence >= 0.7 for '{prompt}', got {confidence}"

    def test_brainstorming_tasks_are_divergent(self):
        """Brainstorming and ideation tasks should be divergent."""
        prompts = [
            "List ways to reduce stress",
            "Brainstorm marketing ideas for a startup",
            "Generate ideas for a birthday party",
            "Explore different approaches to machine learning",
            "Map out possible career paths",
        ]
        for prompt in prompts:
            task_type, confidence, reason = detect_task_type(prompt)
            assert task_type == TaskType.DIVERGENT, f"Expected DIVERGENT for '{prompt}', got {task_type}"

    def test_analysis_tasks_are_divergent(self):
        """Analysis and comparison tasks should be divergent."""
        # High-confidence divergent patterns (>= 0.7 threshold)
        prompts = [
            "Compare Python and JavaScript frameworks",  # compare + and = 0.7
            "What are the pros and cons of remote work?",  # pros and cons = 0.8
            "List the advantages and disadvantages of cloud computing",  # advantages and disadvantages = 0.8
        ]
        for prompt in prompts:
            task_type, confidence, reason = detect_task_type(prompt)
            assert task_type == TaskType.DIVERGENT, f"Expected DIVERGENT for '{prompt}', got {task_type}"

    def test_low_confidence_analysis_tasks_are_uncertain(self):
        """Analysis tasks with low-confidence patterns should be uncertain."""
        # These patterns have confidence < 0.7 threshold
        prompts = [
            "Analyze the impact of social media",  # analyze = 0.6
            "Evaluate different options",  # evaluate = 0.6
        ]
        for prompt in prompts:
            task_type, confidence, reason = detect_task_type(prompt)
            assert task_type == TaskType.UNCERTAIN, f"Expected UNCERTAIN for '{prompt}', got {task_type}"

    def test_open_ended_questions_are_divergent(self):
        """Open-ended hypothetical questions should be divergent."""
        # High-confidence divergent patterns (>= 0.7 threshold)
        prompts = [
            "What are the ways to improve our website?",  # what are ways = 0.8
            "What are the options for our database?",  # what are options = 0.8
            "What if we had unlimited budget?",  # what if = 0.7
            "How might we solve climate change?",  # how might = 0.7
        ]
        for prompt in prompts:
            task_type, confidence, reason = detect_task_type(prompt)
            assert task_type == TaskType.DIVERGENT, f"Expected DIVERGENT for '{prompt}', got {task_type}"

    def test_low_confidence_open_ended_are_uncertain(self):
        """Open-ended questions with low-confidence patterns should be uncertain."""
        # These patterns have confidence < 0.7 threshold
        prompts = [
            "What could we do?",  # what could = 0.6
            "What might happen?",  # what might = 0.6
        ]
        for prompt in prompts:
            task_type, confidence, reason = detect_task_type(prompt)
            assert task_type == TaskType.UNCERTAIN, f"Expected UNCERTAIN for '{prompt}', got {task_type}"

    def test_ambiguous_prompts_are_uncertain(self):
        """Ambiguous prompts should be uncertain."""
        prompts = [
            "Hello",
            "Tell me about cats",
            "Help me with Python",
            "I need advice",
        ]
        for prompt in prompts:
            task_type, confidence, reason = detect_task_type(prompt)
            assert task_type == TaskType.UNCERTAIN, f"Expected UNCERTAIN for '{prompt}', got {task_type}"

    def test_divergent_overrides_convergent(self):
        """Divergent patterns should override convergent patterns."""
        # "What is" is convergent but "write a story" is divergent
        prompt = "Write a story about what is the meaning of life"
        task_type, _, _ = detect_task_type(prompt)
        assert task_type == TaskType.DIVERGENT


class TestShouldBranch:
    """Tests for should_branch()."""

    def test_convergent_tasks_should_not_branch(self):
        """Convergent tasks should return False for should_branch."""
        prompt = "What is the capital of France?"
        branch, reason = should_branch(prompt)
        assert branch is False
        assert "convergent" in reason

    def test_divergent_tasks_should_branch(self):
        """Divergent tasks should return True for should_branch."""
        prompt = "Write a story about a robot"
        branch, reason = should_branch(prompt)
        assert branch is True
        assert "divergent" in reason

    def test_uncertain_tasks_branch_by_default(self):
        """Uncertain tasks should branch by default."""
        config = ConvergenceConfig(branch_on_uncertain=True)
        prompt = "Hello"
        branch, reason = should_branch(prompt, config)
        assert branch is True
        assert "uncertain" in reason

    def test_uncertain_tasks_can_be_configured_not_to_branch(self):
        """Uncertain tasks can be configured not to branch."""
        config = ConvergenceConfig(branch_on_uncertain=False)
        prompt = "Hello"
        branch, reason = should_branch(prompt, config)
        assert branch is False
        assert "uncertain" in reason


class TestConvergenceConfig:
    """Tests for ConvergenceConfig."""

    def test_default_config(self):
        """Default config should have sensible defaults."""
        config = ConvergenceConfig()
        assert config.enabled is True
        assert config.branch_on_uncertain is True
        assert config.convergent_confidence_threshold == 0.7

    def test_disabled_config_returns_uncertain(self):
        """Disabled config should always return uncertain."""
        config = ConvergenceConfig(enabled=False)
        prompt = "What is the capital of France?"
        task_type, confidence, reason = detect_task_type(prompt, config)
        assert task_type == TaskType.UNCERTAIN
        assert reason == "detection_disabled"

    def test_custom_threshold(self):
        """Custom threshold should affect classification."""
        # With high threshold, some matches won't qualify
        config = ConvergenceConfig(convergent_confidence_threshold=0.95)
        prompt = "Define something"  # Pattern weight is 0.7
        task_type, confidence, reason = detect_task_type(prompt, config)
        assert task_type == TaskType.UNCERTAIN  # Below threshold


class TestFormatConvergenceResult:
    """Tests for format_convergence_result()."""

    def test_format_convergent_result(self):
        """Should format convergent result correctly."""
        prompt = "What is the capital of France?"
        result = format_convergence_result(prompt)
        assert "convergent" in result.lower()
        assert "80%" in result or "0.8" in result  # Confidence
        assert "False" in result  # Should not branch

    def test_format_divergent_result(self):
        """Should format divergent result correctly."""
        prompt = "Write a story about dragons"
        result = format_convergence_result(prompt)
        assert "divergent" in result.lower()
        assert "True" in result  # Should branch


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_prompt(self):
        """Empty prompt should be uncertain."""
        task_type, _, _ = detect_task_type("")
        assert task_type == TaskType.UNCERTAIN

    def test_whitespace_prompt(self):
        """Whitespace-only prompt should be uncertain."""
        task_type, _, _ = detect_task_type("   \n\t  ")
        assert task_type == TaskType.UNCERTAIN

    def test_case_insensitive(self):
        """Detection should be case insensitive."""
        lower = "what is the capital of france?"
        upper = "WHAT IS THE CAPITAL OF FRANCE?"
        mixed = "WhAt Is ThE cApItAl Of FrAnCe?"

        for prompt in [lower, upper, mixed]:
            task_type, _, _ = detect_task_type(prompt)
            assert task_type == TaskType.CONVERGENT

    def test_very_long_prompt(self):
        """Should handle very long prompts."""
        prompt = "What is the capital of France? " * 1000
        task_type, _, _ = detect_task_type(prompt)
        assert task_type == TaskType.CONVERGENT

    def test_special_characters(self):
        """Should handle special characters."""
        prompt = "What is 2 + 2?"
        task_type, _, _ = detect_task_type(prompt)
        assert task_type == TaskType.CONVERGENT
