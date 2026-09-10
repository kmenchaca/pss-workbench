"""Tests for the Meta-Harness orchestration layer."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from unified.config import SystemType
from unified.meta_harness import (
    MetaHarness,
    MetaResult,
    TaskAnalysis,
    smart_run,
    analyze_task,
)
from unified.harness import SystemResult


# =============================================================================
# TaskAnalysis Tests
# =============================================================================


class TestTaskAnalysis:
    """Tests for TaskAnalysis dataclass."""

    def test_task_analysis_creation(self):
        """Should create TaskAnalysis with all fields."""
        analysis = TaskAnalysis(
            complexity="high",
            recommended_systems=[SystemType.ARGSWARM, SystemType.PERSONAS],
            parallel_capable=True,
            requires_synthesis=True,
        )

        assert analysis.complexity == "high"
        assert len(analysis.recommended_systems) == 2
        assert analysis.parallel_capable
        assert analysis.requires_synthesis


# =============================================================================
# MetaResult Tests
# =============================================================================


class TestMetaResult:
    """Tests for MetaResult dataclass."""

    def test_meta_result_creation(self):
        """Should create MetaResult with required fields."""
        primary = SystemResult(
            system_type=SystemType.ARGSWARM,
            prompt="Test",
            output={"judgment": "test"},
            success=True,
        )

        result = MetaResult(
            primary_result=primary,
            synthesis="Combined analysis",
            confidence=0.8,
        )

        assert result.primary_result == primary
        assert result.synthesis == "Combined analysis"
        assert result.confidence == 0.8
        assert result.supporting_results == []

    def test_meta_result_with_supporting(self):
        """Should store supporting results."""
        primary = SystemResult(
            system_type=SystemType.ARGSWARM,
            prompt="Test",
            output={},
            success=True,
        )
        supporting = SystemResult(
            system_type=SystemType.PERSONAS,
            prompt="Test",
            output={},
            success=True,
        )

        result = MetaResult(
            primary_result=primary,
            supporting_results=[supporting],
        )

        assert len(result.supporting_results) == 1


# =============================================================================
# MetaHarness Tests
# =============================================================================


class TestMetaHarness:
    """Tests for MetaHarness class."""

    def test_harness_initialization(self):
        """Should initialize with default config."""
        harness = MetaHarness()

        assert harness.config.default_provider == "openrouter"
        assert harness.unified is not None
        assert harness.router is not None
        assert harness.history == []

    def test_harness_custom_config(self):
        """Should accept custom provider and model."""
        harness = MetaHarness(
            provider="anthropic",
            model="claude-3-haiku",
        )

        assert harness.config.default_provider == "anthropic"
        assert harness.config.default_model == "claude-3-haiku"

    def test_analyze_task_low_complexity(self):
        """Should identify low complexity tasks."""
        harness = MetaHarness()

        analysis = harness.analyze_task("What is 2+2?")

        assert analysis.complexity == "low"
        assert len(analysis.recommended_systems) >= 1

    def test_analyze_task_medium_complexity(self):
        """Should identify medium complexity tasks."""
        harness = MetaHarness()

        analysis = harness.analyze_task(
            "Analyze the pros and cons of using microservices architecture"
        )

        assert analysis.complexity in ["medium", "high"]

    def test_analyze_task_high_complexity(self):
        """Should identify high complexity tasks."""
        harness = MetaHarness()

        long_prompt = (
            "Please provide a comprehensive analysis of the various aspects "
            "of implementing a new authentication system. Compare different "
            "approaches and evaluate their security implications. "
            "Consider multiple stakeholder perspectives and synthesize "
            "recommendations based on thorough analysis."
        )

        analysis = harness.analyze_task(long_prompt)

        assert analysis.complexity == "high"
        assert len(analysis.recommended_systems) > 1
        assert analysis.requires_synthesis


class TestComplexityEstimation:
    """Tests for complexity estimation."""

    def test_short_prompt_low_complexity(self):
        """Short prompts should be low complexity."""
        harness = MetaHarness()

        analysis = harness.analyze_task("Hello")
        assert analysis.complexity == "low"

    def test_complex_indicators(self):
        """Prompts with complexity indicators should rate higher."""
        harness = MetaHarness()

        analysis = harness.analyze_task(
            "Provide a comprehensive detailed analysis comparing multiple approaches"
        )

        assert analysis.complexity in ["medium", "high"]


class TestComplementarySystems:
    """Tests for complementary system selection."""

    def test_argswarm_complements(self):
        """Argswarm should complement with personas and redteam."""
        harness = MetaHarness()

        complements = harness._get_complementary_systems(SystemType.ARGSWARM)

        assert SystemType.PERSONAS in complements
        assert SystemType.REDTEAM in complements

    def test_all_systems_have_complements(self):
        """Every system should have complements defined."""
        harness = MetaHarness()

        for sys_type in SystemType:
            complements = harness._get_complementary_systems(sys_type)
            assert len(complements) >= 1


class TestOutputExtraction:
    """Tests for output text extraction."""

    def test_extract_string_output(self):
        """Should extract string output directly."""
        harness = MetaHarness()

        result = SystemResult(
            system_type=SystemType.ARGSWARM,
            prompt="Test",
            output="This is the output",
            success=True,
        )

        text = harness._extract_output_text(result)
        assert text == "This is the output"

    def test_extract_dict_output(self):
        """Should extract dict output as formatted string."""
        harness = MetaHarness()

        result = SystemResult(
            system_type=SystemType.ARGSWARM,
            prompt="Test",
            output={"judgment": "Winner is FOR", "reasoning": "Strong arguments"},
            success=True,
        )

        text = harness._extract_output_text(result)
        assert "judgment" in text
        assert "Winner is FOR" in text

    def test_extract_failed_result(self):
        """Should show error for failed results."""
        harness = MetaHarness()

        result = SystemResult(
            system_type=SystemType.ARGSWARM,
            prompt="Test",
            output=None,
            success=False,
            error="Something went wrong",
        )

        text = harness._extract_output_text(result)
        assert "Error" in text
        assert "Something went wrong" in text


class TestHistoryTracking:
    """Tests for history and statistics tracking."""

    def test_empty_history(self):
        """Should handle empty history."""
        harness = MetaHarness()

        stats = harness.get_history_stats()

        assert stats["total_runs"] == 0

    def test_history_stats(self):
        """Should calculate stats from history."""
        harness = MetaHarness()

        # Add mock history
        harness.history = [
            {"prompt": "Test 1", "systems": ["argswarm"], "success": True},
            {"prompt": "Test 2", "systems": ["personas", "argswarm"], "success": True},
            {"prompt": "Test 3", "systems": ["redteam"], "success": False},
        ]

        stats = harness.get_history_stats()

        assert stats["total_runs"] == 3
        assert stats["success_rate"] == 2 / 3
        assert stats["system_usage"]["argswarm"] == 2
        assert stats["system_usage"]["personas"] == 1
        assert stats["system_usage"]["redteam"] == 1


class TestRecommendations:
    """Tests for recommendation generation."""

    def test_recommend_on_failures(self):
        """Should recommend retrying failures."""
        harness = MetaHarness()

        analysis = TaskAnalysis(
            complexity="medium",
            recommended_systems=[SystemType.ARGSWARM],
            parallel_capable=False,
            requires_synthesis=False,
        )

        results = [
            SystemResult(
                system_type=SystemType.ARGSWARM,
                prompt="Test",
                output=None,
                success=False,
                error="Failed",
            )
        ]

        recs = harness._generate_recommendations(analysis, results)

        assert any("re-running" in r.lower() or "failed" in r.lower() for r in recs)

    def test_recommend_more_perspectives(self):
        """Should recommend more perspectives for single system."""
        harness = MetaHarness()

        analysis = TaskAnalysis(
            complexity="low",
            recommended_systems=[SystemType.ARGSWARM],
            parallel_capable=False,
            requires_synthesis=False,
        )

        results = [
            SystemResult(
                system_type=SystemType.ARGSWARM,
                prompt="Test",
                output={},
                success=True,
            )
        ]

        recs = harness._generate_recommendations(analysis, results)

        assert any("personas" in r.lower() or "perspectives" in r.lower() for r in recs)


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_analyze_task_function(self):
        """analyze_task should work without instantiation."""
        analysis = analyze_task("Debate this topic")

        assert isinstance(analysis, TaskAnalysis)
        assert analysis.complexity in ["low", "medium", "high"]
