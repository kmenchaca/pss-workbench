"""Tests for the Meta-Learner Integration layer."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from unified.config import SystemType
from unified.learner_integration import (
    LearnerControlledHarness,
    LearnerResult,
    LearningStats,
    learner_solve,
    classify_for_routing,
    PROBLEM_TYPE_TO_SYSTEM,
)
from unified.harness import SystemResult


# =============================================================================
# LearnerResult Tests
# =============================================================================


class TestLearnerResult:
    """Tests for LearnerResult dataclass."""

    def test_learner_result_creation(self):
        """Should create LearnerResult with all fields."""
        system_result = SystemResult(
            system_type=SystemType.ARGSWARM,
            prompt="Test",
            output={"judgment": "test"},
            success=True,
        )

        result = LearnerResult(
            result=system_result,
            classification="debate",
            strategy_used="argswarm",
            learning_recorded=True,
            confidence=0.8,
        )

        assert result.result == system_result
        assert result.classification == "debate"
        assert result.strategy_used == "argswarm"
        assert result.learning_recorded
        assert result.confidence == 0.8


# =============================================================================
# LearningStats Tests
# =============================================================================


class TestLearningStats:
    """Tests for LearningStats dataclass."""

    def test_learning_stats_creation(self):
        """Should create LearningStats."""
        stats = LearningStats(
            total_tasks=10,
            success_rate=0.8,
            system_performance={"argswarm": 0.9, "personas": 0.7},
            classification_distribution={"debate": 5, "analytical": 5},
        )

        assert stats.total_tasks == 10
        assert stats.success_rate == 0.8
        assert len(stats.system_performance) == 2
        assert len(stats.classification_distribution) == 2


# =============================================================================
# LearnerControlledHarness Tests
# =============================================================================


class TestLearnerControlledHarness:
    """Tests for LearnerControlledHarness."""

    def test_harness_initialization(self):
        """Should initialize with default settings."""
        harness = LearnerControlledHarness()

        assert harness.provider == "openrouter"
        assert harness.model == "meta-llama/llama-3.1-8b-instruct"
        assert harness.learning_enabled
        assert harness.history == []

    def test_harness_custom_config(self):
        """Should accept custom configuration."""
        harness = LearnerControlledHarness(
            provider="anthropic",
            model="claude-3-haiku",
            learning_enabled=False,
        )

        assert harness.provider == "anthropic"
        assert harness.model == "claude-3-haiku"
        assert not harness.learning_enabled


class TestClassification:
    """Tests for task classification."""

    def test_classify_debate(self):
        """Should classify debate tasks."""
        harness = LearnerControlledHarness()

        result = harness.classify_task("I want to debate whether AI is dangerous")
        assert result == "debate"

    def test_classify_security(self):
        """Should classify security tasks."""
        harness = LearnerControlledHarness()

        result = harness.classify_task("Find security vulnerabilities in this code")
        assert result == "security"

    def test_classify_creative(self):
        """Should classify creative tasks."""
        harness = LearnerControlledHarness()

        result = harness.classify_task("Brainstorm some creative ideas")
        assert result == "creative"

    def test_classify_planning(self):
        """Should classify planning tasks."""
        harness = LearnerControlledHarness()

        result = harness.classify_task("Plan the future scenarios for our product")
        assert result == "planning"

    def test_classify_analytical(self):
        """Should classify analytical tasks."""
        harness = LearnerControlledHarness()

        result = harness.classify_task("Analyze the performance of our system")
        assert result == "analytical"

    def test_classify_default(self):
        """Should default to analytical for unclear tasks."""
        harness = LearnerControlledHarness()

        result = harness.classify_task("Hello world")
        assert result == "analytical"


class TestSystemRecommendation:
    """Tests for system recommendation."""

    def test_recommend_debate_systems(self):
        """Should recommend appropriate systems for debate."""
        harness = LearnerControlledHarness()
        # Disable metalearner to test fallback directly
        harness.metalearner_available = False

        systems = harness.get_recommended_systems("debate")

        assert SystemType.ARGSWARM in systems or SystemType.NEGOTIATE in systems

    def test_recommend_security_systems(self):
        """Should recommend redteam for security."""
        harness = LearnerControlledHarness()
        harness.metalearner_available = False

        systems = harness.get_recommended_systems("security")

        assert SystemType.REDTEAM in systems

    def test_recommend_creative_systems(self):
        """Should recommend dreamlogic for creative."""
        harness = LearnerControlledHarness()
        harness.metalearner_available = False

        systems = harness.get_recommended_systems("creative")

        assert SystemType.DREAMLOGIC in systems or SystemType.EVOLUTION in systems

    def test_fallback_recommendation(self):
        """Should fallback to personas for unknown classification."""
        harness = LearnerControlledHarness()
        harness.metalearner_available = False

        systems = harness.get_recommended_systems("unknown_type")

        assert SystemType.PERSONAS in systems

    def test_returns_non_empty_list(self):
        """Should always return at least one system."""
        harness = LearnerControlledHarness()
        harness.metalearner_available = False

        for classification in ["debate", "security", "creative", "planning", "analytical", "unknown"]:
            systems = harness.get_recommended_systems(classification)
            assert len(systems) >= 1


class TestProblemTypeMapping:
    """Tests for PROBLEM_TYPE_TO_SYSTEM mapping."""

    def test_all_mappings_have_systems(self):
        """All mappings should return non-empty lists."""
        for problem_type, systems in PROBLEM_TYPE_TO_SYSTEM.items():
            assert len(systems) >= 1
            assert all(isinstance(s, SystemType) for s in systems)

    def test_mapping_coverage(self):
        """Should cover common problem types."""
        required_types = [
            "factual", "analytical", "creative", "planning",
            "security", "debate", "optimization",
        ]

        for ptype in required_types:
            assert ptype in PROBLEM_TYPE_TO_SYSTEM


class TestConfidenceCalculation:
    """Tests for confidence calculation."""

    def test_default_confidence(self):
        """Should return default confidence with no history."""
        harness = LearnerControlledHarness()

        confidence = harness._get_confidence("debate", [SystemType.ARGSWARM])

        assert confidence == 0.5

    def test_confidence_with_history(self):
        """Should calculate confidence from history."""
        harness = LearnerControlledHarness()

        # Add mock history
        harness.history = [
            {"classification": "debate", "systems": ["argswarm"], "success": True},
            {"classification": "debate", "systems": ["argswarm"], "success": True},
            {"classification": "debate", "systems": ["argswarm"], "success": False},
        ]

        confidence = harness._get_confidence("debate", [SystemType.ARGSWARM])

        # Should be higher than default due to 2/3 success rate
        assert confidence > 0.5


class TestStats:
    """Tests for statistics gathering."""

    def test_empty_stats(self):
        """Should handle empty history."""
        harness = LearnerControlledHarness()

        stats = harness.get_stats()

        assert stats.total_tasks == 0
        assert stats.success_rate == 0.0

    def test_stats_calculation(self):
        """Should calculate stats from history."""
        harness = LearnerControlledHarness()

        harness.history = [
            {"prompt": "Test 1", "classification": "debate", "systems": ["argswarm"], "success": True},
            {"prompt": "Test 2", "classification": "debate", "systems": ["argswarm"], "success": True},
            {"prompt": "Test 3", "classification": "security", "systems": ["redteam"], "success": False},
        ]

        stats = harness.get_stats()

        assert stats.total_tasks == 3
        assert stats.success_rate == 2 / 3
        assert stats.system_performance["argswarm"] == 1.0
        assert stats.system_performance["redteam"] == 0.0
        assert stats.classification_distribution["debate"] == 2
        assert stats.classification_distribution["security"] == 1


class TestSuggestions:
    """Tests for improvement suggestions."""

    def test_suggest_more_data(self):
        """Should suggest more data when history is small."""
        harness = LearnerControlledHarness()

        harness.history = [
            {"classification": "debate", "systems": ["argswarm"], "success": True},
        ]

        suggestions = harness.suggest_improvements()

        assert any("more tasks" in s.lower() for s in suggestions)

    def test_suggest_avoid_failing_system(self):
        """Should suggest avoiding systems with low success."""
        harness = LearnerControlledHarness()

        # Add enough history to get meaningful suggestions
        for i in range(15):
            harness.history.append({
                "prompt": f"Test {i}",
                "classification": "security",
                "systems": ["redteam"],
                "success": i < 3,  # Only 3 successes = 20%
            })

        suggestions = harness.suggest_improvements()

        # Should suggest avoiding redteam due to low success
        assert any("redteam" in s.lower() for s in suggestions)


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_classify_for_routing(self):
        """classify_for_routing should return classification info."""
        result = classify_for_routing("I want to debate AI regulation")

        assert "classification" in result
        assert "recommended_systems" in result
        assert "primary_system" in result
        assert result["classification"] == "debate"

    def test_classify_for_routing_structure(self):
        """Should return properly structured result."""
        result = classify_for_routing("Analyze this problem")

        assert isinstance(result["classification"], str)
        assert isinstance(result["recommended_systems"], list)
        assert isinstance(result["primary_system"], str)


# =============================================================================
# Strategy Mapping Tests
# =============================================================================


class TestStrategyMapping:
    """Tests for strategy to system mapping."""

    def test_strategy_mapping(self):
        """Should map strategies to systems."""
        harness = LearnerControlledHarness()

        # Test known mappings
        systems = harness._strategy_to_systems("adversarial")
        assert SystemType.REDTEAM in systems or SystemType.ARGSWARM in systems

        systems = harness._strategy_to_systems("analogical")
        assert SystemType.DREAMLOGIC in systems or SystemType.KNOWLEDGE in systems

    def test_unknown_strategy_fallback(self):
        """Should fallback to personas for unknown strategies."""
        harness = LearnerControlledHarness()

        systems = harness._strategy_to_systems("unknown_strategy")
        assert SystemType.PERSONAS in systems
