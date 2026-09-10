"""Comprehensive tests for the meta-learner system."""

import asyncio
import json
import random
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from metalearner import (
    # Types
    Strategy,
    StrategyResult,
    ProblemType,
    ProblemCategory,
    PerformanceRecord,
    StrategyMutation,
    MutationType,
    MetaConfig,
    ClassificationResult,
    RecommendationResult,
    EvaluationResult,
    LearningState,
    # Strategies
    CHAIN_OF_THOUGHT,
    DECOMPOSITION,
    TOOL_HEAVY,
    ANALOGICAL,
    ADVERSARIAL,
    MINIMAL,
    EXHAUSTIVE,
    BUILTIN_STRATEGIES,
    StrategyApplicator,
    apply_strategy,
    get_builtin_strategy,
    list_builtin_strategies,
    create_custom_strategy,
    compose_strategies,
    # Classification
    ProblemClassifier,
    classify_problem,
    extract_features,
    get_problem_type,
    list_problem_types,
    # Mutation
    StrategyMutator,
    mutate_strategy,
    crossover_strategies,
    random_strategy,
    # Evaluation
    StrategyEvaluator,
    ABTest,
    calculate_pass_at_k,
    calculate_token_efficiency,
    calculate_quality_adjusted_score,
    # Database
    PerformanceDB,
    # Recommendation
    StrategyRecommender,
    UCBRecommender,
    ThompsonSamplingRecommender,
    recommend_strategy,
    fallback_strategy,
    # Learning
    LearningLoop,
    IncrementalLearner,
    # Harness
    MetaHarness,
    SimplifiedHarness,
    SolveResult,
    BatchResult,
    # Bootstrap
    BootstrapOptimizer,
    BootstrapResult,
    # Anti-patterns
    AntiPattern,
    AntiPatternDB,
    record_antipattern,
    is_antipattern,
)


# ============================================================================
# Types Tests
# ============================================================================

class TestStrategy:
    """Tests for Strategy dataclass."""

    def test_strategy_creation(self):
        """Test creating a strategy."""
        s = Strategy(
            id="test",
            name="Test Strategy",
            description="A test strategy",
        )
        assert s.id == "test"
        assert s.name == "Test Strategy"
        assert s.parameters == {}

    def test_strategy_serialization(self):
        """Test strategy to_dict and from_dict."""
        s = Strategy(
            id="test",
            name="Test",
            description="Desc",
            parameters={"key": "value"},
            prompt_template="Template {problem}",
        )
        d = s.to_dict()
        assert d["id"] == "test"
        assert d["parameters"]["key"] == "value"

        s2 = Strategy.from_dict(d)
        assert s2.id == s.id
        assert s2.parameters == s.parameters

    def test_strategy_with_parent_ids(self):
        """Test strategy with genealogy."""
        s = Strategy(
            id="child",
            name="Child",
            description="Child strategy",
            parent_ids=["parent1", "parent2"],
        )
        assert len(s.parent_ids) == 2


class TestStrategyResult:
    """Tests for StrategyResult dataclass."""

    def test_result_creation(self):
        """Test creating a result."""
        r = StrategyResult(
            strategy_id="test",
            success=True,
            tokens_used=100,
            quality_score=0.8,
            problem_type="factual",
        )
        assert r.success is True
        assert r.tokens_used == 100

    def test_result_serialization(self):
        """Test result to_dict and from_dict."""
        r = StrategyResult(
            strategy_id="test",
            success=True,
            tokens_used=100,
            quality_score=0.8,
            problem_type="factual",
            output="Test output",
        )
        d = r.to_dict()
        r2 = StrategyResult.from_dict(d)
        assert r2.strategy_id == r.strategy_id
        assert r2.output == r.output


class TestPerformanceRecord:
    """Tests for PerformanceRecord dataclass."""

    def test_empty_record(self):
        """Test empty performance record."""
        record = PerformanceRecord(
            strategy_id="test",
            problem_type="factual",
        )
        assert record.success_rate == 0.0
        assert record.sample_count == 0

    def test_record_with_results(self):
        """Test record with results."""
        results = [
            StrategyResult("test", True, 100, 0.8, "factual"),
            StrategyResult("test", True, 150, 0.9, "factual"),
            StrategyResult("test", False, 200, 0.3, "factual"),
        ]
        record = PerformanceRecord(
            strategy_id="test",
            problem_type="factual",
            results=results,
        )
        assert record.success_rate == 2/3
        assert record.sample_count == 3
        assert record.avg_tokens == 150


class TestMetaConfig:
    """Tests for MetaConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = MetaConfig()
        assert config.learning_rate == 0.1
        assert config.exploration_rate == 0.1

    def test_config_serialization(self):
        """Test config to_dict and from_dict."""
        config = MetaConfig(learning_rate=0.2, mutation_rate=0.3)
        d = config.to_dict()
        config2 = MetaConfig.from_dict(d)
        assert config2.learning_rate == 0.2
        assert config2.mutation_rate == 0.3


# ============================================================================
# Strategies Tests
# ============================================================================

class TestBuiltinStrategies:
    """Tests for built-in strategies."""

    def test_all_builtin_strategies_exist(self):
        """Test that all built-in strategies are defined."""
        assert len(BUILTIN_STRATEGIES) == 7
        assert "chain_of_thought" in BUILTIN_STRATEGIES
        assert "decomposition" in BUILTIN_STRATEGIES
        assert "adversarial" in BUILTIN_STRATEGIES

    def test_get_builtin_strategy(self):
        """Test getting built-in strategy by name."""
        s = get_builtin_strategy("chain_of_thought")
        assert s is not None
        assert s.id == "chain_of_thought"

        s2 = get_builtin_strategy("nonexistent")
        assert s2 is None

    def test_list_builtin_strategies(self):
        """Test listing all built-in strategies."""
        strategies = list_builtin_strategies()
        assert len(strategies) == 7


class TestStrategyApplicator:
    """Tests for StrategyApplicator."""

    def test_apply_basic(self):
        """Test basic strategy application."""
        s = Strategy(
            id="test",
            name="Test",
            description="Test",
            prompt_template="Solve: {problem}",
        )
        applicator = StrategyApplicator(s)
        result = applicator.apply("What is 2+2?")
        assert "Solve: What is 2+2?" in result

    def test_apply_with_context(self):
        """Test application with context."""
        s = Strategy(
            id="test",
            name="Test",
            description="Test",
            prompt_template="Problem: {problem}\nContext: {context}",
        )
        applicator = StrategyApplicator(s)
        result = applicator.apply("Test problem", {"context": "Additional info"})
        assert "Additional info" in result

    def test_apply_function(self):
        """Test apply_strategy function."""
        result = apply_strategy(CHAIN_OF_THOUGHT, "What is 2+2?")
        assert "What is 2+2?" in result
        assert "step" in result.lower()


class TestStrategyComposition:
    """Tests for strategy composition."""

    def test_compose_sequential(self):
        """Test sequential composition."""
        composed = compose_strategies(
            [CHAIN_OF_THOUGHT, ADVERSARIAL],
            "sequential",
        )
        assert "Phase 1" in composed.prompt_template
        assert "Phase 2" in composed.prompt_template
        assert len(composed.parent_ids) == 2

    def test_compose_parallel(self):
        """Test parallel composition."""
        composed = compose_strategies(
            [MINIMAL, EXHAUSTIVE],
            "parallel",
        )
        assert "approaches" in composed.prompt_template.lower()

    def test_compose_single(self):
        """Test composing single strategy returns same strategy."""
        composed = compose_strategies([CHAIN_OF_THOUGHT], "sequential")
        assert composed.id == CHAIN_OF_THOUGHT.id


class TestCustomStrategy:
    """Tests for custom strategy creation."""

    def test_create_custom_strategy(self):
        """Test creating custom strategy."""
        s = create_custom_strategy(
            id="custom",
            name="Custom",
            description="A custom strategy",
            prompt_template="Custom: {problem}",
            parameters={"param1": "value1"},
        )
        assert s.id == "custom"
        assert s.parameters["param1"] == "value1"


# ============================================================================
# Classification Tests
# ============================================================================

class TestProblemClassifier:
    """Tests for ProblemClassifier."""

    def test_classifier_creation(self):
        """Test creating a classifier."""
        classifier = ProblemClassifier()
        assert len(classifier.problem_types) > 0

    def test_classify_factual(self):
        """Test classifying factual questions."""
        result = classify_problem("What is the capital of France?")
        assert result.problem_type.category == ProblemCategory.FACTUAL

    def test_classify_creative(self):
        """Test classifying creative tasks."""
        result = classify_problem("Write a poem about the ocean")
        assert result.problem_type.category == ProblemCategory.CREATIVE

    def test_classify_technical(self):
        """Test classifying technical tasks."""
        result = classify_problem("Debug this Python code that has a syntax error")
        assert result.problem_type.category == ProblemCategory.TECHNICAL

    def test_classify_planning(self):
        """Test classifying planning tasks."""
        result = classify_problem("Create a project plan for building a website")
        assert result.problem_type.category == ProblemCategory.PLANNING

    def test_extract_features(self):
        """Test feature extraction."""
        features = extract_features("What is 2+2?")
        assert "question" in features
        assert "short" in features

    def test_classifier_with_custom_patterns(self):
        """Test classifier with custom patterns."""
        classifier = ProblemClassifier(
            custom_patterns={
                ProblemCategory.TECHNICAL: [r"\bkubernetes\b"],
            }
        )
        result = classifier.classify("How do I deploy to kubernetes?")
        assert result.problem_type.category == ProblemCategory.TECHNICAL


class TestProblemTypes:
    """Tests for problem type management."""

    def test_list_problem_types(self):
        """Test listing problem types."""
        types = list_problem_types()
        assert len(types) > 0

    def test_get_problem_type(self):
        """Test getting problem type by ID."""
        pt = get_problem_type("factual_lookup")
        assert pt is not None
        assert pt.category == ProblemCategory.FACTUAL


# ============================================================================
# Mutation Tests
# ============================================================================

class TestStrategyMutator:
    """Tests for StrategyMutator."""

    def test_mutator_creation(self):
        """Test creating a mutator."""
        mutator = StrategyMutator()
        assert mutator is not None

    def test_parameter_tweak(self):
        """Test parameter tweaking mutation."""
        mutator = StrategyMutator(random.Random(42))
        mutation = mutator.mutate(CHAIN_OF_THOUGHT, MutationType.PARAMETER_TWEAK)
        assert mutation.mutation_type == MutationType.PARAMETER_TWEAK
        assert mutation.new_strategy.id != CHAIN_OF_THOUGHT.id

    def test_prompt_edit(self):
        """Test prompt editing mutation."""
        mutator = StrategyMutator(random.Random(42))
        mutation = mutator.mutate(CHAIN_OF_THOUGHT, MutationType.PROMPT_EDIT)
        assert mutation.mutation_type == MutationType.PROMPT_EDIT

    def test_simplify(self):
        """Test simplification mutation."""
        mutator = StrategyMutator(random.Random(42))
        mutation = mutator.mutate(EXHAUSTIVE, MutationType.SIMPLIFY)
        assert mutation.mutation_type == MutationType.SIMPLIFY

    def test_extend(self):
        """Test extension mutation."""
        mutator = StrategyMutator(random.Random(42))
        mutation = mutator.mutate(MINIMAL, MutationType.EXTEND)
        assert mutation.mutation_type == MutationType.EXTEND

    def test_random_mutation_type(self):
        """Test random mutation type selection."""
        mutator = StrategyMutator()
        mutation = mutator.mutate(CHAIN_OF_THOUGHT)
        assert mutation.mutation_type in list(MutationType)


class TestCrossover:
    """Tests for strategy crossover."""

    def test_crossover_basic(self):
        """Test basic crossover."""
        child = crossover_strategies(CHAIN_OF_THOUGHT, ADVERSARIAL, random.Random(42))
        assert child.id.startswith("crossover_")
        assert len(child.parent_ids) == 2

    def test_crossover_parameters(self):
        """Test that crossover combines parameters."""
        s1 = Strategy("s1", "S1", "Desc", parameters={"a": 1, "b": 2})
        s2 = Strategy("s2", "S2", "Desc", parameters={"b": 3, "c": 4})
        child = crossover_strategies(s1, s2, random.Random(42))
        # Child should have some parameters from each
        assert len(child.parameters) >= 2


class TestRandomStrategy:
    """Tests for random strategy generation."""

    def test_random_strategy(self):
        """Test generating random strategy."""
        s = random_strategy(random.Random(42))
        assert s.id.startswith("random_")
        assert s.prompt_template != ""
        assert len(s.parameters) > 0


# ============================================================================
# Evaluation Tests
# ============================================================================

class TestStrategyEvaluator:
    """Tests for StrategyEvaluator."""

    def test_evaluator_creation(self):
        """Test creating an evaluator."""
        evaluator = StrategyEvaluator()
        assert evaluator is not None

    def test_evaluate_sync(self):
        """Test synchronous evaluation."""
        evaluator = StrategyEvaluator()
        result = evaluator.evaluate_strategy_sync(
            CHAIN_OF_THOUGHT,
            "What is 2+2?",
            "The answer is 4.",
            tokens_used=100,
            ground_truth="4",
        )
        assert result.strategy_id == CHAIN_OF_THOUGHT.id
        assert result.tokens_used == 100

    def test_default_quality_scorer(self):
        """Test default quality scoring."""
        evaluator = StrategyEvaluator()
        # Good output
        result = evaluator.evaluate_strategy_sync(
            CHAIN_OF_THOUGHT,
            "Test problem",
            "Here is a structured answer with steps:\n1. First step\n2. Second step\n\nTherefore, the answer is X.",
        )
        assert result.quality_score > 0.5


class TestABTest:
    """Tests for A/B testing framework."""

    def test_ab_test_creation(self):
        """Test creating A/B test."""
        ab = ABTest(CHAIN_OF_THOUGHT, ADVERSARIAL)
        assert ab.strategy_a.id == CHAIN_OF_THOUGHT.id
        assert ab.strategy_b.id == ADVERSARIAL.id

    def test_ab_test_select_strategy(self):
        """Test strategy selection."""
        ab = ABTest(CHAIN_OF_THOUGHT, ADVERSARIAL)
        selected = ab.select_strategy()
        assert selected.id in [CHAIN_OF_THOUGHT.id, ADVERSARIAL.id]

    def test_ab_test_not_conclusive_initially(self):
        """Test that AB test is not conclusive without data."""
        ab = ABTest(CHAIN_OF_THOUGHT, ADVERSARIAL, min_samples=5)
        assert not ab.is_conclusive()

    def test_ab_test_summary(self):
        """Test getting summary."""
        ab = ABTest(CHAIN_OF_THOUGHT, ADVERSARIAL)
        summary = ab.get_summary()
        assert "strategy_a" in summary
        assert "strategy_b" in summary


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_pass_at_k(self):
        """Test pass@k calculation."""
        results = [
            StrategyResult("s", True, 100, 0.9, "test"),
            StrategyResult("s", False, 100, 0.3, "test"),
            StrategyResult("s", True, 100, 0.8, "test"),
        ]
        assert calculate_pass_at_k(results, k=1) > 0
        assert calculate_pass_at_k([], k=1) == 0.0

    def test_token_efficiency(self):
        """Test token efficiency calculation."""
        results = [
            StrategyResult("s", True, 1000, 0.9, "test"),
            StrategyResult("s", True, 1000, 0.8, "test"),
        ]
        efficiency = calculate_token_efficiency(results)
        assert efficiency == 1.0  # 2 successes / 2000 tokens * 1000

    def test_quality_adjusted_score(self):
        """Test quality adjusted score."""
        results = [
            StrategyResult("s", True, 100, 0.9, "test"),
            StrategyResult("s", True, 100, 0.8, "test"),
        ]
        score = calculate_quality_adjusted_score(results)
        assert 0 <= score <= 1


# ============================================================================
# Database Tests
# ============================================================================

class TestPerformanceDB:
    """Tests for PerformanceDB."""

    def test_db_in_memory(self):
        """Test in-memory database."""
        db = PerformanceDB()
        assert db is not None

    def test_record_result(self):
        """Test recording results."""
        db = PerformanceDB()
        result = StrategyResult("test", True, 100, 0.8, "factual")
        db.record_result(CHAIN_OF_THOUGHT, "factual", result)

        record = db.query_performance(CHAIN_OF_THOUGHT.id, "factual")
        assert record is not None
        assert record.sample_count == 1

    def test_best_strategy_for(self):
        """Test finding best strategy."""
        db = PerformanceDB()

        # Record some results
        for i in range(5):
            db.record_result(
                CHAIN_OF_THOUGHT,
                "factual",
                StrategyResult("cot", True, 100, 0.8, "factual"),
            )
            db.record_result(
                MINIMAL,
                "factual",
                StrategyResult("min", False, 50, 0.3, "factual"),
            )

        best, score = db.best_strategy_for("factual", min_samples=3)
        assert best is not None
        assert best.id == CHAIN_OF_THOUGHT.id

    def test_strategy_rankings(self):
        """Test strategy rankings."""
        db = PerformanceDB()

        for i in range(5):
            db.record_result(
                CHAIN_OF_THOUGHT,
                "factual",
                StrategyResult("cot", True, 100, 0.8, "factual"),
            )

        rankings = db.strategy_rankings("factual")
        assert len(rankings) > 0

    def test_db_persistence(self):
        """Test database persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_db.json"

            db1 = PerformanceDB(db_path)
            db1.record_result(
                CHAIN_OF_THOUGHT,
                "factual",
                StrategyResult("cot", True, 100, 0.8, "factual"),
            )
            db1.save()

            db2 = PerformanceDB(db_path)
            record = db2.query_performance(CHAIN_OF_THOUGHT.id, "factual")
            assert record is not None

    def test_db_statistics(self):
        """Test database statistics."""
        db = PerformanceDB()
        db.record_result(
            CHAIN_OF_THOUGHT,
            "factual",
            StrategyResult("cot", True, 100, 0.8, "factual"),
        )

        stats = db.get_statistics()
        assert stats["total_results"] == 1


# ============================================================================
# Recommendation Tests
# ============================================================================

class TestStrategyRecommender:
    """Tests for StrategyRecommender."""

    def test_recommender_creation(self):
        """Test creating recommender."""
        recommender = StrategyRecommender()
        assert recommender is not None

    def test_recommend_basic(self):
        """Test basic recommendation."""
        result = recommend_strategy("What is the capital of France?")
        assert result.strategy is not None
        assert result.confidence > 0

    def test_recommend_with_fallback(self):
        """Test recommendation includes fallback."""
        result = recommend_strategy("Complex technical problem")
        assert result.fallback is not None or result.strategy is not None

    def test_fallback_strategy(self):
        """Test getting fallback strategy."""
        fb = fallback_strategy("Any problem")
        assert fb is not None


class TestUCBRecommender:
    """Tests for UCB recommender."""

    def test_ucb_creation(self):
        """Test creating UCB recommender."""
        ucb = UCBRecommender()
        assert ucb is not None

    def test_ucb_recommend(self):
        """Test UCB recommendation."""
        ucb = UCBRecommender()
        result = ucb.recommend("Test problem")
        assert result.strategy is not None


class TestThompsonSamplingRecommender:
    """Tests for Thompson Sampling recommender."""

    def test_ts_creation(self):
        """Test creating TS recommender."""
        ts = ThompsonSamplingRecommender()
        assert ts is not None

    def test_ts_recommend(self):
        """Test TS recommendation."""
        ts = ThompsonSamplingRecommender()
        result = ts.recommend("Test problem")
        assert result.strategy is not None


# ============================================================================
# Learning Tests
# ============================================================================

class TestLearningLoop:
    """Tests for LearningLoop."""

    def test_loop_creation(self):
        """Test creating learning loop."""
        loop = LearningLoop()
        assert loop is not None
        assert len(loop.state.strategies) == 7  # Built-in strategies

    def test_discover_strategy(self):
        """Test strategy discovery."""
        loop = LearningLoop()
        # Need some data first
        for i in range(5):
            loop.db.record_result(
                CHAIN_OF_THOUGHT,
                "factual",
                StrategyResult("cot", True, 100, 0.8, "factual"),
            )

        new_strategy = loop.discover_strategy("factual")
        assert new_strategy is not None

    def test_retire_builtin_fails(self):
        """Test that built-in strategies cannot be retired."""
        loop = LearningLoop()
        result = loop.retire_strategy(CHAIN_OF_THOUGHT)
        assert result is False


class TestIncrementalLearner:
    """Tests for IncrementalLearner."""

    def test_learner_creation(self):
        """Test creating incremental learner."""
        learner = IncrementalLearner()
        assert learner is not None

    def test_learn_from_interaction(self):
        """Test learning from interaction."""
        learner = IncrementalLearner()
        result = learner.learn_from_interaction(
            problem="What is 2+2?",
            strategy=CHAIN_OF_THOUGHT,
            output="The answer is 4",
            success=True,
            tokens_used=100,
        )
        assert result.success is True
        assert learner.interaction_count == 1


# ============================================================================
# Harness Tests
# ============================================================================

class TestMetaHarness:
    """Tests for MetaHarness."""

    def test_harness_creation(self):
        """Test creating harness."""
        harness = MetaHarness()
        assert harness is not None

    @pytest.mark.asyncio
    async def test_harness_solve(self):
        """Test solving with harness."""
        async def mock_executor(prompt):
            return {"output": "Paris", "tokens_used": 50}

        harness = MetaHarness()
        harness.set_executor(mock_executor)

        result = await harness.solve("What is the capital of France?")
        assert result.output == "Paris"
        assert result.result.tokens_used == 50

    @pytest.mark.asyncio
    async def test_harness_solve_batch(self):
        """Test batch solving."""
        async def mock_executor(prompt):
            return {"output": "Answer", "tokens_used": 50}

        harness = MetaHarness()
        harness.set_executor(mock_executor)

        problems = [
            ("Question 1?", None),
            ("Question 2?", None),
        ]
        result = await harness.solve_batch(problems, learn=False)
        assert result.problems_solved == 2

    def test_harness_statistics(self):
        """Test getting harness statistics."""
        harness = MetaHarness()
        stats = harness.get_statistics()
        assert "solve_count" in stats


class TestSimplifiedHarness:
    """Tests for SimplifiedHarness."""

    @pytest.mark.asyncio
    async def test_simplified_harness(self):
        """Test simplified harness."""
        async def mock_executor(prompt):
            return "Simple answer"

        harness = SimplifiedHarness(mock_executor)
        result = await harness.solve("Test problem")
        assert result["output"] == "Simple answer"


# ============================================================================
# Bootstrap Tests
# ============================================================================

class TestBootstrapOptimizer:
    """Tests for BootstrapOptimizer."""

    def test_optimizer_creation(self):
        """Test creating bootstrap optimizer."""
        optimizer = BootstrapOptimizer()
        assert optimizer is not None

    @pytest.mark.asyncio
    async def test_bootstrap_basic(self):
        """Test basic bootstrap optimization."""
        async def mock_executor(prompt):
            return {"output": "Answer", "tokens_used": 50}

        optimizer = BootstrapOptimizer(max_depth=1)
        optimizer.set_executor(mock_executor)

        problems = [("Q1?", "A1"), ("Q2?", "A2")]
        result = await optimizer.bootstrap_optimization(problems, iterations=2, depth=0)
        assert isinstance(result, BootstrapResult)
        assert result.iterations == 2


# ============================================================================
# Anti-patterns Tests
# ============================================================================

class TestAntiPatternDB:
    """Tests for AntiPatternDB."""

    def test_db_creation(self):
        """Test creating anti-pattern database."""
        db = AntiPatternDB()
        assert db is not None

    def test_record_antipattern(self):
        """Test recording anti-pattern."""
        db = AntiPatternDB()
        pattern = db.record_antipattern(
            CHAIN_OF_THOUGHT,
            "creative",
            "Too rigid for creative tasks",
            "Write a poem",
        )
        assert pattern.strategy_id == CHAIN_OF_THOUGHT.id
        assert pattern.failure_count == 1

    def test_is_antipattern(self):
        """Test checking anti-pattern."""
        db = AntiPatternDB()
        db.failure_threshold = 2

        # Record multiple failures
        for i in range(3):
            db.record_antipattern(
                MINIMAL,
                "analytical",
                "Too simple for analysis",
            )

        assert db.is_antipattern(MINIMAL, "analytical")
        assert not db.is_antipattern(CHAIN_OF_THOUGHT, "analytical")

    def test_get_blacklist(self):
        """Test getting strategy blacklist."""
        db = AntiPatternDB()
        db.failure_threshold = 1

        db.record_antipattern(MINIMAL, "complex", "Reason")
        db.record_antipattern(MINIMAL, "complex", "Reason 2")

        blacklist = db.get_strategy_blacklist("complex")
        assert MINIMAL.id in blacklist

    def test_remove_antipattern(self):
        """Test removing anti-pattern."""
        db = AntiPatternDB()
        db.record_antipattern(CHAIN_OF_THOUGHT, "test", "Reason")

        removed = db.remove_antipattern(CHAIN_OF_THOUGHT.id, "test")
        assert removed is True

        pattern = db.get_antipattern(CHAIN_OF_THOUGHT.id, "test")
        assert pattern is None

    def test_decay_antipatterns(self):
        """Test decaying anti-pattern severity."""
        db = AntiPatternDB()
        db.record_antipattern(CHAIN_OF_THOUGHT, "test", "Reason", severity=1.0)

        decayed = db.decay_antipatterns(0.5)
        assert decayed == 1

        pattern = db.get_antipattern(CHAIN_OF_THOUGHT.id, "test")
        assert pattern.severity == 0.5


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test full meta-learning workflow."""
        # Setup
        async def mock_executor(prompt):
            if "capital" in prompt.lower():
                return {"output": "Paris", "tokens_used": 50}
            return {"output": "Unknown", "tokens_used": 30}

        harness = MetaHarness()
        harness.set_executor(mock_executor)

        # Solve problems
        result1 = await harness.solve("What is the capital of France?", "Paris")
        assert result1.result.success

        result2 = await harness.solve("What is the capital of Germany?", "Berlin")

        # Check learning
        stats = harness.get_statistics()
        assert stats["solve_count"] == 2

    def test_strategy_lifecycle(self):
        """Test strategy creation, mutation, and evaluation."""
        # Create custom strategy
        s = create_custom_strategy(
            id="custom",
            name="Custom",
            description="Test",
            prompt_template="Solve: {problem}",
        )

        # Mutate it
        mutation = mutate_strategy(s)
        assert mutation.new_strategy.id != s.id

        # Crossover with built-in
        child = crossover_strategies(s, CHAIN_OF_THOUGHT)
        assert len(child.parent_ids) == 2

        # Evaluate
        evaluator = StrategyEvaluator()
        result = evaluator.evaluate_strategy_sync(
            child, "Test problem", "Test output", 100
        )
        assert result.strategy_id == child.id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
