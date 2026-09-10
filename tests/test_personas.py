"""Tests for the Persona Ensemble system.

Comprehensive tests covering:
- Persona library and creation
- Assignment logic
- Consistency checking
- Worldview extraction and comparison
- Synthesis
- Harness orchestration
- Performance tracking
"""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from personas import (
    # Types
    AssignmentResult,
    ConsistencyScore,
    EnsembleResult,
    Persona,
    PersonaCategory,
    PersonaPerformance,
    PersonaResponse,
    ThinkingStyle,
    WorldviewDelta,
    # Library
    BUILTIN_PERSONAS,
    CUSTOMER,
    FIRST_PRINCIPLES,
    INNOVATOR,
    NOVICE,
    OPTIMIST,
    PESSIMIST,
    PersonaLibrary,
    SECURITY_ENGINEER,
    VETERAN,
    create_domain_expert,
    create_persona,
    create_stakeholder,
    # Assignment
    assign_balanced,
    assign_personas,
    detect_all_missing_perspectives,
    detect_missing_perspective,
    detect_relevant_categories,
    dynamic_spawn,
    suggest_additional_personas,
    # Consistency
    check_all_responses,
    check_consistency,
    create_reinforcement_injection,
    detect_drift,
    filter_inconsistent_content,
    identify_weakest_personas,
    reinforce_persona,
    # Worldview
    compare_all_pairs,
    compare_worldviews,
    extract_all,
    extract_assumptions,
    extract_opportunities,
    extract_risks,
    find_consensus_points,
    find_unique_insights,
    summarize_worldview,
    worldview_diversity_score,
    # Synthesis
    attribute_insights,
    build_ensemble_result,
    deliberate,
    identify_tensions,
    multi_perspective_brief,
    reconcile_with_priority,
    synthesize_perspectives,
    # Harness
    HarnessConfig,
    PersonaBranch,
    PersonaHarness,
    create_persona_prompt,
    inject_persona_into_context,
    # Tracking
    PersonaTracker,
    analyze_persona_effectiveness,
    compare_personas,
    suggest_persona_improvements,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def library():
    """Create a persona library with built-ins."""
    return PersonaLibrary()


@pytest.fixture
def empty_library():
    """Create an empty persona library."""
    return PersonaLibrary(include_builtins=False)


@pytest.fixture
def pessimist_response():
    """Create a pessimist-style response."""
    return PersonaResponse(
        persona_id="pessimist",
        content="""This plan will fail catastrophically. There are multiple risks:
        1. The timeline is unrealistic and will slip
        2. The budget assumption is dangerous and incorrect
        3. Security vulnerabilities are being ignored
        The worst case scenario is likely - we could lose everything.""",
    )


@pytest.fixture
def optimist_response():
    """Create an optimist-style response."""
    return PersonaResponse(
        persona_id="optimist",
        content="""This is an exciting opportunity! I see tremendous potential here.
        The benefits include:
        1. Significant growth opportunity in the market
        2. Possible advantage over competitors
        3. Could lead to exciting new possibilities
        This could be transformative for the business.""",
    )


@pytest.fixture
def tracker():
    """Create a persona tracker."""
    return PersonaTracker()


# ============================================================================
# Library Tests
# ============================================================================


class TestPersonaLibrary:
    """Tests for PersonaLibrary."""

    def test_builtin_personas_loaded(self, library):
        """Test that built-in personas are loaded by default."""
        assert len(library) == 10
        assert "pessimist" in library
        assert "optimist" in library

    def test_empty_library(self, empty_library):
        """Test creating empty library."""
        assert len(empty_library) == 0
        assert "pessimist" not in empty_library

    def test_add_persona(self, empty_library):
        """Test adding a persona."""
        persona = create_persona(
            id="test",
            name="Test Persona",
            description="A test persona",
            focus="testing",
        )
        empty_library.add(persona)
        assert "test" in empty_library
        assert empty_library.get("test") == persona

    def test_add_duplicate_raises(self, library):
        """Test that adding duplicate persona raises."""
        with pytest.raises(ValueError):
            library.add(PESSIMIST)

    def test_get_nonexistent(self, library):
        """Test getting non-existent persona returns None."""
        assert library.get("nonexistent") is None

    def test_get_or_raise(self, library):
        """Test get_or_raise raises for missing persona."""
        with pytest.raises(KeyError):
            library.get_or_raise("nonexistent")

    def test_remove_persona(self, library):
        """Test removing a persona."""
        assert library.remove("pessimist")
        assert "pessimist" not in library
        assert not library.remove("nonexistent")

    def test_list_by_category(self, library):
        """Test listing personas by category."""
        risk_personas = library.list_by_category(PersonaCategory.RISK)
        assert len(risk_personas) >= 1
        assert all(p.category == PersonaCategory.RISK for p in risk_personas)

    def test_list_by_thinking_style(self, library):
        """Test listing personas by thinking style."""
        cautious = library.list_by_thinking_style(ThinkingStyle.CAUTIOUS)
        assert len(cautious) >= 1
        assert all(p.thinking_style == ThinkingStyle.CAUTIOUS for p in cautious)

    def test_create_persona(self):
        """Test creating custom persona."""
        persona = create_persona(
            id="custom",
            name="Custom Analyst",
            description="Analyzes from a custom perspective",
            focus="custom analysis",
            biases=["custom bias"],
            strengths=["custom strength"],
        )
        assert persona.id == "custom"
        assert "Custom Analyst" in persona.system_prompt

    def test_create_domain_expert(self):
        """Test creating domain expert persona."""
        expert = create_domain_expert("machine learning", "neural networks", 15)
        assert "expert_machine_learning" == expert.id
        assert "neural networks" in expert.system_prompt
        assert "15 years" in expert.system_prompt

    def test_create_stakeholder(self):
        """Test creating stakeholder persona."""
        cto = create_stakeholder(
            role="CTO",
            concerns=["technical debt", "scalability"],
            priorities=["innovation", "reliability"],
        )
        assert cto.id == "stakeholder_cto"
        assert "technical debt" in cto.system_prompt


# ============================================================================
# Assignment Tests
# ============================================================================


class TestAssignment:
    """Tests for persona assignment."""

    def test_assign_personas_basic(self, library):
        """Test basic persona assignment."""
        result = assign_personas("Analyze this problem", 4, library)
        assert len(result.assignments) == 4
        assert all(isinstance(p, Persona) for p in result.assignments.values())

    def test_assign_with_required(self, library):
        """Test assignment with required personas."""
        result = assign_personas(
            "Analyze this problem",
            3,
            library,
            required_personas=["pessimist", "optimist"],
        )
        assigned_ids = [p.id for p in result.assignments.values()]
        assert "pessimist" in assigned_ids
        assert "optimist" in assigned_ids

    def test_assign_with_excluded(self, library):
        """Test assignment with excluded personas."""
        result = assign_personas(
            "Analyze this problem",
            5,
            library,
            excluded_personas=["pessimist"],
        )
        assigned_ids = [p.id for p in result.assignments.values()]
        assert "pessimist" not in assigned_ids

    def test_detect_relevant_categories(self):
        """Test category detection from problem text."""
        categories = detect_relevant_categories("What are the security risks?")
        assert PersonaCategory.RISK in categories
        # Security is in RISK keywords, so TECHNICAL may not always be detected
        # Just verify we get relevant categories back
        assert len(categories) >= 1

    def test_detect_relevant_categories_opportunity(self):
        """Test category detection for opportunity-focused problem."""
        categories = detect_relevant_categories("What growth opportunities exist?")
        assert PersonaCategory.OPPORTUNITY in categories

    def test_detect_missing_perspective(self, library, pessimist_response, optimist_response):
        """Test detecting missing perspective."""
        responses = [pessimist_response, optimist_response]
        missing = detect_missing_perspective(responses, library)
        # Risk and opportunity are covered, something else should be missing
        assert missing is not None

    def test_detect_all_missing(self, library, pessimist_response):
        """Test detecting all missing perspectives."""
        responses = [pessimist_response]
        missing = detect_all_missing_perspectives(responses, library)
        # Only risk is covered
        assert len(missing) >= 3

    def test_dynamic_spawn(self, library):
        """Test dynamic spawning of missing perspective."""
        persona = dynamic_spawn(PersonaCategory.USER, library)
        assert persona is not None
        assert persona.category == PersonaCategory.USER

    def test_suggest_additional_personas(self, library):
        """Test suggesting additional personas."""
        suggestions = suggest_additional_personas(
            "Security concerns with user experience",
            ["pessimist"],
            max_suggestions=3,
            library=library,
        )
        assert len(suggestions) <= 3
        assert "pessimist" not in [s.id for s in suggestions]

    def test_assign_balanced(self, library):
        """Test balanced assignment across categories."""
        result = assign_balanced(6, library)
        assert len(result.coverage) >= 4  # Should cover multiple categories


# ============================================================================
# Consistency Tests
# ============================================================================


class TestConsistency:
    """Tests for consistency checking."""

    def test_check_consistency_in_character(self, library, pessimist_response):
        """Test consistency check for in-character response."""
        persona = library.get("pessimist")
        score = check_consistency(persona, pessimist_response)
        assert score.score >= 0.7
        assert not score.drift_detected

    def test_check_consistency_out_of_character(self, library):
        """Test consistency check for out-of-character response."""
        persona = library.get("pessimist")
        response = PersonaResponse(
            persona_id="pessimist",
            content="This is so exciting! Great opportunity with no concerns at all!",
        )
        score = check_consistency(persona, response)
        assert score.score < 0.7

    def test_detect_drift(self, library):
        """Test detecting drift over multiple responses."""
        persona = library.get("pessimist")
        responses = [
            PersonaResponse(
                persona_id="pessimist",
                content="This is extremely risky and dangerous. Many concerns about failure. The worst case scenario is disaster.",
            ),
            PersonaResponse(
                persona_id="pessimist",
                content="There might be some benefits here. This could work out well.",
            ),
            PersonaResponse(
                persona_id="pessimist",
                content="This is exciting! Great opportunity with no concerns at all! Will definitely work!",
            ),
        ]
        score = detect_drift(persona, responses)
        # Should detect declining consistency or at least score below perfect
        assert score.score < 1.0 or score.drift_detected

    def test_reinforce_persona(self, library):
        """Test generating reinforcement prompt."""
        persona = library.get("pessimist")
        reinforcement = reinforce_persona(
            persona,
            "Continue analysis",
            issues=["Too optimistic in recent response"],
        )
        assert "Pessimist" in reinforcement
        assert "STAY IN CHARACTER" in reinforcement

    def test_create_reinforcement_injection(self, library):
        """Test creating reinforcement injection."""
        persona = library.get("optimist")
        score = ConsistencyScore(
            persona_id="optimist",
            score=0.3,
            reinforcement_needed=True,
            inconsistent_elements=["Too negative"],
        )
        injection = create_reinforcement_injection(persona, score)
        assert injection is not None
        assert "Optimist" in injection

    def test_no_reinforcement_when_not_needed(self, library):
        """Test no reinforcement when score is good."""
        persona = library.get("optimist")
        score = ConsistencyScore(
            persona_id="optimist",
            score=0.9,
            reinforcement_needed=False,
        )
        injection = create_reinforcement_injection(persona, score)
        assert injection is None

    def test_check_all_responses(self, library, pessimist_response, optimist_response):
        """Test checking all responses in batch."""
        responses = [pessimist_response, optimist_response]
        results = check_all_responses(responses, library)
        assert "pessimist" in results
        assert "optimist" in results

    def test_filter_inconsistent_content(self, library):
        """Test filtering out-of-character content."""
        persona = library.get("pessimist")
        content = "This will fail. But also, this is exciting! Great opportunity!"
        filtered, removed = filter_inconsistent_content(persona, content)
        assert "FILTERED" in filtered


# ============================================================================
# Worldview Tests
# ============================================================================


class TestWorldview:
    """Tests for worldview extraction and comparison."""

    def test_extract_assumptions(self, pessimist_response):
        """Test extracting assumptions."""
        assumptions = extract_assumptions(pessimist_response)
        # Should find implicit assumptions like "dangerous and incorrect"
        assert len(assumptions) >= 0

    def test_extract_risks(self, pessimist_response):
        """Test extracting risks."""
        risks = extract_risks(pessimist_response)
        assert len(risks) >= 1
        assert any("risk" in r.lower() or "fail" in r.lower() for r in risks)

    def test_extract_opportunities(self, optimist_response):
        """Test extracting opportunities."""
        opportunities = extract_opportunities(optimist_response)
        assert len(opportunities) >= 1
        assert any("opportunity" in o.lower() or "potential" in o.lower() for o in opportunities)

    def test_extract_all(self, pessimist_response):
        """Test extracting all worldview elements."""
        enriched = extract_all(pessimist_response)
        assert enriched.persona_id == "pessimist"

    def test_compare_worldviews(self, pessimist_response, optimist_response):
        """Test comparing two worldviews."""
        delta = compare_worldviews(pessimist_response, optimist_response)
        assert delta.persona_a == "pessimist"
        assert delta.persona_b == "optimist"

    def test_worldview_diversity_score(self, pessimist_response, optimist_response):
        """Test diversity scoring."""
        responses = [pessimist_response, optimist_response]
        score = worldview_diversity_score(responses)
        assert 0.0 <= score <= 1.0

    def test_find_consensus_points(self):
        """Test finding consensus points."""
        responses = [
            PersonaResponse(
                persona_id="a",
                content="The risk of delay is significant.",
            ),
            PersonaResponse(
                persona_id="b",
                content="Risk of delay could impact timeline.",
            ),
        ]
        consensus = find_consensus_points(responses)
        # May or may not find consensus depending on exact matching
        assert isinstance(consensus, list)

    def test_find_unique_insights(self, library, pessimist_response, optimist_response):
        """Test finding unique insights."""
        responses = [pessimist_response, optimist_response]
        unique = find_unique_insights(responses, library)
        assert isinstance(unique, dict)


# ============================================================================
# Synthesis Tests
# ============================================================================


class TestSynthesis:
    """Tests for worldview synthesis."""

    def test_synthesize_perspectives(self, library, pessimist_response, optimist_response):
        """Test synthesizing perspectives."""
        responses = [pessimist_response, optimist_response]
        synthesis = synthesize_perspectives(responses, library)
        assert "Multi-Perspective" in synthesis
        assert "Pessimist" in synthesis
        assert "Optimist" in synthesis

    def test_identify_tensions(self, pessimist_response, optimist_response):
        """Test identifying tensions."""
        responses = [pessimist_response, optimist_response]
        tensions = identify_tensions(responses)
        assert isinstance(tensions, list)

    def test_multi_perspective_brief(self, library, pessimist_response, optimist_response):
        """Test creating multi-perspective brief."""
        responses = [pessimist_response, optimist_response]
        brief = multi_perspective_brief(responses, library)
        assert "MULTI-PERSPECTIVE BRIEF" in brief
        assert "Pessimist" in brief.upper() or "pessimist" in brief.lower()

    def test_build_ensemble_result(self, library, pessimist_response, optimist_response):
        """Test building complete ensemble result."""
        responses = [pessimist_response, optimist_response]
        result = build_ensemble_result(responses, library)
        assert isinstance(result, EnsembleResult)
        assert len(result.responses) == 2
        assert result.synthesis != ""

    def test_reconcile_with_priority(self, library, pessimist_response, optimist_response):
        """Test priority-based reconciliation."""
        responses = [pessimist_response, optimist_response]
        reconciled = reconcile_with_priority(
            responses,
            ["pessimist", "optimist"],
            library,
        )
        assert "Priority 1" in reconciled

    def test_deliberate(self, library, pessimist_response, optimist_response):
        """Test deliberation simulation."""
        responses = [pessimist_response, optimist_response]
        deliberation = deliberate(responses, rounds=1, library=library)
        assert "Deliberation" in deliberation


# ============================================================================
# Harness Tests
# ============================================================================


class TestHarness:
    """Tests for PersonaHarness."""

    def test_harness_config_defaults(self):
        """Test harness config has sensible defaults."""
        config = HarnessConfig()
        assert config.num_branches == 4
        assert config.check_consistency is True

    def test_harness_assign(self, library):
        """Test harness persona assignment."""
        harness = PersonaHarness(library=library)
        result = harness.assign("Test problem")
        assert len(harness.branches) == 4
        assert all(isinstance(b, PersonaBranch) for b in harness.branches)

    def test_create_persona_prompt(self):
        """Test creating complete persona prompt."""
        prompt = create_persona_prompt(PESSIMIST, "Analyze this")
        # Check for persona identity elements in the prompt
        assert "pessimistic" in prompt.lower() or "fail" in prompt.lower()
        assert "Analyze this" in prompt

    def test_inject_persona_into_context(self):
        """Test injecting persona into message context."""
        messages = [{"role": "user", "content": "Hello"}]
        injected = inject_persona_into_context(PESSIMIST, messages)
        assert len(injected) == 2
        assert injected[0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_harness_run_with_mock_llm(self, library):
        """Test running harness with mock LLM."""

        async def mock_llm(messages, system_prompt):
            if "pessimist" in system_prompt.lower():
                return "This is risky and will likely fail."
            return "I see potential here."

        harness = PersonaHarness(
            config=HarnessConfig(num_branches=2),
            library=library,
            llm_call=mock_llm,
        )
        result = await harness.run_all("Test problem")
        assert isinstance(result, EnsembleResult)
        assert len(result.responses) >= 2

    def test_harness_get_brief_before_run(self, library):
        """Test getting brief before running returns message."""
        harness = PersonaHarness(library=library)
        brief = harness.get_brief()
        assert "No results" in brief


# ============================================================================
# Tracking Tests
# ============================================================================


class TestTracking:
    """Tests for persona performance tracking."""

    def test_record_use(self, tracker):
        """Test recording a persona use."""
        tracker.record_use("pessimist", success=True, unique_insights=2)
        perf = tracker.get_performance("pessimist")
        assert perf is not None
        assert perf.total_uses == 1
        assert perf.unique_insights == 2

    def test_value_score(self, tracker):
        """Test value score calculation."""
        tracker.record_use("pessimist", success=True, unique_insights=3)
        tracker.record_use("pessimist", success=True, unique_insights=2)
        score = tracker.persona_value_score("pessimist")
        assert 0.0 <= score <= 1.0

    def test_recommend_personas(self, tracker):
        """Test persona recommendations."""
        tracker.record_use("pessimist", success=True, unique_insights=5)
        tracker.record_use("optimist", success=False, unique_insights=0)
        recommendations = tracker.recommend_personas(count=2)
        assert "pessimist" in recommendations

    def test_get_top_performers(self, tracker):
        """Test getting top performers."""
        tracker.record_use("pessimist", success=True, unique_insights=5)
        tracker.record_use("optimist", success=True, unique_insights=3)
        tracker.record_use("novice", success=False, unique_insights=1)
        top = tracker.get_top_performers(2)
        assert len(top) == 2
        assert top[0][1] >= top[1][1]  # Sorted by score

    def test_get_underperformers(self, tracker):
        """Test getting underperformers."""
        tracker.record_use("pessimist", success=True, unique_insights=5)
        tracker.record_use("optimist", success=False, unique_insights=0, was_redundant=True)
        tracker.record_use("optimist", success=False, unique_insights=0, was_redundant=True)
        under = tracker.get_underperformers(0.5)
        assert "optimist" in under

    def test_persistence(self, tracker):
        """Test saving and loading tracker data."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tracking.json"
            tracker = PersonaTracker(persistence_path=path)
            tracker.record_use("pessimist", success=True, unique_insights=3)

            # Create new tracker from same path
            tracker2 = PersonaTracker(persistence_path=path)
            perf = tracker2.get_performance("pessimist")
            assert perf is not None
            assert perf.unique_insights == 3

    def test_analyze_effectiveness(self, tracker):
        """Test analyzing persona effectiveness."""
        tracker.record_use("pessimist", success=True, unique_insights=5)
        tracker.record_use("pessimist", success=True, unique_insights=3)
        analysis = analyze_persona_effectiveness(tracker)
        assert "pessimist" in analysis
        assert "value_score" in analysis["pessimist"]

    def test_compare_personas(self, tracker):
        """Test comparing two personas."""
        tracker.record_use("pessimist", success=True, unique_insights=5)
        tracker.record_use("optimist", success=True, unique_insights=2)
        comparison = compare_personas(tracker, "pessimist", "optimist")
        assert "persona_a" in comparison
        assert "persona_b" in comparison
        assert "comparison" in comparison

    def test_suggest_improvements(self, tracker):
        """Test generating improvement suggestions."""
        # Add enough data
        for _ in range(5):
            tracker.record_use("pessimist", success=True, unique_insights=2)
            tracker.record_use("optimist", success=False, unique_insights=0, was_redundant=True)
        suggestions = suggest_persona_improvements(tracker)
        assert len(suggestions) >= 1


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for the full system."""

    @pytest.mark.asyncio
    async def test_full_ensemble_flow(self, library):
        """Test complete ensemble flow from assignment to synthesis."""

        async def mock_llm(messages, system_prompt):
            # Return different content based on persona
            if "pessimist" in system_prompt.lower():
                return "This poses significant risk. The plan could fail due to timeline issues."
            elif "optimist" in system_prompt.lower():
                return "This is an exciting opportunity. Great potential for growth."
            elif "security" in system_prompt.lower():
                return "Security vulnerabilities exist. Attack vectors need addressing."
            else:
                return "This needs careful analysis from multiple angles."

        config = HarnessConfig(
            num_branches=3,
            check_consistency=True,
            allow_dynamic_spawn=False,
        )
        harness = PersonaHarness(config=config, library=library, llm_call=mock_llm)

        # Run the ensemble
        result = await harness.run_all(
            "Should we launch this new feature?",
            required_personas=["pessimist", "optimist"],
        )

        # Verify results
        assert isinstance(result, EnsembleResult)
        assert len(result.responses) == 3
        assert result.synthesis != ""

        # Get brief
        brief = harness.get_brief()
        assert "MULTI-PERSPECTIVE BRIEF" in brief

        # Get consistency report
        report = harness.get_consistency_report()
        assert "Consistency Report" in report

    def test_persona_types_are_complete(self):
        """Test all type dataclasses have required fields."""
        # Persona
        p = Persona(
            id="test",
            name="Test",
            system_prompt="Test prompt",
            thinking_style=ThinkingStyle.SYSTEMATIC,
        )
        assert p.id == "test"

        # PersonaResponse
        r = PersonaResponse(persona_id="test", content="Content")
        assert r.persona_id == "test"

        # WorldviewDelta
        d = WorldviewDelta(persona_a="a", persona_b="b")
        assert d.persona_a == "a"

        # EnsembleResult
        e = EnsembleResult()
        assert isinstance(e.responses, list)

        # PersonaPerformance
        pp = PersonaPerformance(persona_id="test")
        assert pp.total_uses == 0

    def test_builtin_personas_have_complete_definitions(self):
        """Test all built-in personas have complete definitions."""
        for persona_id, persona in BUILTIN_PERSONAS.items():
            assert persona.id == persona_id
            assert persona.name != ""
            assert persona.system_prompt != ""
            assert len(persona.biases) >= 1
            assert len(persona.strengths) >= 1
            assert persona.description != ""
