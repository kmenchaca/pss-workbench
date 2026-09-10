"""
Comprehensive tests for the Argumentative Swarm system.

Tests cover:
- Position assignment
- Argument construction and parsing
- Refutation generation
- Debate rounds
- Transcript management
- Adjudication
- Full debate flow
"""

import json
import pytest
from datetime import datetime

from argswarm import (
    # Types
    AdjudicationStrategy,
    Argument,
    Crux,
    DebateFormat,
    DebateState,
    Judgment,
    Position,
    Refutation,
    Response,
    RoundType,
    Stance,
    StrengthScore,
    # Positions
    assign_positions,
    balance_positions,
    create_position,
    get_allied_positions,
    get_opposing_positions,
    is_balanced,
    CustomPosition,
    # Arguments
    ArgumentBuilder,
    compare_arguments,
    construct_argument,
    create_argument_from_llm_response,
    evaluate_strength,
    find_weaknesses,
    parse_argument,
    strengthen_argument,
    validate_argument,
    # Debate
    CrossExaminationResult,
    DebateRound,
    RoundManager,
    calculate_position_score,
    cross_examine,
    rebuild,
    refute,
    should_eliminate_position,
    # Transcript
    DebateTranscript,
    TranscriptEntry,
    add_argument,
    add_note,
    add_refutation,
    add_response,
    create_transcript,
    export_to_json,
    export_to_text,
    get_addressed_points,
    get_unaddressed_arguments,
    import_from_json,
    post_to_bulletin_board,
    # Adjudication
    adjudicate,
    find_common_ground,
    identify_cruxes,
    weigh_arguments,
    # Harness
    DebateConfig,
    DebateHarness,
    GateCheck,
    run_simple_debate,
    # Formats
    FormatBuilder,
    OxfordFormat,
    LincolnDouglasFormat,
    SocraticFormat,
    RoundRobinFormat,
    QuickFormat,
    get_format_by_name,
    list_formats,
    describe_format,
    validate_format,
    estimate_debate_duration,
)


# =============================================================================
# Position Tests
# =============================================================================


class TestPositionAssignment:
    """Tests for position assignment functionality."""

    def test_assign_two_positions(self):
        """Two positions should be FOR and AGAINST."""
        positions = assign_positions("Test proposition", 2)
        assert len(positions) == 2
        stances = {p.stance for p in positions}
        assert Stance.FOR in stances
        assert Stance.AGAINST in stances

    def test_assign_single_position(self):
        """Single position should be NEUTRAL."""
        positions = assign_positions("Test proposition", 1)
        assert len(positions) == 1
        assert positions[0].stance == Stance.NEUTRAL

    def test_assign_three_positions_with_special(self):
        """Three positions with special should include SYNTHESIS."""
        positions = assign_positions("Test proposition", 3, include_special=True)
        assert len(positions) == 3
        stances = {p.stance for p in positions}
        assert Stance.SYNTHESIS in stances

    def test_assign_zero_positions(self):
        """Zero positions should return empty list."""
        positions = assign_positions("Test proposition", 0)
        assert positions == []

    def test_custom_stances(self):
        """Custom stances should be respected."""
        custom = [Stance.FOR, Stance.FOR, Stance.AGAINST]
        positions = assign_positions("Test", 3, custom_stances=custom)
        stances = [p.stance for p in positions]
        assert stances == custom

    def test_position_has_branch_id(self):
        """Positions should have branch IDs."""
        positions = assign_positions("Test", 2)
        assert all(p.branch_id is not None for p in positions)

    def test_get_opposing_positions(self):
        """FOR should oppose AGAINST and vice versa."""
        positions = assign_positions("Test", 2)
        for_pos = next(p for p in positions if p.stance == Stance.FOR)
        against_pos = next(p for p in positions if p.stance == Stance.AGAINST)

        opponents = get_opposing_positions(for_pos, positions)
        assert against_pos in opponents

    def test_get_allied_positions(self):
        """Allied positions should share stance."""
        positions = assign_positions("Test", 4)  # 2 FOR, 2 AGAINST
        for_positions = [p for p in positions if p.stance == Stance.FOR]

        if len(for_positions) >= 2:
            allies = get_allied_positions(for_positions[0], positions)
            assert for_positions[1] in allies

    def test_balance_positions(self):
        """Balance should return stance counts."""
        positions = assign_positions("Test", 4)
        balance = balance_positions(positions)
        assert isinstance(balance, dict)
        assert "for" in balance or "against" in balance

    def test_is_balanced(self):
        """Two-position debate should be balanced."""
        positions = assign_positions("Test", 2)
        assert is_balanced(positions)


class TestCustomPosition:
    """Tests for custom position types."""

    def test_create_custom_position(self):
        """Custom positions should be creatable."""
        custom = CustomPosition("Empiricist", "Judges based on empirical evidence")
        position = custom.create("Test proposition")
        assert position.stance == Stance.NEUTRAL
        assert "Empiricist" in position.description

    def test_custom_position_chaining(self):
        """Custom position builder should support chaining."""
        custom = (CustomPosition("Test", "Test desc")
                  .opposes(Stance.FOR)
                  .allies_with(Stance.AGAINST))
        assert Stance.FOR in custom._opposes
        assert Stance.AGAINST in custom._allies


# =============================================================================
# Argument Tests
# =============================================================================


class TestArgumentConstruction:
    """Tests for argument construction."""

    def test_construct_basic_argument(self):
        """Basic argument should be constructable."""
        position = create_position(Stance.FOR, "Test")
        arg = construct_argument(
            position=position,
            claim="X is better than Y",
            evidence="Studies show X outperforms Y",
            warrant="Better performance indicates superiority",
        )
        assert arg.claim == "X is better than Y"
        assert arg.position_id == position.id
        assert 0 <= arg.strength <= 1

    def test_argument_builder(self):
        """ArgumentBuilder should create valid arguments."""
        position = create_position(Stance.FOR, "Test")
        arg = (ArgumentBuilder(position)
               .claim("Test claim")
               .evidence("Test evidence")
               .warrant("Test warrant")
               .qualifier("probably")
               .build())
        assert arg.claim == "Test claim"
        assert arg.qualifier == "probably"

    def test_evaluate_strength_complete(self):
        """Complete arguments should have higher strength."""
        position = create_position(Stance.FOR, "Test")
        complete = construct_argument(
            position=position,
            claim="Test",
            evidence="Strong evidence",
            warrant="Clear reasoning",
            backing="Additional support",
            rebuttal="Unless X happens",
        )
        minimal = construct_argument(
            position=position,
            claim="Test",
            evidence="",
            warrant="",
        )
        assert complete.strength > minimal.strength

    def test_find_weaknesses_missing_evidence(self):
        """Missing evidence should be identified."""
        position = create_position(Stance.FOR, "Test")
        arg = construct_argument(
            position=position,
            claim="Test",
            evidence="",
            warrant="Test",
        )
        weaknesses = find_weaknesses(arg)
        types = [w["type"] for w in weaknesses]
        assert "missing_evidence" in types

    def test_strengthen_argument(self):
        """Strengthened arguments should have higher strength."""
        position = create_position(Stance.FOR, "Test")
        original = construct_argument(
            position=position,
            claim="Test",
            evidence="Basic evidence",
            warrant="Basic warrant",
        )
        strengthened = strengthen_argument(
            original,
            additional_evidence="More detailed evidence with data",
            add_backing="Supported by theory",
        )
        assert strengthened.strength >= original.strength

    def test_validate_argument_valid(self):
        """Valid arguments should pass validation."""
        position = create_position(Stance.FOR, "Test")
        arg = construct_argument(
            position=position,
            claim="Test claim",
            evidence="Evidence",
            warrant="Warrant",
        )
        is_valid, issues = validate_argument(arg)
        assert is_valid
        assert len(issues) == 0

    def test_validate_argument_no_claim(self):
        """Arguments without claims should fail validation."""
        arg = Argument(
            id="test",
            claim="",
            evidence="Evidence",
            warrant="Warrant",
            position_id="pos_1",
        )
        is_valid, issues = validate_argument(arg)
        assert not is_valid
        assert "claim" in issues[0].lower()


class TestArgumentParsing:
    """Tests for parsing arguments from text."""

    def test_parse_basic_argument(self):
        """Basic argument text should be parseable."""
        text = "I argue that X is true. Evidence: studies show X. Because this demonstrates validity."
        components = parse_argument(text)
        assert "claim" in components
        assert components["claim"]  # Non-empty

    def test_parse_with_qualifier(self):
        """Qualifiers should be extracted."""
        text = "X is probably true based on the evidence."
        components = parse_argument(text)
        assert components.get("qualifier") == "probably"

    def test_create_from_llm_response(self):
        """LLM responses should be convertible to arguments."""
        position = create_position(Stance.FOR, "Test")
        response = "I claim that renewable energy is superior. Evidence: costs have dropped 90%."
        arg = create_argument_from_llm_response(response, position)
        assert arg.claim
        assert arg.position_id == position.id


# =============================================================================
# Debate Mechanics Tests
# =============================================================================


class TestRefutation:
    """Tests for refutation functionality."""

    def test_create_refutation(self):
        """Refutations should be creatable."""
        position_for = create_position(Stance.FOR, "Test")
        position_against = create_position(Stance.AGAINST, "Test")
        arg = construct_argument(
            position=position_for,
            claim="Test claim",
            evidence="Evidence",
            warrant="Warrant",
        )
        ref = refute(
            argument=arg,
            refuting_position=position_against,
            counter_claim="This is incorrect because...",
            evidence="Counter evidence",
        )
        assert ref.target_argument_id == arg.id
        assert ref.position_id == position_against.id

    def test_refutation_types(self):
        """Different refutation types should be supported."""
        position_for = create_position(Stance.FOR, "Test")
        position_against = create_position(Stance.AGAINST, "Test")
        arg = construct_argument(position_for, "Claim", "Evidence", "Warrant")

        for ref_type in ["undermine", "rebut", "counter", "deny"]:
            ref = refute(arg, position_against, "Counter", "Evidence", ref_type)
            assert ref.refutation_type == ref_type


class TestDebateRounds:
    """Tests for debate round management."""

    def test_round_manager_start(self):
        """Round manager should start rounds."""
        state = DebateState(proposition="Test")
        state.active_positions = assign_positions("Test", 2)
        manager = RoundManager(state, [RoundType.OPENING, RoundType.CLOSING])

        round = manager.start_round()
        assert round.round_number == 1
        assert round.round_type == RoundType.OPENING

    def test_round_manager_end_round(self):
        """Ending round should advance to next."""
        state = DebateState(proposition="Test")
        state.active_positions = assign_positions("Test", 2)
        manager = RoundManager(state, [RoundType.OPENING, RoundType.CLOSING])

        manager.start_round()
        manager.end_round()
        assert manager.current_round_index == 1

    def test_move_validity(self):
        """Move validity should depend on round type."""
        state = DebateState(proposition="Test")
        state.active_positions = assign_positions("Test", 2)
        manager = RoundManager(state, [RoundType.OPENING])

        manager.start_round()
        position = state.active_positions[0]

        # Arguments allowed in opening
        assert manager.is_move_valid("argument", position)
        # Refutations not allowed in opening
        assert not manager.is_move_valid("refutation", position)


class TestCrossExamination:
    """Tests for cross-examination."""

    def test_cross_examine_generates_questions(self):
        """Cross-examination should generate questions."""
        position_for = create_position(Stance.FOR, "Test")
        position_against = create_position(Stance.AGAINST, "Test")
        arg = construct_argument(position_for, "Claim", "", "")  # Weak argument

        result = cross_examine(position_against, arg)
        assert len(result.questions) > 0

    def test_cross_examine_finds_weaknesses(self):
        """Cross-examination should expose weaknesses."""
        position_for = create_position(Stance.FOR, "Test")
        position_against = create_position(Stance.AGAINST, "Test")
        arg = construct_argument(position_for, "Claim", "", "")

        result = cross_examine(position_against, arg)
        assert len(result.weaknesses_exposed) > 0


class TestRebuild:
    """Tests for argument rebuilding."""

    def test_rebuild_strengthens_argument(self):
        """Rebuilding should strengthen arguments."""
        position = create_position(Stance.FOR, "Test")
        arg = construct_argument(position, "Claim", "Evidence", "Warrant")
        ref = Refutation(
            id="ref_1",
            target_argument_id=arg.id,
            counter_claim="Counter",
            evidence="Counter evidence",
        )

        strengthened, response = rebuild(
            arg, ref,
            additional_evidence="More evidence",
            address_refutation="Response to counter",
        )
        assert strengthened.strength >= arg.strength
        assert response.refutation_id == ref.id


# =============================================================================
# Transcript Tests
# =============================================================================


class TestTranscript:
    """Tests for transcript management."""

    def test_create_transcript(self):
        """Transcripts should be creatable."""
        positions = assign_positions("Test", 2)
        transcript = create_transcript("Test proposition", positions)
        assert transcript.proposition == "Test proposition"
        assert len(transcript.positions) == 2

    def test_add_argument_to_transcript(self):
        """Arguments should be addable to transcript."""
        positions = assign_positions("Test", 2)
        transcript = create_transcript("Test", positions)
        arg = construct_argument(positions[0], "Claim", "Evidence", "Warrant")

        add_argument(transcript, arg, positions[0], 1, RoundType.OPENING)
        assert len(transcript.entries) == 1
        assert transcript.entries[0].entry_type == "argument"

    def test_add_refutation_to_transcript(self):
        """Refutations should be addable to transcript."""
        positions = assign_positions("Test", 2)
        transcript = create_transcript("Test", positions)
        arg = construct_argument(positions[0], "Claim", "Evidence", "Warrant")
        ref = refute(arg, positions[1], "Counter", "Evidence")

        add_refutation(transcript, ref, positions[1], 2, RoundType.REBUTTAL)
        assert len(transcript.entries) == 1
        assert transcript.entries[0].entry_type == "refutation"

    def test_bulletin_board(self):
        """Bulletin board should accept messages."""
        positions = assign_positions("Test", 2)
        transcript = create_transcript("Test", positions)

        post_to_bulletin_board(transcript, "Test message", positions[0])
        assert len(transcript.bulletin_board) == 1

    def test_export_to_text(self):
        """Transcripts should export to text."""
        positions = assign_positions("Test", 2)
        transcript = create_transcript("Test proposition", positions)

        text = export_to_text(transcript)
        assert "Test proposition" in text
        assert "TRANSCRIPT" in text

    def test_export_import_json(self):
        """Transcripts should round-trip through JSON."""
        positions = assign_positions("Test", 2)
        transcript = create_transcript("Test proposition", positions)

        json_str = export_to_json(transcript)
        imported = import_from_json(json_str)

        assert imported.proposition == transcript.proposition
        assert len(imported.positions) == len(transcript.positions)

    def test_get_addressed_points(self):
        """Addressed points should be trackable."""
        positions = assign_positions("Test", 2)
        transcript = create_transcript("Test", positions)
        arg = construct_argument(positions[0], "Claim", "Evidence", "Warrant")
        ref = refute(arg, positions[1], "Counter", "Evidence")

        add_argument(transcript, arg, positions[0], 1, RoundType.OPENING)
        add_refutation(transcript, ref, positions[1], 2, RoundType.REBUTTAL)

        addressed = get_addressed_points(transcript)
        assert arg.id in addressed


# =============================================================================
# Adjudication Tests
# =============================================================================


class TestAdjudication:
    """Tests for adjudication functionality."""

    def test_weigh_arguments(self):
        """Arguments should be weighable."""
        position = create_position(Stance.FOR, "Test")
        args = [
            construct_argument(position, "Strong claim", "Good evidence", "Clear warrant"),
            construct_argument(position, "Weak claim", "", ""),
        ]

        weighted = weigh_arguments(args)
        assert len(weighted) == 2
        # Strong argument should score higher
        assert weighted[0][1] > weighted[1][1]

    def test_adjudicate_produces_judgment(self):
        """Adjudication should produce a judgment."""
        positions = assign_positions("Test", 2)
        state = DebateState(
            proposition="Test",
            active_positions=positions,
            arguments=[
                construct_argument(positions[0], "FOR claim", "Evidence", "Warrant"),
                construct_argument(positions[1], "AGAINST claim", "Evidence", "Warrant"),
            ],
        )
        transcript = create_transcript("Test", positions)

        judgment = adjudicate(transcript, state)
        assert isinstance(judgment, Judgment)
        assert judgment.reasoning

    def test_adjudication_strategies(self):
        """Different strategies should produce judgments."""
        positions = assign_positions("Test", 2)
        state = DebateState(
            proposition="Test",
            active_positions=positions,
            arguments=[
                construct_argument(positions[0], "Claim", "Evidence", "Warrant"),
            ],
        )
        transcript = create_transcript("Test", positions)

        for strategy in AdjudicationStrategy:
            judgment = adjudicate(transcript, state, strategy)
            assert judgment.strategy_used == strategy


# =============================================================================
# Harness Tests
# =============================================================================


class TestDebateHarness:
    """Tests for the main debate harness."""

    def test_create_harness(self):
        """Harness should be creatable."""
        config = DebateConfig(format=QuickFormat())
        harness = DebateHarness("Test proposition", config)
        assert harness.proposition == "Test proposition"
        assert len(harness.positions) == 2

    def test_start_debate(self):
        """Debate should be startable."""
        config = DebateConfig(format=QuickFormat())
        harness = DebateHarness("Test", config)

        round = harness.start_debate()
        assert harness.is_started
        assert round.round_number == 1

    def test_submit_argument(self):
        """Arguments should be submittable."""
        config = DebateConfig(format=QuickFormat())
        harness = DebateHarness("Test", config)
        harness.start_debate()

        position = harness.positions[0]
        arg, gate_check = harness.submit_argument(
            position,
            claim="Test claim",
            evidence="Test evidence",
            warrant="Test warrant",
        )
        assert arg.claim == "Test claim"

    def test_finish_debate(self):
        """Debate should produce judgment when finished."""
        config = DebateConfig(format=QuickFormat())
        harness = DebateHarness("Test", config)
        harness.start_debate()

        # Submit some arguments
        for position in harness.positions:
            harness.submit_argument(
                position,
                claim=f"Claim for {position.stance.value}",
                evidence="Evidence",
                warrant="Warrant",
            )

        judgment = harness.finish_debate()
        assert isinstance(judgment, Judgment)
        assert harness.is_finished


class TestSimpleDebate:
    """Tests for run_simple_debate convenience function."""

    def test_run_simple_debate(self):
        """Simple debate should run to completion."""
        def arg_gen(position, round_num):
            return (
                f"Claim for {position.stance.value}",
                "Evidence",
                "Warrant",
            )

        judgment = run_simple_debate(
            "Test proposition",
            QuickFormat(),
            arg_gen,
        )
        assert isinstance(judgment, Judgment)


# =============================================================================
# Format Tests
# =============================================================================


class TestFormats:
    """Tests for debate formats."""

    def test_oxford_format(self):
        """Oxford format should be valid."""
        format = OxfordFormat()
        assert format.name == "Oxford"
        is_valid, issues = validate_format(format)
        assert is_valid

    def test_lincoln_douglas_format(self):
        """LD format should be valid."""
        format = LincolnDouglasFormat()
        assert format.name == "Lincoln-Douglas"
        is_valid, issues = validate_format(format)
        assert is_valid

    def test_socratic_format(self):
        """Socratic format should be valid."""
        format = SocraticFormat()
        assert format.name == "Socratic"
        assert format.position_count == 3

    def test_round_robin_format(self):
        """Round robin should enable elimination."""
        format = RoundRobinFormat()
        assert format.elimination_enabled

    def test_format_builder(self):
        """Format builder should create valid formats."""
        format = (FormatBuilder("Custom")
                  .with_positions(3)
                  .add_round(RoundType.OPENING)
                  .add_round(RoundType.REBUTTAL)
                  .add_round(RoundType.CLOSING)
                  .with_time_limit(300)
                  .enable_cross_examination()
                  .build())
        assert format.name == "Custom"
        assert format.position_count == 3
        assert format.rounds == 3

    def test_get_format_by_name(self):
        """Formats should be retrievable by name."""
        format = get_format_by_name("oxford")
        assert format is not None
        assert format.name == "Oxford"

    def test_list_formats(self):
        """Format list should be non-empty."""
        formats = list_formats()
        assert len(formats) > 0
        assert "Oxford" in formats

    def test_describe_format(self):
        """Format description should include key info."""
        format = OxfordFormat()
        description = describe_format(format)
        assert "Oxford" in description
        assert "Round" in description

    def test_estimate_duration(self):
        """Duration estimation should be positive."""
        format = OxfordFormat()
        duration = estimate_debate_duration(format)
        assert duration > 0

    def test_validate_format_invalid(self):
        """Invalid formats should fail validation."""
        format = DebateFormat(
            name="Invalid",
            rounds=0,
            position_count=1,
            round_types=[],
        )
        is_valid, issues = validate_format(format)
        assert not is_valid


# =============================================================================
# Integration Tests
# =============================================================================


class TestFullDebateFlow:
    """Integration tests for complete debate flows."""

    def test_full_oxford_debate(self):
        """Full Oxford debate should complete successfully."""
        config = DebateConfig(format=OxfordFormat())
        harness = DebateHarness("AI will benefit humanity", config)

        harness.start_debate()

        # Opening round
        for position in harness.positions:
            harness.submit_argument(
                position,
                claim=f"{position.stance.value.upper()}: AI has major implications",
                evidence="Statistics and studies",
                warrant="Therefore this supports my position",
            )

        harness.end_round()

        # Rebuttal round
        for position in harness.positions:
            opposing_args = harness.get_opposing_arguments(position)
            if opposing_args:
                harness.submit_refutation(
                    position,
                    opposing_args[0],
                    counter_claim="This argument is flawed",
                    evidence="Counter-evidence",
                )

        harness.end_round()

        # Cross-examination
        for position in harness.positions:
            opposing_args = harness.get_opposing_arguments(position)
            if opposing_args:
                harness.conduct_cross_examination(position, opposing_args[0])

        harness.end_round()

        # Closing
        harness.end_round()

        # Finish and judge
        judgment = harness.finish_debate()

        assert judgment is not None
        assert len(judgment.strength_scores) >= 2
        assert harness.is_finished

    def test_debate_with_elimination(self):
        """Debates with elimination should work."""
        format = FormatBuilder("EliminationTest").with_positions(3).add_rounds(
            RoundType.OPENING, RoundType.REBUTTAL, RoundType.CLOSING
        ).enable_elimination(threshold=0.8).build()  # High threshold

        config = DebateConfig(
            format=format,
            elimination_enabled=True,
            elimination_threshold=0.8,
        )
        harness = DebateHarness("Test", config, num_positions=3)

        harness.start_debate()

        # Only one position makes strong argument
        strong_pos = harness.positions[0]
        harness.submit_argument(
            strong_pos,
            claim="Strong claim",
            evidence="Extensive evidence with data and citations",
            warrant="Clear logical connection",
            backing="Additional support",
            rebuttal="Addresses counterarguments",
        )

        harness.end_round()
        harness.end_round()
        harness.end_round()

        judgment = harness.finish_debate()
        assert judgment is not None
