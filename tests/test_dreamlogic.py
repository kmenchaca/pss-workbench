"""
Comprehensive tests for the Dream Logic Generator.

Tests cover all modules: associations, leaps, interestingness,
constraints, styles, boring detection, synthesis, harness,
prompts, curation, and outputs.
"""

import pytest
from unittest.mock import Mock, patch

# Import all modules
from dreamlogic.types import (
    DreamFragment,
    AssociativeLeap,
    InterestingnessScore,
    DreamCollage,
    Constraint,
    DreamStyle,
    DreamBranch,
    DreamTree,
    LeapType,
    ConstraintType,
)
from dreamlogic.associations import (
    AssociationDB,
    Association,
    get_associations,
    surprising_association,
    chain_associations,
    add_association,
)
from dreamlogic.leaps import (
    generate_leap,
    score_leap,
    enforce_leap,
    random_constraint,
    LeapContext,
    leap_types,
    get_leap_description,
)
from dreamlogic.interestingness import (
    InterestingnessModel,
    score_interestingness,
    novelty_score,
    surprise_score,
    aesthetic_score,
    coherence_penalty,
    is_interesting_enough,
)
from dreamlogic.constraints import (
    random_constraint as get_random_constraint,
    chain_constraints,
    apply_constraint,
    constraint_to_prompt,
    constraints_to_prompt,
    check_constraint_satisfied,
    constraint_difficulty_level,
    get_random_by_difficulty,
    all_constraint_types,
)
from dreamlogic.styles import (
    get_style_description,
    get_style_keywords,
    apply_style,
    random_style,
    style_fragment,
    detect_style,
    style_intensity,
    all_styles,
)
from dreamlogic.boring import (
    BoringDetector,
    propagate_boring,
    avoid_patterns,
    boring_score_for_fragment,
    push_toward_strange,
    is_too_boring,
)
from dreamlogic.synthesis import (
    amplify,
    find_resonances,
    collage,
    emergent_themes,
    embrace_contradictions,
    create_collage,
)
from dreamlogic.harness import (
    DreamHarness,
    DreamConfig,
    create_harness,
)
from dreamlogic.prompts import (
    what_does_this_remind_you_of,
    what_is_the_opposite,
    if_this_were_a_myth,
    zoom_to_scale,
    time_shift,
    random_dream_prompt,
    chain_prompts,
)
from dreamlogic.curation import (
    CurationSession,
    InteractiveCurator,
    create_curation_session,
    rate_fragment,
    select_favorites,
)
from dreamlogic.outputs import (
    to_story,
    to_image_prompt,
    to_game_concept,
    to_poem,
    to_raw,
    to_json_serializable,
    all_output_formats,
)


# ============================================================================
# Types Tests
# ============================================================================

class TestTypes:
    """Tests for core type dataclasses."""

    def test_dream_fragment_creation(self):
        """Test creating a dream fragment."""
        fragment = DreamFragment(content="The clock melted into the wall")
        assert fragment.content == "The clock melted into the wall"
        assert fragment.id is not None
        assert fragment.surprise_score == 0.0

    def test_dream_fragment_surprise_score_clamping(self):
        """Test that surprise score is clamped to 0-1."""
        fragment = DreamFragment(content="test", surprise_score=1.5)
        assert fragment.surprise_score == 1.0

        fragment2 = DreamFragment(content="test", surprise_score=-0.5)
        assert fragment2.surprise_score == 0.0

    def test_associative_leap_creation(self):
        """Test creating an associative leap."""
        leap = AssociativeLeap(
            source="clock",
            target="melting time",
            leap_type=LeapType.METAPHOR,
            distance=0.7
        )
        assert leap.source == "clock"
        assert leap.target == "melting time"
        assert leap.distance == 0.7

    def test_interestingness_score_total(self):
        """Test interestingness score calculation."""
        score = InterestingnessScore(
            novelty=0.8,
            surprise=0.7,
            aesthetic=0.6,
            coherence_penalty=0.1
        )
        # (0.8 + 0.7 + 0.6) / 3 - 0.1 = 0.7 - 0.1 = 0.6
        assert abs(score.total - 0.6) < 0.01

    def test_dream_collage_add_fragment(self):
        """Test adding fragments to a collage."""
        collage = DreamCollage()
        fragment = DreamFragment(content="test", surprise_score=0.5)
        collage.add_fragment(fragment)

        assert len(collage.fragments) == 1
        assert collage.total_interestingness == 0.5

    def test_dream_style_enum(self):
        """Test dream style enum values."""
        assert DreamStyle.SURREALIST is not None
        assert DreamStyle.ABSURDIST is not None
        assert DreamStyle.MYTHIC is not None
        assert DreamStyle.LIMINAL is not None
        assert DreamStyle.COSMIC is not None

    def test_dream_branch_interestingness(self):
        """Test branch interestingness calculation."""
        branch = DreamBranch()
        branch.fragments = [
            DreamFragment(content="a", surprise_score=0.4),
            DreamFragment(content="b", surprise_score=0.6),
        ]
        assert branch.interestingness == 0.5

    def test_dream_tree_operations(self):
        """Test dream tree branch management."""
        tree = DreamTree()
        branch1 = DreamBranch()
        branch2 = DreamBranch(parent_id=branch1.id)

        tree.add_branch(branch1)
        tree.add_branch(branch2)

        assert len(tree.branches) == 2
        assert tree.root_id == branch1.id
        assert len(tree.get_alive_branches()) == 2

        tree.terminate_branch(branch1.id)
        assert len(tree.get_alive_branches()) == 1


# ============================================================================
# Associations Tests
# ============================================================================

class TestAssociations:
    """Tests for the association database."""

    def test_association_db_creation(self):
        """Test creating an association database."""
        db = AssociationDB()
        assert db is not None
        assert len(db.all_concepts) > 0

    def test_get_associations_known_concept(self):
        """Test getting associations for a known concept."""
        db = AssociationDB()
        associations = db.get_associations("clock")
        assert len(associations) > 0
        assert all(isinstance(a, Association) for a in associations)

    def test_get_associations_unknown_concept(self):
        """Test getting associations for an unknown concept."""
        db = AssociationDB()
        associations = db.get_associations("xyzzy")
        # Should return generic associations
        assert len(associations) > 0

    def test_add_association(self):
        """Test adding a new association."""
        db = AssociationDB()
        assoc = db.add_association("new_concept", "strange_target", LeapType.METAPHOR)
        assert assoc.source == "new_concept"
        assert assoc.target == "strange_target"

    def test_surprising_association(self):
        """Test getting the most surprising association."""
        db = AssociationDB()
        assoc = db.surprising_association("clock")
        assert assoc is not None
        # Should be the one with highest surprise factor (lowest weight)

    def test_chain_associations(self):
        """Test chaining associations."""
        db = AssociationDB()
        chain = db.chain_associations("clock", depth=2)
        assert len(chain) > 0
        assert all(isinstance(leap, AssociativeLeap) for leap in chain)

    def test_association_surprise_factor(self):
        """Test association surprise factor calculation."""
        assoc = Association("a", "b", LeapType.METAPHOR, weight=0.3)
        assert assoc.surprise_factor == 0.7


# ============================================================================
# Leaps Tests
# ============================================================================

class TestLeaps:
    """Tests for associative leap generation."""

    def test_generate_leap(self):
        """Test generating a leap."""
        context = LeapContext(
            current_concept="mirror",
            history=[],
            style_hints=[],
        )
        leap = generate_leap(context)
        assert leap is not None
        assert leap.source == "mirror" or "mirror" in leap.target.lower()

    def test_score_leap(self):
        """Test scoring a leap."""
        leap = AssociativeLeap(
            source="clock",
            target="melting time that drips like honey",
            leap_type=LeapType.TIME_WARP,
            distance=0.8
        )
        score = score_leap(leap)
        assert 0 <= score <= 1

    def test_enforce_leap_returns_leap_when_needed(self):
        """Test that enforce_leap returns a leap when branch needs one."""
        branch = DreamBranch()
        branch.fragments = [DreamFragment(content="clock", surprise_score=0.1)]

        leap = enforce_leap(branch, minimum_distance=0.3)
        # May or may not return a leap depending on randomness
        # Just check it doesn't error

    def test_random_constraint(self):
        """Test generating a random constraint."""
        constraint = random_constraint()
        assert isinstance(constraint, Constraint)

    def test_leap_types(self):
        """Test getting all leap types."""
        types = leap_types()
        assert LeapType.METAPHOR in types
        assert LeapType.INVERSION in types
        assert len(types) == 6

    def test_get_leap_description(self):
        """Test getting leap type descriptions."""
        desc = get_leap_description(LeapType.METAPHOR)
        assert "parallel" in desc.lower() or "unlike" in desc.lower()


# ============================================================================
# Interestingness Tests
# ============================================================================

class TestInterestingness:
    """Tests for interestingness scoring."""

    def test_score_interestingness(self):
        """Test scoring content interestingness."""
        score = score_interestingness(
            "The mirror reflected shadows that weren't there"
        )
        assert isinstance(score, InterestingnessScore)
        assert 0 <= score.total <= 1

    def test_novelty_score(self):
        """Test novelty scoring with history."""
        history = ["The cat sat on the mat", "A dog ran in the park"]
        score = novelty_score("The quantum flux twisted reality", history)
        assert 0 <= score <= 1

    def test_surprise_score(self):
        """Test surprise scoring with context."""
        score = surprise_score(
            "The void tasted of forgotten colors",
            "It was a normal day"
        )
        assert 0 <= score <= 1

    def test_aesthetic_score(self):
        """Test aesthetic scoring."""
        score = aesthetic_score(
            "Whispers of light danced through the crystalline void"
        )
        assert 0 <= score <= 1

    def test_coherence_penalty(self):
        """Test coherence penalty for logical content."""
        # Boring, logical content should get penalized
        penalty = coherence_penalty(
            "In conclusion, the data clearly shows that therefore we must summarize"
        )
        assert penalty > 0

    def test_is_interesting_enough(self):
        """Test the interestingness threshold check."""
        result = is_interesting_enough(
            "The shadow dreamed of impossible geometries",
            threshold=0.3
        )
        # Just check it returns a boolean
        assert isinstance(result, bool)

    def test_interestingness_model_learning(self):
        """Test that the model can learn from ratings."""
        model = InterestingnessModel()
        for i in range(15):
            model.add_rating(f"Test content {i}", 0.5 + (i * 0.03))
        model.learn_from_ratings()
        # Check weights were adjusted
        # Model should still work after learning


# ============================================================================
# Constraints Tests
# ============================================================================

class TestConstraints:
    """Tests for constraint generation and application."""

    def test_random_constraint(self):
        """Test generating a random constraint."""
        constraint = get_random_constraint()
        assert isinstance(constraint, Constraint)
        assert constraint.type in ConstraintType

    def test_chain_constraints(self):
        """Test generating a chain of constraints."""
        constraints = chain_constraints(3, escalate_difficulty=True)
        assert len(constraints) == 3
        # Difficulty should escalate
        assert constraints[2].difficulty >= constraints[0].difficulty - 0.1

    def test_apply_constraint(self):
        """Test applying a constraint to content."""
        constraint = Constraint(
            type=ConstraintType.MUST_INCLUDE,
            value="a mirror",
            difficulty=0.3
        )
        result = apply_constraint("The dream began", constraint)
        assert "Must include" in result

    def test_constraint_to_prompt(self):
        """Test converting constraint to prompt."""
        constraint = Constraint(
            type=ConstraintType.PERSPECTIVE,
            value="from the viewpoint of an inanimate object",
            difficulty=0.5
        )
        prompt = constraint_to_prompt(constraint)
        assert "Write" in prompt

    def test_check_constraint_satisfied(self):
        """Test checking if constraint is satisfied."""
        constraint = Constraint(
            type=ConstraintType.MUST_INCLUDE,
            value="mirror",
            difficulty=0.3
        )
        assert check_constraint_satisfied("There was a mirror on the wall", constraint)
        assert not check_constraint_satisfied("There was nothing there", constraint)

    def test_constraint_difficulty_level(self):
        """Test difficulty level naming."""
        assert constraint_difficulty_level(0.2) == "easy"
        assert constraint_difficulty_level(0.6) == "challenging"

    def test_all_constraint_types(self):
        """Test getting all constraint types."""
        types = all_constraint_types()
        assert len(types) == 6


# ============================================================================
# Styles Tests
# ============================================================================

class TestStyles:
    """Tests for style injection."""

    def test_get_style_description(self):
        """Test getting style descriptions."""
        desc = get_style_description(DreamStyle.SURREALIST)
        assert "Dali" in desc or "melting" in desc

    def test_get_style_keywords(self):
        """Test getting style keywords."""
        keywords = get_style_keywords(DreamStyle.LIMINAL)
        assert "threshold" in keywords or "between" in keywords

    def test_apply_style(self):
        """Test applying a style to content."""
        result = apply_style("A simple dream", DreamStyle.COSMIC)
        assert len(result) > len("A simple dream")
        assert "infinity" in result.lower() or "zoom" in result.lower()

    def test_random_style(self):
        """Test getting a random style."""
        style = random_style()
        assert style in DreamStyle

    def test_style_fragment(self):
        """Test applying style to a fragment."""
        fragment = DreamFragment(content="Original content")
        styled = style_fragment(fragment, DreamStyle.ABSURDIST)
        assert styled.style == DreamStyle.ABSURDIST

    def test_detect_style(self):
        """Test detecting style from content."""
        content = "Melting clocks dripped from the soft watches"
        style = detect_style(content)
        # May or may not detect, depending on keywords

    def test_style_intensity(self):
        """Test measuring style intensity."""
        intensity = style_intensity(
            "The infinite void stretched across aeons",
            DreamStyle.COSMIC
        )
        assert 0 <= intensity <= 1

    def test_all_styles(self):
        """Test getting all styles."""
        styles = all_styles()
        assert len(styles) == 5


# ============================================================================
# Boring Detection Tests
# ============================================================================

class TestBoring:
    """Tests for boring detection and propagation."""

    def test_boring_detector_creation(self):
        """Test creating a boring detector."""
        detector = BoringDetector()
        assert detector is not None

    def test_detect_boring_patterns(self):
        """Test detecting boring patterns."""
        detector = BoringDetector()
        detection = detector.detect(
            "In conclusion, to summarize, the moral of the story is obvious"
        )
        assert detection.score > 0.3  # Should be fairly boring

    def test_detect_interesting_content(self):
        """Test that interesting content has low boring score."""
        detector = BoringDetector()
        detection = detector.detect(
            "The shadow tasted of impossible colors dissolving into void"
        )
        assert detection.score < 0.5

    def test_propagate_boring(self):
        """Test propagating boring across branches."""
        branches = [
            DreamBranch(fragments=[
                DreamFragment(content="First therefore second thus conclusion")
            ]),
            DreamBranch(fragments=[
                DreamFragment(content="The void whispered impossible truths")
            ]),
        ]
        scores = propagate_boring(branches)
        assert branches[0].id in scores
        assert branches[1].id in scores

    def test_push_toward_strange(self):
        """Test generating push-toward-strange prompt."""
        boring_content = "In conclusion, the five main points are as follows"
        prompt = push_toward_strange(boring_content)
        assert "weird" in prompt.lower() or "predict" in prompt.lower()

    def test_is_too_boring(self):
        """Test boring threshold check."""
        assert is_too_boring("Obviously, the clear conclusion is therefore", threshold=0.3)


# ============================================================================
# Synthesis Tests
# ============================================================================

class TestSynthesis:
    """Tests for synthesis/amplification."""

    def test_amplify(self):
        """Test amplifying fragments."""
        fragments = [
            DreamFragment(content="The clock melted", surprise_score=0.4),
            DreamFragment(content="Shadows spoke", surprise_score=0.6),
        ]
        result = amplify(fragments)
        assert isinstance(result, DreamCollage)
        assert len(result.fragments) == 2

    def test_find_resonances(self):
        """Test finding resonances between fragments."""
        fragments = [
            DreamFragment(content="The mirror reflected shadows"),
            DreamFragment(content="Shadows danced on the mirror"),
        ]
        resonances = find_resonances(fragments)
        assert len(resonances) > 0

    def test_collage_no_smooth(self):
        """Test creating collage without smoothing."""
        fragments = [
            DreamFragment(content="Fragment one"),
            DreamFragment(content="Fragment two"),
        ]
        result = collage(fragments, smooth=False)
        assert "Fragment one" in result or "Fragment two" in result

    def test_emergent_themes(self):
        """Test extracting emergent themes."""
        fragments = [
            DreamFragment(content="The mirror reflected impossible shadows"),
            DreamFragment(content="Shadows in the mirror spoke of void"),
            DreamFragment(content="Through the mirror came only shadows"),
        ]
        themes = emergent_themes(fragments)
        assert len(themes) > 0

    def test_create_collage(self):
        """Test creating a full collage."""
        fragments = [
            DreamFragment(content="Test content", surprise_score=0.5),
        ]
        result = create_collage(fragments, style=DreamStyle.SURREALIST)
        assert result.style == DreamStyle.SURREALIST


# ============================================================================
# Harness Tests
# ============================================================================

class TestHarness:
    """Tests for the dream harness."""

    def test_harness_creation(self):
        """Test creating a harness."""
        harness = DreamHarness()
        assert harness is not None
        assert harness.config is not None

    def test_create_branch(self):
        """Test creating a branch."""
        harness = DreamHarness()
        branch = harness.create_branch()
        assert branch is not None
        assert branch.id in harness.state.tree.branches

    def test_check_gate(self):
        """Test gate checking."""
        harness = DreamHarness()
        branch = harness.create_branch()
        result = harness.check_gate(
            branch,
            "The void tasted of impossible colors"
        )
        assert isinstance(result.interestingness, float)
        assert isinstance(result.passed, bool)

    def test_add_fragment(self):
        """Test adding a fragment to a branch."""
        harness = DreamHarness()
        branch = harness.create_branch()
        fragment = harness.add_fragment(branch, "Test content")
        assert len(branch.fragments) == 1

    def test_build_prompt(self):
        """Test building a prompt with style and constraints."""
        config = DreamConfig(style=DreamStyle.COSMIC, inject_constraints=True)
        harness = DreamHarness(config=config)
        branch = harness.create_branch()
        prompt = harness.build_prompt(branch, "Explore the void")
        assert "COSMIC" in prompt or "void" in prompt.lower()

    def test_synthesize_empty(self):
        """Test synthesizing with no fragments."""
        harness = DreamHarness()
        result = harness.synthesize()
        assert isinstance(result, DreamCollage)

    def test_get_stats(self):
        """Test getting statistics."""
        harness = DreamHarness()
        harness.create_branch()
        stats = harness.get_stats()
        assert "total_branches" in stats
        assert stats["total_branches"] == 1

    def test_create_harness_convenience(self):
        """Test convenience harness creation."""
        harness = create_harness(style=DreamStyle.LIMINAL, max_branches=3)
        assert harness.config.style == DreamStyle.LIMINAL
        assert harness.config.max_branches == 3


# ============================================================================
# Prompts Tests
# ============================================================================

class TestPrompts:
    """Tests for dream prompts."""

    def test_what_does_this_remind_you_of(self):
        """Test association prompt."""
        prompt = what_does_this_remind_you_of("A clock on the wall")
        assert "Given:" in prompt
        assert "clock" in prompt.lower()

    def test_what_is_the_opposite(self):
        """Test inversion prompt."""
        prompt = what_is_the_opposite("Light fills the room")
        assert "Consider:" in prompt

    def test_if_this_were_a_myth(self):
        """Test mythologizing prompt."""
        prompt = if_this_were_a_myth("A hero enters the cave")
        assert "story" in prompt.lower()

    def test_zoom_to_scale(self):
        """Test scale shift prompt."""
        prompt = zoom_to_scale("A grain of sand", scale="cosmic")
        assert "Observe:" in prompt

    def test_time_shift(self):
        """Test time shift prompt."""
        prompt = time_shift("The present moment", direction="past")
        assert "moment" in prompt.lower()

    def test_random_dream_prompt(self):
        """Test random prompt generation."""
        prompt = random_dream_prompt("Any context here")
        assert len(prompt) > 20

    def test_chain_prompts(self):
        """Test generating a chain of prompts."""
        prompts = chain_prompts("Starting context", n=3)
        assert len(prompts) == 3


# ============================================================================
# Curation Tests
# ============================================================================

class TestCuration:
    """Tests for human curation interface."""

    def test_curation_session_creation(self):
        """Test creating a curation session."""
        session = CurationSession()
        assert session is not None
        assert len(session.fragments) == 0

    def test_add_fragments(self):
        """Test adding fragments to session."""
        session = CurationSession()
        fragments = [
            DreamFragment(content="Fragment 1"),
            DreamFragment(content="Fragment 2"),
        ]
        session.add_fragments(fragments)
        assert len(session.fragments) == 2

    def test_rate_fragment(self):
        """Test rating a fragment."""
        session = CurationSession()
        fragment = DreamFragment(content="Test")
        session.add_fragment(fragment)
        rating = session.rate_fragment(fragment.id, 0.8, notes="Very interesting")
        assert rating.rating == 0.8
        assert len(session.ratings) == 1

    def test_select_favorites(self):
        """Test selecting favorites."""
        session = CurationSession()
        f1 = DreamFragment(content="One")
        f2 = DreamFragment(content="Two")
        session.add_fragments([f1, f2])
        session.select_favorites([f1.id])
        assert f1.id in session.favorites
        assert f2.id not in session.favorites

    def test_get_statistics(self):
        """Test getting session statistics."""
        session = CurationSession()
        fragment = DreamFragment(content="Test")
        session.add_fragment(fragment)
        session.rate_fragment(fragment.id, 0.7)
        stats = session.get_statistics()
        assert stats["total_fragments"] == 1
        assert stats["rated_fragments"] == 1

    def test_create_curation_session_from_collage(self):
        """Test creating session from collage."""
        collage = DreamCollage()
        collage.add_fragment(DreamFragment(content="Test"))
        session = create_curation_session(collage=collage)
        assert len(session.fragments) == 1


# ============================================================================
# Outputs Tests
# ============================================================================

class TestOutputs:
    """Tests for output formats."""

    def test_to_story(self):
        """Test converting to story format."""
        collage = DreamCollage(style=DreamStyle.SURREALIST)
        collage.add_fragment(DreamFragment(content="The clock began to melt"))
        story = to_story(collage)
        assert "clock" in story.lower() or "dream" in story.lower()

    def test_to_image_prompt(self):
        """Test converting to image prompt."""
        collage = DreamCollage(style=DreamStyle.COSMIC)
        collage.add_fragment(DreamFragment(content="Stars collapsed inward"))
        prompt = to_image_prompt(collage)
        assert "cosmic" in prompt.lower() or "dream" in prompt.lower()

    def test_to_game_concept(self):
        """Test converting to game concept."""
        collage = DreamCollage()
        collage.add_fragment(DreamFragment(content="Navigate the maze"))
        concept = to_game_concept(collage)
        assert "GAME CONCEPT" in concept

    def test_to_poem_free(self):
        """Test converting to free verse poem."""
        collage = DreamCollage()
        collage.add_fragment(DreamFragment(
            content="The shadow whispered secrets. Ancient and forgotten."
        ))
        poem = to_poem(collage, form="free")
        assert len(poem) > 0

    def test_to_poem_haiku(self):
        """Test converting to haiku chain."""
        collage = DreamCollage()
        collage.add_fragment(DreamFragment(
            content="Moonlight dances on still water beneath ancient trees at midnight"
        ))
        poem = to_poem(collage, form="haiku_chain")
        assert "\n" in poem

    def test_to_raw(self):
        """Test getting raw output."""
        collage = DreamCollage()
        f = DreamFragment(content="Raw content here")
        collage.add_fragment(f)
        raw = to_raw(collage)
        assert "Raw content here" in raw

    def test_to_json_serializable(self):
        """Test JSON serialization."""
        collage = DreamCollage(style=DreamStyle.MYTHIC)
        collage.add_fragment(DreamFragment(content="Test"))
        data = to_json_serializable(collage)
        assert "fragments" in data
        assert data["style"] == "MYTHIC"

    def test_all_output_formats(self):
        """Test getting all format names."""
        formats = all_output_formats()
        assert "story" in formats
        assert "poem" in formats
        assert len(formats) >= 5


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple modules."""

    def test_full_exploration_flow(self):
        """Test a complete exploration flow."""
        # Create harness
        harness = create_harness(style=DreamStyle.SURREALIST)

        # Create branch and add content
        branch = harness.create_branch()
        harness.add_fragment(branch, "The mirror showed something wrong")
        harness.add_fragment(branch, "Reality began to soften at the edges")

        # Check gate
        gate = harness.check_gate(branch, branch.fragments[-1].content)
        assert isinstance(gate.interestingness, float)

        # Synthesize
        result = harness.synthesize()
        assert len(result.fragments) == 2

        # Convert to output
        story = to_story(result)
        assert len(story) > 0

    def test_curation_to_learning(self):
        """Test that curation feeds back to learning."""
        # Create session with fragments
        session = CurationSession()
        for i in range(12):
            session.add_fragment(DreamFragment(
                content=f"Strange content number {i}"
            ))

        # Rate fragments
        for fragment in session.fragments:
            session.rate_fragment(fragment.id, 0.5 + (0.03 * len(fragment.content)))

        # Learn
        session.learn_from_curation()

        # Should not error
        stats = session.get_statistics()
        assert stats["rated_fragments"] == 12

    def test_boring_propagation_affects_branches(self):
        """Test that boring detection affects branch management."""
        harness = DreamHarness()

        # Create boring branch
        boring_branch = harness.create_branch()
        harness.add_fragment(
            boring_branch,
            "In conclusion, therefore, to summarize the five main points"
        )

        # Create interesting branch
        interesting_branch = harness.create_branch()
        harness.add_fragment(
            interesting_branch,
            "The void tasted of impossible geometries collapsing"
        )

        # Propagate boring
        scores = harness.propagate_boring()

        # Boring branch should have higher score
        assert scores[boring_branch.id] > scores[interesting_branch.id]
