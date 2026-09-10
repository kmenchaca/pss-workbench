"""
Adjudication (synthesis) for the Argumentative Swarm system.

Implements multiple strategies for judging debates and producing
reasoned judgments with identified cruxes and common ground.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

from .debate import calculate_position_score
from .transcript import DebateTranscript, get_addressed_points, get_unaddressed_arguments
from .types import (
    AdjudicationStrategy,
    Argument,
    Crux,
    DebateState,
    Judgment,
    Position,
    Refutation,
    Stance,
    StrengthScore,
)


@dataclass
class AdjudicationContext:
    """
    Context for adjudication decisions.

    Attributes:
        transcript: Full debate transcript
        state: Current debate state
        strategy: Adjudication strategy to use
        weights: Custom weights for scoring components
    """
    transcript: DebateTranscript
    state: DebateState
    strategy: AdjudicationStrategy = AdjudicationStrategy.PRAGMATIC
    weights: dict[str, float] = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                "argument_quality": 0.3,
                "refutation_success": 0.25,
                "defense_success": 0.2,
                "evidence_quality": 0.25,
            }


def adjudicate(
    transcript: DebateTranscript,
    state: DebateState,
    strategy: AdjudicationStrategy = AdjudicationStrategy.PRAGMATIC,
) -> Judgment:
    """
    Produce a judgment based on the debate transcript.

    Args:
        transcript: Full record of the debate
        state: Current debate state
        strategy: Strategy for judging

    Returns:
        Judgment with winner, reasoning, cruxes, and scores
    """
    context = AdjudicationContext(
        transcript=transcript,
        state=state,
        strategy=strategy,
    )

    # Calculate strength scores for each position
    strength_scores = _calculate_all_scores(context)

    # Identify key cruxes
    cruxes = identify_cruxes(transcript, state)

    # Find common ground
    common_ground = find_common_ground(state.active_positions, state)

    # Determine winner based on strategy
    winner, reasoning = _determine_winner(context, strength_scores, cruxes)

    # Generate recommendation
    recommendation = _generate_recommendation(
        winner, strength_scores, cruxes, common_ground, strategy
    )

    return Judgment(
        winner=winner,
        reasoning=reasoning,
        key_cruxes=cruxes,
        strength_scores=strength_scores,
        common_ground=common_ground,
        recommendation=recommendation,
        strategy_used=strategy,
    )


def weigh_arguments(
    arguments: list[Argument],
    weights: Optional[dict[str, float]] = None,
) -> list[tuple[str, float]]:
    """
    Compare and rank arguments by weighted strength.

    Args:
        arguments: List of arguments to compare
        weights: Custom weights for different factors

    Returns:
        List of (argument_id, weighted_score) tuples, sorted by score
    """
    if weights is None:
        weights = {
            "base_strength": 0.4,
            "evidence_present": 0.2,
            "warrant_present": 0.2,
            "backing_present": 0.1,
            "rebuttal_present": 0.1,
        }

    scored: list[tuple[str, float]] = []

    for arg in arguments:
        score = arg.strength * weights.get("base_strength", 0.4)

        if arg.evidence:
            score += weights.get("evidence_present", 0.2)
        if arg.warrant:
            score += weights.get("warrant_present", 0.2)
        if arg.backing:
            score += weights.get("backing_present", 0.1)
        if arg.rebuttal:
            score += weights.get("rebuttal_present", 0.1)

        scored.append((arg.id, score))

    return sorted(scored, key=lambda x: x[1], reverse=True)


def identify_cruxes(
    transcript: DebateTranscript,
    state: DebateState,
) -> list[Crux]:
    """
    Identify key points of disagreement in the debate.

    A crux is a fundamental disagreement that, if resolved, would
    likely change the outcome of the debate.

    Args:
        transcript: Full debate transcript
        state: Current debate state

    Returns:
        List of identified cruxes
    """
    cruxes: list[Crux] = []
    crux_counter = 0

    # Find arguments that were heavily contested
    addressed = get_addressed_points(transcript)

    for arg_id, addressing_ids in addressed.items():
        if len(addressing_ids) >= 2:
            # This argument was contested multiple times
            argument = _find_argument_by_id(state, arg_id)
            if argument:
                crux_counter += 1
                positions_involved = _get_contesting_positions(
                    state, arg_id, addressing_ids
                )

                cruxes.append(Crux(
                    id=f"crux_{crux_counter:03d}",
                    description=f"Contested claim: {argument.claim[:100]}...",
                    positions_involved=positions_involved,
                    arguments_involved=[arg_id] + addressing_ids,
                    resolution="",
                ))

    # Find opposing positions with unresolved disagreements
    for_positions = [p for p in state.active_positions if p.stance == Stance.FOR]
    against_positions = [p for p in state.active_positions if p.stance == Stance.AGAINST]

    if for_positions and against_positions:
        # Check for fundamental disagreements
        for_args = [a for a in state.arguments
                    if a.position_id in [p.id for p in for_positions]]
        against_args = [a for a in state.arguments
                        if a.position_id in [p.id for p in against_positions]]

        if for_args and against_args:
            crux_counter += 1
            cruxes.append(Crux(
                id=f"crux_{crux_counter:03d}",
                description="Core disagreement between FOR and AGAINST positions",
                positions_involved=[p.id for p in for_positions + against_positions],
                arguments_involved=[a.id for a in for_args[:3] + against_args[:3]],
                resolution="",
            ))

    return cruxes


def find_common_ground(
    positions: list[Position],
    state: DebateState,
) -> list[str]:
    """
    Identify areas where positions agree.

    Args:
        positions: Positions to analyze
        state: Current debate state

    Returns:
        List of areas of agreement
    """
    common_ground: list[str] = []

    # Look for uncontested arguments
    unaddressed = set()
    for pos in positions:
        pos_unaddressed = get_unaddressed_arguments(
            _create_minimal_transcript(state), pos
        )
        unaddressed.update(pos_unaddressed)

    # Arguments that weren't contested might indicate agreement
    for arg_id in unaddressed:
        argument = _find_argument_by_id(state, arg_id)
        if argument and argument.strength > 0.6:
            common_ground.append(
                f"Uncontested point: {argument.claim[:80]}..."
            )

    # Look for similar claims across opposing positions
    for_args = [a for a in state.arguments
                if any(p.stance == Stance.FOR and p.id == a.position_id
                       for p in positions)]
    against_args = [a for a in state.arguments
                    if any(p.stance == Stance.AGAINST and p.id == a.position_id
                           for p in positions)]

    # Simple keyword overlap check
    for f_arg in for_args:
        for a_arg in against_args:
            overlap = _calculate_claim_overlap(f_arg.claim, a_arg.claim)
            if overlap > 0.3:
                common_ground.append(
                    f"Shared concept: Both sides acknowledge aspects of "
                    f"'{_extract_common_terms(f_arg.claim, a_arg.claim)}'"
                )

    return common_ground[:5]  # Limit to top 5


def _calculate_all_scores(context: AdjudicationContext) -> list[StrengthScore]:
    """Calculate strength scores for all positions."""
    scores: list[StrengthScore] = []

    for position in context.state.active_positions:
        base_score = calculate_position_score(position, context.state)

        # Get detailed scores
        position_args = [a for a in context.state.arguments
                         if a.position_id == position.id]
        position_refs = [r for r in context.state.refutations
                         if r.position_id == position.id]

        arg_quality = (sum(a.strength for a in position_args) / len(position_args)
                       if position_args else 0.0)
        ref_success = (sum(r.strength for r in position_refs) / len(position_refs)
                       if position_refs else 0.0)

        # Calculate defense success
        arg_ids = {a.id for a in position_args}
        received_refs = [r for r in context.state.refutations
                         if r.target_argument_id in arg_ids]
        responses = [r for r in context.state.responses
                     if r.argument_id in arg_ids]
        defense_success = (len(responses) / len(received_refs)
                           if received_refs else 1.0)

        # Evidence quality based on argument completeness
        evidence_quality = (sum(1 for a in position_args if a.evidence and a.warrant)
                            / len(position_args) if position_args else 0.0)

        scores.append(StrengthScore(
            position_id=position.id,
            score=base_score,
            argument_quality=arg_quality,
            refutation_success=ref_success,
            defense_success=min(defense_success, 1.0),
            evidence_quality=evidence_quality,
        ))

    return scores


def _determine_winner(
    context: AdjudicationContext,
    scores: list[StrengthScore],
    cruxes: list[Crux],
) -> tuple[Optional[str], str]:
    """
    Determine the winner based on strategy.

    Returns:
        Tuple of (winner_position_id or None, reasoning)
    """
    if not scores:
        return None, "No positions to evaluate"

    strategy = context.strategy

    if strategy == AdjudicationStrategy.UTILITARIAN:
        return _utilitarian_judgment(context, scores, cruxes)
    elif strategy == AdjudicationStrategy.DEONTOLOGICAL:
        return _deontological_judgment(context, scores, cruxes)
    elif strategy == AdjudicationStrategy.PRAGMATIC:
        return _pragmatic_judgment(context, scores, cruxes)
    elif strategy == AdjudicationStrategy.CONSENSUS:
        return _consensus_judgment(context, scores, cruxes)
    else:
        return _pragmatic_judgment(context, scores, cruxes)


def _utilitarian_judgment(
    context: AdjudicationContext,
    scores: list[StrengthScore],
    cruxes: list[Crux],
) -> tuple[Optional[str], str]:
    """
    Judge based on overall utility and outcomes.

    Emphasizes practical consequences and aggregate benefit.
    """
    # Weight scores by potential impact
    weighted_scores: list[tuple[str, float]] = []

    for score in scores:
        position = _find_position_by_id(context.state, score.position_id)
        if not position:
            continue

        # Utilitarian weights favor evidence and practical arguments
        weighted = (
            score.argument_quality * 0.35 +
            score.evidence_quality * 0.35 +
            score.refutation_success * 0.15 +
            score.defense_success * 0.15
        )
        weighted_scores.append((score.position_id, weighted))

    if not weighted_scores:
        return None, "Unable to evaluate positions"

    weighted_scores.sort(key=lambda x: x[1], reverse=True)
    winner_id = weighted_scores[0][0]
    winner_score = weighted_scores[0][1]

    # Check if it's a clear win
    if len(weighted_scores) > 1 and (winner_score - weighted_scores[1][1]) < 0.1:
        return None, (
            "Utilitarian analysis finds no clear winner. Both positions offer "
            "comparable utility based on evidence and practical considerations."
        )

    winner_pos = _find_position_by_id(context.state, winner_id)
    reasoning = (
        f"Utilitarian analysis favors the {winner_pos.stance.value.upper()} position. "
        f"This position demonstrated superior evidence quality ({weighted_scores[0][1]:.2f}) "
        f"and practical argumentation that would likely produce better outcomes."
    )

    return winner_id, reasoning


def _deontological_judgment(
    context: AdjudicationContext,
    scores: list[StrengthScore],
    cruxes: list[Crux],
) -> tuple[Optional[str], str]:
    """
    Judge based on principles and logical consistency.

    Emphasizes warrant quality and principled reasoning.
    """
    weighted_scores: list[tuple[str, float]] = []

    for score in scores:
        position = _find_position_by_id(context.state, score.position_id)
        if not position:
            continue

        # Deontological weights favor logical consistency (warrants)
        position_args = [a for a in context.state.arguments
                         if a.position_id == position.id]
        warrant_quality = (sum(1 for a in position_args if a.warrant and len(a.warrant) > 50)
                           / len(position_args) if position_args else 0.0)

        weighted = (
            score.argument_quality * 0.25 +
            warrant_quality * 0.40 +
            score.defense_success * 0.20 +
            score.refutation_success * 0.15
        )
        weighted_scores.append((score.position_id, weighted))

    if not weighted_scores:
        return None, "Unable to evaluate positions"

    weighted_scores.sort(key=lambda x: x[1], reverse=True)
    winner_id = weighted_scores[0][0]
    winner_score = weighted_scores[0][1]

    if len(weighted_scores) > 1 and (winner_score - weighted_scores[1][1]) < 0.1:
        return None, (
            "Deontological analysis finds no clear winner. Both positions "
            "demonstrate comparable logical consistency and principled reasoning."
        )

    winner_pos = _find_position_by_id(context.state, winner_id)
    reasoning = (
        f"Deontological analysis favors the {winner_pos.stance.value.upper()} position. "
        f"This position demonstrated superior logical consistency and principled "
        f"reasoning with well-developed warrants connecting evidence to claims."
    )

    return winner_id, reasoning


def _pragmatic_judgment(
    context: AdjudicationContext,
    scores: list[StrengthScore],
    cruxes: list[Crux],
) -> tuple[Optional[str], str]:
    """
    Judge based on overall debate performance.

    Balanced consideration of all factors.
    """
    weighted_scores: list[tuple[str, float]] = []

    weights = context.weights

    for score in scores:
        weighted = (
            score.argument_quality * weights.get("argument_quality", 0.3) +
            score.refutation_success * weights.get("refutation_success", 0.25) +
            score.defense_success * weights.get("defense_success", 0.2) +
            score.evidence_quality * weights.get("evidence_quality", 0.25)
        )
        weighted_scores.append((score.position_id, weighted))

    if not weighted_scores:
        return None, "Unable to evaluate positions"

    weighted_scores.sort(key=lambda x: x[1], reverse=True)
    winner_id = weighted_scores[0][0]
    winner_score = weighted_scores[0][1]

    if len(weighted_scores) > 1 and (winner_score - weighted_scores[1][1]) < 0.1:
        # Close debate - might declare tie or synthesis
        return None, (
            "Pragmatic analysis finds the debate too close to declare a clear winner. "
            f"Top scores: {weighted_scores[0][1]:.2f} vs {weighted_scores[1][1]:.2f}. "
            "A synthesis of positions may be most appropriate."
        )

    winner_pos = _find_position_by_id(context.state, winner_id)
    reasoning = (
        f"Pragmatic analysis declares the {winner_pos.stance.value.upper()} position "
        f"the winner with a score of {winner_score:.2f}. This position excelled in "
        f"argument construction, successful refutations, and defending its claims."
    )

    return winner_id, reasoning


def _consensus_judgment(
    context: AdjudicationContext,
    scores: list[StrengthScore],
    cruxes: list[Crux],
) -> tuple[Optional[str], str]:
    """
    Attempt to find synthesis rather than declaring a winner.

    Emphasizes common ground and resolvable cruxes.
    """
    common = find_common_ground(context.state.active_positions, context.state)

    if len(common) >= 3:
        reasoning = (
            f"Consensus analysis identifies {len(common)} areas of common ground. "
            f"Rather than declaring a winner, both positions can be synthesized: "
            f"{'; '.join(common[:3])}"
        )
        return None, reasoning

    # Fall back to pragmatic if no consensus possible
    if cruxes and all(c.resolution for c in cruxes):
        reasoning = (
            "Consensus analysis notes that key cruxes have been addressed. "
            "A synthesis of positions incorporating resolved disagreements is recommended."
        )
        return None, reasoning

    # If no consensus, use pragmatic
    return _pragmatic_judgment(context, scores, cruxes)


def _generate_recommendation(
    winner: Optional[str],
    scores: list[StrengthScore],
    cruxes: list[Crux],
    common_ground: list[str],
    strategy: AdjudicationStrategy,
) -> str:
    """Generate a final recommendation based on the judgment."""
    parts: list[str] = []

    if winner:
        winning_score = next((s for s in scores if s.position_id == winner), None)
        if winning_score:
            parts.append(
                f"The winning position scored {winning_score.score:.2f} overall."
            )
    else:
        parts.append("No clear winner emerged from this debate.")

    if cruxes:
        unresolved = [c for c in cruxes if not c.resolution]
        if unresolved:
            parts.append(
                f"There are {len(unresolved)} unresolved cruxes that merit further discussion."
            )

    if common_ground:
        parts.append(
            f"Despite disagreements, {len(common_ground)} areas of common ground were identified."
        )

    if strategy == AdjudicationStrategy.CONSENSUS:
        parts.append(
            "Future discussions should build on identified common ground."
        )

    return " ".join(parts)


def _find_argument_by_id(state: DebateState, arg_id: str) -> Optional[Argument]:
    """Find an argument by its ID."""
    for arg in state.arguments:
        if arg.id == arg_id:
            return arg
    return None


def _find_position_by_id(state: DebateState, pos_id: str) -> Optional[Position]:
    """Find a position by its ID."""
    for pos in state.active_positions:
        if pos.id == pos_id:
            return pos
    for pos in state.eliminated_positions:
        if pos.id == pos_id:
            return pos
    return None


def _get_contesting_positions(
    state: DebateState,
    arg_id: str,
    addressing_ids: list[str],
) -> list[str]:
    """Get IDs of positions that contested an argument."""
    positions: set[str] = set()

    # Find the original argument's position
    original_arg = _find_argument_by_id(state, arg_id)
    if original_arg:
        positions.add(original_arg.position_id)

    # Find positions of refutations
    for ref in state.refutations:
        if ref.id in addressing_ids:
            positions.add(ref.position_id)

    return list(positions)


def _create_minimal_transcript(state: DebateState) -> DebateTranscript:
    """Create a minimal transcript from debate state for utility functions."""
    from .transcript import DebateTranscript, TranscriptEntry
    from .types import RoundType

    transcript = DebateTranscript(
        proposition=state.proposition,
        positions=state.active_positions + state.eliminated_positions,
    )

    for arg in state.arguments:
        transcript.entries.append(TranscriptEntry(
            timestamp="",
            round_number=arg.round_number,
            round_type=RoundType.OPENING,
            entry_type="argument",
            position_id=arg.position_id,
            content=arg.claim,
            metadata={"argument_id": arg.id},
        ))

    for ref in state.refutations:
        transcript.entries.append(TranscriptEntry(
            timestamp="",
            round_number=ref.round_number,
            round_type=RoundType.REBUTTAL,
            entry_type="refutation",
            position_id=ref.position_id,
            content=ref.counter_claim,
            references=[ref.target_argument_id],
            metadata={"refutation_id": ref.id},
        ))

    return transcript


def _calculate_claim_overlap(claim1: str, claim2: str) -> float:
    """Calculate word overlap between two claims."""
    words1 = set(claim1.lower().split())
    words2 = set(claim2.lower().split())

    # Remove common words
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                 "being", "have", "has", "had", "do", "does", "did", "will",
                 "would", "could", "should", "may", "might", "must", "shall",
                 "can", "of", "to", "in", "for", "on", "with", "at", "by",
                 "from", "as", "into", "through", "during", "before", "after",
                 "above", "below", "between", "under", "again", "further",
                 "then", "once", "here", "there", "when", "where", "why",
                 "how", "all", "each", "few", "more", "most", "other", "some",
                 "such", "no", "nor", "not", "only", "own", "same", "so",
                 "than", "too", "very", "just", "and", "but", "if", "or",
                 "because", "as", "until", "while", "this", "that", "these",
                 "those", "it", "its"}

    words1 = words1 - stopwords
    words2 = words2 - stopwords

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union) if union else 0.0


def _extract_common_terms(claim1: str, claim2: str) -> str:
    """Extract common meaningful terms between claims."""
    words1 = set(claim1.lower().split())
    words2 = set(claim2.lower().split())

    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                 "being", "have", "has", "had", "do", "does", "did", "will",
                 "would", "could", "should", "may", "might", "must", "shall",
                 "can", "of", "to", "in", "for", "on", "with", "at", "by",
                 "and", "but", "if", "or", "this", "that", "it", "its"}

    common = (words1 & words2) - stopwords

    return ", ".join(list(common)[:3]) if common else "shared concepts"
