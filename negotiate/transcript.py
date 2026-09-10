"""Negotiation transcript recording and analysis."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import json

from .types import (
    Agreement,
    NegotiationMove,
    NegotiationState,
    NegotiationStatus,
    Offer,
    OfferStatus,
    Stakeholder,
)


@dataclass
class NegotiationTranscript:
    """Complete record of a negotiation.

    Attributes:
        negotiation_id: ID of the negotiation.
        stakeholders: Participating stakeholders.
        moves: Ordered list of all moves.
        start_time: When negotiation started.
        end_time: When negotiation ended.
        final_status: How negotiation concluded.
        agreement: Final agreement if reached.
        metadata: Additional contextual information.
    """

    negotiation_id: str
    stakeholders: dict[str, Stakeholder] = field(default_factory=dict)
    moves: list[NegotiationMove] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    final_status: NegotiationStatus = NegotiationStatus.IN_PROGRESS
    agreement: Optional[Agreement] = None
    metadata: dict[str, Any] = field(default_factory=dict)


def create_transcript(state: NegotiationState) -> NegotiationTranscript:
    """Create a transcript from negotiation state.

    Args:
        state: The negotiation state.

    Returns:
        A new transcript.
    """
    return NegotiationTranscript(
        negotiation_id=state.negotiation_id,
        stakeholders=state.stakeholders.copy(),
        moves=state.moves.copy(),
        start_time=state.moves[0].timestamp if state.moves else datetime.now(),
        final_status=state.status,
        agreement=state.agreements[-1] if state.agreements else None,
    )


def add_move(transcript: NegotiationTranscript, move: NegotiationMove) -> None:
    """Add a move to the transcript.

    Args:
        transcript: The transcript to update.
        move: The move to add.
    """
    transcript.moves.append(move)


def finalize_transcript(
    transcript: NegotiationTranscript,
    status: NegotiationStatus,
    agreement: Optional[Agreement] = None,
) -> None:
    """Finalize a transcript when negotiation ends.

    Args:
        transcript: The transcript to finalize.
        status: Final status.
        agreement: Optional final agreement.
    """
    transcript.end_time = datetime.now()
    transcript.final_status = status
    transcript.agreement = agreement


def analyze_concessions(transcript: NegotiationTranscript) -> dict[str, Any]:
    """Analyze concession patterns in the negotiation.

    Args:
        transcript: The transcript to analyze.

    Returns:
        Concession analysis per stakeholder.
    """
    analysis = {}

    for sid in transcript.stakeholders:
        stakeholder_moves = [m for m in transcript.moves if m.stakeholder_id == sid]

        concessions = []
        total_concession = {}

        # Track position over time
        positions = []
        for move in stakeholder_moves:
            if move.move_type == "offer":
                terms = move.details.get("terms", {})
                positions.append(terms)
            elif move.move_type == "counter":
                adjustments = move.details.get("adjustments", {})
                if positions:
                    new_pos = positions[-1].copy()
                    new_pos.update(adjustments)
                    positions.append(new_pos)

        # Calculate concessions between positions
        for i in range(1, len(positions)):
            prev = positions[i - 1]
            curr = positions[i]

            move_concession = {}
            for key in set(prev.keys()) | set(curr.keys()):
                prev_val = prev.get(key, 0)
                curr_val = curr.get(key, 0)
                if prev_val != curr_val:
                    move_concession[key] = curr_val - prev_val
                    total_concession[key] = total_concession.get(key, 0) + (
                        curr_val - prev_val
                    )

            if move_concession:
                concessions.append({
                    "round": i,
                    "concession": move_concession,
                })

        analysis[sid] = {
            "num_positions": len(positions),
            "num_concessions": len(concessions),
            "concessions": concessions,
            "total_concession": total_concession,
            "average_concession_per_round": (
                {
                    k: v / len(concessions) if concessions else 0
                    for k, v in total_concession.items()
                }
            ),
        }

    return analysis


def identify_turning_points(transcript: NegotiationTranscript) -> list[dict[str, Any]]:
    """Identify key turning points in the negotiation.

    Args:
        transcript: The transcript to analyze.

    Returns:
        List of turning points with descriptions.
    """
    turning_points = []

    # Track state changes
    prev_move = None
    rejection_streak = 0
    position_changes = {}

    for i, move in enumerate(transcript.moves):
        is_turning_point = False
        reason = ""

        # First offer
        if move.move_type == "offer" and not any(
            m.move_type == "offer" for m in transcript.moves[:i]
        ):
            is_turning_point = True
            reason = "First offer sets initial anchor"

        # Break in rejection streak
        if move.move_type == "accept" and rejection_streak >= 2:
            is_turning_point = True
            reason = f"Breakthrough after {rejection_streak} rejections"
            rejection_streak = 0

        # Significant concession
        if move.move_type == "counter":
            adjustments = move.details.get("adjustments", {})
            for key, value in adjustments.items():
                if abs(value) > 10:  # Significant change
                    is_turning_point = True
                    reason = f"Major concession on {key}: {value:+.2f}"
                    break

        # Strategy shift detection
        if prev_move and move.stakeholder_id == prev_move.stakeholder_id:
            if prev_move.move_type == "reject" and move.move_type == "counter":
                is_turning_point = True
                reason = "Shift from rejection to counter-proposal"

        # Track rejections
        if move.move_type == "reject":
            rejection_streak += 1

        if is_turning_point:
            turning_points.append({
                "round": move.round_number,
                "move_index": i,
                "move_type": move.move_type,
                "stakeholder": move.stakeholder_id,
                "timestamp": move.timestamp,
                "reason": reason,
            })

        prev_move = move

    # Final acceptance is always a turning point
    final_accepts = [m for m in transcript.moves if m.move_type == "accept"]
    if final_accepts:
        final = final_accepts[-1]
        if not any(tp["move_index"] == transcript.moves.index(final) for tp in turning_points):
            turning_points.append({
                "round": final.round_number,
                "move_index": transcript.moves.index(final),
                "move_type": "accept",
                "stakeholder": final.stakeholder_id,
                "timestamp": final.timestamp,
                "reason": "Final acceptance - agreement reached",
            })

    return turning_points


def export_transcript(
    transcript: NegotiationTranscript,
    format: str = "text",
) -> str:
    """Export transcript in readable format.

    Args:
        transcript: The transcript to export.
        format: Output format ("text", "json", "markdown").

    Returns:
        Formatted transcript string.
    """
    if format == "json":
        return _export_json(transcript)
    elif format == "markdown":
        return _export_markdown(transcript)
    else:
        return _export_text(transcript)


def _export_text(transcript: NegotiationTranscript) -> str:
    """Export as plain text."""
    lines = []
    lines.append(f"Negotiation Transcript: {transcript.negotiation_id}")
    lines.append("=" * 60)
    lines.append("")

    # Stakeholders
    lines.append("PARTICIPANTS:")
    for sid, stakeholder in transcript.stakeholders.items():
        role = f" ({stakeholder.role})" if stakeholder.role else ""
        lines.append(f"  - {stakeholder.name}{role}")
    lines.append("")

    # Timeline
    lines.append("TIMELINE:")
    lines.append(f"  Started: {transcript.start_time}")
    if transcript.end_time:
        duration = transcript.end_time - transcript.start_time
        lines.append(f"  Ended: {transcript.end_time}")
        lines.append(f"  Duration: {duration}")
    lines.append("")

    # Moves
    lines.append("NEGOTIATION MOVES:")
    lines.append("-" * 40)
    for move in transcript.moves:
        stakeholder = transcript.stakeholders.get(move.stakeholder_id)
        name = stakeholder.name if stakeholder else move.stakeholder_id
        lines.append(f"Round {move.round_number}: {name} - {move.move_type.upper()}")

        if move.move_type == "offer":
            terms = move.details.get("terms", {})
            for k, v in terms.items():
                lines.append(f"    {k}: {v}")
        elif move.move_type == "counter":
            adjustments = move.details.get("adjustments", {})
            for k, v in adjustments.items():
                lines.append(f"    {k}: {v:+.2f}")
        elif move.move_type == "reject":
            reason = move.details.get("reason", "No reason given")
            lines.append(f"    Reason: {reason}")

        lines.append("")

    # Outcome
    lines.append("OUTCOME:")
    lines.append(f"  Status: {transcript.final_status.value}")
    if transcript.agreement:
        lines.append("  Agreement Terms:")
        for k, v in transcript.agreement.terms.items():
            lines.append(f"    {k}: {v}")
        lines.append("  Utility Scores:")
        for sid, score in transcript.agreement.utility_scores.items():
            name = transcript.stakeholders.get(sid, Stakeholder(id=sid, name=sid)).name
            lines.append(f"    {name}: {score:.2f}")

    return "\n".join(lines)


def _export_markdown(transcript: NegotiationTranscript) -> str:
    """Export as markdown."""
    lines = []
    lines.append(f"# Negotiation Transcript: {transcript.negotiation_id}")
    lines.append("")

    # Stakeholders
    lines.append("## Participants")
    for sid, stakeholder in transcript.stakeholders.items():
        role = f" *({stakeholder.role})*" if stakeholder.role else ""
        lines.append(f"- **{stakeholder.name}**{role}")
    lines.append("")

    # Timeline
    lines.append("## Timeline")
    lines.append(f"- **Started:** {transcript.start_time}")
    if transcript.end_time:
        duration = transcript.end_time - transcript.start_time
        lines.append(f"- **Ended:** {transcript.end_time}")
        lines.append(f"- **Duration:** {duration}")
    lines.append("")

    # Moves
    lines.append("## Negotiation Moves")
    current_round = -1
    for move in transcript.moves:
        if move.round_number != current_round:
            current_round = move.round_number
            lines.append(f"\n### Round {current_round}")

        stakeholder = transcript.stakeholders.get(move.stakeholder_id)
        name = stakeholder.name if stakeholder else move.stakeholder_id

        if move.move_type == "offer":
            lines.append(f"\n**{name}** makes an offer:")
            terms = move.details.get("terms", {})
            for k, v in terms.items():
                lines.append(f"- {k}: {v}")
        elif move.move_type == "counter":
            lines.append(f"\n**{name}** counters with:")
            adjustments = move.details.get("adjustments", {})
            for k, v in adjustments.items():
                lines.append(f"- {k}: {v:+.2f}")
        elif move.move_type == "accept":
            lines.append(f"\n**{name}** *accepts* the offer")
        elif move.move_type == "reject":
            reason = move.details.get("reason", "")
            lines.append(f"\n**{name}** *rejects* the offer")
            if reason:
                lines.append(f"> {reason}")

    # Outcome
    lines.append("\n## Outcome")
    lines.append(f"**Status:** {transcript.final_status.value}")
    if transcript.agreement:
        lines.append("\n### Agreement Terms")
        for k, v in transcript.agreement.terms.items():
            lines.append(f"- {k}: {v}")
        lines.append("\n### Utility Scores")
        lines.append("| Party | Score |")
        lines.append("|-------|-------|")
        for sid, score in transcript.agreement.utility_scores.items():
            name = transcript.stakeholders.get(sid, Stakeholder(id=sid, name=sid)).name
            lines.append(f"| {name} | {score:.2f} |")

    return "\n".join(lines)


def _export_json(transcript: NegotiationTranscript) -> str:
    """Export as JSON."""
    data = {
        "negotiation_id": transcript.negotiation_id,
        "stakeholders": {
            sid: {
                "id": s.id,
                "name": s.name,
                "role": s.role,
                "objectives": s.objectives,
            }
            for sid, s in transcript.stakeholders.items()
        },
        "moves": [
            {
                "round": m.round_number,
                "stakeholder_id": m.stakeholder_id,
                "move_type": m.move_type,
                "timestamp": m.timestamp.isoformat(),
                "details": m.details,
            }
            for m in transcript.moves
        ],
        "start_time": transcript.start_time.isoformat(),
        "end_time": transcript.end_time.isoformat() if transcript.end_time else None,
        "final_status": transcript.final_status.value,
        "agreement": (
            {
                "id": transcript.agreement.id,
                "terms": transcript.agreement.terms,
                "parties": transcript.agreement.parties,
                "utility_scores": transcript.agreement.utility_scores,
            }
            if transcript.agreement
            else None
        ),
    }
    return json.dumps(data, indent=2)


def summarize_negotiation(transcript: NegotiationTranscript) -> dict[str, Any]:
    """Generate a summary of the negotiation.

    Args:
        transcript: The transcript to summarize.

    Returns:
        Summary statistics and insights.
    """
    total_moves = len(transcript.moves)
    total_rounds = max((m.round_number for m in transcript.moves), default=0) + 1

    # Count move types
    move_counts = {}
    for move in transcript.moves:
        move_counts[move.move_type] = move_counts.get(move.move_type, 0) + 1

    # Moves per stakeholder
    moves_per_stakeholder = {}
    for move in transcript.moves:
        moves_per_stakeholder[move.stakeholder_id] = (
            moves_per_stakeholder.get(move.stakeholder_id, 0) + 1
        )

    # Duration
    duration = None
    if transcript.end_time:
        duration = (transcript.end_time - transcript.start_time).total_seconds()

    # Concession analysis
    concessions = analyze_concessions(transcript)

    # Turning points
    turning_points = identify_turning_points(transcript)

    return {
        "negotiation_id": transcript.negotiation_id,
        "num_stakeholders": len(transcript.stakeholders),
        "total_moves": total_moves,
        "total_rounds": total_rounds,
        "move_counts": move_counts,
        "moves_per_stakeholder": moves_per_stakeholder,
        "duration_seconds": duration,
        "final_status": transcript.final_status.value,
        "reached_agreement": transcript.agreement is not None,
        "num_turning_points": len(turning_points),
        "key_turning_points": turning_points[:3],  # Top 3
        "concession_summary": {
            sid: {
                "total": data["total_concession"],
                "num_concessions": data["num_concessions"],
            }
            for sid, data in concessions.items()
        },
    }
