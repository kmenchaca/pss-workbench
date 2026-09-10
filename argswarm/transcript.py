"""
Debate transcript management for the Argumentative Swarm system.

Provides a complete record of the debate, tracks which points were addressed,
and exports to readable formats.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from .types import (
    Argument,
    DebateState,
    Position,
    Refutation,
    Response,
    RoundType,
    Stance,
)


@dataclass
class TranscriptEntry:
    """
    A single entry in the debate transcript.

    Attributes:
        timestamp: When this entry was created
        round_number: Which round this occurred in
        round_type: Type of round
        entry_type: Type of entry (argument, refutation, response, note)
        position_id: Position that made this entry
        content: The actual content
        references: IDs of related entries (e.g., refutation references argument)
        metadata: Additional entry metadata
    """
    timestamp: str
    round_number: int
    round_type: RoundType
    entry_type: str
    position_id: str
    content: str
    references: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DebateTranscript:
    """
    Full record of a debate.

    Tracks all arguments, refutations, and responses with metadata
    about timing and cross-references.

    Attributes:
        proposition: The topic being debated
        positions: All positions in the debate
        entries: Chronological list of transcript entries
        start_time: When the debate started
        end_time: When the debate ended (if finished)
        notes: General notes about the debate
        bulletin_board: Shared information visible to all positions
    """
    proposition: str
    positions: list[Position] = field(default_factory=list)
    entries: list[TranscriptEntry] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    notes: list[str] = field(default_factory=list)
    bulletin_board: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Set start time if not provided."""
        if not self.start_time:
            self.start_time = datetime.now().isoformat()


def create_transcript(
    proposition: str,
    positions: list[Position],
) -> DebateTranscript:
    """
    Create a new debate transcript.

    Args:
        proposition: The debate topic
        positions: List of positions in the debate

    Returns:
        A new DebateTranscript instance
    """
    return DebateTranscript(
        proposition=proposition,
        positions=positions,
        start_time=datetime.now().isoformat(),
    )


def add_argument(
    transcript: DebateTranscript,
    argument: Argument,
    position: Position,
    round_number: int,
    round_type: RoundType,
) -> TranscriptEntry:
    """
    Add an argument to the transcript.

    Args:
        transcript: The transcript to update
        argument: The argument to add
        position: Position making the argument
        round_number: Current round number
        round_type: Type of round

    Returns:
        The created TranscriptEntry
    """
    content = _format_argument_content(argument)

    entry = TranscriptEntry(
        timestamp=datetime.now().isoformat(),
        round_number=round_number,
        round_type=round_type,
        entry_type="argument",
        position_id=position.id,
        content=content,
        references=[],
        metadata={
            "argument_id": argument.id,
            "strength": argument.strength,
            "has_backing": bool(argument.backing),
            "has_rebuttal": bool(argument.rebuttal),
        },
    )

    transcript.entries.append(entry)
    return entry


def add_refutation(
    transcript: DebateTranscript,
    refutation: Refutation,
    position: Position,
    round_number: int,
    round_type: RoundType,
) -> TranscriptEntry:
    """
    Add a refutation to the transcript.

    Args:
        transcript: The transcript to update
        refutation: The refutation to add
        position: Position making the refutation
        round_number: Current round number
        round_type: Type of round

    Returns:
        The created TranscriptEntry
    """
    content = _format_refutation_content(refutation)

    entry = TranscriptEntry(
        timestamp=datetime.now().isoformat(),
        round_number=round_number,
        round_type=round_type,
        entry_type="refutation",
        position_id=position.id,
        content=content,
        references=[refutation.target_argument_id],
        metadata={
            "refutation_id": refutation.id,
            "refutation_type": refutation.refutation_type,
            "strength": refutation.strength,
            "target_argument_id": refutation.target_argument_id,
        },
    )

    transcript.entries.append(entry)
    return entry


def add_response(
    transcript: DebateTranscript,
    response: Response,
    position: Position,
    round_number: int,
    round_type: RoundType,
) -> TranscriptEntry:
    """
    Add a response to the transcript.

    Args:
        transcript: The transcript to update
        response: The response to add
        position: Position making the response
        round_number: Current round number
        round_type: Type of round

    Returns:
        The created TranscriptEntry
    """
    entry = TranscriptEntry(
        timestamp=datetime.now().isoformat(),
        round_number=round_number,
        round_type=round_type,
        entry_type="response",
        position_id=position.id,
        content=response.response_text,
        references=[response.refutation_id, response.argument_id],
        metadata={
            "response_id": response.id,
            "refutation_id": response.refutation_id,
            "argument_id": response.argument_id,
            "strengthened_claim": response.strengthened_claim,
        },
    )

    transcript.entries.append(entry)
    return entry


def add_note(
    transcript: DebateTranscript,
    note: str,
    round_number: int = 0,
) -> None:
    """
    Add a general note to the transcript.

    Args:
        transcript: The transcript to update
        note: The note content
        round_number: Associated round (0 for general)
    """
    timestamped_note = f"[Round {round_number}] {datetime.now().isoformat()}: {note}"
    transcript.notes.append(timestamped_note)


def post_to_bulletin_board(
    transcript: DebateTranscript,
    message: str,
    position: Position,
    message_type: str = "info",
    visibility: str = "all",
) -> dict[str, Any]:
    """
    Post a message to the shared bulletin board.

    The bulletin board serves as a shared information space where
    positions can communicate discoveries, concessions, or challenges.

    Args:
        transcript: The transcript to update
        message: The message content
        position: Position posting the message
        message_type: Type of message (info, concession, challenge, question)
        visibility: Who can see it (all, opponents, allies)

    Returns:
        The bulletin board entry
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "position_id": position.id,
        "stance": position.stance.value,
        "message": message,
        "type": message_type,
        "visibility": visibility,
    }

    transcript.bulletin_board.append(entry)
    return entry


def get_bulletin_board_for_position(
    transcript: DebateTranscript,
    position: Position,
    all_positions: list[Position],
) -> list[dict[str, Any]]:
    """
    Get bulletin board messages visible to a position.

    Args:
        transcript: The transcript
        position: Position to get messages for
        all_positions: All positions in debate

    Returns:
        List of visible bulletin board entries
    """
    visible: list[dict[str, Any]] = []

    # Find allied and opposing stances
    allied_stances = {position.stance}
    opposing_stances = set()
    if position.stance == Stance.FOR:
        opposing_stances = {Stance.AGAINST}
    elif position.stance == Stance.AGAINST:
        opposing_stances = {Stance.FOR}

    for entry in transcript.bulletin_board:
        visibility = entry.get("visibility", "all")

        if visibility == "all":
            visible.append(entry)
        elif visibility == "opponents":
            # Only visible if poster is opponent
            poster_stance_value = entry.get("stance", "")
            try:
                poster_stance = Stance(poster_stance_value)
                if poster_stance in opposing_stances:
                    visible.append(entry)
            except ValueError:
                pass
        elif visibility == "allies":
            # Only visible if poster is ally
            poster_stance_value = entry.get("stance", "")
            try:
                poster_stance = Stance(poster_stance_value)
                if poster_stance in allied_stances:
                    visible.append(entry)
            except ValueError:
                pass

    return visible


def get_addressed_points(transcript: DebateTranscript) -> dict[str, list[str]]:
    """
    Track which arguments have been addressed by refutations/responses.

    Returns:
        Dictionary mapping argument IDs to list of addressing entry IDs
    """
    addressed: dict[str, list[str]] = {}

    for entry in transcript.entries:
        if entry.entry_type in ("refutation", "response"):
            for ref_id in entry.references:
                if ref_id not in addressed:
                    addressed[ref_id] = []
                addressed[ref_id].append(entry.metadata.get("refutation_id", "")
                                         or entry.metadata.get("response_id", ""))

    return addressed


def get_unaddressed_arguments(
    transcript: DebateTranscript,
    position: Optional[Position] = None,
) -> list[str]:
    """
    Find arguments that haven't been refuted.

    Args:
        transcript: The transcript
        position: If provided, only check this position's arguments

    Returns:
        List of unaddressed argument IDs
    """
    addressed = get_addressed_points(transcript)

    # Get all argument IDs
    argument_entries = [e for e in transcript.entries if e.entry_type == "argument"]

    if position:
        argument_entries = [e for e in argument_entries if e.position_id == position.id]

    argument_ids = [e.metadata.get("argument_id", "") for e in argument_entries]

    return [aid for aid in argument_ids if aid and aid not in addressed]


def export_to_text(transcript: DebateTranscript) -> str:
    """
    Export transcript to readable text format.

    Args:
        transcript: The transcript to export

    Returns:
        Formatted text representation
    """
    lines: list[str] = []

    # Header
    lines.append("=" * 60)
    lines.append("DEBATE TRANSCRIPT")
    lines.append("=" * 60)
    lines.append(f"\nProposition: {transcript.proposition}")
    lines.append(f"Started: {transcript.start_time}")
    if transcript.end_time:
        lines.append(f"Ended: {transcript.end_time}")
    lines.append("")

    # Positions
    lines.append("POSITIONS:")
    for pos in transcript.positions:
        lines.append(f"  - [{pos.stance.value.upper()}] {pos.id}: {pos.description[:50]}...")
    lines.append("")

    # Entries by round
    current_round = 0
    for entry in transcript.entries:
        if entry.round_number != current_round:
            current_round = entry.round_number
            lines.append("-" * 40)
            lines.append(f"ROUND {current_round} ({entry.round_type.value.upper()})")
            lines.append("-" * 40)

        position_stance = _get_position_stance(transcript, entry.position_id)
        lines.append(f"\n[{position_stance}] {entry.entry_type.upper()}:")
        lines.append(f"  {entry.content}")

        if entry.references:
            lines.append(f"  (References: {', '.join(entry.references)})")

    # Bulletin board
    if transcript.bulletin_board:
        lines.append("\n" + "=" * 40)
        lines.append("BULLETIN BOARD")
        lines.append("=" * 40)
        for msg in transcript.bulletin_board:
            lines.append(f"\n[{msg['type'].upper()}] {msg['stance']}: {msg['message']}")

    # Notes
    if transcript.notes:
        lines.append("\n" + "=" * 40)
        lines.append("NOTES")
        lines.append("=" * 40)
        for note in transcript.notes:
            lines.append(f"  {note}")

    return "\n".join(lines)


def export_to_json(transcript: DebateTranscript) -> str:
    """
    Export transcript to JSON format.

    Args:
        transcript: The transcript to export

    Returns:
        JSON string representation
    """
    data = {
        "proposition": transcript.proposition,
        "start_time": transcript.start_time,
        "end_time": transcript.end_time,
        "positions": [
            {
                "id": p.id,
                "stance": p.stance.value,
                "description": p.description,
                "branch_id": p.branch_id,
            }
            for p in transcript.positions
        ],
        "entries": [
            {
                "timestamp": e.timestamp,
                "round_number": e.round_number,
                "round_type": e.round_type.value,
                "entry_type": e.entry_type,
                "position_id": e.position_id,
                "content": e.content,
                "references": e.references,
                "metadata": e.metadata,
            }
            for e in transcript.entries
        ],
        "bulletin_board": transcript.bulletin_board,
        "notes": transcript.notes,
    }

    return json.dumps(data, indent=2)


def import_from_json(json_str: str) -> DebateTranscript:
    """
    Import transcript from JSON format.

    Args:
        json_str: JSON string to import

    Returns:
        DebateTranscript instance
    """
    data = json.loads(json_str)

    positions = [
        Position(
            id=p["id"],
            stance=Stance(p["stance"]),
            proposition=data["proposition"],
            description=p.get("description", ""),
            branch_id=p.get("branch_id"),
        )
        for p in data.get("positions", [])
    ]

    entries = [
        TranscriptEntry(
            timestamp=e["timestamp"],
            round_number=e["round_number"],
            round_type=RoundType(e["round_type"]),
            entry_type=e["entry_type"],
            position_id=e["position_id"],
            content=e["content"],
            references=e.get("references", []),
            metadata=e.get("metadata", {}),
        )
        for e in data.get("entries", [])
    ]

    return DebateTranscript(
        proposition=data["proposition"],
        positions=positions,
        entries=entries,
        start_time=data.get("start_time", ""),
        end_time=data.get("end_time", ""),
        notes=data.get("notes", []),
        bulletin_board=data.get("bulletin_board", []),
    )


def get_round_summary(
    transcript: DebateTranscript,
    round_number: int,
) -> dict[str, Any]:
    """
    Get a summary of a specific round.

    Args:
        transcript: The transcript
        round_number: Round to summarize

    Returns:
        Dictionary with round statistics
    """
    round_entries = [e for e in transcript.entries if e.round_number == round_number]

    arguments = [e for e in round_entries if e.entry_type == "argument"]
    refutations = [e for e in round_entries if e.entry_type == "refutation"]
    responses = [e for e in round_entries if e.entry_type == "response"]

    # Calculate average strengths
    arg_strengths = [e.metadata.get("strength", 0.5) for e in arguments]
    ref_strengths = [e.metadata.get("strength", 0.5) for e in refutations]

    return {
        "round_number": round_number,
        "round_type": round_entries[0].round_type.value if round_entries else None,
        "argument_count": len(arguments),
        "refutation_count": len(refutations),
        "response_count": len(responses),
        "avg_argument_strength": sum(arg_strengths) / len(arg_strengths) if arg_strengths else 0,
        "avg_refutation_strength": sum(ref_strengths) / len(ref_strengths) if ref_strengths else 0,
        "positions_active": list(set(e.position_id for e in round_entries)),
    }


def _format_argument_content(argument: Argument) -> str:
    """Format argument for transcript display."""
    parts = [f"CLAIM: {argument.claim}"]

    if argument.evidence:
        parts.append(f"EVIDENCE: {argument.evidence}")

    if argument.warrant:
        parts.append(f"WARRANT: {argument.warrant}")

    if argument.backing:
        parts.append(f"BACKING: {argument.backing}")

    if argument.qualifier:
        parts.append(f"QUALIFIER: {argument.qualifier}")

    if argument.rebuttal:
        parts.append(f"REBUTTAL: {argument.rebuttal}")

    return " | ".join(parts)


def _format_refutation_content(refutation: Refutation) -> str:
    """Format refutation for transcript display."""
    parts = [
        f"TYPE: {refutation.refutation_type}",
        f"COUNTER: {refutation.counter_claim}",
    ]

    if refutation.evidence:
        parts.append(f"EVIDENCE: {refutation.evidence}")

    return " | ".join(parts)


def _get_position_stance(transcript: DebateTranscript, position_id: str) -> str:
    """Get stance string for a position ID."""
    for pos in transcript.positions:
        if pos.id == position_id:
            return pos.stance.value.upper()
    return "UNKNOWN"
