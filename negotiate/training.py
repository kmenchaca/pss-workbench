"""Training mode for negotiation practice and learning."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional
from uuid import uuid4

from .types import (
    Agreement,
    NegotiationMove,
    NegotiationState,
    NegotiationStatus,
    Offer,
    Stakeholder,
)
from .harness import NegotiationHarness, NegotiationConfig
from .offers import evaluate_offer, make_offer
from .strategy import NegotiationStrategy, get_strategy
from .transcript import (
    NegotiationTranscript,
    analyze_concessions,
    identify_turning_points,
    summarize_negotiation,
)
from .utility import evaluate_terms


@dataclass
class TrainingScenario:
    """A training scenario for practice.

    Attributes:
        id: Scenario identifier.
        name: Scenario name.
        description: What to practice.
        stakeholders: Stakeholder setup.
        target_outcomes: What good outcomes look like.
        difficulty: Difficulty level (1-5).
        learning_objectives: What to learn.
    """

    id: str
    name: str
    description: str
    stakeholders: list[Stakeholder] = field(default_factory=list)
    target_outcomes: dict[str, Any] = field(default_factory=dict)
    difficulty: int = 3
    learning_objectives: list[str] = field(default_factory=list)


@dataclass
class TrainingSession:
    """A practice negotiation session.

    Attributes:
        id: Session identifier.
        scenario: The training scenario.
        human_stakeholder_id: Which stakeholder the human plays.
        state: Negotiation state.
        human_moves: Moves made by human.
        ai_moves: Moves made by AI opponents.
        started_at: When session started.
        completed: Whether session is complete.
        score: Performance score.
    """

    id: str
    scenario: TrainingScenario
    human_stakeholder_id: str
    state: Optional[NegotiationState] = None
    human_moves: list[NegotiationMove] = field(default_factory=list)
    ai_moves: list[NegotiationMove] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    completed: bool = False
    score: Optional[dict[str, Any]] = None


def create_training_scenario(
    name: str,
    description: str,
    stakeholders: list[Stakeholder],
    target_outcomes: dict[str, Any],
    difficulty: int = 3,
    learning_objectives: Optional[list[str]] = None,
) -> TrainingScenario:
    """Create a new training scenario.

    Args:
        name: Scenario name.
        description: What to practice.
        stakeholders: Stakeholder setup.
        target_outcomes: What good outcomes look like.
        difficulty: Difficulty level (1-5).
        learning_objectives: What to learn.

    Returns:
        A new training scenario.
    """
    return TrainingScenario(
        id=str(uuid4())[:8],
        name=name,
        description=description,
        stakeholders=stakeholders,
        target_outcomes=target_outcomes,
        difficulty=difficulty,
        learning_objectives=learning_objectives or [],
    )


def start_training_session(
    scenario: TrainingScenario,
    human_stakeholder_id: str,
    ai_strategies: Optional[dict[str, str]] = None,
) -> TrainingSession:
    """Start a new training session.

    Args:
        scenario: The scenario to practice.
        human_stakeholder_id: Which stakeholder the human plays.
        ai_strategies: Strategy names for AI players.

    Returns:
        A new training session.
    """
    session = TrainingSession(
        id=str(uuid4())[:8],
        scenario=scenario,
        human_stakeholder_id=human_stakeholder_id,
    )

    # Initialize negotiation state
    harness = NegotiationHarness(
        config=NegotiationConfig(
            max_rounds=50,
            track_transcript=True,
        )
    )

    # Set AI strategies
    if ai_strategies:
        for sid, strategy_name in ai_strategies.items():
            if sid != human_stakeholder_id:
                harness.strategies[sid] = get_strategy(strategy_name)

    harness.initialize(scenario.stakeholders)
    session.state = harness.state

    return session


def submit_human_move(
    session: TrainingSession,
    move_type: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    """Submit a move in training session.

    Args:
        session: The training session.
        move_type: Type of move (offer, counter, accept, reject).
        details: Move details (terms, offer_id, etc.).

    Returns:
        Result including AI responses.
    """
    if session.completed:
        return {"error": "Session already completed"}

    human = session.state.stakeholders[session.human_stakeholder_id]

    # Record human move
    human_move = NegotiationMove(
        move_type=move_type,
        stakeholder_id=session.human_stakeholder_id,
        timestamp=datetime.now(),
        details=details,
        round_number=session.state.current_round,
    )
    session.human_moves.append(human_move)
    session.state.moves.append(human_move)

    # Execute the move
    if move_type == "offer":
        make_offer(human, details.get("terms", {}), state=session.state)

    # Get AI responses
    ai_responses = []
    for sid, stakeholder in session.state.stakeholders.items():
        if sid != session.human_stakeholder_id:
            response = _generate_ai_response(session, sid)
            if response:
                ai_responses.append(response)
                session.ai_moves.append(response)
                session.state.moves.append(response)

    # Check if negotiation is complete
    # (simplified - real implementation would use protocol)
    accepts = [m for m in session.state.moves if m.move_type == "accept"]
    if len(accepts) >= len(session.state.stakeholders) - 1:
        session.completed = True
        session.score = score_negotiation(session)

    session.state.current_round += 1

    return {
        "human_move": human_move,
        "ai_responses": ai_responses,
        "round": session.state.current_round,
        "completed": session.completed,
    }


def _generate_ai_response(
    session: TrainingSession,
    ai_stakeholder_id: str,
) -> Optional[NegotiationMove]:
    """Generate AI response to human move.

    Args:
        session: The training session.
        ai_stakeholder_id: AI stakeholder ID.

    Returns:
        AI's response move or None.
    """
    stakeholder = session.state.stakeholders[ai_stakeholder_id]

    # Get last human offer
    human_offers = [
        m
        for m in session.state.moves
        if m.stakeholder_id == session.human_stakeholder_id
        and m.move_type in ["offer", "counter"]
    ]

    if not human_offers:
        return None

    last_human_move = human_offers[-1]
    terms = last_human_move.details.get("terms", {})

    # Evaluate the offer
    evaluation = evaluate_offer(stakeholder, Offer(
        id="temp",
        proposer=session.human_stakeholder_id,
        terms=terms,
    ))

    # Decide response based on difficulty
    difficulty = session.scenario.difficulty

    if evaluation["is_acceptable"] and evaluation["utility"] > 50 - difficulty * 5:
        # Accept good offers
        return NegotiationMove(
            move_type="accept",
            stakeholder_id=ai_stakeholder_id,
            timestamp=datetime.now(),
            details={"offer_id": "human_offer"},
            round_number=session.state.current_round,
        )
    elif evaluation["utility"] < 20:
        # Reject very bad offers
        return NegotiationMove(
            move_type="reject",
            stakeholder_id=ai_stakeholder_id,
            timestamp=datetime.now(),
            details={"reason": "Terms unacceptable"},
            round_number=session.state.current_round,
        )
    else:
        # Counter-offer
        counter_terms = {}
        for key, value in terms.items():
            target = stakeholder.objectives.get(key, value)
            # Move toward AI's preference (more at higher difficulty)
            adjustment = (target - value) * (0.3 + difficulty * 0.1)
            counter_terms[key] = value + adjustment

        return NegotiationMove(
            move_type="counter",
            stakeholder_id=ai_stakeholder_id,
            timestamp=datetime.now(),
            details={"terms": counter_terms},
            round_number=session.state.current_round,
        )


def feedback(
    session: TrainingSession,
    human_moves: list[NegotiationMove],
) -> dict[str, Any]:
    """Provide feedback on human moves.

    Args:
        session: The training session.
        human_moves: Human's moves to analyze.

    Returns:
        Feedback on each move.
    """
    feedback_items = []

    for i, move in enumerate(human_moves):
        move_feedback = {
            "move_index": i,
            "move_type": move.move_type,
            "strengths": [],
            "improvements": [],
            "score": 0,
        }

        if move.move_type == "offer":
            terms = move.details.get("terms", {})
            human = session.state.stakeholders[session.human_stakeholder_id]

            # Check if offer is within constraints
            for key, (min_val, max_val) in human.constraints.items():
                if key in terms:
                    if min_val <= terms[key] <= max_val:
                        move_feedback["strengths"].append(
                            f"Offer on {key} within acceptable range"
                        )
                        move_feedback["score"] += 10
                    else:
                        move_feedback["improvements"].append(
                            f"Offer on {key} outside your acceptable range"
                        )
                        move_feedback["score"] -= 10

            # Check anchoring
            if i == 0:
                # First offer - check if it's a good anchor
                for key, target in human.objectives.items():
                    if key in terms:
                        if target == 0:  # Minimize
                            if terms[key] < 50:
                                move_feedback["strengths"].append(
                                    f"Strong anchor on {key} - started low"
                                )
                                move_feedback["score"] += 15
                        elif target == float("inf"):  # Maximize
                            if terms[key] > 50:
                                move_feedback["strengths"].append(
                                    f"Strong anchor on {key} - started high"
                                )
                                move_feedback["score"] += 15

        elif move.move_type == "counter":
            # Analyze concession pattern
            if i > 0:
                prev_offer = None
                for j in range(i - 1, -1, -1):
                    if human_moves[j].move_type in ["offer", "counter"]:
                        prev_offer = human_moves[j]
                        break

                if prev_offer:
                    prev_terms = prev_offer.details.get("terms", {})
                    curr_terms = move.details.get("terms", {})

                    concession_size = 0
                    for key in prev_terms:
                        if key in curr_terms:
                            concession_size += abs(
                                curr_terms[key] - prev_terms[key]
                            )

                    if concession_size < 5:
                        move_feedback["strengths"].append(
                            "Small concession - maintains negotiating power"
                        )
                        move_feedback["score"] += 10
                    elif concession_size > 20:
                        move_feedback["improvements"].append(
                            "Large concession - may signal desperation"
                        )
                        move_feedback["score"] -= 5

        elif move.move_type == "accept":
            # Check if accepted at good terms
            # Get the offer that was accepted
            human = session.state.stakeholders[session.human_stakeholder_id]
            utility = evaluate_terms(human, move.details.get("terms", {}))

            if utility > 70:
                move_feedback["strengths"].append(
                    f"Accepted at excellent utility ({utility:.1f})"
                )
                move_feedback["score"] += 20
            elif utility > 50:
                move_feedback["strengths"].append(
                    f"Accepted at good utility ({utility:.1f})"
                )
                move_feedback["score"] += 10
            else:
                move_feedback["improvements"].append(
                    f"Accepted at low utility ({utility:.1f}) - could negotiate more"
                )
                move_feedback["score"] -= 10

        feedback_items.append(move_feedback)

    total_score = sum(f["score"] for f in feedback_items)
    max_score = len(feedback_items) * 30  # Max 30 per move

    return {
        "move_feedback": feedback_items,
        "total_score": total_score,
        "max_score": max_score,
        "percentage": (total_score / max_score * 100) if max_score > 0 else 0,
    }


def score_negotiation(session: TrainingSession) -> dict[str, Any]:
    """Score the overall negotiation performance.

    Args:
        session: The completed training session.

    Returns:
        Performance scores.
    """
    human = session.state.stakeholders[session.human_stakeholder_id]

    scores = {
        "outcome_score": 0,
        "process_score": 0,
        "strategy_score": 0,
        "total": 0,
        "max_total": 100,
        "grade": "F",
        "feedback": [],
    }

    # Outcome score (40 points max)
    if session.state.agreements:
        agreement = session.state.agreements[-1]
        utility = agreement.utility_scores.get(session.human_stakeholder_id, 0)

        # Compare to target outcome
        target_utility = session.scenario.target_outcomes.get("min_utility", 50)

        if utility >= target_utility * 1.2:
            scores["outcome_score"] = 40
            scores["feedback"].append("Excellent outcome - exceeded targets!")
        elif utility >= target_utility:
            scores["outcome_score"] = 30
            scores["feedback"].append("Good outcome - met targets")
        elif utility >= target_utility * 0.8:
            scores["outcome_score"] = 20
            scores["feedback"].append("Acceptable outcome - below targets")
        else:
            scores["outcome_score"] = 10
            scores["feedback"].append("Poor outcome - significantly below targets")
    else:
        scores["outcome_score"] = 0
        scores["feedback"].append("No agreement reached")

    # Process score (30 points max)
    num_rounds = session.state.current_round
    target_rounds = session.scenario.target_outcomes.get("max_rounds", 20)

    if num_rounds <= target_rounds * 0.5:
        scores["process_score"] = 30
        scores["feedback"].append("Efficient negotiation")
    elif num_rounds <= target_rounds:
        scores["process_score"] = 20
        scores["feedback"].append("Reasonable pace")
    else:
        scores["process_score"] = 10
        scores["feedback"].append("Negotiation took too long")

    # Strategy score (30 points max)
    move_feedback = feedback(session, session.human_moves)
    strategy_pct = move_feedback["percentage"]

    if strategy_pct >= 80:
        scores["strategy_score"] = 30
    elif strategy_pct >= 60:
        scores["strategy_score"] = 20
    elif strategy_pct >= 40:
        scores["strategy_score"] = 15
    else:
        scores["strategy_score"] = 5

    # Total and grade
    scores["total"] = (
        scores["outcome_score"]
        + scores["process_score"]
        + scores["strategy_score"]
    )

    if scores["total"] >= 90:
        scores["grade"] = "A"
    elif scores["total"] >= 80:
        scores["grade"] = "B"
    elif scores["total"] >= 70:
        scores["grade"] = "C"
    elif scores["total"] >= 60:
        scores["grade"] = "D"
    else:
        scores["grade"] = "F"

    return scores


def suggest_improvement(session: TrainingSession) -> list[dict[str, str]]:
    """Suggest improvements based on session performance.

    Args:
        session: The training session.

    Returns:
        List of improvement suggestions.
    """
    suggestions = []

    # Analyze patterns
    move_feedback = feedback(session, session.human_moves)

    # Check for common issues
    large_concessions = sum(
        1
        for f in move_feedback["move_feedback"]
        if any("Large concession" in imp for imp in f["improvements"])
    )

    if large_concessions > 2:
        suggestions.append({
            "area": "Concession Strategy",
            "issue": "Making too many large concessions",
            "suggestion": "Try smaller, incremental concessions to maintain leverage",
            "priority": "high",
        })

    # Check anchoring
    first_move = session.human_moves[0] if session.human_moves else None
    if first_move and first_move.move_type == "offer":
        terms = first_move.details.get("terms", {})
        human = session.state.stakeholders[session.human_stakeholder_id]

        weak_anchor = False
        for key, target in human.objectives.items():
            if key in terms:
                if target == 0 and terms[key] > 70:  # Minimize but started high
                    weak_anchor = True
                elif target == float("inf") and terms[key] < 30:  # Maximize but started low
                    weak_anchor = True

        if weak_anchor:
            suggestions.append({
                "area": "Anchoring",
                "issue": "Initial offer not aggressive enough",
                "suggestion": "Start with a more extreme anchor to leave room for negotiation",
                "priority": "high",
            })

    # Check pace
    if session.state.current_round > 15:
        suggestions.append({
            "area": "Efficiency",
            "issue": "Negotiation taking too long",
            "suggestion": "Look for creative ways to break impasse earlier",
            "priority": "medium",
        })

    # Check if gave up value
    if session.score and session.score["outcome_score"] < 30:
        suggestions.append({
            "area": "Value Creation",
            "issue": "Final outcome below potential",
            "suggestion": "Look for win-win opportunities and trade-offs",
            "priority": "high",
        })

    return suggestions


# Pre-built training scenarios
BASIC_SCENARIOS = [
    TrainingScenario(
        id="basic_salary",
        name="Salary Negotiation",
        description="Negotiate your starting salary with an employer",
        difficulty=2,
        learning_objectives=[
            "Practice anchoring",
            "Handle counter-offers",
            "Know your BATNA",
        ],
    ),
    TrainingScenario(
        id="car_purchase",
        name="Car Purchase",
        description="Negotiate the price of a used car",
        difficulty=3,
        learning_objectives=[
            "Research and preparation",
            "Information asymmetry",
            "Walk-away point",
        ],
    ),
    TrainingScenario(
        id="contract_renewal",
        name="Contract Renewal",
        description="Renegotiate a service contract",
        difficulty=4,
        learning_objectives=[
            "Leverage existing relationship",
            "Multi-issue negotiation",
            "Long-term value",
        ],
    ),
]
