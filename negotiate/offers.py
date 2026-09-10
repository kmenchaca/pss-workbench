"""Offer mechanics for negotiations."""

from datetime import datetime, timedelta
from typing import Optional
from uuid import uuid4

from .types import (
    CounterOffer,
    NegotiationMove,
    NegotiationState,
    Offer,
    OfferStatus,
    Stakeholder,
)
from .utility import evaluate_terms, UtilityFunction


def make_offer(
    stakeholder: Stakeholder,
    terms: dict[str, float],
    recipients: Optional[list[str]] = None,
    expires_in: Optional[timedelta] = None,
    state: Optional[NegotiationState] = None,
) -> Offer:
    """Create a new offer from a stakeholder.

    Args:
        stakeholder: The stakeholder making the offer.
        terms: The proposed terms.
        recipients: IDs of stakeholders to receive the offer (None = all).
        expires_in: Optional time until offer expires.
        state: Optional negotiation state to record the offer in.

    Returns:
        The new Offer.
    """
    now = datetime.now()
    offer = Offer(
        id=str(uuid4())[:8],
        proposer=stakeholder.id,
        terms=terms.copy(),
        timestamp=now,
        recipients=recipients or [],
        status=OfferStatus.PENDING,
        expires_at=now + expires_in if expires_in else None,
    )

    if state:
        state.offers[offer.id] = offer
        move = NegotiationMove(
            move_type="offer",
            stakeholder_id=stakeholder.id,
            timestamp=now,
            details={"offer_id": offer.id, "terms": terms},
            round_number=state.current_round,
        )
        state.moves.append(move)

    return offer


def evaluate_offer(
    stakeholder: Stakeholder,
    offer: Offer,
    utility_registry: Optional[dict[str, UtilityFunction]] = None,
) -> dict[str, any]:
    """Assess an incoming offer for a stakeholder.

    Args:
        stakeholder: The stakeholder evaluating the offer.
        offer: The offer to evaluate.
        utility_registry: Optional registry of utility functions.

    Returns:
        Dictionary with evaluation results.
    """
    utility = evaluate_terms(stakeholder, offer.terms, utility_registry)

    # Check constraint violations
    violations = []
    for key, (min_val, max_val) in stakeholder.constraints.items():
        if key in offer.terms:
            value = offer.terms[key]
            if value < min_val:
                violations.append(f"{key} below minimum ({value} < {min_val})")
            elif value > max_val:
                violations.append(f"{key} above maximum ({value} > {max_val})")

    # Check against objectives
    objective_gaps = {}
    for key, target in stakeholder.objectives.items():
        if key in offer.terms:
            gap = offer.terms[key] - target
            objective_gaps[key] = gap

    return {
        "offer_id": offer.id,
        "utility": utility,
        "constraint_violations": violations,
        "is_acceptable": len(violations) == 0,
        "objective_gaps": objective_gaps,
        "proposer": offer.proposer,
    }


def counter_offer(
    stakeholder: Stakeholder,
    original_offer: Offer,
    adjustments: dict[str, float],
    justification: Optional[str] = None,
    state: Optional[NegotiationState] = None,
) -> CounterOffer:
    """Respond to an offer with modified terms.

    Args:
        stakeholder: The stakeholder making the counter-offer.
        original_offer: The offer being responded to.
        adjustments: Changes to make to the terms (key -> new value).
        justification: Optional explanation for the counter.
        state: Optional negotiation state to record the counter in.

    Returns:
        The new CounterOffer.
    """
    now = datetime.now()

    # Calculate new terms
    new_terms = original_offer.terms.copy()
    new_terms.update(adjustments)

    # Calculate concessions (how much the original proposer would gain)
    concessions = {}
    for key, new_val in new_terms.items():
        if key in original_offer.terms:
            old_val = original_offer.terms[key]
            if new_val != old_val:
                concessions[key] = new_val - old_val

    counter = CounterOffer(
        id=str(uuid4())[:8],
        original_offer_id=original_offer.id,
        proposer=stakeholder.id,
        new_terms=new_terms,
        concessions=concessions,
        timestamp=now,
        justification=justification,
    )

    # Update original offer status
    original_offer.status = OfferStatus.COUNTERED

    if state:
        state.counter_offers[counter.id] = counter
        move = NegotiationMove(
            move_type="counter",
            stakeholder_id=stakeholder.id,
            timestamp=now,
            details={
                "counter_id": counter.id,
                "original_offer_id": original_offer.id,
                "adjustments": adjustments,
            },
            round_number=state.current_round,
        )
        state.moves.append(move)

    return counter


def accept(
    stakeholder: Stakeholder,
    offer: Offer,
    state: Optional[NegotiationState] = None,
) -> dict[str, any]:
    """Accept an offer.

    Args:
        stakeholder: The stakeholder accepting.
        offer: The offer to accept.
        state: Optional negotiation state to record the acceptance.

    Returns:
        Dictionary with acceptance details.
    """
    now = datetime.now()
    offer.status = OfferStatus.ACCEPTED

    result = {
        "accepted": True,
        "offer_id": offer.id,
        "stakeholder_id": stakeholder.id,
        "terms": offer.terms,
        "timestamp": now,
    }

    if state:
        move = NegotiationMove(
            move_type="accept",
            stakeholder_id=stakeholder.id,
            timestamp=now,
            details={"offer_id": offer.id},
            round_number=state.current_round,
        )
        state.moves.append(move)

    return result


def reject(
    stakeholder: Stakeholder,
    offer: Offer,
    reason: Optional[str] = None,
    state: Optional[NegotiationState] = None,
) -> dict[str, any]:
    """Reject an offer.

    Args:
        stakeholder: The stakeholder rejecting.
        offer: The offer to reject.
        reason: Optional reason for rejection.
        state: Optional negotiation state to record the rejection.

    Returns:
        Dictionary with rejection details.
    """
    now = datetime.now()
    offer.status = OfferStatus.REJECTED

    result = {
        "rejected": True,
        "offer_id": offer.id,
        "stakeholder_id": stakeholder.id,
        "reason": reason,
        "timestamp": now,
    }

    if state:
        move = NegotiationMove(
            move_type="reject",
            stakeholder_id=stakeholder.id,
            timestamp=now,
            details={"offer_id": offer.id, "reason": reason},
            round_number=state.current_round,
        )
        state.moves.append(move)

    return result


def withdraw(
    stakeholder: Stakeholder,
    offer: Offer,
    state: Optional[NegotiationState] = None,
) -> dict[str, any]:
    """Withdraw an offer.

    Args:
        stakeholder: The stakeholder withdrawing (must be the proposer).
        offer: The offer to withdraw.
        state: Optional negotiation state to record the withdrawal.

    Returns:
        Dictionary with withdrawal details.

    Raises:
        ValueError: If stakeholder is not the proposer.
    """
    if offer.proposer != stakeholder.id:
        raise ValueError("Only the proposer can withdraw an offer")

    now = datetime.now()
    offer.status = OfferStatus.WITHDRAWN

    result = {
        "withdrawn": True,
        "offer_id": offer.id,
        "stakeholder_id": stakeholder.id,
        "timestamp": now,
    }

    if state:
        move = NegotiationMove(
            move_type="withdraw",
            stakeholder_id=stakeholder.id,
            timestamp=now,
            details={"offer_id": offer.id},
            round_number=state.current_round,
        )
        state.moves.append(move)

    return result


def check_offer_expired(offer: Offer) -> bool:
    """Check if an offer has expired.

    Args:
        offer: The offer to check.

    Returns:
        True if expired, False otherwise.
    """
    if offer.expires_at is None:
        return False
    return datetime.now() > offer.expires_at


def get_pending_offers(
    state: NegotiationState,
    for_stakeholder: Optional[str] = None,
) -> list[Offer]:
    """Get all pending offers, optionally filtered by recipient.

    Args:
        state: The negotiation state.
        for_stakeholder: Optional stakeholder ID to filter by.

    Returns:
        List of pending offers.
    """
    pending = []
    for offer in state.offers.values():
        if offer.status != OfferStatus.PENDING:
            continue
        if check_offer_expired(offer):
            offer.status = OfferStatus.EXPIRED
            continue
        if for_stakeholder:
            if for_stakeholder in offer.recipients or not offer.recipients:
                pending.append(offer)
        else:
            pending.append(offer)
    return pending


def get_best_offer(
    stakeholder: Stakeholder,
    offers: list[Offer],
    utility_registry: Optional[dict[str, UtilityFunction]] = None,
) -> Optional[Offer]:
    """Find the best offer for a stakeholder from a list.

    Args:
        stakeholder: The stakeholder evaluating.
        offers: List of offers to evaluate.
        utility_registry: Optional registry of utility functions.

    Returns:
        The best offer, or None if list is empty.
    """
    if not offers:
        return None

    best = None
    best_utility = float("-inf")

    for offer in offers:
        evaluation = evaluate_offer(stakeholder, offer, utility_registry)
        if evaluation["is_acceptable"] and evaluation["utility"] > best_utility:
            best_utility = evaluation["utility"]
            best = offer

    return best


def improve_offer(
    stakeholder: Stakeholder,
    current_offer: Offer,
    target_stakeholder: Stakeholder,
    concession_rate: float = 0.1,
) -> dict[str, float]:
    """Suggest improvements to an offer to make it more appealing.

    Args:
        stakeholder: The stakeholder making the offer.
        current_offer: The current offer.
        target_stakeholder: The stakeholder to appeal to.
        concession_rate: How much to concede (0-1).

    Returns:
        Suggested new terms.
    """
    new_terms = current_offer.terms.copy()

    for key, target in target_stakeholder.objectives.items():
        if key not in new_terms:
            continue

        current_val = new_terms[key]

        # Move toward target's preference
        if target == 0.0:  # Target wants to minimize
            new_terms[key] = current_val * (1 - concession_rate)
        elif target == float("inf"):  # Target wants to maximize
            new_terms[key] = current_val * (1 + concession_rate)
        else:
            # Move toward target value
            diff = target - current_val
            new_terms[key] = current_val + (diff * concession_rate)

    return new_terms
