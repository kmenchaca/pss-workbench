"""BATNA (Best Alternative To Negotiated Agreement) discovery and management."""

from dataclasses import dataclass, field
from typing import Any, Optional

from .types import BATNA, NegotiationState, Stakeholder
from .utility import evaluate_terms, UtilityFunction


def discover_batna(
    stakeholder: Stakeholder,
    alternatives: Optional[list[dict[str, Any]]] = None,
    utility_registry: Optional[dict[str, UtilityFunction]] = None,
) -> BATNA:
    """Discover the best alternative for a stakeholder.

    Args:
        stakeholder: The stakeholder to find BATNA for.
        alternatives: List of alternative options to consider.
        utility_registry: Optional registry of utility functions.

    Returns:
        The best BATNA found.
    """
    if not alternatives:
        # Generate default alternatives based on stakeholder type
        alternatives = _generate_default_alternatives(stakeholder)

    best_batna = BATNA(
        stakeholder_id=stakeholder.id,
        alternative={},
        utility=float("-inf"),
    )

    for alt in alternatives:
        # Convert alternative to terms format if needed
        terms = alt.get("terms", alt)
        if not isinstance(terms, dict):
            continue

        utility = evaluate_terms(stakeholder, terms, utility_registry)

        if utility > best_batna.utility:
            best_batna = BATNA(
                stakeholder_id=stakeholder.id,
                alternative=alt.copy(),
                utility=utility,
            )

    return best_batna


def _generate_default_alternatives(stakeholder: Stakeholder) -> list[dict[str, Any]]:
    """Generate default alternatives based on stakeholder role.

    Args:
        stakeholder: The stakeholder.

    Returns:
        List of default alternatives.
    """
    alternatives = []

    # Status quo alternative (do nothing)
    alternatives.append({
        "name": "status_quo",
        "terms": {k: 0.0 for k in stakeholder.objectives.keys()},
        "description": "Walk away, maintain current position",
    })

    # Alternative based on constraints
    if stakeholder.constraints:
        # Worst acceptable deal
        worst_acceptable = {}
        for key, (min_val, max_val) in stakeholder.constraints.items():
            target = stakeholder.objectives.get(key, (min_val + max_val) / 2)
            if target == 0.0:
                worst_acceptable[key] = max_val  # Highest we'd pay
            else:
                worst_acceptable[key] = min_val  # Lowest we'd accept

        alternatives.append({
            "name": "worst_acceptable",
            "terms": worst_acceptable,
            "description": "Minimum acceptable terms",
        })

    # Role-specific alternatives
    role = stakeholder.role
    if role == "BUYER":
        alternatives.append({
            "name": "alternative_supplier",
            "terms": {"price": stakeholder.objectives.get("price", 0) * 1.1},
            "description": "Buy from alternative supplier at higher price",
        })
    elif role == "SELLER":
        alternatives.append({
            "name": "alternative_buyer",
            "terms": {"price": stakeholder.objectives.get("price", 100) * 0.9},
            "description": "Sell to alternative buyer at lower price",
        })

    return alternatives


def evaluate_batna(
    stakeholder: Stakeholder,
    batna: BATNA,
    utility_registry: Optional[dict[str, UtilityFunction]] = None,
) -> dict[str, Any]:
    """Evaluate a BATNA's quality and implications.

    Args:
        stakeholder: The stakeholder whose BATNA this is.
        batna: The BATNA to evaluate.
        utility_registry: Optional registry of utility functions.

    Returns:
        Evaluation results including strength assessment.
    """
    # Recalculate utility to verify
    if "terms" in batna.alternative:
        calculated_utility = evaluate_terms(
            stakeholder, batna.alternative["terms"], utility_registry
        )
    else:
        calculated_utility = batna.utility

    # Assess BATNA strength
    strength = "weak"
    if calculated_utility > 50:
        strength = "moderate"
    if calculated_utility > 80:
        strength = "strong"

    # Calculate walk-away point
    walk_away_point = {}
    for key, target in stakeholder.objectives.items():
        if target == 0.0:  # Minimize
            # Max we'd pay before walking away
            walk_away_point[key] = batna.utility * 1.1
        elif target == float("inf"):  # Maximize
            # Min we'd accept before walking away
            walk_away_point[key] = batna.utility * 0.9
        else:
            walk_away_point[key] = batna.utility

    return {
        "batna": batna,
        "calculated_utility": calculated_utility,
        "strength": strength,
        "walk_away_point": walk_away_point,
        "should_negotiate": calculated_utility < 100,  # If BATNA is too good, why negotiate?
    }


def improve_batna(
    stakeholder: Stakeholder,
    current_batna: BATNA,
    improvement_options: Optional[list[dict[str, Any]]] = None,
) -> BATNA:
    """Attempt to improve stakeholder's BATNA.

    Args:
        stakeholder: The stakeholder.
        current_batna: Current BATNA.
        improvement_options: Possible improvements to consider.

    Returns:
        Improved BATNA (or current if no improvement found).
    """
    if not improvement_options:
        # Generate improvement options
        improvement_options = _generate_improvement_options(stakeholder, current_batna)

    best = current_batna

    for option in improvement_options:
        terms = option.get("terms", {})
        utility = evaluate_terms(stakeholder, terms)

        # Account for improvement cost if specified
        cost = option.get("cost", 0)
        net_utility = utility - cost

        if net_utility > best.utility:
            best = BATNA(
                stakeholder_id=stakeholder.id,
                alternative=option.copy(),
                utility=net_utility,
            )

    return best


def _generate_improvement_options(
    stakeholder: Stakeholder,
    current_batna: BATNA,
) -> list[dict[str, Any]]:
    """Generate options to improve BATNA.

    Args:
        stakeholder: The stakeholder.
        current_batna: Current BATNA.

    Returns:
        List of potential improvements.
    """
    options = []

    # Search for more alternatives
    options.append({
        "name": "market_search",
        "terms": {k: v * 1.1 for k, v in current_batna.alternative.get("terms", {}).items()},
        "cost": 5,  # Time/effort cost
        "description": "Search market for better alternatives",
    })

    # Invest in alternatives
    options.append({
        "name": "develop_alternative",
        "terms": {k: v * 1.2 for k, v in current_batna.alternative.get("terms", {}).items()},
        "cost": 20,
        "description": "Develop/invest in better alternatives",
    })

    # Form coalition for better BATNA
    options.append({
        "name": "coalition_batna",
        "terms": {k: v * 1.15 for k, v in current_batna.alternative.get("terms", {}).items()},
        "cost": 10,
        "description": "Form coalition for stronger position",
    })

    return options


def batna_bluff(
    stakeholder: Stakeholder,
    real_batna: BATNA,
    bluff_factor: float = 1.3,
) -> BATNA:
    """Create a bluffed (inflated) BATNA for negotiation leverage.

    Args:
        stakeholder: The stakeholder.
        real_batna: The actual BATNA.
        bluff_factor: How much to inflate (1.3 = 30% better).

    Returns:
        A bluffed BATNA with higher apparent utility.
    """
    bluffed_terms = {}
    for key, value in real_batna.alternative.get("terms", {}).items():
        bluffed_terms[key] = value * bluff_factor

    return BATNA(
        stakeholder_id=stakeholder.id,
        alternative={
            **real_batna.alternative,
            "terms": bluffed_terms,
            "bluffed": True,
        },
        utility=real_batna.utility * bluff_factor,
        is_bluff=True,
        actual_utility=real_batna.utility,
    )


def detect_batna_bluff(
    claimed_batna: BATNA,
    market_data: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Attempt to detect if a claimed BATNA is a bluff.

    Args:
        claimed_batna: The BATNA being claimed.
        market_data: Optional market data for verification.

    Returns:
        Analysis of whether BATNA might be a bluff.
    """
    indicators = []
    confidence = 0.5  # Start neutral

    # Check if marked as bluff (honest marking)
    if claimed_batna.is_bluff:
        return {
            "is_bluff": True,
            "confidence": 1.0,
            "actual_utility": claimed_batna.actual_utility,
            "indicators": ["Explicitly marked as bluff"],
        }

    # Check against market data
    if market_data:
        market_range = market_data.get("typical_range", (0, 100))
        if claimed_batna.utility > market_range[1] * 1.5:
            indicators.append("Utility significantly above market norm")
            confidence += 0.2

    # Check for unrealistic terms
    terms = claimed_batna.alternative.get("terms", {})
    for key, value in terms.items():
        if value < 0 or value > 10000:
            indicators.append(f"Unrealistic value for {key}: {value}")
            confidence += 0.1

    # Check timing (sudden BATNA improvement is suspicious)
    if "improved_recently" in claimed_batna.alternative:
        indicators.append("BATNA improved suspiciously recently")
        confidence += 0.15

    return {
        "is_bluff": confidence > 0.7,
        "confidence": min(confidence, 1.0),
        "indicators": indicators,
        "claimed_utility": claimed_batna.utility,
    }


def calculate_walk_away_point(
    stakeholder: Stakeholder,
    batna: BATNA,
    risk_tolerance: float = 0.5,
) -> dict[str, float]:
    """Calculate the point at which stakeholder should walk away.

    Args:
        stakeholder: The stakeholder.
        batna: Their BATNA.
        risk_tolerance: How risk-tolerant (0 = averse, 1 = seeking).

    Returns:
        Walk-away thresholds for each dimension.
    """
    walk_away = {}

    for key, target in stakeholder.objectives.items():
        batna_terms = batna.alternative.get("terms", {})
        batna_value = batna_terms.get(key, 0)

        if target == 0.0:  # Wants to minimize
            # Walk away if value exceeds BATNA (adjusted for risk)
            threshold = batna_value * (1 + 0.1 * (1 - risk_tolerance))
            walk_away[key] = threshold
        elif target == float("inf"):  # Wants to maximize
            # Walk away if value falls below BATNA
            threshold = batna_value * (1 - 0.1 * (1 - risk_tolerance))
            walk_away[key] = threshold
        else:
            # Walk away based on distance from target
            threshold = batna_value * (1 - 0.05 * risk_tolerance)
            walk_away[key] = threshold

    return walk_away


def compare_to_batna(
    stakeholder: Stakeholder,
    offer_terms: dict[str, float],
    batna: BATNA,
    utility_registry: Optional[dict[str, UtilityFunction]] = None,
) -> dict[str, Any]:
    """Compare an offer to the stakeholder's BATNA.

    Args:
        stakeholder: The stakeholder.
        offer_terms: Terms of the offer.
        batna: The BATNA to compare against.
        utility_registry: Optional registry of utility functions.

    Returns:
        Comparison results.
    """
    offer_utility = evaluate_terms(stakeholder, offer_terms, utility_registry)

    # Calculate actual BATNA utility if it's a bluff
    actual_batna_utility = batna.actual_utility if batna.is_bluff else batna.utility

    difference = offer_utility - actual_batna_utility
    ratio = offer_utility / actual_batna_utility if actual_batna_utility != 0 else float("inf")

    return {
        "offer_utility": offer_utility,
        "batna_utility": actual_batna_utility,
        "claimed_batna_utility": batna.utility,
        "difference": difference,
        "ratio": ratio,
        "should_accept": offer_utility >= actual_batna_utility,
        "advantage": "offer" if difference > 0 else "batna" if difference < 0 else "equal",
    }
