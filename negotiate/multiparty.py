"""Multi-party negotiation support with coalitions."""

from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4
from itertools import combinations

from .types import (
    Coalition,
    NegotiationState,
    Offer,
    Stakeholder,
)
from .utility import evaluate_terms, UtilityFunction


def form_coalition(
    stakeholders: list[Stakeholder],
    shared_objectives: Optional[dict[str, float]] = None,
    name: Optional[str] = None,
) -> Coalition:
    """Create a coalition of stakeholders.

    Args:
        stakeholders: Stakeholders forming the coalition.
        shared_objectives: Agreed-upon objectives.
        name: Optional coalition name.

    Returns:
        The new coalition.
    """
    member_ids = [s.id for s in stakeholders]

    # Merge objectives if not provided
    if shared_objectives is None:
        shared_objectives = {}
        for s in stakeholders:
            for key, value in s.objectives.items():
                if key in shared_objectives:
                    # Average the objectives
                    shared_objectives[key] = (shared_objectives[key] + value) / 2
                else:
                    shared_objectives[key] = value

    # Default equal split
    equal_share = 1.0 / len(stakeholders) if stakeholders else 0
    internal_split = {s.id: equal_share for s in stakeholders}

    return Coalition(
        id=name or str(uuid4())[:8],
        members=member_ids,
        shared_objectives=shared_objectives,
        internal_split=internal_split,
        stability=1.0,
    )


def coalition_utility(
    coalition: Coalition,
    terms: dict[str, float],
    stakeholders: dict[str, Stakeholder],
    utility_registry: Optional[dict[str, UtilityFunction]] = None,
) -> dict[str, float]:
    """Calculate utility for each coalition member.

    Args:
        coalition: The coalition.
        terms: The proposed terms.
        stakeholders: All stakeholders.
        utility_registry: Optional utility function registry.

    Returns:
        Utility for each member after internal split.
    """
    # Calculate total coalition utility
    total_utility = 0.0
    for sid in coalition.members:
        if sid in stakeholders:
            member_utility = evaluate_terms(
                stakeholders[sid], terms, utility_registry
            )
            total_utility += member_utility

    # Apply internal split
    member_utilities = {}
    for sid in coalition.members:
        share = coalition.internal_split.get(sid, 0)
        member_utilities[sid] = total_utility * share

    return member_utilities


def defection_analysis(
    coalition: Coalition,
    stakeholders: dict[str, Stakeholder],
    current_terms: dict[str, float],
    alternative_offers: Optional[list[Offer]] = None,
    utility_registry: Optional[dict[str, UtilityFunction]] = None,
) -> dict[str, Any]:
    """Analyze stability of coalition against defection.

    Args:
        coalition: The coalition to analyze.
        stakeholders: All stakeholders.
        current_terms: Current negotiated terms.
        alternative_offers: Offers available if defecting.
        utility_registry: Optional utility function registry.

    Returns:
        Defection analysis for each member.
    """
    # Calculate current coalition utilities
    coalition_utils = coalition_utility(
        coalition, current_terms, stakeholders, utility_registry
    )

    defection_risks = {}

    for sid in coalition.members:
        if sid not in stakeholders:
            continue

        stakeholder = stakeholders[sid]
        current_util = coalition_utils.get(sid, 0)

        # Evaluate defection options
        best_defection_util = 0.0
        best_defection_option = None

        if alternative_offers:
            for offer in alternative_offers:
                defect_util = evaluate_terms(stakeholder, offer.terms, utility_registry)
                if defect_util > best_defection_util:
                    best_defection_util = defect_util
                    best_defection_option = offer

        # Individual negotiation value (going alone)
        solo_util = evaluate_terms(stakeholder, current_terms, utility_registry)

        defection_risks[sid] = {
            "coalition_utility": current_util,
            "best_defection_utility": best_defection_util,
            "solo_utility": solo_util,
            "defection_incentive": max(best_defection_util, solo_util) - current_util,
            "likely_to_defect": max(best_defection_util, solo_util) > current_util * 1.1,
            "best_alternative": (
                best_defection_option.id if best_defection_option else "solo"
            ),
        }

    # Update coalition stability
    defecting_count = sum(
        1 for risk in defection_risks.values() if risk["likely_to_defect"]
    )
    coalition.stability = 1.0 - (defecting_count / len(coalition.members))

    return {
        "member_analysis": defection_risks,
        "coalition_stability": coalition.stability,
        "stable": coalition.stability > 0.5,
        "at_risk_members": [
            sid for sid, risk in defection_risks.items() if risk["likely_to_defect"]
        ],
    }


def find_stable_coalitions(
    stakeholders: list[Stakeholder],
    terms: dict[str, float],
    min_size: int = 2,
    utility_registry: Optional[dict[str, UtilityFunction]] = None,
) -> list[Coalition]:
    """Find all stable coalition configurations.

    Args:
        stakeholders: List of all stakeholders.
        terms: Current proposed terms.
        min_size: Minimum coalition size.
        utility_registry: Optional utility function registry.

    Returns:
        List of stable coalitions.
    """
    stable = []
    stakeholder_dict = {s.id: s for s in stakeholders}

    # Try all possible coalition combinations
    for size in range(min_size, len(stakeholders) + 1):
        for combo in combinations(stakeholders, size):
            coalition = form_coalition(list(combo))

            # Analyze stability
            analysis = defection_analysis(
                coalition, stakeholder_dict, terms, utility_registry=utility_registry
            )

            if analysis["stable"]:
                stable.append(coalition)

    return stable


def shapley_value(
    stakeholders: list[Stakeholder],
    coalition_value_func: callable,
) -> dict[str, float]:
    """Calculate Shapley value for each stakeholder.

    The Shapley value is a fair allocation of coalition gains
    based on marginal contributions.

    Args:
        stakeholders: List of stakeholders.
        coalition_value_func: Function that returns value of a coalition.

    Returns:
        Shapley value for each stakeholder.
    """
    import math

    n = len(stakeholders)
    shapley = {s.id: 0.0 for s in stakeholders}

    # For each stakeholder, calculate their Shapley value
    for i, stakeholder in enumerate(stakeholders):
        # For all possible coalitions not including stakeholder
        others = [s for j, s in enumerate(stakeholders) if j != i]

        for size in range(len(others) + 1):
            for combo in combinations(others, size):
                # Value with stakeholder
                with_i = list(combo) + [stakeholder]
                v_with = coalition_value_func(with_i)

                # Value without stakeholder
                v_without = coalition_value_func(list(combo)) if combo else 0

                # Marginal contribution
                marginal = v_with - v_without

                # Shapley weight
                weight = (
                    math.factorial(size)
                    * math.factorial(n - size - 1)
                    / math.factorial(n)
                )

                shapley[stakeholder.id] += weight * marginal

    return shapley


def core_membership(
    coalition: Coalition,
    stakeholders: dict[str, Stakeholder],
    allocation: dict[str, float],
    coalition_value_func: callable,
) -> dict[str, Any]:
    """Check if an allocation is in the core.

    The core is the set of allocations where no sub-coalition
    can do better on its own.

    Args:
        coalition: The grand coalition.
        stakeholders: All stakeholders.
        allocation: Proposed allocation (stakeholder_id -> value).
        coalition_value_func: Function to compute coalition value.

    Returns:
        Core membership analysis.
    """
    members = [stakeholders[sid] for sid in coalition.members if sid in stakeholders]
    n = len(members)

    blocking_coalitions = []

    # Check all possible sub-coalitions
    for size in range(1, n):
        for combo in combinations(members, size):
            # Value this sub-coalition can achieve alone
            sub_value = coalition_value_func(list(combo))

            # What they get in proposed allocation
            allocation_value = sum(allocation.get(s.id, 0) for s in combo)

            # If they can do better alone, this blocks the allocation
            if sub_value > allocation_value:
                blocking_coalitions.append({
                    "members": [s.id for s in combo],
                    "standalone_value": sub_value,
                    "allocation_value": allocation_value,
                    "improvement": sub_value - allocation_value,
                })

    return {
        "is_in_core": len(blocking_coalitions) == 0,
        "blocking_coalitions": blocking_coalitions,
        "num_blocking": len(blocking_coalitions),
    }


def negotiate_internal_split(
    coalition: Coalition,
    stakeholders: dict[str, Stakeholder],
    method: str = "shapley",
    coalition_value_func: Optional[callable] = None,
) -> dict[str, float]:
    """Negotiate how to split coalition gains.

    Args:
        coalition: The coalition.
        stakeholders: All stakeholders.
        method: Split method ("equal", "shapley", "proportional").
        coalition_value_func: Function for coalition value (needed for shapley).

    Returns:
        Proposed internal split.
    """
    members = [stakeholders[sid] for sid in coalition.members if sid in stakeholders]

    if method == "equal":
        equal_share = 1.0 / len(members)
        return {s.id: equal_share for s in members}

    elif method == "shapley" and coalition_value_func:
        shapley = shapley_value(members, coalition_value_func)
        total = sum(shapley.values())
        if total > 0:
            return {sid: val / total for sid, val in shapley.items()}
        return {s.id: 1.0 / len(members) for s in members}

    elif method == "proportional":
        # Proportional to individual standalone value
        standalone = {}
        for s in members:
            standalone[s.id] = sum(s.objectives.values())

        total = sum(standalone.values())
        if total > 0:
            return {sid: val / total for sid, val in standalone.items()}
        return {s.id: 1.0 / len(members) for s in members}

    # Default to equal
    return {s.id: 1.0 / len(members) for s in members}


def multiparty_mediation(
    stakeholders: list[Stakeholder],
    state: NegotiationState,
    utility_registry: Optional[dict[str, UtilityFunction]] = None,
) -> dict[str, Any]:
    """Generate mediation suggestions for multi-party negotiation.

    Args:
        stakeholders: All stakeholders.
        state: Current negotiation state.
        utility_registry: Optional utility function registry.

    Returns:
        Mediation suggestions.
    """
    suggestions = []

    # Find potential coalitions
    stable_coalitions = find_stable_coalitions(
        stakeholders,
        {},  # No current terms
        min_size=2,
        utility_registry=utility_registry,
    )

    if stable_coalitions:
        suggestions.append({
            "type": "coalition_formation",
            "description": "Consider forming coalitions for stronger bargaining",
            "potential_coalitions": [
                {"members": c.members, "stability": c.stability}
                for c in stable_coalitions[:3]
            ],
        })

    # Identify blocking parties
    from .zopa import no_zopa_detection

    zopa_analysis = no_zopa_detection(stakeholders, state=state)

    if zopa_analysis["impossible"]:
        for dim, analysis in zopa_analysis["analysis"].items():
            suggestions.append({
                "type": "bridge_gap",
                "dimension": dim,
                "gap": analysis["gap"],
                "blocking_parties": analysis["blocking_parties"],
                "suggestion": f"Focus on bridging {dim} gap of {analysis['gap']:.2f}",
            })

    # Suggest side payments or trades
    stakeholder_dict = {s.id: s for s in stakeholders}
    for s in stakeholders:
        other_dims = set()
        for other in stakeholders:
            if other.id != s.id:
                other_dims.update(other.objectives.keys())

        own_dims = set(s.objectives.keys())
        tradeable = other_dims - own_dims

        if tradeable:
            suggestions.append({
                "type": "side_trade",
                "stakeholder": s.id,
                "tradeable_dimensions": list(tradeable),
                "suggestion": f"{s.name} could offer concessions on {tradeable}",
            })

    return {
        "suggestions": suggestions,
        "num_stakeholders": len(stakeholders),
        "stable_coalitions_found": len(stable_coalitions),
    }
