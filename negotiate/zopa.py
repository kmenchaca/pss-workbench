"""ZOPA (Zone of Possible Agreement) analysis."""

from dataclasses import dataclass, field
from typing import Any, Optional

from .types import ZOPA, NegotiationState, Stakeholder
from .batna import calculate_walk_away_point, discover_batna


def find_zopa(
    stakeholders: list[Stakeholder],
    dimensions: Optional[list[str]] = None,
    state: Optional[NegotiationState] = None,
) -> dict[str, ZOPA]:
    """Find the Zone of Possible Agreement for all dimensions.

    Args:
        stakeholders: List of stakeholders in negotiation.
        dimensions: Specific dimensions to analyze (None = all).
        state: Optional negotiation state with BATNAs.

    Returns:
        Dictionary of dimension -> ZOPA.
    """
    if not dimensions:
        # Collect all dimensions from stakeholder objectives
        dimensions = set()
        for s in stakeholders:
            dimensions.update(s.objectives.keys())
        dimensions = list(dimensions)

    zopas = {}

    for dim in dimensions:
        zopa = _find_zopa_for_dimension(stakeholders, dim, state)
        zopas[dim] = zopa

    return zopas


def _find_zopa_for_dimension(
    stakeholders: list[Stakeholder],
    dimension: str,
    state: Optional[NegotiationState] = None,
) -> ZOPA:
    """Find ZOPA for a single dimension.

    Args:
        stakeholders: List of stakeholders.
        dimension: The dimension to analyze.
        state: Optional negotiation state.

    Returns:
        ZOPA for this dimension.
    """
    min_acceptable = {}
    max_acceptable = {}

    for s in stakeholders:
        # Get walk-away points based on BATNA
        if state and s.id in state.batnas:
            batna = state.batnas[s.id]
        else:
            batna = discover_batna(s)

        walk_away = calculate_walk_away_point(s, batna)

        # Determine acceptable range
        target = s.objectives.get(dimension, 0)
        constraint = s.constraints.get(dimension, (float("-inf"), float("inf")))

        if target == 0.0:  # Wants to minimize (e.g., buyer on price)
            # Min acceptable is 0 (or constraint min)
            # Max acceptable is walk-away point or constraint max
            min_acceptable[s.id] = max(0, constraint[0])
            max_acceptable[s.id] = min(
                walk_away.get(dimension, constraint[1]), constraint[1]
            )
        elif target == float("inf"):  # Wants to maximize (e.g., seller on price)
            # Min acceptable is walk-away point or constraint min
            # Max acceptable is infinity (or constraint max)
            min_acceptable[s.id] = max(
                walk_away.get(dimension, constraint[0]), constraint[0]
            )
            max_acceptable[s.id] = constraint[1]
        else:
            # Has specific target
            min_acceptable[s.id] = max(constraint[0], target * 0.8)
            max_acceptable[s.id] = min(constraint[1], target * 1.2)

    # Find overlap
    overlap, overlap_range = _calculate_overlap(min_acceptable, max_acceptable)

    return ZOPA(
        dimension=dimension,
        min_acceptable=min_acceptable,
        max_acceptable=max_acceptable,
        overlap=overlap,
        overlap_range=overlap_range,
    )


def _calculate_overlap(
    min_acceptable: dict[str, float],
    max_acceptable: dict[str, float],
) -> tuple[bool, Optional[tuple[float, float]]]:
    """Calculate if there's an overlapping zone.

    Args:
        min_acceptable: Minimum acceptable values per stakeholder.
        max_acceptable: Maximum acceptable values per stakeholder.

    Returns:
        Tuple of (has_overlap, overlap_range).
    """
    if not min_acceptable or not max_acceptable:
        return (False, None)

    # Find the highest minimum and lowest maximum
    highest_min = max(min_acceptable.values())
    lowest_max = min(max_acceptable.values())

    if highest_min <= lowest_max:
        return (True, (highest_min, lowest_max))
    return (False, None)


def no_zopa_detection(
    stakeholders: list[Stakeholder],
    dimensions: Optional[list[str]] = None,
    state: Optional[NegotiationState] = None,
) -> dict[str, Any]:
    """Detect dimensions where no agreement is possible.

    Args:
        stakeholders: List of stakeholders.
        dimensions: Dimensions to check.
        state: Optional negotiation state.

    Returns:
        Analysis of impossible negotiations.
    """
    zopas = find_zopa(stakeholders, dimensions, state)

    no_zopa_dims = []
    analysis = {}

    for dim, zopa in zopas.items():
        if not zopa.overlap:
            no_zopa_dims.append(dim)

            # Analyze the gap
            highest_min = max(zopa.min_acceptable.values()) if zopa.min_acceptable else 0
            lowest_max = min(zopa.max_acceptable.values()) if zopa.max_acceptable else 0
            gap = highest_min - lowest_max

            # Identify blocking parties
            blocking_parties = []
            for sid, min_val in zopa.min_acceptable.items():
                if min_val == highest_min and gap > 0:
                    blocking_parties.append(sid)

            analysis[dim] = {
                "gap": gap,
                "blocking_parties": blocking_parties,
                "highest_min": highest_min,
                "lowest_max": lowest_max,
                "can_be_bridged": gap < 10,  # Arbitrary threshold
            }

    return {
        "impossible": len(no_zopa_dims) > 0,
        "no_zopa_dimensions": no_zopa_dims,
        "analysis": analysis,
        "possible_dimensions": [d for d in zopas if zopas[d].overlap],
    }


def expand_zopa(
    stakeholders: list[Stakeholder],
    dimension: str,
    state: Optional[NegotiationState] = None,
    expansion_strategies: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Attempt to expand the ZOPA through value creation.

    Args:
        stakeholders: List of stakeholders.
        dimension: Dimension to expand.
        state: Optional negotiation state.
        expansion_strategies: Strategies to try.

    Returns:
        Expansion results and recommendations.
    """
    current_zopa = _find_zopa_for_dimension(stakeholders, dimension, state)

    if expansion_strategies is None:
        expansion_strategies = [
            "add_issues",
            "improve_batnas",
            "contingent_agreements",
            "cost_cutting",
            "bridging",
        ]

    recommendations = []
    expanded_ranges = {}

    for strategy in expansion_strategies:
        if strategy == "add_issues":
            # Adding issues can create trade-offs
            recommendations.append({
                "strategy": "add_issues",
                "description": "Add additional issues for trade-offs",
                "potential_expansion": 0.1,  # 10% expansion estimate
                "example": f"Consider adding delivery timing, payment terms to trade against {dimension}",
            })

        elif strategy == "improve_batnas":
            # If BATNAs are bluffs or weak, they could be corrected
            recommendations.append({
                "strategy": "improve_batnas",
                "description": "Help parties realize true BATNAs",
                "potential_expansion": 0.15,
            })

        elif strategy == "contingent_agreements":
            # Make agreements contingent on uncertain events
            recommendations.append({
                "strategy": "contingent_agreements",
                "description": "Make terms contingent on future events",
                "potential_expansion": 0.2,
                "example": f"Agree on {dimension} with adjustments based on market conditions",
            })

        elif strategy == "cost_cutting":
            # Find ways to reduce costs for parties
            recommendations.append({
                "strategy": "cost_cutting",
                "description": "Find ways to reduce costs for all parties",
                "potential_expansion": 0.25,
            })

        elif strategy == "bridging":
            # Create new options that meet underlying interests
            recommendations.append({
                "strategy": "bridging",
                "description": "Create new options meeting underlying interests",
                "potential_expansion": 0.3,
            })

    # Calculate potential expanded ZOPA
    if current_zopa.overlap_range:
        low, high = current_zopa.overlap_range
        # Estimate expansion
        total_expansion = sum(r.get("potential_expansion", 0) for r in recommendations)
        expanded_range = (low * (1 - total_expansion * 0.3), high * (1 + total_expansion * 0.3))
    else:
        # No current overlap - estimate what's needed
        highest_min = max(current_zopa.min_acceptable.values())
        lowest_max = min(current_zopa.max_acceptable.values())
        gap = highest_min - lowest_max
        expanded_range = None

        # Check if expansion could close the gap
        if gap < sum(r.get("potential_expansion", 0) for r in recommendations) * highest_min:
            midpoint = (highest_min + lowest_max) / 2
            expanded_range = (midpoint - gap * 0.1, midpoint + gap * 0.1)

    return {
        "current_zopa": current_zopa,
        "has_overlap": current_zopa.overlap,
        "recommendations": recommendations,
        "expanded_range_estimate": expanded_range,
        "strategies_tried": expansion_strategies,
    }


def visualize_zopa(
    stakeholders: list[Stakeholder],
    dimension: str,
    state: Optional[NegotiationState] = None,
    width: int = 60,
) -> str:
    """Create ASCII visualization of ZOPA.

    Args:
        stakeholders: List of stakeholders.
        dimension: Dimension to visualize.
        state: Optional negotiation state.
        width: Width of visualization in characters.

    Returns:
        ASCII representation of ZOPA.
    """
    zopa = _find_zopa_for_dimension(stakeholders, dimension, state)

    # Get all values to determine scale
    all_values = []
    all_values.extend(zopa.min_acceptable.values())
    all_values.extend(zopa.max_acceptable.values())

    if not all_values:
        return f"No data for dimension: {dimension}"

    min_val = min(all_values) * 0.9
    max_val = max(all_values) * 1.1
    scale = max_val - min_val

    if scale == 0:
        scale = 1

    lines = []
    lines.append(f"ZOPA Analysis: {dimension}")
    lines.append("=" * width)

    # Scale line
    scale_line = f"{min_val:.1f}" + " " * (width - 20) + f"{max_val:.1f}"
    lines.append(scale_line)
    lines.append("-" * width)

    # Draw each stakeholder's range
    for sid in zopa.min_acceptable.keys():
        min_pos = int((zopa.min_acceptable[sid] - min_val) / scale * (width - 2))
        max_pos = int((zopa.max_acceptable[sid] - min_val) / scale * (width - 2))

        min_pos = max(0, min(width - 2, min_pos))
        max_pos = max(0, min(width - 2, max_pos))

        # Create range visualization
        line = [" "] * width
        for i in range(min_pos, max_pos + 1):
            line[i] = "-"
        line[min_pos] = "["
        if max_pos < width:
            line[max_pos] = "]"

        lines.append(f"{sid[:8]:8} |{''.join(line)}|")

    # Draw overlap zone
    lines.append("-" * width)
    if zopa.overlap and zopa.overlap_range:
        overlap_min, overlap_max = zopa.overlap_range
        min_pos = int((overlap_min - min_val) / scale * (width - 2))
        max_pos = int((overlap_max - min_val) / scale * (width - 2))

        min_pos = max(0, min(width - 2, min_pos))
        max_pos = max(0, min(width - 2, max_pos))

        line = [" "] * width
        for i in range(min_pos, max_pos + 1):
            line[i] = "#"

        lines.append(f"{'ZOPA':8} |{''.join(line)}|")
        lines.append(f"Overlap: [{overlap_min:.2f}, {overlap_max:.2f}]")
    else:
        lines.append("NO OVERLAP - No ZOPA exists")

    return "\n".join(lines)


def find_optimal_point(
    stakeholders: list[Stakeholder],
    dimension: str,
    state: Optional[NegotiationState] = None,
    fairness_criterion: str = "equal_surplus",
) -> Optional[dict[str, Any]]:
    """Find optimal agreement point within ZOPA.

    Args:
        stakeholders: List of stakeholders.
        dimension: Dimension to optimize.
        state: Optional negotiation state.
        fairness_criterion: How to determine "optimal" (equal_surplus, nash, kalai).

    Returns:
        Optimal point details or None if no ZOPA.
    """
    zopa = _find_zopa_for_dimension(stakeholders, dimension, state)

    if not zopa.overlap or not zopa.overlap_range:
        return None

    low, high = zopa.overlap_range

    if fairness_criterion == "equal_surplus":
        # Split the surplus equally
        optimal = (low + high) / 2

    elif fairness_criterion == "nash":
        # Nash bargaining solution (product of surpluses)
        # Simplified: geometric mean of range
        optimal = (low * high) ** 0.5

    elif fairness_criterion == "kalai":
        # Kalai-Smorodinsky: proportional gains
        # Map to proportional position in ZOPA
        optimal = (low + high) / 2  # Simplified

    else:
        optimal = (low + high) / 2

    # Calculate surplus for each stakeholder at optimal point
    surpluses = {}
    for sid in stakeholders:
        s = next((x for x in stakeholders if isinstance(x, Stakeholder) and x.id == sid.id), sid)
        batna_value = 0
        if state and s.id in state.batnas:
            batna_terms = state.batnas[s.id].alternative.get("terms", {})
            batna_value = batna_terms.get(dimension, 0)

        target = s.objectives.get(dimension, 0)
        if target == 0.0:  # Minimizer
            surpluses[s.id] = batna_value - optimal
        else:  # Maximizer
            surpluses[s.id] = optimal - batna_value

    return {
        "optimal_point": optimal,
        "zopa_range": (low, high),
        "fairness_criterion": fairness_criterion,
        "surpluses": surpluses,
        "total_surplus": sum(surpluses.values()),
    }
