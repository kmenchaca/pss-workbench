"""Stakeholder definition and management for negotiations."""

from typing import Any, Optional
from uuid import uuid4

from .types import Stakeholder


# Built-in stakeholder templates
BUYER = {
    "role": "BUYER",
    "default_objectives": {"price": 0.0},  # Minimize price
    "default_constraints": {"price": (0.0, float("inf"))},
    "typical_private_info": ["max_budget", "urgency", "alternatives"],
}

SELLER = {
    "role": "SELLER",
    "default_objectives": {"price": float("inf")},  # Maximize price
    "default_constraints": {"price": (0.0, float("inf"))},
    "typical_private_info": ["min_acceptable", "cost_basis", "other_buyers"],
}

REGULATOR = {
    "role": "REGULATOR",
    "default_objectives": {"compliance": 1.0, "fairness": 1.0},
    "default_constraints": {"compliance": (0.8, 1.0)},
    "typical_private_info": ["enforcement_priority", "flexibility"],
}

MEDIATOR = {
    "role": "MEDIATOR",
    "default_objectives": {"agreement_reached": 1.0, "fairness": 1.0},
    "default_constraints": {},
    "typical_private_info": ["known_batnas", "suggested_zopa"],
}

TEMPLATES = {
    "BUYER": BUYER,
    "SELLER": SELLER,
    "REGULATOR": REGULATOR,
    "MEDIATOR": MEDIATOR,
}


def define_stakeholder(
    name: str,
    objectives: dict[str, float],
    constraints: Optional[dict[str, tuple[float, float]]] = None,
    private_info: Optional[dict[str, Any]] = None,
    utility_function: Optional[str] = None,
    role: Optional[str] = None,
    stakeholder_id: Optional[str] = None,
) -> Stakeholder:
    """Create a new stakeholder with given properties.

    Args:
        name: Human-readable name for the stakeholder.
        objectives: What the stakeholder wants (key -> target value).
        constraints: Hard limits (key -> (min, max)).
        private_info: Information hidden from other parties.
        utility_function: Name of utility function to use.
        role: Optional role template to apply.
        stakeholder_id: Optional custom ID (auto-generated if not provided).

    Returns:
        A new Stakeholder instance.
    """
    sid = stakeholder_id or str(uuid4())[:8]
    constraints = constraints or {}
    private_info = private_info or {}

    return Stakeholder(
        id=sid,
        name=name,
        objectives=objectives.copy(),
        constraints=constraints.copy(),
        private_info=private_info.copy(),
        utility_function=utility_function,
        role=role,
    )


def create_from_template(
    name: str,
    template: str,
    objectives_override: Optional[dict[str, float]] = None,
    constraints_override: Optional[dict[str, tuple[float, float]]] = None,
    private_info: Optional[dict[str, Any]] = None,
) -> Stakeholder:
    """Create a stakeholder from a built-in template.

    Args:
        name: Human-readable name for the stakeholder.
        template: Template name (BUYER, SELLER, REGULATOR, MEDIATOR).
        objectives_override: Override default objectives.
        constraints_override: Override default constraints.
        private_info: Additional private information.

    Returns:
        A new Stakeholder instance based on the template.

    Raises:
        ValueError: If template name is not recognized.
    """
    if template not in TEMPLATES:
        raise ValueError(f"Unknown template: {template}. Valid: {list(TEMPLATES.keys())}")

    tmpl = TEMPLATES[template]
    objectives = tmpl["default_objectives"].copy()
    constraints = tmpl["default_constraints"].copy()

    if objectives_override:
        objectives.update(objectives_override)
    if constraints_override:
        constraints.update(constraints_override)

    return define_stakeholder(
        name=name,
        objectives=objectives,
        constraints=constraints,
        private_info=private_info or {},
        role=template,
    )


def reveal_info(
    stakeholder: Stakeholder,
    info_key: str,
    recipient_id: Optional[str] = None,
) -> tuple[str, Any]:
    """Strategically disclose private information.

    Args:
        stakeholder: The stakeholder revealing info.
        info_key: Key of the private info to reveal.
        recipient_id: Optional specific recipient (None = all parties).

    Returns:
        Tuple of (info_key, info_value) that was revealed.

    Raises:
        KeyError: If info_key is not in stakeholder's private_info.
    """
    if info_key not in stakeholder.private_info:
        raise KeyError(f"No private info with key '{info_key}' for {stakeholder.name}")

    value = stakeholder.private_info[info_key]
    # In a real system, this would update state to mark info as revealed
    return (info_key, value)


def hide_info(stakeholder: Stakeholder, key: str, value: Any) -> None:
    """Add new private information to a stakeholder.

    Args:
        stakeholder: The stakeholder to update.
        key: Key for the private info.
        value: Value of the private info.
    """
    stakeholder.private_info[key] = value


def update_objectives(
    stakeholder: Stakeholder,
    new_objectives: dict[str, float],
    merge: bool = True,
) -> None:
    """Update a stakeholder's objectives.

    Args:
        stakeholder: The stakeholder to update.
        new_objectives: New objective values.
        merge: If True, merge with existing; if False, replace entirely.
    """
    if merge:
        stakeholder.objectives.update(new_objectives)
    else:
        stakeholder.objectives = new_objectives.copy()


def update_constraints(
    stakeholder: Stakeholder,
    new_constraints: dict[str, tuple[float, float]],
    merge: bool = True,
) -> None:
    """Update a stakeholder's constraints.

    Args:
        stakeholder: The stakeholder to update.
        new_constraints: New constraint values.
        merge: If True, merge with existing; if False, replace entirely.
    """
    if merge:
        stakeholder.constraints.update(new_constraints)
    else:
        stakeholder.constraints = new_constraints.copy()


def check_constraints(
    stakeholder: Stakeholder,
    terms: dict[str, float],
) -> tuple[bool, list[str]]:
    """Check if terms satisfy stakeholder constraints.

    Args:
        stakeholder: The stakeholder whose constraints to check.
        terms: The proposed terms to validate.

    Returns:
        Tuple of (all_satisfied, list_of_violated_constraints).
    """
    violations = []
    for key, (min_val, max_val) in stakeholder.constraints.items():
        if key in terms:
            value = terms[key]
            if value < min_val or value > max_val:
                violations.append(f"{key}: {value} not in [{min_val}, {max_val}]")

    return (len(violations) == 0, violations)


def get_stakeholder_summary(stakeholder: Stakeholder) -> dict[str, Any]:
    """Get a summary of stakeholder's public information.

    Args:
        stakeholder: The stakeholder to summarize.

    Returns:
        Dictionary with public stakeholder information.
    """
    return {
        "id": stakeholder.id,
        "name": stakeholder.name,
        "role": stakeholder.role,
        "objectives": stakeholder.objectives,
        "constraints": stakeholder.constraints,
        # private_info is intentionally excluded
    }


def clone_stakeholder(
    stakeholder: Stakeholder,
    new_name: Optional[str] = None,
    new_id: Optional[str] = None,
) -> Stakeholder:
    """Create a copy of a stakeholder.

    Args:
        stakeholder: The stakeholder to clone.
        new_name: Optional new name for the clone.
        new_id: Optional new ID for the clone.

    Returns:
        A new Stakeholder instance with copied properties.
    """
    return Stakeholder(
        id=new_id or str(uuid4())[:8],
        name=new_name or f"{stakeholder.name}_copy",
        objectives=stakeholder.objectives.copy(),
        constraints=stakeholder.constraints.copy(),
        private_info=stakeholder.private_info.copy(),
        utility_function=stakeholder.utility_function,
        role=stakeholder.role,
    )
