"""
Observation processing utilities.

Functions for cleaning, structuring, and comparing observations
from embodied environments.
"""

from typing import Any, Optional
import json
import hashlib

from .types import Observation


def process_observation(raw: Any, env_type: str) -> Observation:
    """Process raw environment data into a structured observation.

    Args:
        raw: Raw observation data from environment
        env_type: Type of environment ("mock", "text_game", "browser", "api")

    Returns:
        Processed Observation object
    """
    processors = {
        "mock": _process_mock_observation,
        "text_game": _process_text_game_observation,
        "browser": _process_browser_observation,
        "api": _process_api_observation,
    }

    processor = processors.get(env_type, _process_generic_observation)
    return processor(raw)


def _process_mock_observation(raw: Any) -> Observation:
    """Process mock environment observation."""
    if isinstance(raw, Observation):
        return raw

    if isinstance(raw, dict):
        return Observation(
            state={
                "position": raw.get("position", (0, 0)),
                "goal": raw.get("goal", (0, 0)),
                "inventory": raw.get("inventory", []),
                "nearby": raw.get("nearby", []),
            },
            raw_data=raw,
        )

    return Observation(state={"raw": str(raw)}, raw_data=raw)


def _process_text_game_observation(raw: Any) -> Observation:
    """Process text game observation."""
    if isinstance(raw, Observation):
        return raw

    if isinstance(raw, dict):
        return Observation(
            state={
                "room": raw.get("room", "unknown"),
                "description": raw.get("description", ""),
                "exits": raw.get("exits", []),
                "items": raw.get("items", []),
                "inventory": raw.get("inventory", []),
            },
            raw_data=raw,
        )

    # Handle string descriptions
    if isinstance(raw, str):
        return Observation(
            state={
                "description": raw,
            },
            raw_data=raw,
        )

    return Observation(state={"raw": str(raw)}, raw_data=raw)


def _process_browser_observation(raw: Any) -> Observation:
    """Process browser observation."""
    if isinstance(raw, Observation):
        return raw

    if isinstance(raw, dict):
        return Observation(
            state={
                "url": raw.get("url", ""),
                "title": raw.get("title", ""),
                "content": raw.get("content", raw.get("content_preview", ""))[:500],
                "elements": raw.get("elements", []),
            },
            raw_data=raw,
        )

    return Observation(state={"raw": str(raw)}, raw_data=raw)


def _process_api_observation(raw: Any) -> Observation:
    """Process API observation."""
    if isinstance(raw, Observation):
        return raw

    if isinstance(raw, dict):
        return Observation(
            state={
                "status": raw.get("last_status", 0),
                "response": raw.get("last_response"),
                "endpoints": raw.get("endpoints", []),
            },
            raw_data=raw,
        )

    return Observation(state={"raw": str(raw)}, raw_data=raw)


def _process_generic_observation(raw: Any) -> Observation:
    """Process generic observation."""
    if isinstance(raw, Observation):
        return raw

    if isinstance(raw, dict):
        return Observation(state=raw, raw_data=raw)

    return Observation(state={"raw": str(raw)}, raw_data=raw)


def observation_to_text(obs: Observation) -> str:
    """Convert observation to text for LLM consumption.

    Args:
        obs: Observation to convert

    Returns:
        Human-readable text description
    """
    lines = ["Current State:"]

    for key, value in obs.state.items():
        if isinstance(value, list):
            if value:
                lines.append(f"  {key}: {', '.join(str(v) for v in value)}")
            else:
                lines.append(f"  {key}: (none)")
        elif isinstance(value, dict):
            lines.append(f"  {key}: {json.dumps(value, indent=4)}")
        elif value is not None:
            lines.append(f"  {key}: {value}")

    return "\n".join(lines)


def diff_observations(o1: Observation, o2: Observation) -> dict[str, Any]:
    """Compute the difference between two observations.

    Args:
        o1: First (previous) observation
        o2: Second (current) observation

    Returns:
        Dict with 'added', 'removed', 'changed' keys
    """
    diff = {
        "added": {},
        "removed": {},
        "changed": {},
    }

    keys1 = set(o1.state.keys())
    keys2 = set(o2.state.keys())

    # Added keys
    for key in keys2 - keys1:
        diff["added"][key] = o2.state[key]

    # Removed keys
    for key in keys1 - keys2:
        diff["removed"][key] = o1.state[key]

    # Changed keys
    for key in keys1 & keys2:
        v1 = o1.state[key]
        v2 = o2.state[key]

        if v1 != v2:
            diff["changed"][key] = {
                "from": v1,
                "to": v2,
            }

    return diff


def diff_to_text(diff: dict[str, Any]) -> str:
    """Convert observation diff to human-readable text.

    Args:
        diff: Diff dict from diff_observations

    Returns:
        Human-readable description of changes
    """
    lines = []

    if diff["added"]:
        lines.append("New:")
        for key, value in diff["added"].items():
            lines.append(f"  + {key}: {value}")

    if diff["removed"]:
        lines.append("Gone:")
        for key, value in diff["removed"].items():
            lines.append(f"  - {key}: {value}")

    if diff["changed"]:
        lines.append("Changed:")
        for key, change in diff["changed"].items():
            lines.append(f"  {key}: {change['from']} -> {change['to']}")

    if not lines:
        return "No changes"

    return "\n".join(lines)


def extract_state_features(obs: Observation) -> dict[str, Any]:
    """Extract key features from observation for planning.

    Args:
        obs: Observation to extract features from

    Returns:
        Dict of extracted features useful for planning
    """
    features = {}

    state = obs.state

    # Position features
    if "position" in state:
        pos = state["position"]
        features["position"] = pos
        if "goal" in state:
            goal = state["goal"]
            # Manhattan distance to goal
            features["distance_to_goal"] = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    # Inventory features
    if "inventory" in state:
        inv = state["inventory"]
        features["has_items"] = len(inv) > 0
        features["inventory_size"] = len(inv)
        features["inventory"] = inv

    # Navigation features
    if "exits" in state:
        features["available_directions"] = state["exits"]
        features["num_exits"] = len(state["exits"])

    # Nearby features
    if "nearby" in state:
        features["nearby_count"] = len(state["nearby"])
        features["nearby"] = state["nearby"]

    # API features
    if "status" in state:
        features["last_status_ok"] = 200 <= state["status"] < 300
        features["last_status"] = state["status"]

    if "response" in state and state["response"]:
        features["has_response"] = True
        if isinstance(state["response"], dict):
            if "error" in state["response"]:
                features["has_error"] = True
                features["error"] = state["response"]["error"]
            if "items" in state["response"]:
                features["num_items"] = len(state["response"]["items"])

    # Browser features
    if "url" in state:
        features["url"] = state["url"]

    if "elements" in state:
        features["num_elements"] = len(state["elements"])

    return features


def observation_hash(obs: Observation) -> str:
    """Compute a hash of the observation state for comparison.

    Args:
        obs: Observation to hash

    Returns:
        Hex digest of state hash
    """
    # Convert state to stable JSON string
    state_str = json.dumps(obs.state, sort_keys=True, default=str)
    return hashlib.sha256(state_str.encode()).hexdigest()[:16]


def states_equivalent(o1: Observation, o2: Observation) -> bool:
    """Check if two observations represent equivalent states.

    Args:
        o1: First observation
        o2: Second observation

    Returns:
        True if states are equivalent
    """
    return observation_hash(o1) == observation_hash(o2)


def summarize_observation(obs: Observation, max_length: int = 100) -> str:
    """Create a short summary of the observation.

    Args:
        obs: Observation to summarize
        max_length: Maximum length of summary

    Returns:
        Short summary string
    """
    parts = []

    state = obs.state

    # Position summary
    if "position" in state:
        parts.append(f"at {state['position']}")

    # Room summary
    if "room" in state:
        parts.append(f"in {state['room']}")

    # URL summary
    if "url" in state:
        url = state["url"]
        if len(url) > 30:
            url = url[:27] + "..."
        parts.append(f"on {url}")

    # Inventory summary
    if "inventory" in state and state["inventory"]:
        parts.append(f"carrying {len(state['inventory'])} items")

    # Status summary
    if "status" in state and state["status"]:
        parts.append(f"status {state['status']}")

    summary = ", ".join(parts) if parts else "unknown state"

    if len(summary) > max_length:
        summary = summary[:max_length - 3] + "..."

    return summary
