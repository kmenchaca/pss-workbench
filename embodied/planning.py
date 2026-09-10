"""
Action planning with LLM guidance.

Functions for selecting and generating actions using LLM reasoning.
"""

from typing import Any, Callable, Optional
import json
import re

from .types import Action, ActionResult, Goal, Observation
from .environment import ActionSpace


# Type for LLM call function
LLMCallable = Callable[[str, str], str]


def plan_action(
    observation: Observation,
    goal: Goal,
    history: list[tuple[Action, ActionResult]],
    action_space: ActionSpace,
    llm_call: Optional[LLMCallable] = None,
) -> Action:
    """Select the next action using LLM reasoning.

    Args:
        observation: Current observation
        goal: What we're trying to achieve
        history: Previous actions and results
        action_space: Available actions
        llm_call: Function to call LLM (system_prompt, user_prompt) -> response

    Returns:
        Action to take next
    """
    if llm_call is None:
        # Default to simple heuristic planning
        return _heuristic_plan(observation, goal, history, action_space)

    # Build prompt for LLM
    system_prompt = _build_planning_system_prompt(action_space)
    user_prompt = _build_planning_user_prompt(observation, goal, history)

    # Call LLM
    response = llm_call(system_prompt, user_prompt)

    # Parse response into action
    action = action_from_text(response, action_space)

    return action


def _build_planning_system_prompt(action_space: ActionSpace) -> str:
    """Build system prompt for action planning."""
    return f"""You are an agent that selects actions to achieve goals.

{action_space.describe()}

Respond with a JSON object specifying your chosen action:
{{"action": "action_type", "parameters": {{"param1": "value1"}}}}

Think step by step about:
1. What is the current state?
2. What is the goal?
3. What actions have been tried?
4. What action will make progress?

Then provide your action choice as JSON."""


def _build_planning_user_prompt(
    observation: Observation,
    goal: Goal,
    history: list[tuple[Action, ActionResult]],
) -> str:
    """Build user prompt with current state and history."""
    lines = [f"Goal: {goal.description}", "", "Current observation:"]

    for key, value in observation.state.items():
        lines.append(f"  {key}: {value}")

    if history:
        lines.append("")
        lines.append("Recent actions:")
        for action, result in history[-5:]:  # Last 5 actions
            status = "succeeded" if result.success else f"failed: {result.error}"
            lines.append(f"  {action} -> {status}")

    lines.append("")
    lines.append("What action should I take next?")

    return "\n".join(lines)


def _heuristic_plan(
    observation: Observation,
    goal: Goal,
    history: list[tuple[Action, ActionResult]],
    action_space: ActionSpace,
) -> Action:
    """Simple heuristic planning without LLM.

    Uses basic rules to select actions when no LLM is available.
    """
    state = observation.state

    # Check for grid navigation (mock environment)
    if "position" in state and "goal" in state:
        return _plan_grid_navigation(state)

    # Check for text adventure
    if "room" in state and "exits" in state:
        return _plan_text_adventure(state, history)

    # Check for API environment
    if "endpoints" in state:
        return _plan_api_exploration(state, history)

    # Default: wait
    if "wait" in action_space.action_types:
        return Action(action_type="wait")

    # Pick first available action
    if action_space.action_types:
        return Action(action_type=action_space.action_types[0])

    return Action(action_type="noop")


def _plan_grid_navigation(state: dict[str, Any]) -> Action:
    """Plan movement in grid environment."""
    pos = state["position"]
    goal = state["goal"]

    dx = goal[0] - pos[0]
    dy = goal[1] - pos[1]

    # Prioritize larger distance
    if abs(dx) >= abs(dy):
        direction = "east" if dx > 0 else "west"
    else:
        direction = "north" if dy > 0 else "south"

    return Action(action_type="move", parameters={"direction": direction})


def _plan_text_adventure(
    state: dict[str, Any],
    history: list[tuple[Action, ActionResult]],
) -> Action:
    """Plan actions for text adventure."""
    # First, try to pick up items
    items = state.get("items", [])
    inventory = state.get("inventory", [])

    if items:
        return Action(action_type="take", parameters={"item": items[0]})

    # Then explore exits we haven't visited
    exits = state.get("exits", [])
    visited_directions = set()

    for action, result in history:
        if action.action_type == "go" and result.success:
            visited_directions.add(action.parameters.get("direction", ""))

    for exit_dir in exits:
        if exit_dir not in visited_directions:
            return Action(action_type="go", parameters={"direction": exit_dir})

    # If all exits visited, pick one at random
    if exits:
        return Action(action_type="go", parameters={"direction": exits[0]})

    return Action(action_type="look")


def _plan_api_exploration(
    state: dict[str, Any],
    history: list[tuple[Action, ActionResult]],
) -> Action:
    """Plan actions for API exploration."""
    endpoints = state.get("endpoints", [])
    called_endpoints = set()

    for action, result in history:
        if action.action_type in ["get", "post", "put", "delete", "request"]:
            path = action.parameters.get("path", "")
            method = action.parameters.get("method", action.action_type.upper())
            called_endpoints.add(f"{method} {path}")

    # Find an uncalled endpoint
    for endpoint in endpoints:
        # Parse "GET /items: List all items" format
        parts = endpoint.split(":")
        if parts:
            method_path = parts[0].strip()
            if method_path not in called_endpoints:
                method, path = method_path.split(" ", 1)
                return Action(
                    action_type=method.lower(),
                    parameters={"path": path},
                )

    # Default to GET /items
    return Action(action_type="get", parameters={"path": "/items"})


def action_from_text(text: str, action_space: ActionSpace) -> Action:
    """Parse LLM text response into an action.

    Args:
        text: LLM response text
        action_space: Valid action space

    Returns:
        Parsed Action object
    """
    # Try to extract JSON - handle nested braces
    # First try to find balanced braces
    start_idx = text.find('{')
    if start_idx != -1:
        depth = 0
        end_idx = start_idx
        for i, char in enumerate(text[start_idx:], start_idx):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break

        if end_idx > start_idx:
            json_str = text[start_idx:end_idx]
            try:
                data = json.loads(json_str)
                action_type = data.get("action", data.get("action_type", ""))
                parameters = data.get("parameters", data.get("params", {}))

                if action_type and action_type in action_space.action_types:
                    return Action(action_type=action_type, parameters=parameters)
            except json.JSONDecodeError:
                pass

    # Try to match action patterns in text
    text_lower = text.lower()

    for action_type in action_space.action_types:
        if action_type.lower() in text_lower:
            # Try to extract parameters from text
            parameters = _extract_parameters_from_text(text, action_type, action_space)
            return Action(action_type=action_type, parameters=parameters)

    # Default to first action type
    if action_space.action_types:
        return Action(action_type=action_space.action_types[0])

    return Action(action_type="noop")


def _extract_parameters_from_text(
    text: str,
    action_type: str,
    action_space: ActionSpace,
) -> dict[str, Any]:
    """Extract action parameters from natural language text."""
    parameters = {}

    spec = action_space.parameter_specs.get(action_type, {})
    required = spec.get("required", [])

    # Common parameter patterns
    direction_pattern = r'\b(north|south|east|west|up|down|left|right)\b'
    item_pattern = r'(?:pick up|take|get|grab|use|drop)\s+(?:the\s+)?(\w+)'
    url_pattern = r'(https?://[^\s]+|/[^\s]+)'

    for param in required:
        if param == "direction":
            match = re.search(direction_pattern, text, re.IGNORECASE)
            if match:
                parameters["direction"] = match.group(1).lower()

        elif param == "item":
            match = re.search(item_pattern, text, re.IGNORECASE)
            if match:
                parameters["item"] = match.group(1).lower()

        elif param in ["url", "path"]:
            match = re.search(url_pattern, text)
            if match:
                parameters[param] = match.group(1)

    return parameters


def explain_action(action: Action, context: dict[str, Any]) -> str:
    """Generate explanation for why an action was chosen.

    Args:
        action: Action to explain
        context: Context including observation, goal, history

    Returns:
        Human-readable explanation
    """
    observation = context.get("observation")
    goal = context.get("goal")

    explanations = []

    if action.action_type == "move":
        direction = action.parameters.get("direction", "")
        if observation and "position" in observation.state and "goal" in observation.state:
            pos = observation.state["position"]
            goal_pos = observation.state["goal"]
            explanations.append(
                f"Moving {direction} to get closer to goal at {goal_pos} from {pos}"
            )
        else:
            explanations.append(f"Moving {direction} to explore")

    elif action.action_type == "take":
        item = action.parameters.get("item", "")
        explanations.append(f"Picking up {item} which may be useful")

    elif action.action_type == "go":
        direction = action.parameters.get("direction", "")
        explanations.append(f"Going {direction} to explore new area")

    elif action.action_type == "get":
        path = action.parameters.get("path", "")
        explanations.append(f"Fetching data from {path}")

    elif action.action_type == "post":
        path = action.parameters.get("path", "")
        explanations.append(f"Creating resource at {path}")

    else:
        explanations.append(f"Performing {action.action_type}")

    if goal:
        explanations.append(f"Goal: {goal.description}")

    return " | ".join(explanations)


def replan_on_failure(
    action_result: ActionResult,
    history: list[tuple[Action, ActionResult]],
    observation: Observation,
    goal: Goal,
    action_space: ActionSpace,
    llm_call: Optional[LLMCallable] = None,
) -> Action:
    """Generate alternative action after a failure.

    Args:
        action_result: Failed action result
        history: Action history
        observation: Current observation
        goal: Goal we're trying to achieve
        action_space: Available actions
        llm_call: Optional LLM for smarter replanning

    Returns:
        Alternative action to try
    """
    # Get the failed action
    failed_action_id = action_result.action_id
    failed_action = None
    for action, _ in history:
        if action.id == failed_action_id:
            failed_action = action
            break

    if llm_call:
        # Use LLM to replan
        system_prompt = f"""You are replanning after a failed action.

The action "{failed_action}" failed with error: {action_result.error}

{action_space.describe()}

Suggest an alternative action that avoids the same failure.
Respond with JSON: {{"action": "action_type", "parameters": {{}}, "reason": "..."}}"""

        user_prompt = _build_planning_user_prompt(observation, goal, history)
        response = llm_call(system_prompt, user_prompt)
        return action_from_text(response, action_space)

    # Heuristic replanning
    if failed_action:
        # Avoid the same action type for now
        alternative_types = [
            at for at in action_space.action_types
            if at != failed_action.action_type
        ]

        if alternative_types:
            # Simple fallback to different action
            if "wait" in alternative_types:
                return Action(action_type="wait")
            if "look" in alternative_types:
                return Action(action_type="look")
            return Action(action_type=alternative_types[0])

    # Last resort: try original planning
    return _heuristic_plan(observation, goal, history, action_space)
