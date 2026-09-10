"""
API testing environment.

Provides an environment for testing REST APIs with
GET, POST, PUT, DELETE operations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional
import json

from ..environment import ActionSpace, Environment
from ..types import Action, ActionResult, EnvironmentState, Observation


@dataclass
class APIEndpoint:
    """Definition of an API endpoint."""
    method: str
    path: str
    handler: Callable[[dict[str, Any]], tuple[int, dict[str, Any]]]
    description: str = ""


class APIEnvironment(Environment):
    """Environment for testing REST APIs.

    Can be configured with mock endpoints or connected to real APIs.
    Useful for testing API exploration and interaction patterns.

    Attributes:
        base_url: Base URL for the API
        endpoints: List of available endpoints
        mock_mode: If True, use mock handlers instead of real HTTP
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        endpoints: Optional[list[APIEndpoint]] = None,
        mock_mode: bool = True,
        max_requests: int = 100,
    ):
        super().__init__()
        self.base_url = base_url
        self.endpoints = endpoints or self._default_endpoints()
        self.mock_mode = mock_mode
        self.max_requests = max_requests

        self._requests: int = 0
        self._last_response: Optional[dict[str, Any]] = None
        self._last_status: int = 0
        self._terminal: bool = False
        self._mock_data: dict[str, Any] = {}

    def _default_endpoints(self) -> list[APIEndpoint]:
        """Create default mock endpoints for testing."""
        def get_items(params: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            items = self._mock_data.get("items", [])
            return 200, {"items": items, "count": len(items)}

        def post_item(params: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            items = self._mock_data.setdefault("items", [])
            new_item = {
                "id": len(items) + 1,
                "name": params.get("body", {}).get("name", "unnamed"),
                "value": params.get("body", {}).get("value", 0),
            }
            items.append(new_item)
            return 201, {"item": new_item}

        def get_item(params: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            item_id = params.get("path_params", {}).get("id")
            items = self._mock_data.get("items", [])
            for item in items:
                if str(item["id"]) == str(item_id):
                    return 200, {"item": item}
            return 404, {"error": "Item not found"}

        def delete_item(params: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            item_id = params.get("path_params", {}).get("id")
            items = self._mock_data.get("items", [])
            for i, item in enumerate(items):
                if str(item["id"]) == str(item_id):
                    deleted = items.pop(i)
                    return 200, {"deleted": deleted}
            return 404, {"error": "Item not found"}

        def update_item(params: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            item_id = params.get("path_params", {}).get("id")
            items = self._mock_data.get("items", [])
            for item in items:
                if str(item["id"]) == str(item_id):
                    body = params.get("body", {})
                    if "name" in body:
                        item["name"] = body["name"]
                    if "value" in body:
                        item["value"] = body["value"]
                    return 200, {"item": item}
            return 404, {"error": "Item not found"}

        return [
            APIEndpoint("GET", "/items", get_items, "List all items"),
            APIEndpoint("POST", "/items", post_item, "Create a new item"),
            APIEndpoint("GET", "/items/{id}", get_item, "Get a specific item"),
            APIEndpoint("PUT", "/items/{id}", update_item, "Update an item"),
            APIEndpoint("DELETE", "/items/{id}", delete_item, "Delete an item"),
        ]

    def _create_action_space(self) -> ActionSpace:
        return ActionSpace(
            action_types=["request", "get", "post", "put", "delete", "auth"],
            parameter_specs={
                "request": {
                    "required": ["method", "path"],
                    "allowed": ["method", "path", "body", "headers", "params"],
                },
                "get": {
                    "required": ["path"],
                    "allowed": ["path", "params", "headers"],
                },
                "post": {
                    "required": ["path"],
                    "allowed": ["path", "body", "headers"],
                },
                "put": {
                    "required": ["path"],
                    "allowed": ["path", "body", "headers"],
                },
                "delete": {
                    "required": ["path"],
                    "allowed": ["path", "headers"],
                },
                "auth": {
                    "required": ["token"],
                    "allowed": ["token", "type"],
                },
            },
        )

    def reset(self) -> Observation:
        """Reset API environment."""
        self._requests = 0
        self._last_response = None
        self._last_status = 0
        self._terminal = False
        self._mock_data = {
            "items": [
                {"id": 1, "name": "initial_item", "value": 100},
            ]
        }
        return self.observe()

    def observe(self) -> Observation:
        """Get current observation."""
        endpoint_list = [
            f"{e.method} {e.path}: {e.description}"
            for e in self.endpoints
        ]

        return Observation(
            state={
                "base_url": self.base_url,
                "endpoints": endpoint_list,
                "last_status": self._last_status,
                "last_response": self._last_response,
                "requests_made": self._requests,
            },
            raw_data={
                "mock_data": self._mock_data.copy() if self.mock_mode else None,
            },
        )

    def step(self, action: Action) -> ActionResult:
        """Execute an API request."""
        self._requests += 1

        # Normalize action to request
        if action.action_type in ["get", "post", "put", "delete"]:
            method = action.action_type.upper()
            path = action.parameters.get("path", "/")
            body = action.parameters.get("body", {})
            params = action.parameters.get("params", {})
            headers = action.parameters.get("headers", {})
        elif action.action_type == "request":
            method = action.parameters.get("method", "GET").upper()
            path = action.parameters.get("path", "/")
            body = action.parameters.get("body", {})
            params = action.parameters.get("params", {})
            headers = action.parameters.get("headers", {})
        elif action.action_type == "auth":
            # Store auth token for future requests
            token = action.parameters.get("token", "")
            self._mock_data["_auth_token"] = token
            return ActionResult(
                action_id=action.id,
                success=True,
                new_observation=self.observe(),
                reward=0.0,
            )
        else:
            return ActionResult(
                action_id=action.id,
                success=False,
                new_observation=self.observe(),
                reward=-0.1,
                error=f"Unknown action type: {action.action_type}",
            )

        return self._execute_request(action, method, path, body, params, headers)

    def _execute_request(
        self,
        action: Action,
        method: str,
        path: str,
        body: dict[str, Any],
        params: dict[str, Any],
        headers: dict[str, Any],
    ) -> ActionResult:
        """Execute the actual request."""
        if self.mock_mode:
            return self._execute_mock_request(action, method, path, body, params)
        else:
            return self._execute_real_request(action, method, path, body, params, headers)

    def _execute_mock_request(
        self,
        action: Action,
        method: str,
        path: str,
        body: dict[str, Any],
        params: dict[str, Any],
    ) -> ActionResult:
        """Execute request against mock handlers."""
        # Find matching endpoint
        endpoint = None
        path_params = {}

        for ep in self.endpoints:
            if ep.method != method:
                continue

            # Check for exact match or pattern match
            if ep.path == path:
                endpoint = ep
                break

            # Check for path parameter match (e.g., /items/{id})
            if "{" in ep.path:
                pattern_parts = ep.path.split("/")
                path_parts = path.split("/")

                if len(pattern_parts) != len(path_parts):
                    continue

                match = True
                for pp, pa in zip(pattern_parts, path_parts):
                    if pp.startswith("{") and pp.endswith("}"):
                        param_name = pp[1:-1]
                        path_params[param_name] = pa
                    elif pp != pa:
                        match = False
                        break

                if match:
                    endpoint = ep
                    break

        if endpoint is None:
            self._last_status = 404
            self._last_response = {"error": f"No endpoint found for {method} {path}"}
            return ActionResult(
                action_id=action.id,
                success=False,
                new_observation=self.observe(),
                reward=-0.1,
                error=f"Endpoint not found: {method} {path}",
            )

        # Execute handler
        try:
            status, response = endpoint.handler({
                "path_params": path_params,
                "params": params,
                "body": body,
            })

            self._last_status = status
            self._last_response = response

            success = 200 <= status < 300
            reward = 0.1 if success else -0.05

            return ActionResult(
                action_id=action.id,
                success=success,
                new_observation=self.observe(),
                reward=reward,
                error=None if success else response.get("error", "Request failed"),
            )
        except Exception as e:
            self._last_status = 500
            self._last_response = {"error": str(e)}
            return ActionResult(
                action_id=action.id,
                success=False,
                new_observation=self.observe(),
                reward=-0.1,
                error=str(e),
            )

    def _execute_real_request(
        self,
        action: Action,
        method: str,
        path: str,
        body: dict[str, Any],
        params: dict[str, Any],
        headers: dict[str, Any],
    ) -> ActionResult:
        """Execute real HTTP request. Requires httpx."""
        # This would use httpx for real requests
        # For now, return an error indicating real mode isn't implemented
        return ActionResult(
            action_id=action.id,
            success=False,
            new_observation=self.observe(),
            reward=-0.1,
            error="Real HTTP mode not implemented. Use mock_mode=True.",
        )

    def is_terminal(self) -> bool:
        """Check if session is over."""
        return self._terminal or self._requests >= self.max_requests

    def checkpoint(self) -> EnvironmentState:
        """Save API state."""
        return EnvironmentState(
            data={
                "requests": self._requests,
                "last_status": self._last_status,
                "last_response": self._last_response,
                "terminal": self._terminal,
                "mock_data": json.loads(json.dumps(self._mock_data)),  # Deep copy
            }
        )

    def restore(self, state: EnvironmentState) -> None:
        """Restore API state."""
        data = state.data
        self._requests = data["requests"]
        self._last_status = data["last_status"]
        self._last_response = data["last_response"]
        self._terminal = data["terminal"]
        self._mock_data = json.loads(json.dumps(data["mock_data"]))  # Deep copy

    def add_endpoint(self, endpoint: APIEndpoint) -> None:
        """Add a new endpoint to the API."""
        self.endpoints.append(endpoint)
