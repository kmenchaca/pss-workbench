"""Pydantic models for PSS Web API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class RunRequest(BaseModel):
    """Request to start a new exploration."""

    prompt: str
    preset: str | None = None
    config_overrides: dict[str, Any] | None = None


class CommandRequest(BaseModel):
    """Request to send a command."""

    type: str  # kill, inject, promote, pause, resume, quit
    target: str | None = None
    payload: str | None = None


class ContextInfo(BaseModel):
    """Serialized context information."""

    id: str
    parent_id: str | None
    status: str
    token_count: int
    cost: float
    branch_reason: str | None
    output: str | None
    last_message: str | None = None  # Last assistant message for preview
    termination_reason: str | None = None


class TreeState(BaseModel):
    """Serialized tree state."""

    contexts: list[ContextInfo]
    leaves: list[str]
    root_id: str | None


class StatusInfo(BaseModel):
    """Status information."""

    running_count: int
    terminated_count: int
    branched_count: int
    killed_count: int
    leaf_count: int
    total_cost: float
    elapsed_seconds: float
    status_messages: list[str]


class StateResponse(BaseModel):
    """Full state response."""

    tree: TreeState
    status: StatusInfo
    paused: bool
    quit_requested: bool
    exploration_active: bool


class CommandResponse(BaseModel):
    """Response to a command."""

    success: bool
    message: str


class ContextDetailResponse(BaseModel):
    """Detailed context information including messages."""

    id: str
    parent_id: str | None
    status: str
    token_count: int
    cost: float
    branch_reason: str | None
    output: str | None
    messages: list[dict[str, str]]
    gates_seen: int
    termination_reason: str | None = None
