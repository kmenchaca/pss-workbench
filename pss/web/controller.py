"""Web controller for PSS - extends InteractiveController for web use."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from pss.interactive import InteractiveController, InteractiveState
from pss.web.models import (
    ContextDetailResponse,
    ContextInfo,
    StateResponse,
    StatusInfo,
    TreeState,
)

if TYPE_CHECKING:
    from pss.config import PSSConfig
    from pss.types import Context


class WebController(InteractiveController):
    """Web-friendly interactive controller.

    Extends InteractiveController with serialization methods for JSON responses.
    """

    def __init__(self, config: PSSConfig):
        super().__init__(config)
        self.exploration_active = False

    def get_serializable_state(self) -> StateResponse:
        """Serialize the complete state for JSON response."""
        with self.state.lock:
            return StateResponse(
                tree=self._serialize_tree(),
                status=self._serialize_status(),
                paused=self.state.paused,
                quit_requested=self.state.quit_requested,
                exploration_active=self.exploration_active,
            )

    def _serialize_tree(self) -> TreeState:
        """Serialize tree structure for JSON."""
        contexts = []
        for ctx in self.state.tree.contexts.values():
            contexts.append(self._serialize_context(ctx))

        return TreeState(
            contexts=contexts,
            leaves=list(self.state.tree.leaves),
            root_id=self.state.tree.root_id
            or next(
                (
                    c.id
                    for c in self.state.tree.contexts.values()
                    if c.parent_id is None
                ),
                None,
            ),
        )

    def _serialize_context(self, ctx: Context) -> ContextInfo:
        """Serialize a single context for JSON."""
        # Extract last assistant message for preview
        last_message = None
        for msg in reversed(ctx.messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False) if content else ""
                if content:
                    # Truncate for preview
                    last_message = content[-200:] if len(content) > 200 else content
                    break

        return ContextInfo(
            id=ctx.id,
            parent_id=ctx.parent_id,
            status=ctx.status if ctx.id not in self.state.killed else "killed",
            token_count=ctx.token_count,
            cost=ctx.usage.cost if ctx.usage else 0.0,
            branch_reason=ctx.branch_reason,
            output=ctx.output[:500]
            if ctx.output and len(ctx.output) > 500
            else ctx.output,
            last_message=last_message,
            termination_reason=ctx.termination_reason.value
            if ctx.termination_reason
            else None,
        )

    def _serialize_status(self) -> StatusInfo:
        """Serialize status information for JSON."""
        contexts = self.state.tree.contexts
        running_count = sum(
            1
            for c in contexts.values()
            if c.status == "running" and c.id not in self.state.killed
        )
        terminated_count = sum(1 for c in contexts.values() if c.status == "terminated")
        branched_count = sum(1 for c in contexts.values() if c.status == "branched")
        killed_count = len(self.state.killed)
        leaf_count = len(self.state.tree.leaves)

        return StatusInfo(
            running_count=running_count,
            terminated_count=terminated_count,
            branched_count=branched_count,
            killed_count=killed_count,
            leaf_count=leaf_count,
            total_cost=self.state.total_cost,
            elapsed_seconds=time.time() - self.state.start_time,
            status_messages=list(self.state.status_messages[-10:]),
        )

    def get_context_detail(self, ctx_id: str) -> ContextDetailResponse | None:
        """Get detailed information about a specific context."""
        with self.state.lock:
            ctx = self.state.tree.contexts.get(ctx_id)
            if not ctx:
                # Try partial match
                matches = [cid for cid in self.state.tree.contexts if ctx_id in cid]
                if len(matches) == 1:
                    ctx = self.state.tree.contexts[matches[0]]
                else:
                    return None

            # Preserve full messages for inspection and export; live requests are bounded.
            messages = []
            for msg in ctx.messages:
                content = msg.get("content", "")
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False) if content else ""
                messages.append(
                    {
                        "role": msg.get("role", "unknown"),
                        "content": content,
                    }
                )

            return ContextDetailResponse(
                id=ctx.id,
                parent_id=ctx.parent_id,
                status=ctx.status if ctx.id not in self.state.killed else "killed",
                token_count=ctx.token_count,
                cost=ctx.usage.cost if ctx.usage else 0.0,
                branch_reason=ctx.branch_reason,
                output=ctx.output,
                messages=messages,
                gates_seen=ctx.gates_seen,
                termination_reason=ctx.termination_reason.value
                if ctx.termination_reason
                else None,
            )

    def set_exploration_active(self, active: bool) -> None:
        """Set whether exploration is currently active."""
        self.exploration_active = active

    def reset_state(self) -> None:
        """Reset state for a new exploration run."""
        with self.state.lock:
            self.state = InteractiveState()
        self.exploration_active = False
