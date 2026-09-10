"""Tests for PSS Web UI module."""

import pytest
import time

from pss.config import PSSConfig
from pss.types import Context, TokenUsage
from pss.web.controller import WebController
from pss.web.models import (
    CommandRequest,
    ContextInfo,
    RunRequest,
    StateResponse,
    StatusInfo,
    TreeState,
)


class TestWebController:
    """Tests for WebController."""

    def test_controller_init(self):
        """Test controller initialization."""
        config = PSSConfig()
        controller = WebController(config)
        assert controller.config == config
        assert not controller.exploration_active
        assert not controller.is_paused()

    def test_get_serializable_state_empty(self):
        """Test getting state when no exploration is running."""
        config = PSSConfig()
        controller = WebController(config)
        state = controller.get_serializable_state()

        assert isinstance(state, StateResponse)
        assert state.paused is False
        assert state.quit_requested is False
        assert state.exploration_active is False
        assert len(state.tree.contexts) == 0

    def test_get_serializable_state_with_contexts(self):
        """Test getting state with contexts."""
        config = PSSConfig()
        controller = WebController(config)

        # Add a root context (branched status)
        root = Context(id="root", parent_id=None, messages=[], status="branched")
        root.usage = TokenUsage(input_tokens=100, output_tokens=50, cost=0.001)
        controller.on_context_update(root)

        # Add a child context (running status)
        child = Context(id="root->abc", parent_id="root", messages=[], status="running")
        child.usage = TokenUsage(input_tokens=200, output_tokens=100, cost=0.002)
        child.branch_reason = "Test branch"
        controller.on_context_update(child)

        state = controller.get_serializable_state()

        assert len(state.tree.contexts) == 2
        assert state.status.running_count == 1  # only child is running

        # Check context serialization
        ctx_ids = [c.id for c in state.tree.contexts]
        assert "root" in ctx_ids
        assert "root->abc" in ctx_ids

    def test_get_context_detail(self):
        """Test getting detailed context information."""
        config = PSSConfig()
        controller = WebController(config)

        # Add a context with messages
        ctx = Context(
            id="root",
            parent_id=None,
            messages=[
                {"role": "user", "content": "Test prompt"},
                {"role": "assistant", "content": "Test response"},
            ],
            status="terminated",
            output="Final output",
        )
        ctx.usage = TokenUsage(input_tokens=100, output_tokens=50, cost=0.001)
        ctx.gates_seen = 2
        controller.on_context_update(ctx)

        detail = controller.get_context_detail("root")

        assert detail is not None
        assert detail.id == "root"
        assert detail.status == "terminated"
        assert detail.output == "Final output"
        assert len(detail.messages) == 2
        assert detail.gates_seen == 2

    def test_get_context_detail_partial_match(self):
        """Test getting context detail with partial ID match."""
        config = PSSConfig()
        controller = WebController(config)

        ctx = Context(id="root->abc123", parent_id="root", messages=[])
        ctx.usage = TokenUsage()
        controller.on_context_update(ctx)

        # Partial match should work
        detail = controller.get_context_detail("abc")
        assert detail is not None
        assert detail.id == "root->abc123"

    def test_get_context_detail_not_found(self):
        """Test getting context detail for non-existent context."""
        config = PSSConfig()
        controller = WebController(config)

        detail = controller.get_context_detail("nonexistent")
        assert detail is None

    def test_set_exploration_active(self):
        """Test setting exploration active state."""
        config = PSSConfig()
        controller = WebController(config)

        assert not controller.exploration_active

        controller.set_exploration_active(True)
        assert controller.exploration_active

        state = controller.get_serializable_state()
        assert state.exploration_active is True

        controller.set_exploration_active(False)
        assert not controller.exploration_active

    def test_reset_state(self):
        """Test resetting controller state."""
        config = PSSConfig()
        controller = WebController(config)

        # Add some state
        ctx = Context(id="root", parent_id=None, messages=[])
        ctx.usage = TokenUsage(cost=0.01)
        controller.on_context_update(ctx)
        controller.set_exploration_active(True)

        # Verify state exists
        state1 = controller.get_serializable_state()
        assert len(state1.tree.contexts) == 1
        assert state1.exploration_active is True

        # Reset
        controller.reset_state()

        # Verify state is cleared
        state2 = controller.get_serializable_state()
        assert len(state2.tree.contexts) == 0
        assert state2.exploration_active is False

    def test_killed_context_in_state(self):
        """Test that killed contexts show correct status."""
        config = PSSConfig()
        controller = WebController(config)

        ctx = Context(id="root", parent_id=None, messages=[], status="running")
        ctx.usage = TokenUsage()
        controller.on_context_update(ctx)

        # Kill the context
        from pss.interactive import Command, CommandType
        controller.process_command(Command(type=CommandType.KILL, target="root"))

        state = controller.get_serializable_state()

        # Find the context in state
        root_ctx = next(c for c in state.tree.contexts if c.id == "root")
        assert root_ctx.status == "killed"
        assert state.status.killed_count == 1


class TestWebModels:
    """Tests for Pydantic models."""

    def test_run_request_minimal(self):
        """Test RunRequest with minimal fields."""
        req = RunRequest(prompt="Test prompt")
        assert req.prompt == "Test prompt"
        assert req.preset is None
        assert req.config_overrides is None

    def test_run_request_full(self):
        """Test RunRequest with all fields."""
        req = RunRequest(
            prompt="Test prompt",
            preset="explore",
            config_overrides={"max_contexts": 10},
        )
        assert req.prompt == "Test prompt"
        assert req.preset == "explore"
        assert req.config_overrides == {"max_contexts": 10}

    def test_command_request(self):
        """Test CommandRequest model."""
        req = CommandRequest(type="kill", target="root->abc")
        assert req.type == "kill"
        assert req.target == "root->abc"
        assert req.payload is None

    def test_context_info(self):
        """Test ContextInfo model."""
        info = ContextInfo(
            id="root",
            parent_id=None,
            status="running",
            token_count=1000,
            cost=0.01,
            branch_reason=None,
            output=None,
        )
        assert info.id == "root"
        assert info.status == "running"
        assert info.cost == 0.01

    def test_tree_state(self):
        """Test TreeState model."""
        tree = TreeState(
            contexts=[
                ContextInfo(
                    id="root",
                    parent_id=None,
                    status="branched",
                    token_count=500,
                    cost=0.005,
                    branch_reason=None,
                    output=None,
                )
            ],
            leaves=["root->abc"],
            root_id="root",
        )
        assert len(tree.contexts) == 1
        assert tree.root_id == "root"

    def test_status_info(self):
        """Test StatusInfo model."""
        status = StatusInfo(
            running_count=2,
            terminated_count=1,
            branched_count=3,
            killed_count=0,
            leaf_count=1,
            total_cost=0.05,
            elapsed_seconds=45.5,
            status_messages=["Branch spawned", "Context killed"],
        )
        assert status.running_count == 2
        assert status.total_cost == 0.05
        assert len(status.status_messages) == 2

    def test_state_response(self):
        """Test StateResponse model."""
        state = StateResponse(
            tree=TreeState(contexts=[], leaves=[], root_id=None),
            status=StatusInfo(
                running_count=0,
                terminated_count=0,
                branched_count=0,
                killed_count=0,
                leaf_count=0,
                total_cost=0.0,
                elapsed_seconds=0.0,
                status_messages=[],
            ),
            paused=False,
            quit_requested=False,
            exploration_active=False,
        )
        assert state.exploration_active is False
        assert state.paused is False


class TestStaticFiles:
    """Tests for static file existence."""

    def test_static_files_exist(self):
        """Test that static files exist."""
        from pathlib import Path
        from pss.web.server import STATIC_DIR

        assert STATIC_DIR.exists()
        assert (STATIC_DIR / "index.html").exists()
        assert (STATIC_DIR / "style.css").exists()
        assert (STATIC_DIR / "app.js").exists()
        assert (STATIC_DIR / "examples.js").exists()
        assert (STATIC_DIR / "graph.js").exists()
        assert (STATIC_DIR / "mark.svg").exists()

    def test_index_html_content(self):
        """Test that index.html has expected content."""
        from pss.web.server import STATIC_DIR

        index = STATIC_DIR / "index.html"
        content = index.read_text(encoding="utf-8")

        assert "Pepe Silvia Search" in content
        assert 'id="graph"' in content
        assert 'id="detail"' in content
        assert 'id="compare-dialog"' in content
        assert "Curated example" in content
        assert "/static/style.css" in content
        assert "/static/app.js" in content

    def test_app_js_content(self):
        """Test that app.js has expected content."""
        from pss.web.server import STATIC_DIR

        app_js = STATIC_DIR / "app.js"
        content = app_js.read_text(encoding="utf-8")

        assert "renderGraph" in content
        assert "x-pss-token" in content
        assert "/api/context/" in content
        assert "/api/run" in content
        assert "/api/command" in content
