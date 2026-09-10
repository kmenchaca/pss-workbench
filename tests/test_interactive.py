"""Tests for interactive mode."""

import pytest
import threading
import time

from pss.interactive import (
    InteractiveState,
    InteractiveController,
    Command,
    CommandType,
    parse_command,
    create_tree_view,
    create_status_panel,
)
from pss.config import PSSConfig
from pss.types import Context, SearchTree, TokenUsage


class TestCommandParsing:
    """Tests for command parsing."""

    def test_parse_quit_commands(self):
        """Test parsing quit commands."""
        assert parse_command("q").type == CommandType.QUIT
        assert parse_command("quit").type == CommandType.QUIT
        assert parse_command("exit").type == CommandType.QUIT

    def test_parse_help_commands(self):
        """Test parsing help commands."""
        assert parse_command("?").type == CommandType.HELP
        assert parse_command("help").type == CommandType.HELP
        assert parse_command("h").type == CommandType.HELP

    def test_parse_pause_resume(self):
        """Test parsing pause/resume commands."""
        assert parse_command("pause").type == CommandType.PAUSE
        assert parse_command("resume").type == CommandType.RESUME

    def test_parse_kill_command(self):
        """Test parsing kill command."""
        cmd = parse_command("k root")
        assert cmd.type == CommandType.KILL
        assert cmd.target == "root"

        cmd = parse_command("kill root->abc123")
        assert cmd.type == CommandType.KILL
        assert cmd.target == "root->abc123"

    def test_parse_inject_command(self):
        """Test parsing inject command."""
        cmd = parse_command("i root focus on the code")
        assert cmd.type == CommandType.INJECT
        assert cmd.target == "root"
        assert cmd.payload == "focus on the code"

        cmd = parse_command("inject ctx123 please explore option B")
        assert cmd.type == CommandType.INJECT
        assert cmd.target == "ctx123"
        assert cmd.payload == "please explore option B"

    def test_parse_promote_command(self):
        """Test parsing promote command."""
        cmd = parse_command("p root->abc")
        assert cmd.type == CommandType.PROMOTE
        assert cmd.target == "root->abc"

    def test_parse_empty_returns_none(self):
        """Test that empty input returns None."""
        assert parse_command("") is None
        assert parse_command("   ") is None

    def test_parse_invalid_returns_none(self):
        """Test that invalid commands return None."""
        # Kill without target
        assert parse_command("k") is None
        # Inject without message
        assert parse_command("i root") is None


class TestInteractiveState:
    """Tests for InteractiveState."""

    def test_initial_state(self):
        """Test initial state values."""
        state = InteractiveState()
        assert state.paused is False
        assert state.quit_requested is False
        assert len(state.killed) == 0
        assert len(state.injections) == 0
        assert len(state.promoted) == 0

    def test_thread_safety(self):
        """Test that state can be accessed from multiple threads."""
        state = InteractiveState()

        def modify_state():
            with state.lock:
                state.killed.add("test")
                time.sleep(0.01)
                state.paused = True

        threads = [threading.Thread(target=modify_state) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert "test" in state.killed
        assert state.paused is True


class TestInteractiveController:
    """Tests for InteractiveController."""

    def test_controller_init(self):
        """Test controller initialization."""
        config = PSSConfig()
        controller = InteractiveController(config)
        assert controller.config == config
        assert not controller.is_paused()
        assert not controller.is_quit_requested()

    def test_process_pause_resume(self):
        """Test pause/resume commands."""
        config = PSSConfig()
        controller = InteractiveController(config)

        msg = controller.process_command(Command(type=CommandType.PAUSE))
        assert "paused" in msg.lower()
        assert controller.is_paused()

        msg = controller.process_command(Command(type=CommandType.RESUME))
        assert "resumed" in msg.lower()
        assert not controller.is_paused()

    def test_process_quit(self):
        """Test quit command."""
        config = PSSConfig()
        controller = InteractiveController(config)

        msg = controller.process_command(Command(type=CommandType.QUIT))
        assert controller.is_quit_requested()

    def test_process_kill(self):
        """Test kill command."""
        config = PSSConfig()
        controller = InteractiveController(config)

        # Add a context to the tree
        ctx = Context(id="root", parent_id=None, messages=[])
        controller.on_context_update(ctx)

        msg = controller.process_command(Command(type=CommandType.KILL, target="root"))
        assert controller.should_kill("root")
        assert "killed" in msg.lower()

    def test_process_kill_partial_match(self):
        """Test kill with partial ID match."""
        config = PSSConfig()
        controller = InteractiveController(config)

        # Add contexts
        ctx1 = Context(id="root->abc123", parent_id="root", messages=[])
        ctx2 = Context(id="root->def456", parent_id="root", messages=[])
        controller.on_context_update(ctx1)
        controller.on_context_update(ctx2)

        # Partial match should work if unique
        msg = controller.process_command(Command(type=CommandType.KILL, target="abc"))
        assert controller.should_kill("root->abc123")

        # Ambiguous match should fail
        msg = controller.process_command(Command(type=CommandType.KILL, target="root"))
        assert "ambiguous" in msg.lower()

    def test_process_inject(self):
        """Test inject command."""
        config = PSSConfig()
        controller = InteractiveController(config)

        # Add a running context
        ctx = Context(id="root", parent_id=None, messages=[], status="running")
        controller.on_context_update(ctx)

        msg = controller.process_command(
            Command(type=CommandType.INJECT, target="root", payload="focus on code")
        )
        assert "injected" in msg.lower()

        # Get the injection
        injection = controller.get_injection("root")
        assert injection == "focus on code"

        # Second get should return None (consumed)
        assert controller.get_injection("root") is None

    def test_process_promote(self):
        """Test promote command."""
        config = PSSConfig()
        controller = InteractiveController(config)

        # Add a context
        ctx = Context(id="root", parent_id=None, messages=[])
        controller.on_context_update(ctx)

        msg = controller.process_command(Command(type=CommandType.PROMOTE, target="root"))
        assert "promoted" in msg.lower()

        # Check should_promote returns True once
        assert controller.should_promote("root")
        # Second check should return False (consumed)
        assert not controller.should_promote("root")

    def test_on_context_update(self):
        """Test context update callback."""
        config = PSSConfig()
        controller = InteractiveController(config)

        ctx = Context(id="test", parent_id=None, messages=[])
        ctx.usage = TokenUsage(input_tokens=100, output_tokens=50, cost=0.001)

        controller.on_context_update(ctx)

        assert "test" in controller.state.tree.contexts
        assert controller.state.total_cost == 0.001

    def test_on_leaf_added(self):
        """Test leaf added callback."""
        config = PSSConfig()
        controller = InteractiveController(config)

        controller.on_leaf_added("test_leaf")
        assert "test_leaf" in controller.state.tree.leaves

    def test_on_branch_spawned(self):
        """Test branch spawned callback."""
        config = PSSConfig()
        controller = InteractiveController(config)

        controller.on_branch_spawned("root", "root->child", "test branch")
        assert len(controller.state.status_messages) == 1
        assert "spawned" in controller.state.status_messages[0].lower()


class TestTreeView:
    """Tests for tree visualization."""

    def test_empty_tree(self):
        """Test rendering empty tree."""
        state = InteractiveState()
        tree = create_tree_view(state)
        # Should render without error
        assert tree is not None

    def test_tree_with_contexts(self):
        """Test rendering tree with contexts."""
        state = InteractiveState()

        # Add root
        root = Context(id="root", parent_id=None, messages=[], status="branched")
        root.usage = TokenUsage(input_tokens=100, output_tokens=50, cost=0.001)
        state.tree.contexts["root"] = root

        # Add child
        child = Context(id="root->abc", parent_id="root", messages=[], status="running")
        child.usage = TokenUsage(input_tokens=200, output_tokens=100, cost=0.002)
        child.branch_reason = "Test branch"
        state.tree.contexts["root->abc"] = child

        # Add terminated leaf
        leaf = Context(id="root->def", parent_id="root", messages=[], status="terminated", output="Result")
        leaf.usage = TokenUsage(input_tokens=150, output_tokens=75, cost=0.0015)
        state.tree.contexts["root->def"] = leaf
        state.tree.leaves.append("root->def")

        tree = create_tree_view(state)
        assert tree is not None

    def test_killed_context_display(self):
        """Test that killed contexts show as killed."""
        state = InteractiveState()

        ctx = Context(id="root", parent_id=None, messages=[], status="running")
        ctx.usage = TokenUsage()
        state.tree.contexts["root"] = ctx
        state.killed.add("root")

        tree = create_tree_view(state)
        # Tree should render - actual display tested manually
        assert tree is not None


class TestStatusPanel:
    """Tests for status panel."""

    def test_status_panel_creation(self):
        """Test status panel creates without error."""
        state = InteractiveState()
        panel = create_status_panel(state)
        assert panel is not None

    def test_status_panel_with_contexts(self):
        """Test status panel with contexts."""
        state = InteractiveState()
        state.paused = True
        state.total_cost = 0.05
        state.status_messages = ["Test message 1", "Test message 2"]

        # Add some contexts
        ctx = Context(id="root", parent_id=None, messages=[], status="running")
        ctx.usage = TokenUsage()
        state.tree.contexts["root"] = ctx

        panel = create_status_panel(state)
        assert panel is not None
