"""Interactive mode for PSS - human-in-the-loop branch control.

Provides a terminal UI for:
- Live tree visualization
- Kill: Stop branches going off track
- Inject: Add guidance to running branches
- Promote: Spawn more branches like a good one
- Pause/Resume: Stop the run, inspect, continue

v0.8.0
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from pss.config import PSSConfig
from pss.types import Context, SearchTree


class CommandType(Enum):
    """Types of interactive commands."""
    KILL = "kill"
    INJECT = "inject"
    PROMOTE = "promote"
    PAUSE = "pause"
    RESUME = "resume"
    QUIT = "quit"
    HELP = "help"
    STATUS = "status"


@dataclass
class Command:
    """A command from the user."""
    type: CommandType
    target: str | None = None  # Context ID for kill/inject/promote
    payload: str | None = None  # Message for inject


@dataclass
class InteractiveState:
    """Shared state between controller and harness."""
    tree: SearchTree = field(default_factory=SearchTree)
    paused: bool = False
    quit_requested: bool = False
    pending_commands: list[Command] = field(default_factory=list)
    # Messages injected into specific contexts
    injections: dict[str, str] = field(default_factory=dict)  # ctx_id -> message
    # Contexts to kill
    killed: set[str] = field(default_factory=set)
    # Contexts to promote (spawn more like)
    promoted: set[str] = field(default_factory=set)
    # Status messages for display
    status_messages: list[str] = field(default_factory=list)
    # Lock for thread safety
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Total cost so far
    total_cost: float = 0.0
    # Start time
    start_time: float = field(default_factory=time.time)


def create_tree_view(state: InteractiveState) -> Tree:
    """Create a rich Tree showing the exploration state."""
    tree = Tree("[bold blue]PSS Exploration[/bold blue]")

    with state.lock:
        if not state.tree.contexts:
            tree.add("[dim]No contexts yet[/dim]")
            return tree

        # Build tree structure
        context_nodes: dict[str, Any] = {}

        # Find root(s) - contexts with no parent
        roots = [c for c in state.tree.contexts.values() if c.parent_id is None]

        for root in roots:
            root_node = _add_context_to_tree(tree, root, state)
            context_nodes[root.id] = root_node
            _add_children(root_node, root.id, state, context_nodes)

    return tree


def _add_context_to_tree(parent_node: Tree, ctx: Context, state: InteractiveState) -> Tree:
    """Add a context node to the tree."""
    # Status indicator
    if ctx.id in state.killed:
        status = "[red]✗ KILLED[/red]"
    elif ctx.status == "terminated":
        if ctx.output:
            status = "[green]✓ LEAF[/green]"
        else:
            status = "[dim]terminated[/dim]"
    elif ctx.status == "branched":
        status = "[yellow]⑂ branched[/yellow]"
    elif ctx.status == "running":
        status = "[cyan]● running[/cyan]"
    else:
        status = f"[dim]{ctx.status}[/dim]"

    # Context info
    tokens = f"{ctx.token_count:,} tok"
    cost = f"${ctx.usage.cost:.4f}" if ctx.usage.cost > 0 else ""

    # Truncate branch reason
    reason = ""
    if ctx.branch_reason:
        reason = f" - {ctx.branch_reason[:40]}..." if len(ctx.branch_reason) > 40 else f" - {ctx.branch_reason}"

    # Promoted indicator
    promoted = " [magenta]★[/magenta]" if ctx.id in state.promoted else ""

    label = f"{status} [bold]{ctx.id}[/bold]{promoted} ({tokens}{', ' + cost if cost else ''}){reason}"

    return parent_node.add(label)


def _add_children(parent_node: Tree, parent_id: str, state: InteractiveState, context_nodes: dict[str, Any]) -> None:
    """Recursively add child contexts to the tree."""
    children = [c for c in state.tree.contexts.values() if c.parent_id == parent_id]

    for child in children:
        child_node = _add_context_to_tree(parent_node, child, state)
        context_nodes[child.id] = child_node
        _add_children(child_node, child.id, state, context_nodes)


def create_status_panel(state: InteractiveState) -> Panel:
    """Create a status panel showing run stats."""
    with state.lock:
        contexts = state.tree.contexts
        running = sum(1 for c in contexts.values() if c.status == "running" and c.id not in state.killed)
        terminated = sum(1 for c in contexts.values() if c.status == "terminated")
        leaves = len(state.tree.leaves)
        killed = len(state.killed)
        elapsed = time.time() - state.start_time

        status = "PAUSED" if state.paused else "RUNNING"
        status_style = "yellow bold" if state.paused else "green bold"

        table = Table.grid(padding=(0, 2))
        table.add_column(style="dim")
        table.add_column()

        table.add_row("Status:", f"[{status_style}]{status}[/{status_style}]")
        table.add_row("Contexts:", f"{len(contexts)} total ({running} running, {terminated} terminated, {killed} killed)")
        table.add_row("Leaves:", str(leaves))
        table.add_row("Cost:", f"${state.total_cost:.4f}")
        table.add_row("Elapsed:", f"{elapsed:.1f}s")

        # Recent status messages
        if state.status_messages:
            table.add_row("", "")
            for msg in state.status_messages[-3:]:
                table.add_row("", f"[dim]{msg}[/dim]")

    return Panel(table, title="[bold]Status[/bold]", border_style="blue")


def create_help_panel() -> Panel:
    """Create a help panel showing available commands."""
    help_text = """[bold]Commands:[/bold]
  [cyan]k <id>[/cyan]  Kill a branch (e.g., k root→abc123)
  [cyan]i <id> <msg>[/cyan]  Inject guidance into a branch
  [cyan]p <id>[/cyan]  Promote - spawn more branches like this one
  [cyan]pause[/cyan]   Pause exploration
  [cyan]resume[/cyan]  Resume exploration
  [cyan]status[/cyan]  Show detailed status
  [cyan]q[/cyan]       Quit (abort exploration)
  [cyan]?[/cyan]       Show this help"""

    return Panel(help_text, title="[bold]Help[/bold]", border_style="dim")


def create_layout(state: InteractiveState) -> Layout:
    """Create the full terminal layout."""
    layout = Layout()

    layout.split_column(
        Layout(name="tree", ratio=3),
        Layout(name="bottom", ratio=1),
    )

    layout["bottom"].split_row(
        Layout(name="status", ratio=2),
        Layout(name="help", ratio=1),
    )

    layout["tree"].update(Panel(create_tree_view(state), title="[bold]Exploration Tree[/bold]", border_style="green"))
    layout["status"].update(create_status_panel(state))
    layout["help"].update(create_help_panel())

    return layout


def parse_command(input_str: str) -> Command | None:
    """Parse a command string into a Command object."""
    input_str = input_str.strip()
    if not input_str:
        return None

    parts = input_str.split(maxsplit=2)
    cmd = parts[0].lower()

    if cmd in ("q", "quit", "exit"):
        return Command(type=CommandType.QUIT)
    elif cmd in ("?", "help", "h"):
        return Command(type=CommandType.HELP)
    elif cmd in ("pause", "p") and len(parts) == 1:
        return Command(type=CommandType.PAUSE)
    elif cmd in ("resume", "r") and len(parts) == 1:
        return Command(type=CommandType.RESUME)
    elif cmd == "status":
        return Command(type=CommandType.STATUS)
    elif cmd in ("k", "kill"):
        if len(parts) < 2:
            return None
        return Command(type=CommandType.KILL, target=parts[1])
    elif cmd in ("i", "inject"):
        if len(parts) < 3:
            return None
        return Command(type=CommandType.INJECT, target=parts[1], payload=parts[2])
    elif cmd in ("p", "promote") and len(parts) >= 2:
        return Command(type=CommandType.PROMOTE, target=parts[1])

    return None


class InteractiveController:
    """Controller for interactive PSS sessions."""

    def __init__(self, config: PSSConfig):
        self.config = config
        self.state = InteractiveState()
        self.console = Console()
        self.command_queue: queue.Queue[Command] = queue.Queue()
        self._input_thread: threading.Thread | None = None
        self._stop_input = threading.Event()

    def on_context_update(self, ctx: Context) -> None:
        """Callback when a context is updated."""
        with self.state.lock:
            self.state.tree.contexts[ctx.id] = ctx
            self.state.total_cost = sum(c.usage.cost for c in self.state.tree.contexts.values())

    def on_leaf_added(self, ctx_id: str) -> None:
        """Callback when a leaf is added."""
        with self.state.lock:
            if ctx_id not in self.state.tree.leaves:
                self.state.tree.leaves.append(ctx_id)

    def on_branch_spawned(self, parent_id: str, child_id: str, reason: str) -> None:
        """Callback when a branch is spawned."""
        with self.state.lock:
            self.state.status_messages.append(f"Branch {child_id} spawned from {parent_id}")
            # Keep only last 10 messages
            if len(self.state.status_messages) > 10:
                self.state.status_messages = self.state.status_messages[-10:]

    def should_kill(self, ctx_id: str) -> bool:
        """Check if a context should be killed."""
        with self.state.lock:
            return ctx_id in self.state.killed

    def get_injection(self, ctx_id: str) -> str | None:
        """Get any injected message for a context."""
        with self.state.lock:
            return self.state.injections.pop(ctx_id, None)

    def should_promote(self, ctx_id: str) -> bool:
        """Check if a context should be promoted (spawn more like it)."""
        with self.state.lock:
            if ctx_id in self.state.promoted:
                self.state.promoted.remove(ctx_id)
                return True
            return False

    def is_paused(self) -> bool:
        """Check if exploration is paused."""
        with self.state.lock:
            return self.state.paused

    def wait_while_paused(self) -> bool:
        """Block while paused. Returns False if quit requested."""
        while True:
            with self.state.lock:
                if self.state.quit_requested:
                    return False
                if not self.state.paused:
                    return True
            time.sleep(0.1)

    def is_quit_requested(self) -> bool:
        """Check if quit was requested."""
        with self.state.lock:
            return self.state.quit_requested

    def process_command(self, cmd: Command) -> str:
        """Process a command and return status message."""
        with self.state.lock:
            if cmd.type == CommandType.QUIT:
                self.state.quit_requested = True
                return "Quit requested - stopping exploration..."

            elif cmd.type == CommandType.PAUSE:
                self.state.paused = True
                return "Exploration paused"

            elif cmd.type == CommandType.RESUME:
                self.state.paused = False
                return "Exploration resumed"

            elif cmd.type == CommandType.KILL:
                if cmd.target:
                    # Support partial matching
                    matches = [cid for cid in self.state.tree.contexts.keys()
                              if cmd.target in cid]
                    if len(matches) == 1:
                        self.state.killed.add(matches[0])
                        return f"Killed context {matches[0]}"
                    elif len(matches) > 1:
                        return f"Ambiguous: matches {matches}"
                    else:
                        return f"No context matching '{cmd.target}'"
                return "Kill requires a context ID"

            elif cmd.type == CommandType.INJECT:
                if cmd.target and cmd.payload:
                    matches = [cid for cid in self.state.tree.contexts.keys()
                              if cmd.target in cid and self.state.tree.contexts[cid].status == "running"]
                    if len(matches) == 1:
                        self.state.injections[matches[0]] = cmd.payload
                        return f"Injected into {matches[0]}: {cmd.payload[:30]}..."
                    elif len(matches) > 1:
                        return f"Ambiguous: matches {matches}"
                    else:
                        return f"No running context matching '{cmd.target}'"
                return "Inject requires context ID and message"

            elif cmd.type == CommandType.PROMOTE:
                if cmd.target:
                    matches = [cid for cid in self.state.tree.contexts.keys()
                              if cmd.target in cid]
                    if len(matches) == 1:
                        self.state.promoted.add(matches[0])
                        return f"Promoted context {matches[0]} - will spawn similar branches"
                    elif len(matches) > 1:
                        return f"Ambiguous: matches {matches}"
                    else:
                        return f"No context matching '{cmd.target}'"
                return "Promote requires a context ID"

            elif cmd.type == CommandType.HELP:
                return "Help shown in panel"

            elif cmd.type == CommandType.STATUS:
                contexts = self.state.tree.contexts
                running = [c for c in contexts.values() if c.status == "running"]
                return f"Running: {[c.id for c in running]}"

        return "Unknown command"

    def _input_loop(self, live: Live) -> None:
        """Background thread for reading user input."""
        while not self._stop_input.is_set():
            try:
                # Non-blocking input with timeout
                import sys
                import select

                # On Windows, use msvcrt
                if sys.platform == "win32":
                    import msvcrt
                    if msvcrt.kbhit():
                        # Read a line
                        line = ""
                        while True:
                            if msvcrt.kbhit():
                                char = msvcrt.getwch()
                                if char == '\r':
                                    break
                                elif char == '\x03':  # Ctrl+C
                                    self.command_queue.put(Command(type=CommandType.QUIT))
                                    return
                                line += char
                        if line:
                            cmd = parse_command(line)
                            if cmd:
                                self.command_queue.put(cmd)
                else:
                    # Unix - use select
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if rlist:
                        line = sys.stdin.readline().strip()
                        if line:
                            cmd = parse_command(line)
                            if cmd:
                                self.command_queue.put(cmd)

                time.sleep(0.1)
            except Exception:
                pass

    def run_display(self, run_exploration: Callable[[], None]) -> None:
        """Run the interactive display with exploration in background."""
        self._stop_input.clear()

        with Live(create_layout(self.state), console=self.console, refresh_per_second=4) as live:
            # Start input thread
            self._input_thread = threading.Thread(target=self._input_loop, args=(live,), daemon=True)
            self._input_thread.start()

            # Run exploration in background thread
            exploration_thread = threading.Thread(target=run_exploration, daemon=True)
            exploration_thread.start()

            # Main display loop
            while exploration_thread.is_alive() or not self.command_queue.empty():
                # Process any pending commands
                try:
                    while True:
                        cmd = self.command_queue.get_nowait()
                        msg = self.process_command(cmd)
                        with self.state.lock:
                            self.state.status_messages.append(msg)
                        if cmd.type == CommandType.QUIT:
                            break
                except queue.Empty:
                    pass

                # Update display
                live.update(create_layout(self.state))

                # Check for quit
                if self.is_quit_requested():
                    break

                time.sleep(0.1)

            self._stop_input.set()

        # Final summary
        self.console.print("\n[bold]Exploration Complete[/bold]")
        self.console.print(create_tree_view(self.state))
        with self.state.lock:
            leaves = len(self.state.tree.leaves)
            cost = self.state.total_cost
        self.console.print(f"\n[green]Leaves: {leaves}[/green] | [cyan]Cost: ${cost:.4f}[/cyan]")


def run_interactive(
    initial_prompt: str,
    config: PSSConfig,
    provider: "Provider",
) -> list[Context]:
    """
    Run PSS in interactive mode with live tree view and commands.

    Args:
        initial_prompt: The prompt to explore
        config: PSS configuration
        provider: LLM provider

    Returns:
        List of leaf contexts
    """
    from pss.harness import run_pss

    controller = InteractiveController(config)
    result_holder: list[Context] = []

    def exploration():
        # Monkey-patch the harness to use our callbacks
        result = run_pss(
            initial_prompt,
            config,
            provider,
            on_context_update=controller.on_context_update,
        )

        # Handle both list and PSSResult return types
        if isinstance(result, list):
            result_holder.extend(result)
        else:
            result_holder.extend(result.leaves)

    controller.run_display(exploration)

    return result_holder
