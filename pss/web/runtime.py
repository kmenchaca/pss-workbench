"""Single-session runtime; source histories stay untouched and no state hits disk."""

from __future__ import annotations

import os
import threading
import time
from typing import Any

from pss.config import PSSConfig
from pss.interactive import Command, CommandType
from pss.types import TerminationReason
from pss.web.controller import WebController
from pss.web.gpu_runtime import gpu_key, gpu_ready
from pss.web.workbench_provider import (
    ProviderFailure,
    RunLimitReached,
    WorkbenchProvider,
)


def safe_config(max_contexts: int = 6) -> PSSConfig:
    return PSSConfig(
        provider="openrouter",
        model=os.getenv("PSS_MODEL", "qwen3.5:27b" if os.getenv("PSS_PROVIDER") == "ollama" else "meta-llama/llama-3.1-8b-instruct"),
        soft_gate_tokens=150 if os.getenv("PSS_PROVIDER") in ("ollama", "llamacpp") else 600,
        hard_gate_tokens=400 if os.getenv("PSS_PROVIDER") in ("ollama", "llamacpp") else 1400,
        per_context_max=4000,
        total_max=20_000,
        max_contexts=max_contexts,
        parallel=False,
        max_parallel=1,
        diversity_enabled=False,
        diversity_provider="none",
        agentic_enabled=False,
        synthesis_enabled=False,
        timeout_seconds=300,
        system_prompt=(
            "Choose ONE concrete approach to the user's question and develop it. "
            "Do not list multiple approaches. Write a short title, the core idea, "
            "one specific example, and the most important tradeoff. Keep your entire "
            "response under 90 words. Treat proposals as possibilities, not facts."
        ),
    )


class Runtime:
    def __init__(self):
        self.lock = threading.Lock()
        self.controller = WebController(safe_config())
        self.thread: threading.Thread | None = None
        self.provider: WorkbenchProvider | None = None
        self.stop = threading.Event()
        self.prompt = ""
        self.session_id = ""
        self.outcome = "idle"
        self.message = ""
        self.started = 0.0
        self.finished = 0.0

    @property
    def active(self):
        return self.thread is not None and self.thread.is_alive()

    def settings(self) -> dict[str, Any]:
        backend = os.getenv("PSS_PROVIDER", "openrouter")
        gpu = backend == "llamacpp"
        local = backend in ("ollama", "llamacpp")
        configured = gpu_ready() if gpu else backend == "ollama" or (backend == "openrouter" and bool(os.getenv("OPENROUTER_API_KEY")))
        enabled = os.getenv("PSS_ENABLE_LIVE", "").lower() == "true"
        return {
            "live_enabled": configured and enabled,
            "provider": "llama.cpp (local)" if gpu else "Ollama (local)" if local else "OpenRouter",
            "gpu_only": gpu,
            "local": local,
            "model": safe_config().model,
            "max_calls": WorkbenchProvider.MAX_CALLS,
            "max_output_tokens_per_call": WorkbenchProvider.LOCAL_OUTPUT_TOKENS if backend == "ollama" else WorkbenchProvider.MAX_OUTPUT_TOKENS,
            "max_seconds": WorkbenchProvider.MAX_SECONDS,
            "notice": "Runs through your configured local llama.cpp server. No CPU or cloud fallback is selected by PSS. Hardware allocation belongs to your model server. No provider charges." if gpu else "Runs on this computer through Ollama. No cloud fallback or provider charges. The first response may take longer while the model loads." if local else (
                "Live runs send your prompt and branch conversation to OpenRouter. "
                "Provider billing includes input tokens. These are request/output limits, "
                "not a dollar-price guarantee. Set a provider-side spending cap."
            ),
            "setup": "Set PSS_LOCAL_URL and PSS_LOCAL_KEY_FILE for your existing authenticated llama.cpp server, then restart PSS. See docs/SETUP.md." if gpu else (
                "For local runs, set PSS_PROVIDER=ollama, PSS_MODEL to an installed model, "
                "and PSS_ENABLE_LIVE=true in this project's .env, then restart. "
                "Cloud runs instead require OPENROUTER_API_KEY. Examples need neither."
            ),
        }

    def start(self, prompt: str, max_contexts: int):
        import uuid

        with self.lock:
            if self.active:
                raise RuntimeError(
                    "An exploration is still running. Stop it before starting another."
                )
            if not self.settings()["live_enabled"]:
                raise PermissionError(
                    "Live mode is not enabled. Guided examples are available without a key."
                )
            self.controller = WebController(safe_config(max_contexts))
            self.provider = None
            self.stop = threading.Event()
            self.prompt = prompt
            self.session_id = uuid.uuid4().hex
            self.outcome, self.message = "running", ""
            self.started, self.finished = time.monotonic(), 0.0
            # Capture controller and stop flag: a later session must never receive old callbacks.
            control, stop = self.controller, self.stop
            control.set_exploration_active(True)
            self.thread = threading.Thread(
                target=self._run, args=(control, stop), daemon=True
            )
            self.thread.start()

    def _run(self, control: WebController, stop: threading.Event):
        from pss.harness import run_pss

        provider = None
        try:
            provider = WorkbenchProvider(
                control.config.model,
                gpu_key() if os.getenv("PSS_PROVIDER") == "llamacpp" else os.getenv("OPENROUTER_API_KEY", ""), stop,
                backend=os.getenv("PSS_PROVIDER", "openrouter"),
            )
            self.provider = provider
            run_pss(
                self.prompt,
                control.config,
                provider,
                on_context_update=control.on_context_update,
                interactive_controller=control,
            )
            self.outcome = "stopped" if stop.is_set() else "complete"
            self.message = (
                "Stopped; partial branches are preserved."
                if stop.is_set()
                else "Exploration finished."
            )
        except RunLimitReached as exc:
            self.outcome, self.message = "stopped", str(exc)
        except ProviderFailure as exc:
            self.outcome, self.message = "error", str(exc)
        except Exception:  # noqa: BLE001 - final worker boundary must redact provider data.
            # Do not leak exception text that may contain provider responses or credentials.
            self.outcome, self.message = (
                "error",
                "The exploration could not finish. Your partial branches are preserved.",
            )
        finally:
            if provider:
                provider.close()
            with control.state.lock:
                for ctx in control.state.tree.contexts.values():
                    if ctx.status == "running":
                        ctx.status = "terminated"
                        ctx.termination_reason = TerminationReason.BUDGET
                control.state.status_messages.append(self.message)
            control.set_exploration_active(False)
            self.finished = time.monotonic()

    def state(self) -> dict:
        state = self.controller.get_serializable_state().model_dump()
        state.update(
            {
                "prompt": self.prompt,
                "session_id": self.session_id,
                "mode": "live",
                "outcome": self.outcome,
                "message": self.message,
                "request_count": self.provider.calls if self.provider else 0,
                "input_tokens": self.provider.input_tokens if self.provider else 0,
                "output_tokens": self.provider.output_tokens if self.provider else 0,
                "model": self.controller.config.model,
                "provider": "llama.cpp (local)" if os.getenv("PSS_PROVIDER") == "llamacpp" else os.getenv("PSS_PROVIDER", "openrouter"),
            }
        )
        state["status"]["elapsed_seconds"] = (
            (self.finished or time.monotonic()) - self.started if self.started else 0
        )
        return state

    def command(self, kind: str, target: str | None, payload: str | None) -> str:
        if kind in ("pause", "resume", "quit") and not self.active:
            raise ValueError("No exploration is running.")
        if kind in ("kill", "inject", "promote"):
            with self.controller.state.lock:
                ctx = self.controller.state.tree.contexts.get(target or "")
                if not ctx:
                    raise ValueError("Choose an existing branch.")
                if ctx.status != "running" and kind != "kill":
                    raise ValueError("Only a running branch can receive live guidance.")
                if kind == "inject" and not (payload or "").strip():
                    raise ValueError("Add guidance before sending.")
            if kind != "kill" and not self.active:
                raise ValueError(
                    "Start a new run to explore further; completed branches are not restarted."
                )
        if kind == "quit":
            self.stop.set()
        mapping = {
            "kill": CommandType.KILL,
            "inject": CommandType.INJECT,
            "promote": CommandType.PROMOTE,
            "pause": CommandType.PAUSE,
            "resume": CommandType.RESUME,
            "quit": CommandType.QUIT,
        }
        if kind not in mapping:
            raise ValueError("Unknown command.")
        return self.controller.process_command(Command(mapping[kind], target, payload))
