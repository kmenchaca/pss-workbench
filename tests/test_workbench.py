"""Release safety and integration checks; all provider traffic is mocked."""

import json
import threading

import httpx
import pytest
from fastapi.testclient import TestClient

from pss.types import Context, TerminationReason
from pss.web import server
from pss.web.runtime import Runtime, safe_config
from pss.web.workbench_provider import (
    ProviderFailure,
    RunLimitReached,
    WorkbenchProvider,
)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("PSS_PROVIDER", "openrouter")
    monkeypatch.delenv("PSS_ENABLE_LIVE", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(server, "runtime", Runtime())
    with TestClient(server.app, base_url="http://127.0.0.1:4311") as instance:
        instance.headers["x-pss-token"] = instance.get("/api/config").json()[
            "csrf_token"
        ]
        yield instance


def test_default_offline_and_headers(client):
    response = client.get("/api/config")
    assert not response.json()["live_enabled"]
    assert response.headers["x-frame-options"] == "DENY"
    assert "script-src 'self'" in response.headers["content-security-policy"]
    assert client.post("/api/run", json={"prompt": "Hello"}).status_code == 403
    assert client.get("/api/state").json()["outcome"] == "idle"


@pytest.mark.parametrize(
    "headers",
    [
        {"Host": "attacker.test"},
        {"Origin": "https://attacker.test"},
        {"Origin": "http://127.0.0.1:9999"},
        {"Sec-Fetch-Site": "cross-site"},
    ],
)
def test_rejects_cross_origin(client, headers):
    assert client.get("/api/config", headers=headers).status_code == 403
    assert (
        client.get("/api/config", headers={"Origin": "http://[invalid"}).status_code
        == 403
    )


def test_mutation_token_and_size(client):
    assert (
        client.post(
            "/api/run", json={"prompt": "x"}, headers=[(b"x-pss-token", b"\xff")]
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/run", json={"prompt": "x"}, headers={"x-pss-token": "bad"}
        ).status_code
        == 403
    )
    assert client.post("/api/run", content=b"x" * 33000).status_code == 413
    assert (
        client.post("/api/run", content=iter([b"x" * 17000, b"y" * 17000])).status_code
        == 413
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": " "},
        {"prompt": "x" * 6001},
        {"prompt": "x", "preset": "agentic"},
        {"prompt": "x", "config_overrides": {"provider": "other"}},
        {"prompt": "x", "config_overrides": {"max_contexts": True}},
        {"prompt": "x", "config_overrides": {"max_contexts": 7}},
    ],
)
def test_run_validation(client, payload):
    assert client.post("/api/run", json=payload).status_code == 422


def test_missing_detail_command_and_traversal(client):
    assert client.get("/api/context/missing").status_code == 404
    assert (
        client.post(
            "/api/command", json={"type": "inject", "target": "missing"}
        ).status_code
        == 422
    )
    assert client.post("/api/command", json={"type": "quit"}).status_code == 422
    assert client.get("/static/%2e%2e/%2e%2e/pyproject.toml").status_code == 404


def test_websocket_requires_secret(client):
    from starlette.websockets import WebSocketDisconnect

    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect("ws://127.0.0.1:4311/ws"),
    ):
        pass
    with client.websocket_connect(
        "ws://127.0.0.1:4311/ws", subprotocols=[f"pss.{server.csrf_token}"]
    ) as ws:
        assert ws.receive_json()["outcome"] == "idle"


def test_details_preserve_messages_and_root():
    runtime = Runtime()
    ctx = Context(
        "root",
        None,
        messages=[
            {"role": "assistant", "content": "x" * 5000},
            {"role": "tool", "content": None},
        ],
        status="terminated",
        termination_reason=TerminationReason.BUDGET,
    )
    runtime.controller.on_context_update(ctx)
    assert runtime.state()["tree"]["root_id"] == "root"
    detail = runtime.controller.get_context_detail("root")
    assert len(detail.messages[0]["content"]) == 5000
    assert detail.messages[1]["content"] == ""
    assert detail.termination_reason == "budget"


def test_safe_config():
    config = safe_config()
    assert config.max_contexts == 6 and not config.parallel
    assert (
        not config.agentic_enabled
        and not config.diversity_enabled
        and not config.synthesis_enabled
    )


def provider(handler):
    return WorkbenchProvider(
        "test-model", "test-key", threading.Event(), httpx.MockTransport(handler)
    )


def test_provider_protocol_and_usage():
    def respond(request):
        assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
        payload = json.loads(request.content)
        assert payload["max_tokens"] == 768
        assert payload["tool_choice"]["function"]["name"] == "checkpoint_decision"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "Notes",
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "function": {
                                        "name": "checkpoint_decision",
                                        "arguments": '{"action":"continue"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 13, "completion_tokens": 7},
            },
        )

    p = provider(respond)
    try:
        text, calls, tokens, usage = p.chat(
            [], [{"name": "checkpoint_decision", "input_schema": {}}], "required"
        )
        assert text == "Notes" and calls[0]["arguments"]["action"] == "continue"
        assert tokens == 20 and usage.cost == 0 and p.calls == 1
    finally:
        p.close()


@pytest.mark.parametrize("status", [401, 402, 403, 429, 500])
def test_provider_errors_sanitized_no_retry(status):
    p = provider(lambda request: httpx.Response(status, text="SECRET test-key"))
    try:
        with pytest.raises(ProviderFailure) as error:
            p.chat([])
        assert "SECRET" not in str(error.value) and "test-key" not in str(error.value)
        assert p.calls == 1
    finally:
        p.close()


def test_limits_prevent_dispatch():
    def forbidden(request):
        raise AssertionError("No dispatch permitted")

    p = provider(forbidden)
    try:
        p.stop.set()
        with pytest.raises(RunLimitReached):
            p.chat([])
        p.stop.clear()
        with pytest.raises(RunLimitReached):
            p.chat([{"role": "user", "content": "x" * 49000}])
        with pytest.raises(ProviderFailure):
            p.chat([], [{"name": "shell"}])
        p.calls = 18
        with pytest.raises(RunLimitReached):
            p.chat([])
        p.calls = 0
        p.started -= 301
        with pytest.raises(RunLimitReached):
            p.chat([])
    finally:
        p.close()


def test_runtime_real_harness_with_mock_provider(monkeypatch):
    """Actual PSS gates/branches execute through our bounded provider without networking."""
    import pss.web.runtime as module
    monkeypatch.setenv("PSS_PROVIDER", "openrouter")

    def respond(request):
        payload = json.loads(request.content)
        if payload.get("tools"):
            decision = {
                "action": "branch",
                "branches": ["Try cooperation", "Try solitude"],
                "also_continue": False,
            }
            message = {
                "content": "Compare two directions.",
                "tool_calls": [
                    {
                        "id": "gate",
                        "function": {
                            "name": "checkpoint_decision",
                            "arguments": json.dumps(decision),
                        },
                    }
                ],
            }
        else:
            message = {"content": "A concrete alternative with a tradeoff."}
        return httpx.Response(
            200,
            json={
                "choices": [{"message": message}],
                "usage": {"prompt_tokens": 350, "completion_tokens": 350},
            },
        )

    monkeypatch.setattr(
        module,
        "WorkbenchProvider",
        lambda model, key, stop, **kwargs: WorkbenchProvider(
            model, key, stop, httpx.MockTransport(respond), **kwargs
        ),
    )
    # settings references class constants; keep that on the real class while starting.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    runtime = Runtime()
    runtime.prompt = "Explore a fictional village"
    runtime.controller.set_exploration_active(True)
    runtime._run(runtime.controller, runtime.stop)
    state = runtime.state()
    assert state["outcome"] in ("complete", "stopped"), state["message"]
    assert state["request_count"] <= 18 and state["request_count"] > 1
    assert state["tree"]["root_id"] == "root"
    assert 3 <= len(state["tree"]["contexts"]) <= 6
    assert not state["exploration_active"]
    assert all(c["status"] != "running" for c in state["tree"]["contexts"])
