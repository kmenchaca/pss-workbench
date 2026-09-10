import json
import threading

import httpx

from pss.web.workbench_provider import WorkbenchProvider
from pss.web.runtime import Runtime


def test_local_transport_never_sends_key_or_cloud_request():
    def respond(request):
        assert str(request.url) == "http://127.0.0.1:11434/v1/chat/completions"
        assert "authorization" not in request.headers
        payload = json.loads(request.content)
        assert payload["reasoning_effort"] == "none"
        assert payload["max_tokens"] == 192
        return httpx.Response(200, json={"choices": [{"message": {"content": "Local thought"}}]})
    provider = WorkbenchProvider("qwen3.5:27b", "must-not-leave", threading.Event(), httpx.MockTransport(respond), backend="ollama")
    try:
        assert provider.chat([{"role": "user", "content": "Hello"}])[0] == "Local thought"
    finally:
        provider.close()


def test_gpu_adapter_has_one_pinned_endpoint_and_no_fallback():
    def respond(request):
        assert str(request.url) == "http://127.0.0.1:4317/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer private-test-key"
        payload = json.loads(request.content)
        assert payload["reasoning_effort"] == "none"
        assert payload["max_tokens"] == 768
        return httpx.Response(200, json={"choices": [{"message": {"content": "GPU thought"}}]})
    provider = WorkbenchProvider("pss-gpu", "private-test-key", threading.Event(), httpx.MockTransport(respond), backend="llamacpp")
    try:
        assert provider.chat([{"role": "user", "content": "Hello"}])[0] == "GPU thought"
    finally:
        provider.close()


def test_missing_gpu_does_not_enable_ollama_or_cloud(monkeypatch):
    import pss.web.runtime as module
    monkeypatch.setenv("PSS_PROVIDER", "llamacpp")
    monkeypatch.setenv("PSS_ENABLE_LIVE", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "present-but-must-not-be-used")
    monkeypatch.setattr(module, "gpu_ready", lambda: False)
    settings=Runtime().settings()
    assert settings["gpu_only"]
    assert not settings["live_enabled"]
    assert "No CPU or cloud fallback" in settings["notice"]
