import threading

import httpx
import pytest

from pss.web.gpu_runtime import gpu_key, local_url, gpu_ready
from pss.web.workbench_provider import WorkbenchProvider


def test_custom_loopback_port_and_key_file(monkeypatch, tmp_path):
    keyfile = tmp_path / 'key.txt'
    keyfile.write_text('synthetic-fixture-key\n')
    monkeypatch.setenv('PSS_LOCAL_KEY_FILE', str(keyfile))
    monkeypatch.setenv('PSS_LOCAL_URL', 'http://127.0.0.1:18417/')
    requests = []

    def respond(request):
        requests.append(request)
        assert str(request.url) == 'http://127.0.0.1:18417/v1/chat/completions'
        assert request.headers['authorization'] == 'Bearer synthetic-fixture-key'
        return httpx.Response(200, json={'choices': [{'message': {'content': 'One direction'}}]})

    provider = WorkbenchProvider('local-test', gpu_key(), threading.Event(), httpx.MockTransport(respond), backend='llamacpp')
    try:
        assert provider.chat([{'role': 'user', 'content': 'Explore'}])[0] == 'One direction'
        assert len(requests) == 1
    finally:
        provider.close()


@pytest.mark.parametrize('url', [
    'http://example.com', 'http://192.168.1.2:4317', 'https://127.0.0.1',
    'http://user:pass@localhost', 'http://127.0.0.1/v1',
    'http://127.0.0.1?secret=x', 'http://localhost#fragment', 'http://localhost:bad',
])
def test_invalid_local_origin_rejected_before_request(monkeypatch, url):
    monkeypatch.setenv('PSS_LOCAL_URL', url)
    with pytest.raises(ValueError):
        local_url()


def test_explicit_missing_key_keeps_live_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv('PSS_LOCAL_KEY_FILE', str(tmp_path / 'missing.txt'))
    assert not gpu_ready()


def test_unauthenticated_local_transport_sends_no_dummy_key(monkeypatch):
    monkeypatch.setenv('PSS_LOCAL_URL', 'http://127.0.0.1:18417')
    def respond(request):
        assert 'authorization' not in request.headers
        return httpx.Response(200, json={'choices': [{'message': {'content': 'Local thought'}}]})
    provider = WorkbenchProvider('local-test', '', threading.Event(), httpx.MockTransport(respond), backend='llamacpp')
    try:
        assert provider.chat([{'role': 'user', 'content': 'Explore'}])[0] == 'Local thought'
    finally:
        provider.close()


def test_console_launch_reads_working_directory_env_without_overriding_shell(monkeypatch, tmp_path):
    import os
    import uvicorn
    from pss.web import server
    (tmp_path / '.env').write_text('PSS_MODEL=portable-fixture\nPSS_ENABLE_LIVE=true\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('PSS_MODEL', raising=False)
    monkeypatch.setenv('PSS_ENABLE_LIVE', 'false')
    monkeypatch.setattr('sys.argv', ['pss-web'])
    seen = []
    monkeypatch.setattr(uvicorn, 'run', lambda *args, **kwargs: seen.append((os.getenv('PSS_MODEL'), os.getenv('PSS_ENABLE_LIVE'))))
    server.main()
    assert seen == [('portable-fixture', 'false')]
