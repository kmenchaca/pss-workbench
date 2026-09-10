"""Read-only connection to the explicitly launched PSS GPU runner; never falls back."""
from pathlib import Path
import os
from urllib.parse import urlsplit

import httpx

PRIVATE_DIR = Path(__file__).resolve().parents[2] / "private-runtime"


def local_url() -> str:
    """Only an explicitly configured loopback runner; never follow redirects."""
    url = os.getenv("PSS_LOCAL_URL", "http://127.0.0.1:4317").rstrip("/")
    parsed = urlsplit(url)
    if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
            or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment):
        raise ValueError("PSS_LOCAL_URL must be an HTTP loopback origin without a path or credentials.")
    _ = parsed.port  # Reject malformed ports before dispatch.
    return url


def gpu_key() -> str:
    try:
        path = Path(os.getenv("PSS_LOCAL_KEY_FILE", str(PRIVATE_DIR / "api-key.txt")))
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def gpu_ready() -> bool:
    if os.getenv("PSS_LOCAL_KEY_FILE") and not gpu_key():
        return False
    try:
        with httpx.Client(timeout=1.0, trust_env=False, follow_redirects=False) as client:
            headers = {"Authorization": f"Bearer {gpu_key()}"} if gpu_key() else {}
            response = client.get(f"{local_url()}/health", headers=headers)
            return response.status_code == 200 and response.json().get("status") == "ok"
    except (httpx.HTTPError, ValueError):
        return False
