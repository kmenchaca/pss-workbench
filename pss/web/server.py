"""Loopback-only, same-origin PSS workbench. Live spending is explicitly opt-in."""

from __future__ import annotations

import asyncio
import hmac
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pss.interactive import Command, CommandType
from pss.web.models import CommandRequest, CommandResponse, RunRequest
from pss.web.runtime import Runtime

STATIC_DIR = Path(__file__).parent / "static"
runtime = Runtime()
csrf_token = secrets.token_urlsafe(32)


def allowed_origin(url, headers) -> bool:
    if url.hostname not in ("127.0.0.1", "localhost", "::1"):
        return False
    origin = headers.get("origin")
    if headers.get("sec-fetch-site") == "cross-site":
        return False
    if origin:
        try:
            parsed = urlsplit(origin)
        except ValueError:
            return False
        expected_scheme = "http" if url.scheme in ("http", "ws") else "https"
        if parsed.scheme != expected_scheme or parsed.netloc != url.netloc:
            return False
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if runtime.active:
        runtime.stop.set()
        runtime.controller.process_command(Command(CommandType.QUIT))


app = FastAPI(
    title="Pepe Silvia Search",
    version="0.9.1",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)


@app.middleware("http")
async def guard(request: Request, call_next):
    if not allowed_origin(request.url, request.headers):
        return JSONResponse(
            {"detail": "This app is for same-origin loopback use only."},
            status_code=403,
        )
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        token = request.headers.get("x-pss-token", "")
        if not hmac.compare_digest(token.encode("utf-8"), csrf_token.encode("ascii")):
            return JSONResponse(
                {"detail": "Refresh the page before making changes."}, status_code=403
            )
        try:
            if int(request.headers.get("content-length", "0")) > 32_768:
                return JSONResponse({"detail": "Request too large."}, status_code=413)
        except ValueError:
            return JSONResponse({"detail": "Invalid content length."}, status_code=400)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 32_768:
                return JSONResponse({"detail": "Request too large."}, status_code=413)
        request._body = bytes(body)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; font-src 'self'; connect-src 'self'; "
        "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
    )
    return response


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "product.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/api/config")
async def config():
    return {**runtime.settings(), "csrf_token": csrf_token}


@app.get("/api/state")
async def state():
    return runtime.state()


@app.post("/api/run")
async def start_run(request: RunRequest):
    prompt = request.prompt.strip()
    if not prompt or len(prompt) > 6000:
        raise HTTPException(422, "Use a prompt between 1 and 6,000 characters.")
    if request.preset not in (None, "explore"):
        raise HTTPException(422, "Only text exploration is supported by the workbench.")
    overrides = request.config_overrides or {}
    if set(overrides) - {"max_contexts"}:
        raise HTTPException(422, "Only the branch limit can be changed here.")
    cap = overrides.get("max_contexts", 6)
    if type(cap) is not int or cap < 2 or cap > 6:
        raise HTTPException(422, "Choose 2–6 total contexts, including the root.")
    try:
        runtime.start(prompt, cap)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"success": True, "session_id": runtime.session_id}


@app.post("/api/command", response_model=CommandResponse)
async def command(request: CommandRequest):
    if len(request.payload or "") > 2000 or len(request.target or "") > 500:
        raise HTTPException(422, "That command is too long.")
    try:
        message = runtime.command(request.type, request.target, request.payload)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return CommandResponse(success=True, message=message)


@app.get("/api/context/{ctx_id}")
async def detail(ctx_id: str, session_id: str | None = None):
    if session_id is not None and session_id != runtime.session_id:
        raise HTTPException(409, "That exploration is no longer the server's active session.")
    result = runtime.controller.get_context_detail(ctx_id)
    if result is None:
        raise HTTPException(404, "Branch not found.")
    return {**result.model_dump(), "session_id": runtime.session_id}


@app.websocket("/ws")
async def websocket(websocket: WebSocket):
    protocols = [
        p.strip()
        for p in websocket.headers.get("sec-websocket-protocol", "").split(",")
    ]
    expected = f"pss.{csrf_token}"
    if (
        not allowed_origin(websocket.url, websocket.headers)
        or expected not in protocols
    ):
        await websocket.close(code=1008)
        return
    await websocket.accept(subprotocol=expected)
    try:
        while True:
            await websocket.send_json(runtime.state())
            await asyncio.sleep(0.5)
    except (WebSocketDisconnect, RuntimeError):
        pass


def main():
    import argparse

    import uvicorn
    from dotenv import load_dotenv

    # The installed console script must honor configuration in the launch folder,
    # not only files next to the source or inside site-packages.
    load_dotenv(Path.cwd() / ".env", override=False)

    parser = argparse.ArgumentParser(description="Local PSS workbench")
    parser.add_argument(
        "--host", choices=["127.0.0.1", "localhost", "::1"], default="127.0.0.1"
    )
    parser.add_argument("--port", type=int, default=4311)
    args = parser.parse_args()
    print(f"Pepe Silvia Search: http://{args.host}:{args.port}")
    print(
        "Guided examples work offline. Live requests require explicit local configuration."
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
