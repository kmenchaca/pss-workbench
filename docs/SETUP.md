# Model setup

The default is offline. Copy `.env.example` to `.env` in the repository and restart `uv run pss-web` after changes. Keep credentials on the server and out of Git. Environment variables already set in your shell take precedence over `.env`.

## OpenRouter

```dotenv
PSS_ENABLE_LIVE=true
PSS_PROVIDER=openrouter
PSS_MODEL=your-selected-tool-capable-model
OPENROUTER_API_KEY=your-own-key
```

Choose a currently available model that supports function/tool calls. PSS uses `checkpoint_decision` calls to control branching. Requests go to OpenRouter and its selected provider; do not send private material without considering that boundary. Set a provider-side spending limit. Adapter regression tests use mocked responses; the public-release verification did not spend money or validate every available model.

## Ollama

With an existing tool-capable model served at `http://127.0.0.1:11434`:

```dotenv
PSS_ENABLE_LIVE=true
PSS_PROVIDER=ollama
PSS_MODEL=your-installed-model
```

PSS does not install, start, or download models. This adapter limits each response to 192 tokens; checkpoint quality depends on the model. Configuration alone does not establish readiness, and a failed request is shown in the workbench. This adapter is regression-tested with mocks, not a compatibility promise for every Ollama model.

## Local llama.cpp server

Use an existing tool-capable llama.cpp server with `/health` and `/v1/chat/completions` endpoints:

```dotenv
PSS_ENABLE_LIVE=true
PSS_PROVIDER=llamacpp
PSS_MODEL=your-server-model-alias
PSS_LOCAL_URL=http://127.0.0.1:4317
# If your server requires authentication:
# PSS_LOCAL_KEY_FILE=/absolute/path/to/your/server-key.txt
```

On Windows, a path such as `C:/models/server-key.txt` is accepted. The optional key file contains only the existing server's API key. If a key file is explicitly configured but unreadable or empty, live mode stays disabled. Without a configured key, the server must allow unauthenticated loopback requests. The URL must be an HTTP loopback origin, with no path, credentials or query. Custom loopback ports are supported. PSS never redirects or falls back to a cloud service.

Hardware allocation, chat templates and GPU offload are configured in your model server, not by PSS. No particular graphics card is required by the workbench. The recorded example used a local Qwen model on an RTX 3090; this establishes one working configuration, not universal model compatibility.

## Common problems

- **Port already in use:** run `uv run pss-web --port 4321` and open that port. Do not stop an unrelated service.
- **Model setup stays disabled:** check the provider name, explicit `PSS_ENABLE_LIVE=true`, key configuration, and restart the workbench.
- **A run stops at a limit:** inspect and keep its partial thoughts, or start a smaller follow-up question. No automatic retry is made.
- **Missing previous work:** browser storage belongs to the exact browser and origin, including port. Import your JSON backup at a new origin.
