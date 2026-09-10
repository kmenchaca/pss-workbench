# Release verification

Verified September 10, 2026 on Windows with Python 3.11 and Node.js. These checks establish the listed behavior, not general model quality.

- Clean locked installation with `uv sync --frozen --extra web --extra dev`.
- Python suite: **1,427 passed, 1 skipped**. Two existing warnings remain: a helper class with a constructor is not a collected test class, and a dependency emits a deprecation warning.
- Frontend suite: **32 passed**.
- Source distribution and wheel build succeeded. The wheel was installed in a separate environment and launched outside the source tree. HTML, JavaScript and image assets were served successfully, with live mode disabled and configuration read from the launch folder.
- Browser: authored example, Paths/Map, keeping a thought, annotations and next move, persistence after reload, actual Markdown/JSON downloads, and native file-input import. Imported copies preserved existing work and restored annotations.
- Desktop and 390-pixel mobile rendering inspected. No page errors or document-level horizontal overflow in the checked mobile path. This is not a comprehensive accessibility or cross-browser audit.
- A real local-model museum exploration is included separately, with its prompt and limits documented beside the recording. No paid model calls were used.
- Public source scan found no credential-shaped values, private runtime files or personal machine paths. Flagged email/IP examples were reviewed as test fixtures. Binary assets and dependencies are not covered by a text-pattern scan; this is not a comprehensive security certification.

CI runs the Python suite, frontend suite and package build on Linux and Windows. The workflow has no ignored test failures. See the repository's Actions tab for actual hosted-run results rather than treating the workflow's existence as a pass.

OpenRouter and Ollama transports were tested with mocked responses. Local llama.cpp was exercised live. Compatible tool-call behavior varies by model and serving configuration. Output correctness, creativity and comparative usefulness remain unvalidated.
