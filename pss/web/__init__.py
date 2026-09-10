"""PSS Web UI package.

Provides a web-based interface for interactive mode, accessible from any device
on the local network (including phones).

Usage:
    pip install pss[web]
    pss web --port 8000
"""

from pss.web.controller import WebController

__all__ = ["WebController"]
