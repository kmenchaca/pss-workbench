#!/usr/bin/env python3
"""Entry point for the unified interface CLI.

Usage:
    python -m unified argswarm "AI will replace most jobs"
    python -m unified --list
"""

from .cli import main

if __name__ == "__main__":
    main()
