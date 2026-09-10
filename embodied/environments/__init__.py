"""
Concrete environment implementations.

Provides mock, text game, browser, and API environments
for embodied agent exploration.
"""

from .mock import MockEnvironment
from .text_game import TextGameEnvironment
from .browser import BrowserEnvironment
from .api import APIEnvironment

__all__ = [
    "MockEnvironment",
    "TextGameEnvironment",
    "BrowserEnvironment",
    "APIEnvironment",
]
