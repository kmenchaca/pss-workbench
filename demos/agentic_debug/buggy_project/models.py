"""Data models for auth service."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class User:
    """User account model."""
    id: int
    username: str
    email: str
    password_hash: str
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None


@dataclass
class Session:
    """User session model."""
    id: str
    user_id: int
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str


@dataclass
class LoginAttempt:
    """Record of a login attempt."""
    user_id: int
    timestamp: datetime
    success: bool
    ip_address: str
    reason: Optional[str] = None
