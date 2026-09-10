"""Authentication service with login flow."""

import secrets
from datetime import datetime, timedelta
from typing import Optional, Tuple

from .config import SESSION_DURATION_HOURS, SESSION_ID_LENGTH
from .models import LoginAttempt, Session, User
from .password import verify_password
from .rate_limit import (
    increment_failed_attempts,
    is_account_locked,
    lock_account,
    reset_failed_attempts,
    should_lock_account,
)


# In-memory storage for demo (would be database in production)
_users: dict[int, User] = {}
_sessions: dict[str, Session] = {}
_salts: dict[int, str] = {}  # user_id -> salt
_login_attempts: list[LoginAttempt] = []


def register_user(user: User, password_salt: str) -> None:
    """Register a user in the system."""
    _users[user.id] = user
    _salts[user.id] = password_salt


def get_user_by_username(username: str) -> Optional[User]:
    """Look up a user by username."""
    for user in _users.values():
        if user.username == username:
            return user
    return None


def get_user_salt(user_id: int) -> Optional[str]:
    """Get the password salt for a user."""
    return _salts.get(user_id)


def create_session(user: User, ip_address: str, user_agent: str) -> Session:
    """Create a new session for a user."""
    session = Session(
        id=secrets.token_hex(SESSION_ID_LENGTH),
        user_id=user.id,
        created_at=datetime.now(),
        expires_at=datetime.now() + timedelta(hours=SESSION_DURATION_HOURS),
        ip_address=ip_address,
        user_agent=user_agent,
    )
    _sessions[session.id] = session
    return session


def record_login_attempt(
    user_id: int, success: bool, ip_address: str, reason: Optional[str] = None
) -> None:
    """Record a login attempt for auditing."""
    attempt = LoginAttempt(
        user_id=user_id,
        timestamp=datetime.now(),
        success=success,
        ip_address=ip_address,
        reason=reason,
    )
    _login_attempts.append(attempt)


def login(
    username: str, password: str, ip_address: str, user_agent: str
) -> Tuple[Optional[Session], Optional[str]]:
    """
    Attempt to log in a user.

    Returns:
        (session, None) on success
        (None, error_message) on failure
    """
    # Look up user
    user = get_user_by_username(username)
    if user is None:
        return None, "Invalid username or password"

    # Check if account is locked
    if is_account_locked(user):
        record_login_attempt(user.id, False, ip_address, "account_locked")
        return None, "Account is temporarily locked due to too many failed attempts"

    # Get password salt
    salt = get_user_salt(user.id)
    if salt is None:
        return None, "Internal error"

    # BUG: reset_failed_attempts is called BEFORE password verification
    # This means failed_attempts never accumulates!
    # The attacker can try unlimited passwords because the counter
    # resets on every attempt, not just successful ones.
    user = reset_failed_attempts(user)
    _users[user.id] = user

    # Verify password
    if not verify_password(password, user.password_hash, salt):
        # Increment failed attempts
        user = increment_failed_attempts(user)

        # Check if we should lock the account
        if should_lock_account(user):
            user = lock_account(user)
            _users[user.id] = user
            record_login_attempt(user.id, False, ip_address, "locked_after_failures")
            return None, "Account locked due to too many failed attempts"

        _users[user.id] = user
        record_login_attempt(user.id, False, ip_address, "wrong_password")
        return None, "Invalid username or password"

    # Successful login
    user.last_login = datetime.now()
    _users[user.id] = user

    session = create_session(user, ip_address, user_agent)
    record_login_attempt(user.id, True, ip_address)

    return session, None


def logout(session_id: str) -> bool:
    """Log out by invalidating a session."""
    if session_id in _sessions:
        del _sessions[session_id]
        return True
    return False


def get_session(session_id: str) -> Optional[Session]:
    """Get a session by ID, returning None if expired or not found."""
    session = _sessions.get(session_id)
    if session is None:
        return None

    if datetime.now() > session.expires_at:
        del _sessions[session_id]
        return None

    return session


def clear_all() -> None:
    """Clear all stored data (for testing)."""
    _users.clear()
    _sessions.clear()
    _salts.clear()
    _login_attempts.clear()
