"""Rate limiting logic for login attempts."""

from datetime import datetime, timedelta
from typing import Optional

from .config import MAX_FAILED_ATTEMPTS, LOCKOUT_DURATION_MINUTES
from .models import User


def is_account_locked(user: User) -> bool:
    """Check if a user account is currently locked."""
    if user.locked_until is None:
        return False
    return datetime.now() < user.locked_until


def should_lock_account(user: User) -> bool:
    """Check if account should be locked based on failed attempts."""
    return user.failed_attempts >= MAX_FAILED_ATTEMPTS


def calculate_lockout_expiry() -> datetime:
    """Calculate when a lockout should expire."""
    return datetime.now() + timedelta(minutes=LOCKOUT_DURATION_MINUTES)


def increment_failed_attempts(user: User) -> User:
    """Increment failed login attempts for a user."""
    user.failed_attempts += 1
    return user


def reset_failed_attempts(user: User) -> User:
    """Reset failed login attempts after successful login."""
    user.failed_attempts = 0
    user.locked_until = None
    return user


def lock_account(user: User) -> User:
    """Lock a user account."""
    user.locked_until = calculate_lockout_expiry()
    user.is_active = False
    return user


def unlock_account(user: User) -> User:
    """Unlock a user account."""
    user.locked_until = None
    user.is_active = True
    user.failed_attempts = 0
    return user


def get_remaining_attempts(user: User) -> int:
    """Get the number of remaining login attempts before lockout."""
    return max(0, MAX_FAILED_ATTEMPTS - user.failed_attempts)


def format_lockout_message(user: User) -> Optional[str]:
    """Format a message about account lockout status."""
    if not is_account_locked(user):
        return None

    remaining = user.locked_until - datetime.now()
    minutes = int(remaining.total_seconds() / 60)
    return f"Account locked. Try again in {minutes} minutes."
