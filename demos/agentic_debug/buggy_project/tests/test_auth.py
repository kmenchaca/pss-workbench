"""Tests for authentication service."""

import pytest
from datetime import datetime

from ..models import User
from ..password import create_password_hash
from ..auth import (
    clear_all,
    login,
    logout,
    register_user,
    get_session,
)
from ..config import MAX_FAILED_ATTEMPTS


@pytest.fixture(autouse=True)
def clean_state():
    """Clear all data before each test."""
    clear_all()
    yield
    clear_all()


def create_test_user(username: str = "testuser", password: str = "SecurePass123!") -> User:
    """Helper to create and register a test user."""
    password_hash, salt = create_password_hash(password)
    user = User(
        id=1,
        username=username,
        email=f"{username}@example.com",
        password_hash=password_hash,
        created_at=datetime.now(),
    )
    register_user(user, salt)
    return user


class TestLogin:
    """Tests for the login function."""

    def test_successful_login(self):
        """Test that valid credentials result in a session."""
        create_test_user("alice", "MyPassword123!")

        session, error = login("alice", "MyPassword123!", "127.0.0.1", "TestAgent")

        assert error is None
        assert session is not None
        assert session.user_id == 1

    def test_wrong_password(self):
        """Test that wrong password returns error."""
        create_test_user("alice", "MyPassword123!")

        session, error = login("alice", "WrongPassword", "127.0.0.1", "TestAgent")

        assert session is None
        assert error == "Invalid username or password"

    def test_nonexistent_user(self):
        """Test that login fails for nonexistent user."""
        session, error = login("nobody", "password", "127.0.0.1", "TestAgent")

        assert session is None
        assert error == "Invalid username or password"


class TestRateLimiting:
    """Tests for rate limiting / brute force protection."""

    def test_account_locks_after_max_failures(self):
        """
        CRITICAL SECURITY TEST: Account should lock after MAX_FAILED_ATTEMPTS.

        This test simulates a brute force attack. After 5 wrong passwords,
        the account should be locked.
        """
        create_test_user("victim", "SecretPassword!")

        # Try wrong passwords MAX_FAILED_ATTEMPTS times
        for i in range(MAX_FAILED_ATTEMPTS):
            session, error = login("victim", f"wrong_guess_{i}", "127.0.0.1", "Attacker")
            assert session is None, f"Should not succeed with wrong password on attempt {i+1}"

        # The next attempt should report the account is locked
        session, error = login("victim", "another_wrong", "127.0.0.1", "Attacker")

        assert session is None
        assert "locked" in error.lower(), f"Expected lockout message, got: {error}"

    def test_successful_login_resets_attempts(self):
        """Test that successful login resets the failure counter."""
        create_test_user("bob", "GoodPassword!")

        # Fail a few times (but not enough to lock)
        for i in range(MAX_FAILED_ATTEMPTS - 2):
            login("bob", "wrong", "127.0.0.1", "TestAgent")

        # Successful login
        session, error = login("bob", "GoodPassword!", "127.0.0.1", "TestAgent")
        assert session is not None

        # Now fail a few more times - should NOT lock because counter was reset
        for i in range(MAX_FAILED_ATTEMPTS - 2):
            login("bob", "wrong", "127.0.0.1", "TestAgent")

        # Should still be able to login
        session, error = login("bob", "GoodPassword!", "127.0.0.1", "TestAgent")
        assert session is not None
        assert error is None

    def test_brute_force_attack_blocked(self):
        """
        SECURITY TEST: Simulate a brute force attack trying 100 passwords.

        The attacker should be blocked after MAX_FAILED_ATTEMPTS tries,
        preventing them from trying all 100 passwords.
        """
        create_test_user("target", "RealPassword123!")
        passwords_to_try = [f"password{i}" for i in range(100)]

        blocked_at = None
        for i, pwd in enumerate(passwords_to_try):
            session, error = login("target", pwd, "10.0.0.1", "BruteForcer")

            if error and "locked" in error.lower():
                blocked_at = i
                break

        assert blocked_at is not None, "Attacker was never blocked!"
        assert blocked_at == MAX_FAILED_ATTEMPTS, \
            f"Should block after {MAX_FAILED_ATTEMPTS} attempts, blocked at {blocked_at}"


class TestSession:
    """Tests for session management."""

    def test_session_valid_after_login(self):
        """Test that session is valid after login."""
        create_test_user("carol", "Password123!")

        session, _ = login("carol", "Password123!", "127.0.0.1", "TestAgent")

        retrieved = get_session(session.id)
        assert retrieved is not None
        assert retrieved.user_id == 1

    def test_logout_invalidates_session(self):
        """Test that logout invalidates the session."""
        create_test_user("dave", "Password123!")

        session, _ = login("dave", "Password123!", "127.0.0.1", "TestAgent")
        session_id = session.id

        result = logout(session_id)

        assert result is True
        assert get_session(session_id) is None
