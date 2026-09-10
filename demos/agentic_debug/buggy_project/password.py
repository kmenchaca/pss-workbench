"""Password hashing utilities."""

import hashlib
import secrets
from typing import Tuple


def generate_salt() -> str:
    """Generate a random salt for password hashing."""
    return secrets.token_hex(16)


def hash_password(password: str, salt: str) -> str:
    """Hash a password with the given salt."""
    # Use SHA-256 with salt
    combined = f"{salt}{password}"
    return hashlib.sha256(combined.encode()).hexdigest()


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verify a password against a stored hash."""
    computed_hash = hash_password(password, salt)
    return computed_hash == stored_hash


def create_password_hash(password: str) -> Tuple[str, str]:
    """Create a password hash with a new salt. Returns (hash, salt)."""
    salt = generate_salt()
    password_hash = hash_password(password, salt)
    return password_hash, salt
