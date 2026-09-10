"""Configuration for auth service."""

# Rate limiting
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_DURATION_MINUTES = 30

# Session settings
SESSION_DURATION_HOURS = 24
SESSION_ID_LENGTH = 32

# Password requirements
MIN_PASSWORD_LENGTH = 8
REQUIRE_SPECIAL_CHARS = True
