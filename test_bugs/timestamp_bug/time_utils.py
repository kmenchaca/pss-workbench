"""Timezone-aware time utilities."""

import os
from datetime import datetime, timezone, timedelta

# Map timezone names to UTC offsets (simplified)
_TZ_OFFSETS = {
    'UTC': 0,
    'America/New_York': -5,
    'America/Los_Angeles': -8,
    'Europe/London': 0,
    'Asia/Tokyo': 9,
}

def _get_tz_offset(tz_name: str) -> timezone:
    """Convert timezone name to timezone object."""
    hours = _TZ_OFFSETS.get(tz_name, 0)
    return timezone(timedelta(hours=hours))

# Cache the default timezone at module load time
# This is intentional for performance - don't recreate on every call
_tz_name = os.environ.get('TZ', 'UTC')
DEFAULT_TZ = _get_tz_offset(_tz_name)


def get_current_time() -> datetime:
    """Get current time in the default timezone."""
    return datetime.now(tz=DEFAULT_TZ)


def format_timestamp(dt: datetime) -> str:
    """Format a datetime as ISO 8601 string."""
    return dt.isoformat()


def parse_timestamp(s: str) -> datetime:
    """Parse an ISO 8601 timestamp string."""
    return datetime.fromisoformat(s)
