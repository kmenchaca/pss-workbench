"""Configuration loader with timezone-aware timestamp parsing."""

import os
from datetime import datetime

# Simulated config data (in production this would be loaded from file)
_RAW_CONFIG = {
    "api_key": "sk-xxx-xxx",
    "rate_limit": 100,
    "cache_ttl": 300,
    # These timestamps are in Eastern Time (the ops team is in NYC)
    "maintenance_start": "2024-01-15T02:00:00",
    "maintenance_end": "2024-01-15T06:00:00",
}


def _parse_config_timestamps(config: dict) -> dict:
    """
    Parse timestamp fields in config.

    The ops team writes timestamps in Eastern Time, so we temporarily
    set TZ to parse them correctly, then convert to UTC for storage.
    """
    from time_utils import parse_timestamp, get_current_time

    result = config.copy()

    timestamp_fields = ['maintenance_start', 'maintenance_end']
    for field in timestamp_fields:
        if field in result and isinstance(result[field], str):
            # Parse the timestamp
            dt = parse_timestamp(result[field])
            # Store as ISO string
            result[field] = dt.isoformat()

    return result


def load_config() -> dict:
    """
    Load and parse the application configuration.

    Handles timezone conversion for timestamp fields.
    """
    # The config file timestamps are in Eastern Time
    # Temporarily set TZ so the parser interprets them correctly
    old_tz = os.environ.get('TZ')
    os.environ['TZ'] = 'America/New_York'

    try:
        config = _parse_config_timestamps(_RAW_CONFIG)
    finally:
        # Restore original TZ
        if old_tz is not None:
            os.environ['TZ'] = old_tz
        elif 'TZ' in os.environ:
            del os.environ['TZ']

    return config


# Pre-load config at module import for fast access
CONFIG = load_config()


def get_config() -> dict:
    """Get the loaded configuration."""
    return CONFIG
