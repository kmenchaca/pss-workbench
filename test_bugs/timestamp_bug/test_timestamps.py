"""Test script that demonstrates the timestamp bug."""

import os
from datetime import datetime, timezone


def test_direct_import():
    """
    Test timestamps when time_utils is imported directly first.

    This works correctly because time_utils loads before config_loader
    manipulates the TZ environment variable.
    """
    print("=== Test 1: Direct import (event_logger first) ===")

    # Fresh import - event_logger imports time_utils first
    import importlib
    import sys

    # Clear any cached imports
    for mod in list(sys.modules.keys()):
        if mod in ('time_utils', 'config_loader', 'event_logger', 'api_endpoints'):
            del sys.modules[mod]

    # Import event_logger first (imports time_utils with correct TZ)
    from event_logger import log_event
    # Then import config (too late to affect time_utils)
    from config_loader import get_config

    event = log_event("test_event", {"test": True})

    # Parse the timestamp and check the offset
    dt = datetime.fromisoformat(event.timestamp)

    print(f"Event timestamp: {event.timestamp}")
    print(f"UTC offset: {dt.utcoffset()}")
    print(f"Expected: UTC (0:00:00)")
    print(f"Result: {'PASS' if str(dt.utcoffset()) == '0:00:00' else 'FAIL'}")
    print()

    return str(dt.utcoffset()) == '0:00:00'


def test_api_import():
    """
    Test timestamps when api_endpoints is imported.

    BUG: This fails because api_endpoints imports config_loader,
    which sets TZ='America/New_York' before importing time_utils.
    The timezone offset gets cached incorrectly.
    """
    print("=== Test 2: API import (config_loader first) ===")

    import importlib
    import sys

    # Clear cached imports
    for mod in list(sys.modules.keys()):
        if mod in ('time_utils', 'config_loader', 'event_logger', 'api_endpoints'):
            del sys.modules[mod]

    # Ensure TZ is not set
    if 'TZ' in os.environ:
        del os.environ['TZ']

    # Import api_endpoints first
    # This imports config_loader, which sets TZ before time_utils loads
    from api_endpoints import UserAPI

    api = UserAPI()
    result = api.create_user("testuser", "test@example.com")

    # Parse the timestamp and check the offset
    dt = datetime.fromisoformat(result["created_at"])

    print(f"Event timestamp: {result['created_at']}")
    print(f"UTC offset: {dt.utcoffset()}")
    print(f"Expected: UTC (0:00:00)")
    actual_offset = str(dt.utcoffset())
    print(f"Result: {'PASS' if actual_offset == '0:00:00' else 'FAIL - got ' + actual_offset}")
    print()

    return actual_offset == '0:00:00'


def test_mixed_endpoints():
    """
    Test that shows the bug more clearly - same server, different results.
    """
    print("=== Test 3: Mixed endpoints (simulating production) ===")

    import sys
    for mod in list(sys.modules.keys()):
        if mod in ('time_utils', 'config_loader', 'event_logger', 'api_endpoints'):
            del sys.modules[mod]

    if 'TZ' in os.environ:
        del os.environ['TZ']

    # In production, api_endpoints is typically imported at startup
    from api_endpoints import UserAPI, OrderAPI

    user_api = UserAPI()
    order_api = OrderAPI()

    # Create events through both APIs
    user_result = user_api.create_user("alice", "alice@example.com")
    order_result = order_api.create_order("user_123", ["item1", "item2"])

    user_dt = datetime.fromisoformat(user_result["created_at"])
    order_dt = datetime.fromisoformat(order_result["created_at"])

    print(f"User API timestamp:  {user_result['created_at']}")
    print(f"Order API timestamp: {order_result['created_at']}")
    print(f"User API offset:  {user_dt.utcoffset()}")
    print(f"Order API offset: {order_dt.utcoffset()}")
    print()

    # Both should be UTC, but they're both wrong because time_utils
    # was poisoned at import time
    if str(user_dt.utcoffset()) != '0:00:00':
        print("BUG CONFIRMED: Timestamps are not in UTC!")
        print(f"They're offset by {user_dt.utcoffset()} (Eastern Time)")
        print()
        print("This happens because:")
        print("1. api_endpoints imports config_loader")
        print("2. config_loader sets TZ='America/New_York' temporarily")
        print("3. config_loader imports time_utils (first import)")
        print("4. time_utils caches DEFAULT_TZ with the wrong value")
        print("5. All subsequent timestamps use the wrong timezone")
        return False

    return True


if __name__ == "__main__":
    print("Timestamp Bug Reproduction")
    print("=" * 50)
    print()

    # Test 1 should pass
    test1_pass = test_direct_import()

    # Test 2 should fail (demonstrates the bug)
    test2_pass = test_api_import()

    # Test 3 shows production behavior
    test3_pass = test_mixed_endpoints()

    print("=" * 50)
    print("Summary:")
    print(f"  Test 1 (direct import):  {'PASS' if test1_pass else 'FAIL'}")
    print(f"  Test 2 (API import):     {'PASS' if test2_pass else 'FAIL'}")
    print(f"  Test 3 (mixed):          {'PASS' if test3_pass else 'FAIL'}")

    if not test2_pass or not test3_pass:
        print()
        print("The bug is in the import order / initialization sequence.")
