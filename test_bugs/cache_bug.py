"""
A caching layer for user profile data.

BUG REPORT:
Users are seeing stale profile data after updates. The cache TTL is 5 minutes,
but users report seeing old data for much longer - sometimes hours.

We've tried:
- Reducing TTL to 1 minute (didn't help)
- Adding cache.invalidate() calls after updates (didn't help)
- Checking Redis connectivity (it's fine)

The issue seems intermittent - some users see fresh data, others don't.
Happens more often on mobile than desktop, but we can't figure out why.

Stack trace from a user complaint:
  File "cache_bug.py", line 89, in get_user_profile
    return self.cache.get(cache_key) or self._fetch_and_cache(user_id)
  File "cache_bug.py", line 94, in _fetch_and_cache
    self.cache.set(cache_key, profile, ttl=self.ttl)

The cache operations look correct. We're stumped.
"""

import hashlib
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class UserProfile:
    user_id: str
    username: str
    email: str
    avatar_url: str
    updated_at: float


class SimpleCache:
    """In-memory cache with TTL support."""

    def __init__(self):
        self._store: dict[str, tuple[Any, float]] = {}  # key -> (value, expires_at)

    def get(self, key: str) -> Any | None:
        if key not in self._store:
            return None
        value, expires_at = self._store[key]
        if time.time() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        expires_at = time.time() + ttl
        self._store[key] = (value, expires_at)

    def invalidate(self, key: str) -> bool:
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> None:
        self._store.clear()


class UserProfileService:
    """Service for fetching and caching user profiles."""

    def __init__(self, cache: SimpleCache, ttl: int = 300):
        self.cache = cache
        self.ttl = ttl
        self._db = {}  # Simulated database

    def _generate_cache_key(self, user_id: str, request_context: dict) -> str:
        """Generate a cache key for the user profile."""
        # Include request context for proper cache segmentation
        # Mobile and desktop might need different profile formats
        platform = request_context.get("platform", "unknown")
        version = request_context.get("app_version", "0.0.0")

        # Create a deterministic key
        key_parts = f"profile:{user_id}:{platform}:{version}"
        return hashlib.md5(key_parts.encode()).hexdigest()[:16]

    def get_user_profile(self, user_id: str, request_context: dict = None) -> UserProfile | None:
        """Get user profile, using cache if available."""
        request_context = request_context or {}
        cache_key = self._generate_cache_key(user_id, request_context)

        cached = self.cache.get(cache_key)
        if cached:
            return cached

        return self._fetch_and_cache(user_id, cache_key)

    def _fetch_and_cache(self, user_id: str, cache_key: str) -> UserProfile | None:
        """Fetch from DB and cache the result."""
        profile = self._db.get(user_id)
        if profile:
            # Cache a copy to avoid reference issues
            from copy import deepcopy
            cached_copy = deepcopy(profile)
            self.cache.set(cache_key, cached_copy, ttl=self.ttl)
            return cached_copy
        return profile

    def update_user_profile(self, user_id: str, **updates) -> UserProfile | None:
        """Update user profile and invalidate cache."""
        profile = self._db.get(user_id)
        if not profile:
            return None

        # Update fields
        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        profile.updated_at = time.time()

        # Invalidate cache - but what key?
        # We need to invalidate, but we don't have request_context here...
        # Just use a simple key pattern
        cache_key = f"profile:{user_id}"
        self.cache.invalidate(cache_key)

        return profile

    def create_user(self, user_id: str, username: str, email: str) -> UserProfile:
        """Create a new user profile."""
        profile = UserProfile(
            user_id=user_id,
            username=username,
            email=email,
            avatar_url=f"https://avatars.example.com/{user_id}",
            updated_at=time.time()
        )
        self._db[user_id] = profile
        return profile


def test_cache_staleness():
    """
    Test that demonstrates the staleness bug.

    Expected: After update, both mobile and desktop should see new data.
    Actual: They see stale data.
    """
    cache = SimpleCache()
    service = UserProfileService(cache, ttl=300)

    # Create a user
    service.create_user("user123", "alice", "alice@example.com")

    # Mobile app fetches profile
    mobile_context = {"platform": "ios", "app_version": "2.1.0"}
    profile1 = service.get_user_profile("user123", mobile_context)
    print(f"Mobile sees: {profile1.username}")

    # Desktop fetches profile
    desktop_context = {"platform": "web", "app_version": "1.0.0"}
    profile2 = service.get_user_profile("user123", desktop_context)
    print(f"Desktop sees: {profile2.username}")

    # User updates their username via API
    service.update_user_profile("user123", username="alice_updated")
    print("Username updated to 'alice_updated'")

    # Mobile fetches again - should see new username
    profile3 = service.get_user_profile("user123", mobile_context)
    print(f"Mobile sees after update: {profile3.username}")

    # Desktop fetches again - should see new username
    profile4 = service.get_user_profile("user123", desktop_context)
    print(f"Desktop sees after update: {profile4.username}")

    # Check if bug is present
    if profile3.username != "alice_updated" or profile4.username != "alice_updated":
        print("\nBUG CONFIRMED: Stale data after update!")
        print(f"   Expected: 'alice_updated'")
        print(f"   Mobile got: '{profile3.username}'")
        print(f"   Desktop got: '{profile4.username}'")
        return False
    else:
        print("\nCache invalidation working correctly")
        return True


if __name__ == "__main__":
    test_cache_staleness()
