"""Bug finding benchmark for PSS evaluation.

Tests the agentic "investigate" and "debug" presets with planted bugs
in synthetic codebases. Success is measured by bug detection rate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class PlantedBug:
    """A planted bug in code."""

    bug_id: str
    description: str
    location: str  # file:line
    bug_type: str  # off_by_one, null_check, race_condition, etc.
    severity: str  # low, medium, high
    keywords: list[str] = field(default_factory=list)  # Keywords that indicate detection


@dataclass
class BugfindProblem:
    """A bug finding problem with synthetic code."""

    name: str
    description: str
    files: dict[str, str]  # filename -> content
    bugs: list[PlantedBug]
    difficulty: str  # easy, medium, hard


# Synthetic codebase 1: Simple Python utility functions
BUGGY_UTILS_CODE = '''"""Utility functions for data processing."""

def find_average(numbers):
    """Calculate average of a list of numbers."""
    total = 0
    for i in range(len(numbers)):
        total += numbers[i]
    return total / len(numbers)  # BUG: Division by zero if empty list


def binary_search(arr, target):
    """Find target in sorted array, return index or -1."""
    left = 0
    right = len(arr)  # BUG: Should be len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1


def remove_duplicates(items):
    """Remove duplicates from list while preserving order."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result  # This one is actually correct


def parse_config(config_str):
    """Parse config string into dict."""
    result = {}
    lines = config_str.split("\\n")
    for line in lines:
        if "=" in line:
            key, value = line.split("=")  # BUG: Fails if value contains =
            result[key.strip()] = value.strip()
    return result


def merge_sorted(arr1, arr2):
    """Merge two sorted arrays."""
    result = []
    i = j = 0

    while i < len(arr1) and j < len(arr2):
        if arr1[i] <= arr2[j]:
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1

    # BUG: Only appends remaining from arr1, forgets arr2
    while i < len(arr1):
        result.append(arr1[i])
        i += 1

    return result
'''

BUGGY_UTILS_BUGS = [
    PlantedBug(
        bug_id="utils_1",
        description="Division by zero when calculating average of empty list",
        location="utils.py:8",
        bug_type="missing_check",
        severity="high",
        keywords=["empty", "division", "zero", "ZeroDivisionError", "len(numbers)"],
    ),
    PlantedBug(
        bug_id="utils_2",
        description="Off-by-one error in binary search: right should be len(arr)-1",
        location="utils.py:14",
        bug_type="off_by_one",
        severity="high",
        keywords=["off-by-one", "len(arr) - 1", "right = len", "IndexError", "bounds"],
    ),
    PlantedBug(
        bug_id="utils_3",
        description="Config parser fails when value contains = character",
        location="utils.py:31",
        bug_type="parsing_error",
        severity="medium",
        keywords=["split", "=", "value contains", "maxsplit", "multiple ="],
    ),
    PlantedBug(
        bug_id="utils_4",
        description="merge_sorted doesn't append remaining elements from arr2",
        location="utils.py:48",
        bug_type="incomplete_logic",
        severity="high",
        keywords=["arr2", "remaining", "while j <", "forget", "missing"],
    ),
]


# Synthetic codebase 2: Simple web API handlers
BUGGY_API_CODE = '''"""API handlers for user management."""

import json
from typing import Optional

class UserService:
    def __init__(self):
        self.users = {}
        self.sessions = {}

    def create_user(self, username: str, password: str) -> dict:
        """Create a new user."""
        if username in self.users:
            return {"error": "User exists"}

        # BUG: Password stored in plaintext
        self.users[username] = {
            "username": username,
            "password": password,
            "created_at": "2024-01-01"
        }
        return {"success": True, "username": username}

    def login(self, username: str, password: str) -> Optional[str]:
        """Login and return session token."""
        user = self.users.get(username)
        if not user:
            return None

        # BUG: Timing attack vulnerability - early return reveals user existence
        if user["password"] != password:
            return None

        import random
        token = str(random.randint(100000, 999999))  # BUG: Weak token generation
        self.sessions[token] = username
        return token

    def get_user(self, token: str, target_username: str) -> Optional[dict]:
        """Get user info (requires auth)."""
        if token not in self.sessions:
            return None

        # BUG: No authorization check - any logged in user can view any profile
        user = self.users.get(target_username)
        if user:
            return {"username": user["username"]}
        return None

    def delete_user(self, token: str, username: str) -> bool:
        """Delete a user account."""
        if token not in self.sessions:
            return False

        requesting_user = self.sessions[token]

        # BUG: IDOR - user can delete other users
        if username in self.users:
            del self.users[username]
            return True

        return False
'''

BUGGY_API_BUGS = [
    PlantedBug(
        bug_id="api_1",
        description="Password stored in plaintext instead of hashed",
        location="api.py:17",
        bug_type="security",
        severity="high",
        keywords=["plaintext", "hash", "bcrypt", "password", "cleartext"],
    ),
    PlantedBug(
        bug_id="api_2",
        description="Weak session token generation using random.randint",
        location="api.py:33",
        bug_type="security",
        severity="high",
        keywords=["random", "weak", "token", "predictable", "secrets", "uuid"],
    ),
    PlantedBug(
        bug_id="api_3",
        description="IDOR vulnerability - any user can view any other user's profile",
        location="api.py:43",
        bug_type="authorization",
        severity="high",
        keywords=["IDOR", "authorization", "access control", "any user", "no check"],
    ),
    PlantedBug(
        bug_id="api_4",
        description="IDOR vulnerability - user can delete other users' accounts",
        location="api.py:54",
        bug_type="authorization",
        severity="high",
        keywords=["IDOR", "delete", "other user", "authorization", "requesting_user"],
    ),
]


# Synthetic codebase 3: Data structure implementation
BUGGY_DS_CODE = '''"""Custom data structure implementations."""

class CircularBuffer:
    """Fixed-size circular buffer."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.head = 0
        self.tail = 0
        self.size = 0

    def push(self, item):
        """Add item to buffer."""
        if self.size == self.capacity:
            # BUG: Should overwrite head, not reject
            raise Exception("Buffer full")

        self.buffer[self.tail] = item
        self.tail = (self.tail + 1) % self.capacity
        self.size += 1

    def pop(self):
        """Remove and return oldest item."""
        if self.size == 0:
            return None

        item = self.buffer[self.head]
        self.head = (self.head + 1) % self.capacity
        self.size -= 1
        return item

    def __len__(self):
        return self.size


class LRUCache:
    """Least Recently Used cache."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.order = []  # List to track access order

    def get(self, key):
        """Get value from cache."""
        if key in self.cache:
            # BUG: Should update order on access
            return self.cache[key]
        return None

    def put(self, key, value):
        """Add/update value in cache."""
        if key in self.cache:
            self.cache[key] = value
            # BUG: Should move to end of order list
            return

        if len(self.cache) >= self.capacity:
            # Evict least recently used
            oldest = self.order.pop(0)
            del self.cache[oldest]

        self.cache[key] = value
        self.order.append(key)


class MinHeap:
    """Min heap implementation."""

    def __init__(self):
        self.heap = []

    def push(self, item):
        """Add item to heap."""
        self.heap.append(item)
        self._bubble_up(len(self.heap) - 1)

    def pop(self):
        """Remove and return minimum item."""
        if not self.heap:
            return None

        min_item = self.heap[0]
        last = self.heap.pop()

        if self.heap:
            self.heap[0] = last
            self._bubble_down(0)

        return min_item

    def _bubble_up(self, index):
        """Move item up to maintain heap property."""
        parent = (index - 1) // 2
        while index > 0 and self.heap[index] < self.heap[parent]:
            self.heap[index], self.heap[parent] = self.heap[parent], self.heap[index]
            index = parent
            parent = (index - 1) // 2

    def _bubble_down(self, index):
        """Move item down to maintain heap property."""
        size = len(self.heap)
        while True:
            smallest = index
            left = 2 * index + 1
            right = 2 * index + 2

            # BUG: Should be < size, not <= size
            if left <= size and self.heap[left] < self.heap[smallest]:
                smallest = left
            if right <= size and self.heap[right] < self.heap[smallest]:
                smallest = right

            if smallest == index:
                break

            self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
            index = smallest
'''

BUGGY_DS_BUGS = [
    PlantedBug(
        bug_id="ds_1",
        description="CircularBuffer should overwrite oldest element when full, not raise exception",
        location="ds.py:16",
        bug_type="incorrect_behavior",
        severity="medium",
        keywords=["overwrite", "circular", "full", "oldest", "should not raise"],
    ),
    PlantedBug(
        bug_id="ds_2",
        description="LRUCache.get() doesn't update access order - items never become 'recently used'",
        location="ds.py:49",
        bug_type="incorrect_behavior",
        severity="high",
        keywords=["order", "access", "recently used", "update", "move"],
    ),
    PlantedBug(
        bug_id="ds_3",
        description="LRUCache.put() doesn't move existing key to end of order list",
        location="ds.py:56",
        bug_type="incorrect_behavior",
        severity="medium",
        keywords=["order", "move", "end", "existing key", "update"],
    ),
    PlantedBug(
        bug_id="ds_4",
        description="MinHeap._bubble_down uses <= instead of < for bounds check",
        location="ds.py:101",
        bug_type="off_by_one",
        severity="high",
        keywords=["<= size", "< size", "IndexError", "bounds", "off-by-one"],
    ),
]


BUGFIND_PROBLEMS: list[BugfindProblem] = [
    BugfindProblem(
        name="utils_bugs",
        description="Find bugs in utility functions",
        files={"utils.py": BUGGY_UTILS_CODE},
        bugs=BUGGY_UTILS_BUGS,
        difficulty="easy",
    ),
    BugfindProblem(
        name="api_security",
        description="Find security vulnerabilities in API code",
        files={"api.py": BUGGY_API_CODE},
        bugs=BUGGY_API_BUGS,
        difficulty="medium",
    ),
    BugfindProblem(
        name="data_structures",
        description="Find bugs in data structure implementations",
        files={"ds.py": BUGGY_DS_CODE},
        bugs=BUGGY_DS_BUGS,
        difficulty="hard",
    ),
]


def format_bugfind_prompt(problem: BugfindProblem) -> str:
    """Format a bugfinding problem as a prompt."""
    files_text = ""
    for filename, content in problem.files.items():
        files_text += f"\n--- {filename} ---\n{content}\n"

    return f"""Review the following code for bugs and issues.

{problem.description}

{files_text}

Analyze the code carefully and identify:
1. Logic errors
2. Edge cases not handled
3. Security vulnerabilities
4. Off-by-one errors
5. Resource leaks

For each bug found, describe:
- What the bug is
- Where it's located (file and line if possible)
- How to fix it
- The potential impact

Be thorough and systematic in your analysis."""


def check_bug_found(output: str, bug: PlantedBug) -> bool:
    """
    Check if an output identifies a specific planted bug.

    Args:
        output: The model's analysis
        bug: The planted bug to check for

    Returns:
        True if the output identifies this bug
    """
    output_lower = output.lower()

    # Check if any keywords are mentioned
    keyword_matches = sum(1 for kw in bug.keywords if kw.lower() in output_lower)

    # Need at least 2 keyword matches to count as found
    # This reduces false positives from generic statements
    if keyword_matches >= 2:
        return True

    # Also check for the bug description or location
    if bug.location.lower() in output_lower:
        return True

    # Check for description fragments
    desc_words = bug.description.lower().split()
    desc_matches = sum(1 for word in desc_words if len(word) > 4 and word in output_lower)
    if desc_matches >= 3:
        return True

    return False


def evaluate_bugfind(
    outputs: list[str],
    problem: BugfindProblem,
) -> dict:
    """
    Evaluate bug finding outputs.

    Returns metrics about detection performance.
    """
    all_found_bugs = set()
    per_output_results = []

    for output in outputs:
        found_in_output = []
        for bug in problem.bugs:
            if check_bug_found(output, bug):
                all_found_bugs.add(bug.bug_id)
                found_in_output.append(bug.bug_id)
        per_output_results.append(found_in_output)

    total_bugs = len(problem.bugs)
    bugs_found = len(all_found_bugs)

    return {
        "total_bugs": total_bugs,
        "bugs_found": bugs_found,
        "detection_rate": bugs_found / total_bugs if total_bugs > 0 else 0,
        "found_bug_ids": list(all_found_bugs),
        "per_output_results": per_output_results,
        "any_output_found_all": any(
            len(r) == total_bugs for r in per_output_results
        ),
    }


def get_problems_by_difficulty(difficulty: str) -> list[BugfindProblem]:
    """Get problems filtered by difficulty."""
    return [p for p in BUGFIND_PROBLEMS if p.difficulty == difficulty]
