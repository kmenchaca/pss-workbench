#!/usr/bin/env python3
"""Agentic Debugging Demo: PSS with real tools investigates a real bug.

This demo shows PSS branches using actual file reading, test running,
and code searching to find a security bug in an auth service.

The bug: reset_failed_attempts() is called BEFORE password verification,
so the rate limiter never actually limits anything.
"""

import os
import sys
import subprocess

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pss.config import PSSConfig
from pss.providers import create_provider
from pss.harness import run_pss

# Path to the buggy project
BUGGY_PROJECT_DIR = os.path.join(os.path.dirname(__file__), "buggy_project")

GROUND_TRUTH = {
    "bug_file": "auth.py",
    "bug_function": "login",
    "bug_description": "reset_failed_attempts called before password verification",
    "root_cause": "failed_attempts counter resets on every login attempt, not just successful ones",
    "security_impact": "rate limiting is completely bypassed, unlimited password attempts allowed",
    "correct_keywords": [
        "reset_failed_attempts",
        "before verification",
        "before password",
        "resets on every",
        "never accumulates",
        "always resets",
        "unconditionally",
        "order of operations",
    ],
    "red_herrings": [
        "hash collision",
        "timing attack",
        "session fixation",
        "sql injection",
        "xss",
    ],
}


BUG_REPORT = """You are debugging a Python authentication service. The security tests are failing.

SYMPTOM: Rate limiting tests fail. The tests expect the account to lock after 5 failed login attempts, but it never locks.

PROJECT STRUCTURE:
- buggy_project/
  - auth.py (main login flow)
  - rate_limit.py (rate limiting logic)
  - password.py (password hashing)
  - config.py (settings like MAX_FAILED_ATTEMPTS=5)
  - models.py (User, Session dataclasses)
  - tests/test_auth.py (failing tests)

YOUR TASK: Find the ROOT CAUSE of why rate limiting doesn't work.

TOOLS AVAILABLE:
- read_file: Read the contents of any file
- search_files: Search for patterns across files
- run_command: Run pytest or other commands

START by running the tests to see what fails, then investigate the code.
"""


def run_tests_manually() -> str:
    """Run the tests and return output."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v"],
        cwd=BUGGY_PROJECT_DIR,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.stdout + result.stderr


def score_response(text: str) -> tuple[float, str, list[str]]:
    """Score how well the response found the real bug."""
    text_lower = text.lower()

    # Check for correct insights
    correct_found = []
    for keyword in GROUND_TRUTH["correct_keywords"]:
        if keyword.lower() in text_lower:
            correct_found.append(keyword)

    # Check for red herrings
    herrings_found = []
    for herring in GROUND_TRUTH["red_herrings"]:
        if herring.lower() in text_lower:
            herrings_found.append(herring)

    # Key check: did it identify the core issue?
    found_reset_bug = any(x in text_lower for x in [
        "reset_failed_attempts",
        "reset failed attempts",
        "resets the counter",
        "counter is reset",
        "counter gets reset",
        "reset before",
        "always reset",
    ])

    found_order_issue = any(x in text_lower for x in [
        "before verification",
        "before password",
        "before verify",
        "order of operations",
        "called first",
        "unconditionally",
        "every attempt",
        "every login",
    ])

    # Score
    if found_reset_bug and found_order_issue:
        return 1.0, "FOUND IT - identified reset_failed_attempts ordering bug", correct_found
    elif found_reset_bug:
        return 0.75, "PARTIAL - found reset function but not full explanation", correct_found
    elif found_order_issue:
        return 0.5, "CLOSE - noticed order issue but not specific function", correct_found
    elif len(correct_found) >= 3:
        return 0.4, f"HINTS - found relevant keywords ({', '.join(correct_found[:3])})", correct_found
    elif len(herrings_found) >= 2:
        return 0.1, f"MISDIRECTED - chased red herrings ({', '.join(herrings_found)})", correct_found
    else:
        return 0.2, "UNCLEAR - no clear diagnosis", correct_found


def main():
    print("=" * 70)
    print("AGENTIC DEBUGGING DEMO")
    print("PSS with tools vs reasoning-only")
    print("=" * 70)
    print()
    print("GROUND TRUTH:")
    print(f"  Bug location: {GROUND_TRUTH['bug_file']}:{GROUND_TRUTH['bug_function']}")
    print(f"  Root cause: {GROUND_TRUTH['root_cause']}")
    print()

    # First, run tests to show what's failing
    print("Running tests to show failures...")
    print("-" * 70)
    test_output = run_tests_manually()
    # Show summary
    for line in test_output.split('\n'):
        if 'PASSED' in line or 'FAILED' in line or 'ERROR' in line:
            print(f"  {line.strip()}")
    print("-" * 70)
    print()

    # ========================================================================
    # Test 1: Reasoning only (no tools)
    # ========================================================================
    print("=" * 70)
    print("Test 1: REASONING ONLY (no tools)")
    print("=" * 70)
    print()

    # For reasoning-only, include code snippets in the prompt
    reasoning_prompt = BUG_REPORT + """

Here are the key code files for reference:

=== auth.py (login function) ===
```python
def login(username, password, ip_address, user_agent):
    user = get_user_by_username(username)
    if user is None:
        return None, "Invalid username or password"

    if is_account_locked(user):
        record_login_attempt(user.id, False, ip_address, "account_locked")
        return None, "Account is temporarily locked"

    salt = get_user_salt(user.id)
    if salt is None:
        return None, "Internal error"

    user = reset_failed_attempts(user)
    _users[user.id] = user

    if not verify_password(password, user.password_hash, salt):
        user = increment_failed_attempts(user)
        if should_lock_account(user):
            user = lock_account(user)
            _users[user.id] = user
            return None, "Account locked due to too many failed attempts"
        _users[user.id] = user
        return None, "Invalid username or password"

    user.last_login = datetime.now()
    _users[user.id] = user
    session = create_session(user, ip_address, user_agent)
    return session, None
```

=== rate_limit.py ===
```python
def reset_failed_attempts(user):
    user.failed_attempts = 0
    user.locked_until = None
    return user

def increment_failed_attempts(user):
    user.failed_attempts += 1
    return user

def should_lock_account(user):
    return user.failed_attempts >= MAX_FAILED_ATTEMPTS
```

=== Failing test ===
```python
def test_account_locks_after_max_failures(self):
    create_test_user("victim", "SecretPassword!")
    for i in range(MAX_FAILED_ATTEMPTS):
        session, error = login("victim", f"wrong_guess_{i}", "127.0.0.1", "Attacker")
    session, error = login("victim", "another_wrong", "127.0.0.1", "Attacker")
    assert "locked" in error.lower()  # FAILS - account never locks!
```

Analyze this code. What is the root cause of the rate limiting failure?
"""

    provider_reasoning = create_provider("openrouter", "meta-llama/llama-3.1-8b-instruct")

    config_reasoning = PSSConfig(
        provider="openrouter",
        model="meta-llama/llama-3.1-8b-instruct",
        agentic_enabled=False,  # No tools
        convergence_enabled=False,
        max_contexts=1,
        per_context_max=4000,
        total_max=4000,
    )

    result_reasoning = run_pss(reasoning_prompt, config_reasoning, provider_reasoning)
    reasoning_output = result_reasoning.leaves[0].output if result_reasoning.leaves else ""
    reasoning_score, reasoning_diagnosis, reasoning_insights = score_response(reasoning_output)

    print(f"Diagnosis: {reasoning_diagnosis}")
    print(f"Score: {reasoning_score:.0%}")
    print(f"Keywords found: {reasoning_insights or 'none'}")
    print(f"\nOutput preview:\n{reasoning_output[:1000]}...")
    print(f"\nTokens: {result_reasoning.total_usage.total_tokens:,}")

    # ========================================================================
    # Test 2: Agentic (with tools)
    # ========================================================================
    print()
    print("=" * 70)
    print("Test 2: AGENTIC (with tools)")
    print("=" * 70)
    print()

    provider_agentic = create_provider("openrouter", "openai/gpt-4o-mini")

    config_agentic = PSSConfig(
        provider="openrouter",
        model="openai/gpt-4o-mini",  # GPT-4o-mini for reliable tool calling
        agentic_enabled=True,
        agentic_tools=["read_file", "run_command", "search_files"],
        agentic_working_dir=BUGGY_PROJECT_DIR,
        # Enable branching
        sibling_diversity_enabled=True,
        sibling_similarity_threshold=0.7,
        # Enable synthesis
        synthesis_enabled=True,
        synthesis_strategy="deliberate",
        # Budget
        per_context_max=8000,
        total_max=50000,
        soft_gate_tokens=2000,
        hard_gate_tokens=6000,
        max_contexts=4,
    )

    result_agentic = run_pss(BUG_REPORT, config_agentic, provider_agentic)

    # Score each leaf
    best_score = 0.0
    best_diagnosis = ""
    best_insights = []
    best_output = ""

    print(f"Analyzing {len(result_agentic.leaves)} leaves...")
    for i, leaf in enumerate(result_agentic.leaves):
        output = leaf.output or ""
        score, diagnosis, insights = score_response(output)
        print(f"  Leaf {i+1} ({leaf.branch_reason or 'root'}): {score:.0%} - {diagnosis}")

        # Show tool usage
        if leaf.tool_traces:
            tools_used = [t.tool_name for t in leaf.tool_traces]
            print(f"    Tools used: {tools_used[:5]}...")

        if score > best_score:
            best_score = score
            best_diagnosis = diagnosis
            best_insights = insights
            best_output = output

    # Score synthesis if available
    if result_agentic.synthesis:
        synth_score, synth_diagnosis, synth_insights = score_response(
            result_agentic.synthesis.unified_output
        )
        print(f"  Synthesis: {synth_score:.0%} - {synth_diagnosis}")

        if synth_score > best_score:
            best_score = synth_score
            best_diagnosis = synth_diagnosis
            best_insights = synth_insights
            best_output = result_agentic.synthesis.unified_output

    print(f"\nBest score: {best_score:.0%} - {best_diagnosis}")
    print(f"Best keywords: {best_insights or 'none'}")
    print(f"\nBest output preview:\n{best_output[:1000]}...")
    print(f"\nTokens: {result_agentic.total_usage.total_tokens:,}")

    # ========================================================================
    # Comparison
    # ========================================================================
    print()
    print("=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print()
    print(f"Reasoning only: {reasoning_score:.0%} - {reasoning_diagnosis}")
    print(f"Agentic (tools): {best_score:.0%} - {best_diagnosis}")
    print()

    if best_score > reasoning_score:
        improvement = (best_score - reasoning_score) / max(reasoning_score, 0.01)
        print(f"[+] AGENTIC DEBUGGING WON")
        print(f"    Score improvement: {improvement:.0%}")
        if best_score >= 0.75:
            print("    Tools helped find the reset_failed_attempts ordering bug!")
    elif best_score == reasoning_score:
        print("[=] TIE")
    else:
        print("[-] Reasoning did better (unexpected)")

    print()
    print(f"Token cost: Reasoning={result_reasoning.total_usage.total_tokens:,}, "
          f"Agentic={result_agentic.total_usage.total_tokens:,}")

    # Show tool traces
    print()
    print("-" * 70)
    print("TOOL USAGE ANALYSIS")
    print("-" * 70)
    all_traces = []
    for leaf in result_agentic.leaves:
        all_traces.extend(leaf.tool_traces)

    if all_traces:
        print(f"Total tool calls: {len(all_traces)}")
        tool_counts = {}
        for trace in all_traces:
            tool_counts[trace.tool_name] = tool_counts.get(trace.tool_name, 0) + 1
        for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
            print(f"  {tool}: {count} calls")

        # Show what files were read
        files_read = set()
        for trace in all_traces:
            if trace.tool_name == "read_file":
                path = trace.arguments.get("path", "")
                files_read.add(path)
        if files_read:
            print(f"\nFiles investigated: {sorted(files_read)}")
    else:
        print("No tools were used (unexpected)")


if __name__ == "__main__":
    main()
