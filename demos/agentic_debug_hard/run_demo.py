#!/usr/bin/env python3
"""Hard Agentic Debugging Demo: Pytest fixture scope bug.

This demo tests a harder debugging scenario where:
- The error message points to the wrong file
- Finding the bug requires understanding pytest fixture scopes
- Multiple files look like they could contain the bug

The bug: A pytest fixture has scope="module" instead of scope="function",
causing test pollution between tests.
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
    "bug_file": "conftest.py",
    "bug_line": 21,
    "bug_description": "fixture scope='module' instead of scope='function'",
    "root_cause": "pytest fixture shared across tests due to module scope",
    "correct_keywords": [
        "scope",
        "module",
        "function",
        "fixture",
        "conftest",
        "shared",
        "isolation",
        "pytest.fixture",
    ],
    "partial_keywords": [
        "test pollution",
        "not isolated",
        "persists",
        "accumulates",
        "between tests",
    ],
    "red_herrings": [
        "catalog.py",
        "load_catalog",
        "clear_catalog",
        "global variable",
        "_catalog",
    ],
}


# Minimal prompt - no code snippets, just the problem
BUG_REPORT = """You are debugging a Python test suite. One test is failing due to "test pollution".

SYMPTOM: The test `test_catalog_still_has_five_products` fails.
ERROR MESSAGE: "TEST POLLUTION: Expected 5 products but got 6. Products: ['LAPTOP', 'MOUSE', 'KEYBOARD', 'HEADPHONES', 'BOOK', 'CUSTOM']"

The previous test adds a 'CUSTOM' product, which should not be visible to subsequent tests.

PROJECT STRUCTURE:
- buggy_project/
  - models.py (Product, Order dataclasses)
  - catalog.py (product catalog with load/clear functions)
  - orders.py (order processing)
  - discounts.py (discount calculations)
  - tax.py (tax calculations)
  - tests/
    - conftest.py (pytest fixtures)
    - test_orders.py (the failing tests)

YOUR TASK: Find the ROOT CAUSE of why products persist between tests.

TOOLS AVAILABLE:
- read_file: Read any file
- search_files: Search for patterns
- run_command: Run pytest or other commands

IMPORTANT: The error message tells you WHAT is wrong, but not WHERE the bug is.
You need to investigate multiple files to find the actual cause.

START by running the tests, then investigate the test infrastructure.
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

    partial_found = []
    for keyword in GROUND_TRUTH["partial_keywords"]:
        if keyword.lower() in text_lower:
            partial_found.append(keyword)

    # Check for red herrings
    herrings_found = []
    for herring in GROUND_TRUTH["red_herrings"]:
        if herring.lower() in text_lower:
            herrings_found.append(herring)

    # Key checks - look for the core concepts
    found_fixture_scope = any(x in text_lower for x in [
        "scope=\"module\"",
        "scope='module'",
        "scope=module",
        'scope="module"',
        "module scope",
        "fixture scope",
        "scope should be",
        "scope of 'module'",
        "scope of \"module\"",
        "has a scope of",
        "scoped to module",
        "module-scoped",
    ])

    found_conftest = "conftest" in text_lower

    found_needs_function_scope = any(x in text_lower for x in [
        "scope=\"function\"",
        "scope='function'",
        "function scope",
        "scope should be function",
        "change scope to function",
        "default scope",
        "scope to 'function'",
        "scope to \"function\"",
        "changing the scope to",
        "change the scope",
    ])

    # Score
    if found_fixture_scope and found_conftest and found_needs_function_scope:
        return 1.0, "FOUND IT - identified fixture scope bug in conftest.py", correct_found
    elif found_fixture_scope and found_conftest:
        return 0.85, "VERY CLOSE - found scope issue in conftest but didn't specify fix", correct_found
    elif found_fixture_scope:
        return 0.7, "CLOSE - identified scope issue but not in conftest", correct_found
    elif found_conftest and len(correct_found) >= 2:
        return 0.5, "PARTIAL - found conftest but didn't identify scope issue", correct_found
    elif len(correct_found) >= 3:
        return 0.4, f"HINTS - found relevant keywords ({', '.join(correct_found[:3])})", correct_found
    elif len(partial_found) >= 2:
        return 0.3, f"UNDERSTANDING - understands pollution but not cause", partial_found
    elif len(herrings_found) >= 2 and len(correct_found) < 2:
        return 0.1, f"MISDIRECTED - focused on red herrings ({', '.join(herrings_found[:2])})", correct_found
    else:
        return 0.2, "UNCLEAR - no clear diagnosis", correct_found


def main():
    print("=" * 70)
    print("HARD AGENTIC DEBUGGING DEMO")
    print("Pytest fixture scope bug - requires multi-file investigation")
    print("=" * 70)
    print()
    print("GROUND TRUTH:")
    print(f"  Bug location: {GROUND_TRUTH['bug_file']}:{GROUND_TRUTH['bug_line']}")
    print(f"  Root cause: {GROUND_TRUTH['root_cause']}")
    print()

    # Run tests to show failures
    print("Running tests to show failures...")
    print("-" * 70)
    test_output = run_tests_manually()
    for line in test_output.split('\n'):
        if 'PASSED' in line or 'FAILED' in line or 'ERROR' in line or 'TEST POLLUTION' in line:
            print(f"  {line.strip()[:100]}")
    print("-" * 70)
    print()

    # ========================================================================
    # Test 1: Reasoning only (no code, just error message)
    # ========================================================================
    print("=" * 70)
    print("Test 1: REASONING ONLY (no tools, no code)")
    print("=" * 70)
    print()

    # For reasoning-only, give ONLY the error message - no code
    reasoning_prompt = """A Python test suite has a failing test.

TEST OUTPUT:
```
tests/test_orders.py::TestCatalogIsolation::test_catalog_still_has_five_products FAILED

AssertionError: TEST POLLUTION: Expected 5 products but got 6.
Products: ['LAPTOP', 'MOUSE', 'KEYBOARD', 'HEADPHONES', 'BOOK', 'CUSTOM']
```

The test before this one (`test_add_custom_product_isolated`) adds a 'CUSTOM' product.
The failing test expects only 5 products but sees 6.

This is a Python project using pytest. The tests use a fixture called `catalog_with_products`
that is defined in `conftest.py`.

What is the most likely root cause of this test pollution?
Where would you look to fix it?
"""

    provider_reasoning = create_provider("openrouter", "meta-llama/llama-3.1-8b-instruct")

    config_reasoning = PSSConfig(
        provider="openrouter",
        model="meta-llama/llama-3.1-8b-instruct",
        agentic_enabled=False,
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
    print(f"\nOutput preview:\n{reasoning_output[:1200]}...")
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
        model="openai/gpt-4o-mini",
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
        per_context_max=10000,
        total_max=80000,
        soft_gate_tokens=2500,
        hard_gate_tokens=8000,
        max_contexts=5,
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
        branch_name = (leaf.branch_reason or 'root')[:50]
        print(f"  Leaf {i+1} ({branch_name}): {score:.0%} - {diagnosis}")

        if leaf.tool_traces:
            tools_used = [t.tool_name for t in leaf.tool_traces]
            print(f"    Tools: {tools_used[:5]}...")

        if score > best_score:
            best_score = score
            best_diagnosis = diagnosis
            best_insights = insights
            best_output = output

    # Score synthesis
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
    print(f"\nBest output preview:\n{best_output[:1200]}...")
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
        print(f"    Score improvement: +{(best_score - reasoning_score)*100:.0f}pp")
        if best_score >= 0.85:
            print("    Tools helped trace through conftest.py to find the scope bug!")
    elif best_score == reasoning_score:
        print("[=] TIE")
        if best_score >= 0.7:
            print("    Both found the fixture scope issue")
        else:
            print("    Neither fully diagnosed the bug")
    else:
        print("[-] Reasoning did better (unexpected)")

    print()
    print(f"Token cost: Reasoning={result_reasoning.total_usage.total_tokens:,}, "
          f"Agentic={result_agentic.total_usage.total_tokens:,}")

    # Tool analysis
    print()
    print("-" * 70)
    print("TOOL USAGE ANALYSIS")
    print("-" * 70)
    all_traces = []
    for leaf in result_agentic.leaves:
        all_traces.extend(leaf.tool_traces)

    if all_traces:
        print(f"Total tool calls: {len(all_traces)}")

        # Which files were read?
        files_read = set()
        for trace in all_traces:
            if trace.tool_name == "read_file":
                path = trace.arguments.get("path", "")
                files_read.add(path)

        if files_read:
            print(f"\nFiles investigated:")
            for f in sorted(files_read):
                # Check if it's the right file
                marker = " <-- BUG IS HERE" if "conftest" in f.lower() else ""
                print(f"  - {f}{marker}")

        # Did any branch read conftest.py?
        read_conftest = any("conftest" in f.lower() for f in files_read)
        if read_conftest:
            print("\n[+] At least one branch investigated conftest.py")
        else:
            print("\n[-] No branch read conftest.py (missed the bug location)")
    else:
        print("No tools were used")

    # Show what the bug actually is
    print()
    print("-" * 70)
    print("THE ACTUAL BUG (for reference)")
    print("-" * 70)
    print("""
In conftest.py line 21:

    @pytest.fixture(scope="module")  # <-- BUG
    def catalog_with_products():

Should be:

    @pytest.fixture(scope="function")  # <-- FIX (or just @pytest.fixture)
    def catalog_with_products():

The 'module' scope means the fixture is shared across ALL tests in the module.
When one test modifies the catalog, subsequent tests see those modifications.
""")


if __name__ == "__main__":
    main()
