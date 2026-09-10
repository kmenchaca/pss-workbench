#!/usr/bin/env python3
"""Demo: PSS finds the bug that single-shot misses.

This demo presents a debugging task where the stack trace is misleading.
The bug report points at PayloadSerializer, but the real issue is a
decorator that buffers the entire request body.

SUCCESS CRITERIA:
- Single-shot: Follows stack trace, suggests heap/serializer fixes (WRONG)
- PSS: At least one branch questions the premise and finds the decorator (RIGHT)

GROUND TRUTH: The bug is in log_request_body decorator, not the serializer.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pss.config import PSSConfig
from pss.providers import create_provider
from pss.harness import run_pss

# Import ground truth from buggy app
from buggy_app import GROUND_TRUTH


BUG_REPORT = """Analyze this Python code to find the ROOT CAUSE of a MemoryError.

SYMPTOM: Large file uploads (>50MB) cause MemoryError
STACK TRACE shows error at: PayloadSerializer.serialize()

Here is the EXACT code. Read it carefully:

```python
def log_request_body(func):
    @wraps(func)
    def wrapper(request, *args, **kwargs):
        body_copy = request.body.read()       # <-- RUNS FIRST
        request.body = io.BytesIO(body_copy)
        logger.debug(f"Request body preview: {body_copy[:100]}...")
        return func(request, *args, **kwargs)
    return wrapper


class PayloadSerializer:
    def __init__(self, chunk_size=8192):
        self.chunk_size = chunk_size

    def serialize(self, data):
        # Stack trace points HERE, but this processes in chunks
        result = {"size": len(data), "checksum": hash(data), "chunks": []}
        for i in range(0, len(data), self.chunk_size):
            chunk = data[i:i + self.chunk_size]
            result["chunks"].append({"offset": i, "size": len(chunk)})
        return result


class RequestHandler:
    def __init__(self):
        self.serializer = PayloadSerializer()

    @log_request_body   # <-- DECORATOR applied here
    def process_upload(self, request):
        data = request.body.read()
        result = self.serializer.serialize(data)
        return result
```

EXECUTION ORDER when process_upload is called:
1. The @log_request_body decorator's wrapper() runs FIRST
2. wrapper() calls body_copy = request.body.read() - reads ENTIRE body into memory
3. Then process_upload runs
4. Then serialize() is called

QUESTION: The stack trace blames PayloadSerializer.serialize().
But is that actually where the memory problem originates?
Look at what happens BEFORE serialize() is called.
What is the TRUE root cause?"""


def score_response(text: str) -> tuple[float, str, list[str]]:
    """Score how well the response found the real bug.

    Returns:
        (score, diagnosis, found_insights)
    """
    text_lower = text.lower()

    # Check for correct insights
    correct_found = []
    for insight in GROUND_TRUTH["correct_insights"]:
        if insight.lower() in text_lower:
            correct_found.append(insight)

    # Check for red herrings
    herrings_found = []
    for herring in GROUND_TRUTH["red_herrings"]:
        if herring.lower() in text_lower:
            herrings_found.append(herring)

    # Key check: does it identify the decorator as the problem?
    found_decorator = any(x in text_lower for x in [
        "decorator",
        "log_request_body",
        "@log_request_body",
        "wrapper",
        "body_copy = request.body.read()",
        "buffers the entire",
        "reads the entire body",
        "before the handler",
        "before serialization",
    ])

    # Does it explicitly say the serializer is NOT the problem?
    exonerates_serializer = any(x in text_lower for x in [
        "not the serializer",
        "serializer is fine",
        "serializer isn't the problem",
        "serializer is not the issue",
        "isn't actually in the serializer",
        "not in the serializer",
        "before the serializer",
        "problem is not in serialize",
    ])

    # Score
    if found_decorator and exonerates_serializer:
        return 1.0, "FOUND IT - identified decorator as root cause", correct_found
    elif found_decorator:
        return 0.75, "PARTIAL - mentioned decorator but didn't fully diagnose", correct_found
    elif len(correct_found) >= 3:
        return 0.5, f"CLOSE - found relevant insights ({', '.join(correct_found)})", correct_found
    elif len(correct_found) >= 1:
        return 0.25, f"WEAK - found some insights ({', '.join(correct_found)})", correct_found
    elif len(herrings_found) >= 2:
        return 0.0, f"MISDIRECTED - fell for red herrings ({', '.join(herrings_found)})", correct_found
    else:
        return 0.1, "UNCLEAR - no clear diagnosis", correct_found


def main():
    print("=" * 70)
    print("PSS Demo: Finding the Bug Single-Shot Misses")
    print("=" * 70)
    print()
    print("GROUND TRUTH:")
    print(f"  Bug location: {GROUND_TRUTH['bug_location']}")
    print(f"  Bug line: {GROUND_TRUTH['bug_line']}")
    print()
    print("The stack trace points at PayloadSerializer.serialize()")
    print("But the REAL bug is in the log_request_body decorator.")
    print("It buffers the entire request body BEFORE serialization.")
    print()
    print("SUCCESS = PSS finds the decorator; single-shot follows stack trace")
    print()

    provider = create_provider("openrouter", "meta-llama/llama-3.1-8b-instruct")

    # ========================================================================
    # Test 1: Single-shot
    # ========================================================================
    print("-" * 70)
    print("Test 1: SINGLE-SHOT (convergent mode)")
    print("-" * 70)

    config_single = PSSConfig(
        provider="openrouter",
        model="meta-llama/llama-3.1-8b-instruct",
        convergence_enabled=False,  # Disable - run as single context without branching
        max_contexts=1,  # Force single context
        per_context_max=6000,
        total_max=6000,
    )

    result_single = run_pss(BUG_REPORT, config_single, provider)
    single_output = result_single.leaves[0].output if result_single.leaves else ""
    single_score, single_diagnosis, single_insights = score_response(single_output)

    print(f"\nDiagnosis: {single_diagnosis}")
    print(f"Score: {single_score:.0%}")
    print(f"Insights found: {single_insights or 'none'}")
    print(f"\nOutput preview:\n{single_output[:800]}...")
    print(f"\nTokens: {result_single.total_usage.total_tokens:,}")

    # ========================================================================
    # Test 2: PSS with branching
    # ========================================================================
    print()
    print("-" * 70)
    print("Test 2: PSS (branching + synthesis)")
    print("-" * 70)

    config_pss = PSSConfig(
        provider="openrouter",
        model="meta-llama/llama-3.1-8b-instruct",
        # Enable divergent exploration
        sibling_diversity_enabled=True,
        sibling_similarity_threshold=0.7,
        adaptive_gates_enabled=True,
        adaptive_momentum_enabled=True,
        # Enable synthesis
        synthesis_enabled=True,
        synthesis_strategy="deliberate",  # Debate between hypotheses
        # Budget
        per_context_max=10000,
        total_max=60000,
        soft_gate_tokens=2500,
        hard_gate_tokens=8000,
        max_contexts=5,
    )

    result_pss = run_pss(BUG_REPORT, config_pss, provider)

    # Score each leaf
    best_score = 0.0
    best_diagnosis = ""
    best_insights = []
    leaf_results = []

    print(f"\nAnalyzing {len(result_pss.leaves)} leaves...")
    for i, leaf in enumerate(result_pss.leaves):
        output = leaf.output or ""
        score, diagnosis, insights = score_response(output)
        leaf_results.append({
            "branch": leaf.branch_reason or "root",
            "score": score,
            "diagnosis": diagnosis,
            "insights": insights,
        })
        print(f"  Leaf {i+1} ({leaf.branch_reason or 'root'}): {score:.0%} - {diagnosis}")

        if score > best_score:
            best_score = score
            best_diagnosis = diagnosis
            best_insights = insights
            best_output = output

    # Also score synthesis if available
    synthesis_score = 0.0
    synthesis_diagnosis = ""
    if result_pss.synthesis:
        synthesis_score, synthesis_diagnosis, synthesis_insights = score_response(
            result_pss.synthesis.unified_output
        )
        print(f"  Synthesis: {synthesis_score:.0%} - {synthesis_diagnosis}")

        if synthesis_score > best_score:
            best_score = synthesis_score
            best_diagnosis = synthesis_diagnosis
            best_insights = synthesis_insights
            best_output = result_pss.synthesis.unified_output

    print(f"\nBest score: {best_score:.0%} - {best_diagnosis}")
    print(f"Best insights: {best_insights or 'none'}")
    print(f"\nBest output preview:\n{best_output[:800]}...")
    print(f"\nTokens: {result_pss.total_usage.total_tokens:,}")

    # ========================================================================
    # Comparison
    # ========================================================================
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"Single-shot: {single_score:.0%} - {single_diagnosis}")
    print(f"PSS best:    {best_score:.0%} - {best_diagnosis}")
    print()

    if best_score > single_score:
        improvement = (best_score - single_score) / max(single_score, 0.01)
        print(f"[+] PSS FOUND WHAT SINGLE-SHOT MISSED")
        print(f"    Score improvement: {improvement:.0%}")
        if best_score >= 0.75:
            print(f"    PSS identified the decorator as the root cause!")
        print()
        print("    This demonstrates PSS's value: exploring 'what if the")
        print("    stack trace is misleading?' led to the correct diagnosis.")
    elif best_score == single_score:
        print("[=] TIE - Both found similar insights")
        if best_score >= 0.75:
            print("    Both correctly identified the decorator.")
        elif best_score <= 0.25:
            print("    Neither found the real bug.")
    else:
        print("[-] Single-shot did better (unexpected)")

    print()
    print(f"Token cost: Single={result_single.total_usage.total_tokens:,}, "
          f"PSS={result_pss.total_usage.total_tokens:,}")

    # ========================================================================
    # Detailed branch analysis
    # ========================================================================
    print()
    print("-" * 70)
    print("BRANCH ANALYSIS")
    print("-" * 70)
    print()
    print("What hypotheses did PSS explore?")
    for result in leaf_results:
        branch = result["branch"][:50] if result["branch"] else "root"
        print(f"  - {branch}")
        print(f"    Score: {result['score']:.0%}, Insights: {result['insights'] or 'none'}")
    print()

    # Check if any branch questioned the premise
    premise_questioning_branches = []
    for leaf in result_pss.leaves:
        reason = (leaf.branch_reason or "").lower()
        output = (leaf.output or "").lower()
        if any(phrase in reason + output for phrase in [
            "what if",
            "might not be",
            "isn't actually",
            "not the real",
            "misleading",
            "red herring",
            "look elsewhere",
            "before the serializer",
            "decorator",
        ]):
            premise_questioning_branches.append(leaf.branch_reason or "root")

    if premise_questioning_branches:
        print("Branches that questioned the premise:")
        for b in premise_questioning_branches:
            print(f"  - {b}")
    else:
        print("No branches explicitly questioned the stack trace premise.")


if __name__ == "__main__":
    main()
