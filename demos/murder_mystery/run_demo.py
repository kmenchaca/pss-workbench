#!/usr/bin/env python3
"""Demo: PSS holds incompatible hypotheses and finds the collaboration.

This demo presents a murder mystery where:
- Clues strongly implicate Suspect A (Elena)
- Other clues strongly implicate Suspect B (David)
- Neither can have acted alone (timeline contradictions)
- The solution: they collaborated

SUCCESS CRITERIA:
- Single-shot: Commits to one suspect, rationalizes contradictions (WRONG)
- PSS: At least one branch considers "what if they worked together?" (RIGHT)

This is the "genuinely impressive" demo - solving a puzzle that requires
holding incompatible hypotheses until you realize they're compatible.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pss.config import PSSConfig
from pss.providers import create_provider
from pss.harness import run_pss

from case_file import CASE_FILE, GROUND_TRUTH, score_solution


INVESTIGATION_PROMPT = f"""{CASE_FILE}

You are a detective analyzing this case. Based on all the evidence:

1. What is your theory of what happened?
2. Who is responsible for Marcus Chen's death?
3. How do you explain the timeline contradictions?

Think through the evidence carefully before concluding."""


def main():
    print("=" * 70)
    print("PSS Demo: Murder Mystery with Incompatible Hypotheses")
    print("=" * 70)
    print()
    print("THE PUZZLE:")
    print("  - Elena was in Marcus's office 8:25-8:50 PM")
    print("  - But she has VIDEO ALIBI for time of death (9:00-9:30 PM)")
    print("  - David left the building at 8:45 PM (before death)")
    print("  - But he has motive, means, and the poison required his expertise")
    print()
    print("THE TRAP:")
    print("  Neither can have acted ALONE. The evidence contradicts itself.")
    print("  Single-shot will pick one and rationalize.")
    print()
    print("THE SOLUTION:")
    print("  They collaborated. Elena drugged Marcus at 8:30 PM (delayed poison).")
    print("  David returned via service entrance at 9:00 PM when alarm was disabled.")
    print()
    print("SUCCESS = PSS considers collaboration hypothesis")
    print()

    provider = create_provider("openrouter", "meta-llama/llama-3.1-8b-instruct")

    # ========================================================================
    # Test 1: Single-shot
    # ========================================================================
    print("-" * 70)
    print("Test 1: SINGLE-SHOT")
    print("-" * 70)

    config_single = PSSConfig(
        provider="openrouter",
        model="meta-llama/llama-3.1-8b-instruct",
        convergence_enabled=True,
        per_context_max=10000,
        total_max=10000,
    )

    result_single = run_pss(INVESTIGATION_PROMPT, config_single, provider)
    single_output = result_single.leaves[0].output if result_single.leaves else ""
    single_score, single_diagnosis, single_insights = score_solution(single_output)

    print(f"\nVerdict: {single_diagnosis}")
    print(f"Score: {single_score:.0%}")
    print(f"Key insights found: {single_insights[:5] if single_insights else 'none'}")
    print(f"\nSolution excerpt:\n{single_output[:1000]}...")
    print(f"\nTokens: {result_single.total_usage.total_tokens:,}")

    # ========================================================================
    # Test 2: PSS with branching
    # ========================================================================
    print()
    print("-" * 70)
    print("Test 2: PSS (branching + synthesis)")
    print("-" * 70)

    # Prompt that encourages exploring multiple hypotheses
    pss_prompt = f"""{CASE_FILE}

You are a detective analyzing this case.

IMPORTANT: Consider MULTIPLE hypotheses before concluding. The evidence seems contradictory - explore why.

Possible hypotheses to consider:
- Elena acted alone (explain her alibi)
- David acted alone (explain how he returned)
- A third party is involved
- The timeline evidence is wrong
- Something else entirely

For each hypothesis, identify what evidence supports it and what contradicts it.
Then determine the most likely explanation.

Who killed Marcus Chen and how?"""

    config_pss = PSSConfig(
        provider="openrouter",
        model="meta-llama/llama-3.1-8b-instruct",
        # Enable divergent exploration
        sibling_diversity_enabled=True,
        sibling_similarity_threshold=0.6,  # Stricter - want different theories
        adaptive_gates_enabled=True,
        adaptive_momentum_enabled=True,
        # Enable synthesis to combine hypotheses
        synthesis_enabled=True,
        synthesis_strategy="deliberate",  # Debate between theories
        synthesis_max_tokens=3000,
        # Budget for exploration
        per_context_max=12000,
        total_max=70000,
        soft_gate_tokens=3000,
        hard_gate_tokens=10000,
        max_contexts=5,
    )

    result_pss = run_pss(pss_prompt, config_pss, provider)

    # Analyze each branch
    best_score = 0.0
    best_diagnosis = ""
    best_insights = []
    best_output = ""
    branch_results = []

    print(f"\nAnalyzing {len(result_pss.leaves)} branches...")
    for i, leaf in enumerate(result_pss.leaves):
        output = leaf.output or ""
        score, diagnosis, insights = score_solution(output)
        branch_name = leaf.branch_reason or "root"

        branch_results.append({
            "branch": branch_name,
            "score": score,
            "diagnosis": diagnosis,
            "insights": insights,
        })

        # Summarize branch hypothesis
        hypothesis = "unknown"
        output_lower = output.lower()
        if "elena" in output_lower and "david" in output_lower and any(w in output_lower for w in ["together", "both", "collaborat"]):
            hypothesis = "COLLABORATION"
        elif "elena" in output_lower and "guilty" in output_lower:
            hypothesis = "Elena alone"
        elif "david" in output_lower and "guilty" in output_lower:
            hypothesis = "David alone"
        elif "third" in output_lower or "someone else" in output_lower:
            hypothesis = "Third party"

        print(f"  Branch {i+1} [{branch_name[:30]}]")
        print(f"    Hypothesis: {hypothesis}")
        print(f"    Score: {score:.0%} - {diagnosis}")

        if score > best_score:
            best_score = score
            best_diagnosis = diagnosis
            best_insights = insights
            best_output = output

    # Score synthesis
    synthesis_score = 0.0
    if result_pss.synthesis:
        synthesis_score, synthesis_diagnosis, synthesis_insights = score_solution(
            result_pss.synthesis.unified_output
        )
        print(f"  Synthesis:")
        print(f"    Score: {synthesis_score:.0%} - {synthesis_diagnosis}")

        if synthesis_score > best_score:
            best_score = synthesis_score
            best_diagnosis = synthesis_diagnosis
            best_insights = synthesis_insights
            best_output = result_pss.synthesis.unified_output

    print(f"\nBest score: {best_score:.0%} - {best_diagnosis}")
    print(f"Key insights found: {best_insights[:5] if best_insights else 'none'}")
    print(f"\nBest solution excerpt:\n{best_output[:1000]}...")
    print(f"\nTokens: {result_pss.total_usage.total_tokens:,}")

    # ========================================================================
    # Analysis: Did PSS explore the collaboration hypothesis?
    # ========================================================================
    print()
    print("-" * 70)
    print("HYPOTHESIS EXPLORATION ANALYSIS")
    print("-" * 70)
    print()

    # Check what hypotheses were explored
    hypotheses_explored = {
        "elena_alone": False,
        "david_alone": False,
        "collaboration": False,
        "third_party": False,
        "timeline_wrong": False,
    }

    for leaf in result_pss.leaves:
        output_lower = (leaf.output or "").lower()
        branch_lower = (leaf.branch_reason or "").lower()
        combined = output_lower + " " + branch_lower

        if any(w in combined for w in ["elena alone", "elena is guilty", "elena did it", "elena killed"]):
            hypotheses_explored["elena_alone"] = True
        if any(w in combined for w in ["david alone", "david is guilty", "david did it", "david killed"]):
            hypotheses_explored["david_alone"] = True
        if any(w in combined for w in ["together", "both", "collaborat", "conspir", "partner"]):
            hypotheses_explored["collaboration"] = True
        if any(w in combined for w in ["third party", "someone else", "unknown person"]):
            hypotheses_explored["third_party"] = True
        if any(w in combined for w in ["timeline wrong", "coroner wrong", "time of death is"]):
            hypotheses_explored["timeline_wrong"] = True

    print("Hypotheses explored by PSS:")
    for hyp, explored in hypotheses_explored.items():
        status = "[x]" if explored else "[ ]"
        print(f"  {status} {hyp.replace('_', ' ').title()}")

    # ========================================================================
    # Final Comparison
    # ========================================================================
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"Single-shot: {single_score:.0%} - {single_diagnosis}")
    print(f"PSS best:    {best_score:.0%} - {best_diagnosis}")
    print()

    # Did PSS find collaboration?
    pss_found_collaboration = hypotheses_explored["collaboration"] or best_score >= 0.75

    if pss_found_collaboration and single_score < 0.75:
        print("[+] PSS SOLVED WHAT SINGLE-SHOT COULDN'T")
        print("    Single-shot committed to one suspect.")
        print("    PSS explored multiple hypotheses and found the collaboration.")
        print()
        print("    This demonstrates PSS's unique value: holding incompatible")
        print("    hypotheses until you realize they're actually compatible.")
    elif pss_found_collaboration and single_score >= 0.75:
        print("[=] BOTH FOUND THE COLLABORATION")
        print("    This case may have been too obvious - both methods solved it.")
    elif not pss_found_collaboration and single_score >= 0.75:
        print("[-] SINGLE-SHOT FOUND IT, PSS MISSED IT")
        print("    Unexpected result - PSS should have explored more hypotheses.")
    else:
        print("[~] NEITHER FULLY SOLVED IT")
        print("    Neither method identified the collaboration.")
        if hypotheses_explored["elena_alone"] or hypotheses_explored["david_alone"]:
            print("    PSS at least explored individual suspects.")
        print("    The puzzle may need tuning or the model may need more tokens.")

    print()
    print(f"Token cost: Single={result_single.total_usage.total_tokens:,}, "
          f"PSS={result_pss.total_usage.total_tokens:,}")

    # ========================================================================
    # Ground Truth Reveal
    # ========================================================================
    print()
    print("-" * 70)
    print("GROUND TRUTH")
    print("-" * 70)
    print()
    print("THE SOLUTION:")
    print(f"  {GROUND_TRUTH['solution']}")
    print()
    print("KEY EVIDENCE:")
    print("  - Poison was DELAYED-ACTION (30-45 min)")
    print("  - Elena was with Marcus at 8:30 PM - she administered it")
    print("  - Service entrance alarm was disabled 8:55-9:10 PM")
    print("  - David could have returned undetected to ensure death")
    print("  - Both benefit: Elena gets insurance, David keeps his position")


if __name__ == "__main__":
    main()
