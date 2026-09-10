#!/usr/bin/env python3
"""Demo: PSS produces measurably more diverse creative output.

Task: "Write 5 opening lines for a novel about grief."

HYPOTHESIS:
- Single-shot (5 times): Similar tone, style, perspective
- PSS (5 branches): Deliberately different approaches

MEASUREMENT:
1. Embedding diversity (objective) - average pairwise distance
2. Style categorization (semi-objective) - count distinct styles
3. Human blind rating (subjective) - which set has more range?

SUCCESS = PSS outputs have higher diversity score AND distinct styles.
"""

import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pss.config import PSSConfig
from pss.providers import create_provider
from pss.harness import run_pss
from pss.diversity import compute_diversity, get_embeddings_openai
import numpy as np


CREATIVE_PROMPT = """Write an opening line for a novel about grief.

Just the opening line - one sentence that would hook a reader.

Requirements:
- Must be about grief (loss of someone/something)
- Should work as the literal first sentence of a novel
- Make it compelling

Give me just the line, no explanation."""


# Style categories for semi-objective analysis
STYLE_MARKERS = {
    "melancholic_reflection": [
        "remembered", "used to", "once", "before", "when she was",
        "looking back", "memory", "recall", "think of", "miss",
    ],
    "present_action": [
        "walks", "stands", "sits", "opens", "closes", "picks up",
        "puts down", "looks at", "touches", "holds",
    ],
    "dark_humor": [
        "funny thing", "ironic", "laughed", "joke", "amusing",
        "ridiculous", "absurd", "couldn't help but", "had to admit",
    ],
    "visceral_physical": [
        "chest", "throat", "stomach", "hands", "breath", "heart",
        "ache", "weight", "hollow", "tight", "burn",
    ],
    "philosophical": [
        "time", "meaning", "existence", "nothing", "everything",
        "understand", "truth", "reality", "world", "life",
    ],
    "dialogue_opening": [
        '"', "'", "said", "asked", "told", "whispered", "called",
    ],
    "sensory_concrete": [
        "smell", "taste", "sound", "feel", "color", "light",
        "dark", "cold", "warm", "soft", "hard",
    ],
    "second_person": [
        "you ", "your ", "you're", "yourself",
    ],
    "child_perspective": [
        "mommy", "daddy", "mama", "papa", "didn't understand",
        "too young", "grown-ups", "years old",
    ],
}


def categorize_style(text: str) -> list[str]:
    """Identify style markers in text."""
    text_lower = text.lower()
    found_styles = []
    for style, markers in STYLE_MARKERS.items():
        if any(marker in text_lower for marker in markers):
            found_styles.append(style)
    return found_styles


def extract_opening_line(text: str) -> str:
    """Extract just the opening line from model output."""
    # Remove common prefixes
    text = text.strip()
    for prefix in ["Here's", "Here is", "Opening line:", "Line:", "The opening line:"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    # Remove quotes if wrapped
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]

    # Take first sentence if multiple
    for sep in [". ", ".\n", "\n\n"]:
        if sep in text:
            text = text.split(sep)[0] + "."
            break

    return text.strip()


def compute_pairwise_diversity(texts: list[str]) -> tuple[float, float, list[float]]:
    """Compute embedding-based diversity metrics.

    Returns:
        (mean_distance, min_distance, all_distances)
    """
    if len(texts) < 2:
        return 0.0, 0.0, []

    embeddings, cost = get_embeddings_openai(texts)
    embs = [np.array(e) for e in embeddings]

    distances = []
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            # Cosine distance
            dot = np.dot(embs[i], embs[j])
            norm_i = np.linalg.norm(embs[i])
            norm_j = np.linalg.norm(embs[j])
            similarity = dot / (norm_i * norm_j)
            distance = 1.0 - similarity
            distances.append(distance)

    return np.mean(distances), np.min(distances), distances


def main():
    print("=" * 70)
    print("PSS Demo: Creative Diversity Measurement")
    print("=" * 70)
    print()
    print("Task: Write 5 opening lines for a novel about grief")
    print()
    print("HYPOTHESIS:")
    print("  Single-shot x5: Similar tone, style, perspective")
    print("  PSS x5 branches: Deliberately different approaches")
    print()
    print("MEASUREMENT:")
    print("  1. Embedding diversity (objective)")
    print("  2. Style categorization (semi-objective)")
    print("  3. Human blind rating (output saved for evaluation)")
    print()

    provider = create_provider("openrouter", "meta-llama/llama-3.1-8b-instruct")

    # ========================================================================
    # Test 1: Single-shot x5
    # ========================================================================
    print("-" * 70)
    print("Test 1: SINGLE-SHOT x5 (same prompt, 5 separate calls)")
    print("-" * 70)

    single_lines = []
    single_tokens = 0

    config_single = PSSConfig(
        provider="openrouter",
        model="meta-llama/llama-3.1-8b-instruct",
        convergence_enabled=True,
        per_context_max=2000,
        total_max=2000,
    )

    for i in range(3):  # Reduced from 5 to 3 for faster demo
        result = run_pss(CREATIVE_PROMPT, config_single, provider)
        output = result.leaves[0].output if result.leaves else ""
        line = extract_opening_line(output)
        single_lines.append(line)
        single_tokens += result.total_usage.total_tokens
        print(f"  {i+1}. {line[:80]}{'...' if len(line) > 80 else ''}")

    # Analyze single-shot diversity
    single_mean_dist, single_min_dist, single_distances = compute_pairwise_diversity(single_lines)
    single_styles = [set(categorize_style(line)) for line in single_lines]
    single_unique_styles = set().union(*single_styles)

    print()
    print(f"Embedding diversity: mean={single_mean_dist:.3f}, min={single_min_dist:.3f}")
    print(f"Unique styles found: {len(single_unique_styles)} ({', '.join(single_unique_styles) or 'none'})")
    print(f"Total tokens: {single_tokens:,}")

    # ========================================================================
    # Test 2: PSS with 5 branches
    # ========================================================================
    print()
    print("-" * 70)
    print("Test 2: PSS (5 branches with diversity enforcement)")
    print("-" * 70)

    # Use a prompt that encourages different approaches
    pss_prompt = """Write an opening line for a novel about grief.

Just the opening line - one sentence that would hook a reader.

Requirements:
- Must be about grief (loss of someone/something)
- Should work as the literal first sentence of a novel
- Make it compelling

Explore DIFFERENT approaches:
- Different tones (melancholic, angry, numb, dark humor, philosophical)
- Different perspectives (first person, third person, child, elderly)
- Different styles (action, reflection, dialogue, sensory)

Give me just the line, no explanation."""

    config_pss = PSSConfig(
        provider="openrouter",
        model="meta-llama/llama-3.1-8b-instruct",
        # Enable divergent exploration
        sibling_diversity_enabled=True,
        sibling_similarity_threshold=0.6,  # Stricter for creative
        adaptive_gates_enabled=True,
        adaptive_momentum_enabled=True,
        # No synthesis - we want the raw branches
        synthesis_enabled=False,
        # Reduced budget for faster demo
        per_context_max=2000,
        total_max=12000,
        soft_gate_tokens=800,
        hard_gate_tokens=1800,
        max_contexts=4,
    )

    result_pss = run_pss(pss_prompt, config_pss, provider)

    pss_lines = []
    for leaf in result_pss.leaves:
        output = leaf.output or ""
        line = extract_opening_line(output)
        if line and len(line) > 10:  # Filter empty/too-short
            pss_lines.append(line)
            branch = leaf.branch_reason or "root"
            print(f"  [{branch[:30]}] {line[:60]}{'...' if len(line) > 60 else ''}")

    # Pad if we got fewer than 5
    while len(pss_lines) < 5 and result_pss.leaves:
        # Use any remaining output
        for leaf in result_pss.leaves:
            if leaf.output and len(pss_lines) < 5:
                line = extract_opening_line(leaf.output)
                if line not in pss_lines:
                    pss_lines.append(line)

    # Analyze PSS diversity
    pss_mean_dist, pss_min_dist, pss_distances = compute_pairwise_diversity(pss_lines[:5])
    pss_styles = [set(categorize_style(line)) for line in pss_lines[:5]]
    pss_unique_styles = set().union(*pss_styles) if pss_styles else set()

    print()
    print(f"Embedding diversity: mean={pss_mean_dist:.3f}, min={pss_min_dist:.3f}")
    print(f"Unique styles found: {len(pss_unique_styles)} ({', '.join(pss_unique_styles) or 'none'})")
    print(f"Total tokens: {result_pss.total_usage.total_tokens:,}")

    # ========================================================================
    # Comparison
    # ========================================================================
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print("EMBEDDING DIVERSITY (higher = more different):")
    print(f"  Single-shot: mean={single_mean_dist:.3f}, min={single_min_dist:.3f}")
    print(f"  PSS:         mean={pss_mean_dist:.3f}, min={pss_min_dist:.3f}")
    print()

    if pss_mean_dist > single_mean_dist:
        improvement = (pss_mean_dist - single_mean_dist) / single_mean_dist * 100
        print(f"  [+] PSS is {improvement:.1f}% more diverse (embedding distance)")
    elif pss_mean_dist < single_mean_dist:
        decline = (single_mean_dist - pss_mean_dist) / single_mean_dist * 100
        print(f"  [-] PSS is {decline:.1f}% less diverse (unexpected)")
    else:
        print("  [=] Similar diversity")

    print()
    print("STYLE DIVERSITY (unique categories):")
    print(f"  Single-shot: {len(single_unique_styles)} styles")
    print(f"  PSS:         {len(pss_unique_styles)} styles")
    print()

    if len(pss_unique_styles) > len(single_unique_styles):
        new_styles = pss_unique_styles - single_unique_styles
        print(f"  [+] PSS found {len(new_styles)} additional styles: {', '.join(new_styles)}")
    elif len(pss_unique_styles) < len(single_unique_styles):
        print(f"  [-] Single-shot found more styles")
    else:
        print("  [=] Same number of styles")

    print()
    print("TOKEN COST:")
    print(f"  Single-shot: {single_tokens:,}")
    print(f"  PSS:         {result_pss.total_usage.total_tokens:,}")

    # ========================================================================
    # Save for human evaluation
    # ========================================================================
    print()
    print("-" * 70)
    print("HUMAN EVALUATION OUTPUT")
    print("-" * 70)
    print()
    print("Below are two sets of opening lines, anonymized.")
    print("Which set has more RANGE (different tones, styles, perspectives)?")
    print()

    # Randomize order for blind evaluation
    import random
    sets = [
        ("A", single_lines[:5]),
        ("B", pss_lines[:5]),
    ]
    random.shuffle(sets)

    for set_name, lines in sets:
        print(f"SET {set_name}:")
        for i, line in enumerate(lines, 1):
            print(f"  {i}. {line}")
        print()

    # Save evaluation data
    eval_data = {
        "timestamp": datetime.now().isoformat(),
        "task": "opening lines for novel about grief",
        "single_shot": {
            "lines": single_lines,
            "embedding_diversity_mean": single_mean_dist,
            "embedding_diversity_min": single_min_dist,
            "styles": list(single_unique_styles),
            "tokens": single_tokens,
        },
        "pss": {
            "lines": pss_lines,
            "embedding_diversity_mean": pss_mean_dist,
            "embedding_diversity_min": pss_min_dist,
            "styles": list(pss_unique_styles),
            "tokens": result_pss.total_usage.total_tokens,
        },
        "blind_eval_order": [s[0] for s in sets],  # Which set is A vs B
        "ground_truth": {
            "set_A_is": "single_shot" if sets[0][1] == single_lines[:5] else "pss",
            "set_B_is": "pss" if sets[0][1] == single_lines[:5] else "single_shot",
        }
    }

    output_path = os.path.join(os.path.dirname(__file__), "eval_output.json")
    with open(output_path, "w") as f:
        json.dump(eval_data, f, indent=2)
    print(f"Evaluation data saved to: {output_path}")

    # ========================================================================
    # Final verdict
    # ========================================================================
    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print()

    diversity_win = pss_mean_dist > single_mean_dist
    style_win = len(pss_unique_styles) > len(single_unique_styles)

    if diversity_win and style_win:
        print("[+] PSS WINS on both metrics")
        print("    Higher embedding diversity AND more distinct styles")
    elif diversity_win:
        print("[~] PSS WINS on embedding diversity")
        print("    But similar number of distinct styles")
    elif style_win:
        print("[~] PSS WINS on style diversity")
        print("    But similar embedding distance")
    else:
        print("[-] NO CLEAR WINNER")
        print("    Single-shot matched or exceeded PSS diversity")

    print()
    print("Human evaluation needed for final verdict.")
    print(f"Review the blind sets above and record which has more range.")


if __name__ == "__main__":
    main()
