"""Brainstorm benchmark for PSS evaluation.

Tests the "explore" preset use case: generating diverse approaches
to open-ended problems. The key metric is diversity, not correctness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class BrainstormProblem:
    """A brainstorming problem."""

    prompt: str
    topic: str
    min_ideas: int = 5  # Minimum expected ideas
    category: str = "general"  # general, technical, creative
    seed_ideas: list[str] | None = None  # Optional known good ideas


BRAINSTORM_PROBLEMS: list[BrainstormProblem] = [
    # Technical problems
    BrainstormProblem(
        prompt="""List different approaches to reduce latency in a web application.

For each approach, briefly describe:
1. What it involves
2. When it's most effective
3. Potential tradeoffs

Be specific and practical.""",
        topic="web_latency",
        category="technical",
        min_ideas=8,
        seed_ideas=[
            "CDN caching",
            "Database indexing",
            "Connection pooling",
            "Lazy loading",
            "Code splitting",
            "Redis caching",
            "HTTP/2",
            "Server-side rendering",
        ],
    ),
    BrainstormProblem(
        prompt="""What are different strategies for handling errors in distributed systems?

Consider:
- Detection methods
- Recovery strategies
- Prevention approaches

List distinct approaches with brief explanations.""",
        topic="distributed_errors",
        category="technical",
        min_ideas=6,
        seed_ideas=[
            "Circuit breakers",
            "Retry with backoff",
            "Saga pattern",
            "Event sourcing",
            "Bulkhead isolation",
            "Health checks",
        ],
    ),
    BrainstormProblem(
        prompt="""List different ways to secure an API endpoint.

Include:
- Authentication methods
- Authorization approaches
- Rate limiting strategies
- Data validation techniques

Be comprehensive.""",
        topic="api_security",
        category="technical",
        min_ideas=10,
        seed_ideas=[
            "JWT tokens",
            "OAuth 2.0",
            "API keys",
            "Rate limiting",
            "Input validation",
            "HTTPS",
            "CORS",
            "WAF",
        ],
    ),
    # Creative problems
    BrainstormProblem(
        prompt="""Brainstorm different story premises involving a lighthouse.

Each premise should be genuinely different in:
- Genre (horror, romance, mystery, etc.)
- Time period
- Central conflict

List distinct story ideas.""",
        topic="lighthouse_stories",
        category="creative",
        min_ideas=5,
        seed_ideas=[
            "Ghost story",
            "Romance during WWII",
            "Sci-fi last human",
            "Murder mystery",
            "Coming of age",
        ],
    ),
    BrainstormProblem(
        prompt="""List different ways to visually represent time in a user interface.

Consider:
- Linear vs non-linear
- Analog vs digital
- Abstract vs concrete
- Static vs animated

Describe each approach briefly.""",
        topic="time_ui",
        category="creative",
        min_ideas=7,
        seed_ideas=[
            "Timeline",
            "Calendar grid",
            "Analog clock",
            "Progress bar",
            "Spiral visualization",
            "Color gradient",
            "Animation sequence",
        ],
    ),
    # General problems
    BrainstormProblem(
        prompt="""What are different approaches to reduce household energy consumption?

Include:
- Behavioral changes
- Technology solutions
- Structural modifications

List practical, actionable ideas.""",
        topic="energy_saving",
        category="general",
        min_ideas=10,
        seed_ideas=[
            "LED bulbs",
            "Smart thermostat",
            "Insulation",
            "Solar panels",
            "Unplug devices",
            "Energy audit",
            "Timer switches",
            "Efficient appliances",
        ],
    ),
    BrainstormProblem(
        prompt="""List different methods for learning a new programming language.

Consider:
- Structured vs unstructured
- Solo vs collaborative
- Passive vs active
- Project-based vs concept-based

Describe varied approaches.""",
        topic="learn_programming",
        category="general",
        min_ideas=8,
        seed_ideas=[
            "Online courses",
            "Build projects",
            "Read documentation",
            "Pair programming",
            "Code reviews",
            "Tutorials",
            "Open source contribution",
            "Coding challenges",
        ],
    ),
    BrainstormProblem(
        prompt="""What are different approaches to improve team communication in a remote work environment?

Include:
- Synchronous methods
- Asynchronous methods
- Tools and processes
- Cultural practices

Be specific.""",
        topic="remote_communication",
        category="general",
        min_ideas=8,
        seed_ideas=[
            "Daily standups",
            "Documentation",
            "Video calls",
            "Slack channels",
            "Async updates",
            "Virtual coffee",
            "Written RFCs",
            "Screen sharing",
        ],
    ),
]


def format_brainstorm_prompt(problem: BrainstormProblem) -> str:
    """Format a brainstorming problem as a prompt."""
    return f"""{problem.prompt}

Format your response as a numbered list. Each idea should be on its own line:
1. [Idea name]: [Brief description]
2. [Idea name]: [Brief description]
...

List at least {problem.min_ideas} distinct ideas."""


def count_unique_ideas(output: str, problem: BrainstormProblem | None = None) -> int:
    """
    Count the number of unique ideas in an output.

    This uses heuristics to identify distinct ideas:
    - Numbered lists
    - Bullet points
    - Paragraph breaks with distinct concepts

    Args:
        output: The model's output text
        problem: Optional problem for context

    Returns:
        Estimated number of unique ideas
    """
    ideas = extract_ideas(output)
    return len(ideas)


def extract_ideas(output: str) -> list[str]:
    """
    Extract individual ideas from brainstorm output.

    Returns a list of idea strings.
    """
    ideas = []

    # First try to find numbered list items
    numbered = re.findall(
        r"(?:^|\n)\s*\d+[\.\)]\s*\*?\*?([^\n]+?)(?:\*?\*?)(?:\n|$)",
        output,
        re.MULTILINE,
    )
    if numbered:
        ideas.extend([item.strip() for item in numbered if item.strip()])

    # Also try bullet points
    bullets = re.findall(
        r"(?:^|\n)\s*[\-\*\u2022]\s*\*?\*?([^\n]+?)(?:\*?\*?)(?:\n|$)",
        output,
        re.MULTILINE,
    )
    if bullets:
        ideas.extend([item.strip() for item in bullets if item.strip()])

    # If we didn't find structured ideas, try to split by paragraph
    if not ideas:
        paragraphs = output.split("\n\n")
        for para in paragraphs:
            para = para.strip()
            # Filter out very short or very long paragraphs
            if 20 < len(para) < 500:
                ideas.append(para)

    # Deduplicate while preserving order
    seen = set()
    unique_ideas = []
    for idea in ideas:
        # Normalize for comparison
        normalized = idea.lower().strip()[:50]  # First 50 chars
        if normalized not in seen and len(idea) > 10:
            seen.add(normalized)
            unique_ideas.append(idea)

    return unique_ideas


def evaluate_brainstorm_diversity(
    outputs: list[str],
    problem: BrainstormProblem,
) -> dict:
    """
    Evaluate brainstorming outputs for diversity and coverage.

    Returns metrics about the combined output quality.
    """
    all_ideas = []
    for output in outputs:
        ideas = extract_ideas(output)
        all_ideas.extend(ideas)

    # Deduplicate across all outputs
    unique_ideas = []
    seen_normalized = set()
    for idea in all_ideas:
        normalized = idea.lower().strip()[:50]
        if normalized not in seen_normalized:
            seen_normalized.add(normalized)
            unique_ideas.append(idea)

    # Check coverage of seed ideas (if available)
    coverage = 0
    if problem.seed_ideas:
        for seed in problem.seed_ideas:
            seed_lower = seed.lower()
            for idea in unique_ideas:
                if seed_lower in idea.lower():
                    coverage += 1
                    break

    return {
        "total_ideas": len(all_ideas),
        "unique_ideas": len(unique_ideas),
        "ideas_per_output": len(all_ideas) / len(outputs) if outputs else 0,
        "meets_minimum": len(unique_ideas) >= problem.min_ideas,
        "seed_coverage": coverage / len(problem.seed_ideas) if problem.seed_ideas else None,
    }


def get_problems_by_category(category: str) -> list[BrainstormProblem]:
    """Get problems filtered by category."""
    return [p for p in BRAINSTORM_PROBLEMS if p.category == category]
