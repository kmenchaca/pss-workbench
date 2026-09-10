"""Problem classification for the meta-learner system.

This module provides functionality to classify problems into categories
and extract features that help determine which strategies work best.
"""

import re
from typing import Any, Callable

from .types import (
    ClassificationResult,
    ProblemCategory,
    ProblemType,
)


# Feature patterns for classification
FACTUAL_PATTERNS = [
    r"\bwhat is\b",
    r"\bwho is\b",
    r"\bwhen did\b",
    r"\bwhere is\b",
    r"\bdefine\b",
    r"\bcapital of\b",
    r"\bhow many\b",
    r"\blist the\b",
    r"\bname the\b",
    r"\bwhat are the\b",
]

CREATIVE_PATTERNS = [
    r"\bwrite a\b",
    r"\bcreate a\b",
    r"\bdesign a\b",
    r"\bimagine\b",
    r"\bstory about\b",
    r"\bpoem about\b",
    r"\bgenerate\b",
    r"\bbrainstorm\b",
    r"\binvent\b",
    r"\bcome up with\b",
]

ANALYTICAL_PATTERNS = [
    r"\banalyze\b",
    r"\bcompare\b",
    r"\bevaluate\b",
    r"\bwhy does\b",
    r"\bexplain why\b",
    r"\bwhat causes\b",
    r"\bpros and cons\b",
    r"\badvantages\b",
    r"\bdisadvantages\b",
    r"\bcritique\b",
]

TECHNICAL_PATTERNS = [
    r"\bcode\b",
    r"\bprogram\b",
    r"\balgorithm\b",
    r"\bfunction\b",
    r"\bimplement\b",
    r"\bdebug\b",
    r"\bfix the\b",
    r"\bsyntax\b",
    r"\bapi\b",
    r"\bdatabase\b",
]

PLANNING_PATTERNS = [
    r"\bplan\b",
    r"\bschedule\b",
    r"\bstrategy\b",
    r"\bsteps to\b",
    r"\bhow to\b",
    r"\bprocess for\b",
    r"\bworkflow\b",
    r"\broadmap\b",
    r"\bproject\b",
    r"\borganize\b",
]


# Built-in problem types
BUILTIN_PROBLEM_TYPES: dict[str, ProblemType] = {
    "factual_lookup": ProblemType(
        id="factual_lookup",
        name="Factual Lookup",
        category=ProblemCategory.FACTUAL,
        features=["simple_question", "single_answer", "verifiable"],
        recommended_strategies=["minimal", "tool_heavy"],
        description="Simple questions with definite, verifiable answers.",
    ),
    "factual_complex": ProblemType(
        id="factual_complex",
        name="Complex Factual",
        category=ProblemCategory.FACTUAL,
        features=["multi_part", "requires_synthesis", "verifiable"],
        recommended_strategies=["decomposition", "chain_of_thought"],
        description="Factual questions requiring synthesis of multiple facts.",
    ),
    "creative_writing": ProblemType(
        id="creative_writing",
        name="Creative Writing",
        category=ProblemCategory.CREATIVE,
        features=["open_ended", "subjective", "style_matters"],
        recommended_strategies=["exhaustive", "analogical"],
        description="Creative writing tasks with subjective quality.",
    ),
    "creative_design": ProblemType(
        id="creative_design",
        name="Creative Design",
        category=ProblemCategory.CREATIVE,
        features=["open_ended", "constraints", "novel_solution"],
        recommended_strategies=["decomposition", "adversarial"],
        description="Design problems requiring novel solutions.",
    ),
    "analytical_comparison": ProblemType(
        id="analytical_comparison",
        name="Analytical Comparison",
        category=ProblemCategory.ANALYTICAL,
        features=["multiple_options", "criteria", "judgment"],
        recommended_strategies=["exhaustive", "adversarial"],
        description="Comparing options on multiple criteria.",
    ),
    "analytical_causal": ProblemType(
        id="analytical_causal",
        name="Causal Analysis",
        category=ProblemCategory.ANALYTICAL,
        features=["cause_effect", "explanation", "reasoning"],
        recommended_strategies=["chain_of_thought", "analogical"],
        description="Understanding cause and effect relationships.",
    ),
    "technical_implementation": ProblemType(
        id="technical_implementation",
        name="Technical Implementation",
        category=ProblemCategory.TECHNICAL,
        features=["code", "specific_language", "testable"],
        recommended_strategies=["decomposition", "adversarial", "tool_heavy"],
        description="Implementing code or technical solutions.",
    ),
    "technical_debugging": ProblemType(
        id="technical_debugging",
        name="Technical Debugging",
        category=ProblemCategory.TECHNICAL,
        features=["existing_code", "error", "fix_needed"],
        recommended_strategies=["chain_of_thought", "tool_heavy"],
        description="Finding and fixing bugs in code.",
    ),
    "planning_project": ProblemType(
        id="planning_project",
        name="Project Planning",
        category=ProblemCategory.PLANNING,
        features=["multi_step", "dependencies", "timeline"],
        recommended_strategies=["decomposition", "exhaustive"],
        description="Planning multi-step projects.",
    ),
    "planning_strategy": ProblemType(
        id="planning_strategy",
        name="Strategy Planning",
        category=ProblemCategory.PLANNING,
        features=["goals", "constraints", "optimization"],
        recommended_strategies=["adversarial", "exhaustive"],
        description="Developing strategies to achieve goals.",
    ),
}


class ProblemClassifier:
    """Classifies problems into types based on content analysis."""

    def __init__(
        self,
        custom_types: dict[str, ProblemType] | None = None,
        custom_patterns: dict[ProblemCategory, list[str]] | None = None,
    ):
        """Initialize the classifier.

        Args:
            custom_types: Additional problem types to include.
            custom_patterns: Additional patterns for classification.
        """
        self.problem_types = dict(BUILTIN_PROBLEM_TYPES)
        if custom_types:
            self.problem_types.update(custom_types)

        self.patterns = {
            ProblemCategory.FACTUAL: list(FACTUAL_PATTERNS),
            ProblemCategory.CREATIVE: list(CREATIVE_PATTERNS),
            ProblemCategory.ANALYTICAL: list(ANALYTICAL_PATTERNS),
            ProblemCategory.TECHNICAL: list(TECHNICAL_PATTERNS),
            ProblemCategory.PLANNING: list(PLANNING_PATTERNS),
        }
        if custom_patterns:
            for category, patterns in custom_patterns.items():
                self.patterns[category].extend(patterns)

        # Compile patterns for efficiency
        self.compiled_patterns: dict[ProblemCategory, list[re.Pattern]] = {
            category: [re.compile(p, re.IGNORECASE) for p in patterns]
            for category, patterns in self.patterns.items()
        }

    def extract_features(self, problem: str) -> list[str]:
        """Extract features from a problem string.

        Args:
            problem: The problem text.

        Returns:
            List of extracted feature strings.
        """
        features = []
        problem_lower = problem.lower()

        # Length-based features
        word_count = len(problem.split())
        if word_count < 20:
            features.append("short")
        elif word_count < 50:
            features.append("medium")
        else:
            features.append("long")

        # Question type features
        if "?" in problem:
            features.append("question")
        if problem.strip().endswith("?"):
            features.append("ends_with_question")

        # Sentence count
        sentences = [s for s in re.split(r"[.!?]", problem) if s.strip()]
        if len(sentences) == 1:
            features.append("single_sentence")
        elif len(sentences) <= 3:
            features.append("few_sentences")
        else:
            features.append("many_sentences")

        # Check for lists or bullet points
        if re.search(r"^\s*[-*\d.]\s", problem, re.MULTILINE):
            features.append("has_list")

        # Check for code
        if re.search(r"```|\bdef\s+\w+|function\s+\w+|\bclass\s+\w+", problem):
            features.append("has_code")

        # Check for specific request types
        if re.search(r"\bgive me\b|\bprovide\b|\btell me\b", problem_lower):
            features.append("direct_request")

        if re.search(r"\bexample\b|\bsample\b", problem_lower):
            features.append("wants_examples")

        if re.search(r"\bstep\s*by\s*step\b|\bwalk.*through\b", problem_lower):
            features.append("wants_steps")

        return features

    def score_category(self, problem: str, category: ProblemCategory) -> float:
        """Score how well a problem matches a category.

        Args:
            problem: The problem text.
            category: The category to score.

        Returns:
            Score from 0.0 to 1.0.
        """
        patterns = self.compiled_patterns.get(category, [])
        if not patterns:
            return 0.0

        matches = sum(1 for p in patterns if p.search(problem))
        return min(1.0, matches / max(1, len(patterns) * 0.3))

    def classify(self, problem: str) -> ClassificationResult:
        """Classify a problem into a problem type.

        Args:
            problem: The problem text.

        Returns:
            ClassificationResult with type and confidence.
        """
        features = self.extract_features(problem)

        # Score each category
        category_scores: dict[ProblemCategory, float] = {}
        for category in ProblemCategory:
            category_scores[category] = self.score_category(problem, category)

        # Find best category
        if not any(category_scores.values()):
            # Default to analytical if no patterns match
            best_category = ProblemCategory.ANALYTICAL
            confidence = 0.3
        else:
            best_category = max(category_scores, key=category_scores.get)
            confidence = category_scores[best_category]

        # Find best problem type within category
        matching_types = [
            pt for pt in self.problem_types.values()
            if pt.category == best_category
        ]

        if not matching_types:
            # Fallback to first type of category
            matching_types = [
                pt for pt in self.problem_types.values()
            ]

        # Score types based on feature overlap
        type_scores = []
        for pt in matching_types:
            overlap = len(set(features) & set(pt.features))
            score = overlap / max(1, len(pt.features))
            type_scores.append((pt, score))

        type_scores.sort(key=lambda x: x[1], reverse=True)
        best_type, type_confidence = type_scores[0] if type_scores else (matching_types[0], 0.5)

        # Combine confidences
        final_confidence = (confidence + type_confidence) / 2

        # Get alternative types
        alternatives = [
            (pt, score) for pt, score in type_scores[1:4]
            if score > 0.1
        ]

        return ClassificationResult(
            problem_type=best_type,
            confidence=final_confidence,
            features=features,
            alternative_types=alternatives,
        )

    def add_problem_type(self, problem_type: ProblemType) -> None:
        """Add a custom problem type.

        Args:
            problem_type: The problem type to add.
        """
        self.problem_types[problem_type.id] = problem_type

    def add_pattern(self, category: ProblemCategory, pattern: str) -> None:
        """Add a pattern for a category.

        Args:
            category: The category to add the pattern to.
            pattern: Regex pattern string.
        """
        self.patterns[category].append(pattern)
        self.compiled_patterns[category].append(re.compile(pattern, re.IGNORECASE))


# Global classifier instance
_default_classifier: ProblemClassifier | None = None


def get_classifier() -> ProblemClassifier:
    """Get the default classifier instance."""
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = ProblemClassifier()
    return _default_classifier


def classify_problem(problem: str) -> ClassificationResult:
    """Classify a problem using the default classifier.

    Args:
        problem: The problem text.

    Returns:
        ClassificationResult with type and confidence.
    """
    return get_classifier().classify(problem)


def extract_features(problem: str) -> list[str]:
    """Extract features from a problem.

    Args:
        problem: The problem text.

    Returns:
        List of extracted features.
    """
    return get_classifier().extract_features(problem)


def get_problem_type(type_id: str) -> ProblemType | None:
    """Get a problem type by ID.

    Args:
        type_id: The problem type ID.

    Returns:
        The problem type if found, None otherwise.
    """
    return get_classifier().problem_types.get(type_id)


def list_problem_types() -> list[ProblemType]:
    """Get all registered problem types.

    Returns:
        List of all problem types.
    """
    return list(get_classifier().problem_types.values())


def register_problem_type(problem_type: ProblemType) -> None:
    """Register a custom problem type.

    Args:
        problem_type: The problem type to register.
    """
    get_classifier().add_problem_type(problem_type)


def register_pattern(category: ProblemCategory, pattern: str) -> None:
    """Register a pattern for classification.

    Args:
        category: The category for the pattern.
        pattern: Regex pattern string.
    """
    get_classifier().add_pattern(category, pattern)


# LLM-based classification for more nuanced cases
async def classify_with_llm(
    problem: str,
    llm_call: Callable[[str], Any],
) -> ClassificationResult:
    """Use an LLM to classify a problem (for complex cases).

    Args:
        problem: The problem text.
        llm_call: Async function to call the LLM.

    Returns:
        ClassificationResult from LLM analysis.
    """
    prompt = f"""Classify this problem into one of these categories:
- FACTUAL: Questions with definite, verifiable answers
- CREATIVE: Open-ended creative tasks
- ANALYTICAL: Analysis, comparison, evaluation tasks
- TECHNICAL: Code, implementation, debugging tasks
- PLANNING: Project planning, strategy tasks

Also identify the key features of the problem.

Problem: {problem}

Respond with:
CATEGORY: [category name]
CONFIDENCE: [0.0-1.0]
FEATURES: [comma-separated list]
REASONING: [brief explanation]"""

    response = await llm_call(prompt)

    # Parse response
    lines = response.split("\n")
    category_str = "ANALYTICAL"
    confidence = 0.5
    features = []

    for line in lines:
        if line.startswith("CATEGORY:"):
            category_str = line.split(":", 1)[1].strip().upper()
        elif line.startswith("CONFIDENCE:"):
            try:
                confidence = float(line.split(":", 1)[1].strip())
            except ValueError:
                confidence = 0.5
        elif line.startswith("FEATURES:"):
            features = [f.strip() for f in line.split(":", 1)[1].split(",")]

    # Map to category
    try:
        category = ProblemCategory(category_str.lower())
    except ValueError:
        category = ProblemCategory.ANALYTICAL

    # Find matching problem type
    classifier = get_classifier()
    matching_types = [
        pt for pt in classifier.problem_types.values()
        if pt.category == category
    ]
    best_type = matching_types[0] if matching_types else list(classifier.problem_types.values())[0]

    return ClassificationResult(
        problem_type=best_type,
        confidence=confidence,
        features=features,
        alternative_types=[],
    )
