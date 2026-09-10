"""
Argument construction and parsing for the Argumentative Swarm system.

Implements the Toulmin model of argumentation with support for
constructing, parsing, evaluating, and analyzing arguments.
"""

import re
import uuid
from typing import Any, Callable, Optional

from .types import Argument, Position


def generate_argument_id() -> str:
    """Generate a unique argument ID."""
    return f"arg_{uuid.uuid4().hex[:8]}"


def construct_argument(
    position: Position,
    claim: str,
    evidence: str,
    warrant: str,
    backing: str = "",
    qualifier: str = "",
    rebuttal: str = "",
    round_number: int = 0,
) -> Argument:
    """
    Build a structured argument following the Toulmin model.

    The Toulmin model consists of:
    - Claim: The assertion being made
    - Evidence (Data): Facts supporting the claim
    - Warrant: Reasoning connecting evidence to claim
    - Backing: Additional support for the warrant
    - Qualifier: Degree of certainty
    - Rebuttal: Conditions under which claim wouldn't hold

    Args:
        position: The position this argument supports
        claim: The main assertion
        evidence: Supporting data/facts
        warrant: Reasoning connecting evidence to claim
        backing: Additional warrant support
        qualifier: Certainty qualifier (e.g., "probably")
        rebuttal: Potential exceptions
        round_number: Which debate round this is made in

    Returns:
        A new Argument instance with calculated strength
    """
    argument = Argument(
        id=generate_argument_id(),
        claim=claim,
        evidence=evidence,
        warrant=warrant,
        position_id=position.id,
        strength=0.5,  # Initial strength, will be recalculated
        backing=backing,
        qualifier=qualifier,
        rebuttal=rebuttal,
        round_number=round_number,
    )

    # Calculate initial strength based on completeness
    argument.strength = evaluate_strength(argument)

    return argument


def parse_argument(text: str) -> dict[str, str]:
    """
    Extract claim/evidence/warrant from unstructured text.

    Attempts to identify Toulmin model components from natural language.

    Args:
        text: Unstructured argument text

    Returns:
        Dictionary with extracted components (claim, evidence, warrant, etc.)
    """
    components: dict[str, str] = {
        "claim": "",
        "evidence": "",
        "warrant": "",
        "backing": "",
        "qualifier": "",
        "rebuttal": "",
    }

    # Look for explicit markers
    claim_patterns = [
        r"(?:I argue|I claim|My position is|Therefore|Thus|Hence)[:\s]+(.+?)(?:\.|$)",
        r"(?:The claim is|The argument is)[:\s]+(.+?)(?:\.|$)",
    ]

    evidence_patterns = [
        r"(?:Evidence|Data|Facts?|Studies? show|Research indicates)[:\s]+(.+?)(?:\.|$)",
        r"(?:According to|Based on)[:\s]+(.+?)(?:\.|$)",
    ]

    warrant_patterns = [
        r"(?:Because|Since|This is because|The reason is)[:\s]+(.+?)(?:\.|$)",
        r"(?:This shows|This demonstrates|This proves)[:\s]+(.+?)(?:\.|$)",
    ]

    qualifier_patterns = [
        r"\b(probably|likely|certainly|possibly|presumably|generally|typically)\b",
    ]

    rebuttal_patterns = [
        r"(?:Unless|Except when|However, if)[:\s]+(.+?)(?:\.|$)",
        r"(?:This may not hold if|An exception would be)[:\s]+(.+?)(?:\.|$)",
    ]

    # Extract components using patterns
    for pattern in claim_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            components["claim"] = match.group(1).strip()
            break

    for pattern in evidence_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            components["evidence"] = match.group(1).strip()
            break

    for pattern in warrant_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            components["warrant"] = match.group(1).strip()
            break

    for pattern in qualifier_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            components["qualifier"] = match.group(1).strip()
            break

    for pattern in rebuttal_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            components["rebuttal"] = match.group(1).strip()
            break

    # If no claim found, use the first sentence
    if not components["claim"]:
        sentences = text.split(".")
        if sentences:
            components["claim"] = sentences[0].strip()

    # If no evidence found, look for factual statements
    if not components["evidence"]:
        # Look for sentences with numbers or citations
        for sentence in text.split("."):
            if re.search(r"\d+|%|study|research|data", sentence, re.IGNORECASE):
                components["evidence"] = sentence.strip()
                break

    return components


def evaluate_strength(argument: Argument) -> float:
    """
    Score argument quality based on completeness and structure.

    Evaluation criteria:
    - Presence of core components (claim, evidence, warrant)
    - Presence of optional components (backing, qualifier, rebuttal)
    - Length and specificity of components

    Args:
        argument: The argument to evaluate

    Returns:
        Quality score from 0.0 to 1.0
    """
    score = 0.0

    # Core components (60% of score)
    if argument.claim:
        claim_score = min(len(argument.claim) / 100, 1.0) * 0.2
        score += claim_score

    if argument.evidence:
        evidence_score = min(len(argument.evidence) / 150, 1.0) * 0.2
        score += evidence_score

    if argument.warrant:
        warrant_score = min(len(argument.warrant) / 100, 1.0) * 0.2
        score += warrant_score

    # Optional components (40% of score)
    if argument.backing:
        score += 0.15

    if argument.qualifier:
        score += 0.1

    if argument.rebuttal:
        score += 0.15

    return min(score, 1.0)


def find_weaknesses(argument: Argument) -> list[dict[str, str]]:
    """
    Identify attackable points in an argument.

    Analyzes the argument for:
    - Missing components
    - Weak warrants
    - Unsupported claims
    - Logical gaps

    Args:
        argument: The argument to analyze

    Returns:
        List of weaknesses with type and description
    """
    weaknesses: list[dict[str, str]] = []

    # Check for missing components
    if not argument.evidence:
        weaknesses.append({
            "type": "missing_evidence",
            "description": "Claim lacks supporting evidence",
            "attack_strategy": "Demand evidence or provide counter-evidence",
        })

    if not argument.warrant:
        weaknesses.append({
            "type": "missing_warrant",
            "description": "No reasoning connecting evidence to claim",
            "attack_strategy": "Challenge the logical connection",
        })

    if not argument.backing:
        weaknesses.append({
            "type": "unsupported_warrant",
            "description": "Warrant itself is not backed up",
            "attack_strategy": "Question the validity of the reasoning",
        })

    if not argument.rebuttal:
        weaknesses.append({
            "type": "no_exceptions",
            "description": "Argument doesn't acknowledge exceptions",
            "attack_strategy": "Present counter-examples or edge cases",
        })

    # Check for weak qualifiers
    strong_qualifiers = ["certainly", "definitely", "absolutely", "always"]
    if argument.qualifier.lower() in strong_qualifiers:
        weaknesses.append({
            "type": "overconfident",
            "description": f"Overconfident qualifier: '{argument.qualifier}'",
            "attack_strategy": "Present exceptions to challenge absolutism",
        })

    # Check for short/weak components
    if argument.evidence and len(argument.evidence) < 50:
        weaknesses.append({
            "type": "thin_evidence",
            "description": "Evidence is brief and may lack detail",
            "attack_strategy": "Question the sufficiency of evidence",
        })

    if argument.warrant and len(argument.warrant) < 30:
        weaknesses.append({
            "type": "weak_warrant",
            "description": "Warrant is underdeveloped",
            "attack_strategy": "Challenge the logical leap",
        })

    return weaknesses


def strengthen_argument(
    argument: Argument,
    additional_evidence: str = "",
    stronger_warrant: str = "",
    add_backing: str = "",
    add_rebuttal: str = "",
) -> Argument:
    """
    Create a strengthened version of an argument.

    Args:
        argument: The original argument
        additional_evidence: New evidence to add
        stronger_warrant: Improved warrant
        add_backing: Backing to add
        add_rebuttal: Rebuttal to add

    Returns:
        A new, strengthened Argument instance
    """
    new_evidence = argument.evidence
    if additional_evidence:
        new_evidence = f"{argument.evidence} {additional_evidence}"

    new_warrant = stronger_warrant if stronger_warrant else argument.warrant
    new_backing = add_backing if add_backing else argument.backing
    new_rebuttal = add_rebuttal if add_rebuttal else argument.rebuttal

    strengthened = Argument(
        id=generate_argument_id(),
        claim=argument.claim,
        evidence=new_evidence,
        warrant=new_warrant,
        position_id=argument.position_id,
        backing=new_backing,
        qualifier=argument.qualifier,
        rebuttal=new_rebuttal,
        round_number=argument.round_number,
    )

    strengthened.strength = evaluate_strength(strengthened)
    return strengthened


def compare_arguments(arg1: Argument, arg2: Argument) -> dict[str, Any]:
    """
    Compare two arguments and identify differences.

    Args:
        arg1: First argument
        arg2: Second argument

    Returns:
        Comparison results with strength comparison and key differences
    """
    return {
        "strength_difference": arg1.strength - arg2.strength,
        "stronger": arg1.id if arg1.strength > arg2.strength else arg2.id,
        "arg1_weaknesses": find_weaknesses(arg1),
        "arg2_weaknesses": find_weaknesses(arg2),
        "arg1_has_backing": bool(arg1.backing),
        "arg2_has_backing": bool(arg2.backing),
        "arg1_has_rebuttal": bool(arg1.rebuttal),
        "arg2_has_rebuttal": bool(arg2.rebuttal),
    }


class ArgumentBuilder:
    """
    Fluent builder for constructing arguments.

    Example:
        argument = (ArgumentBuilder(position)
            .claim("X is better than Y")
            .evidence("Studies show X outperforms Y by 20%")
            .warrant("Higher performance indicates superiority")
            .qualifier("generally")
            .build())
    """

    def __init__(self, position: Position):
        """Initialize builder with a position."""
        self._position = position
        self._claim = ""
        self._evidence = ""
        self._warrant = ""
        self._backing = ""
        self._qualifier = ""
        self._rebuttal = ""
        self._round_number = 0

    def claim(self, claim: str) -> "ArgumentBuilder":
        """Set the claim."""
        self._claim = claim
        return self

    def evidence(self, evidence: str) -> "ArgumentBuilder":
        """Set the evidence/data."""
        self._evidence = evidence
        return self

    def warrant(self, warrant: str) -> "ArgumentBuilder":
        """Set the warrant."""
        self._warrant = warrant
        return self

    def backing(self, backing: str) -> "ArgumentBuilder":
        """Set the backing."""
        self._backing = backing
        return self

    def qualifier(self, qualifier: str) -> "ArgumentBuilder":
        """Set the qualifier."""
        self._qualifier = qualifier
        return self

    def rebuttal(self, rebuttal: str) -> "ArgumentBuilder":
        """Set the rebuttal."""
        self._rebuttal = rebuttal
        return self

    def in_round(self, round_number: int) -> "ArgumentBuilder":
        """Set the round number."""
        self._round_number = round_number
        return self

    def build(self) -> Argument:
        """Build and return the argument."""
        return construct_argument(
            position=self._position,
            claim=self._claim,
            evidence=self._evidence,
            warrant=self._warrant,
            backing=self._backing,
            qualifier=self._qualifier,
            rebuttal=self._rebuttal,
            round_number=self._round_number,
        )


def create_argument_from_llm_response(
    response: str,
    position: Position,
    round_number: int = 0,
    parser: Optional[Callable[[str], dict[str, str]]] = None,
) -> Argument:
    """
    Create an argument from an LLM response.

    Uses the parser (defaults to parse_argument) to extract components.

    Args:
        response: Raw LLM response text
        position: Position this argument supports
        round_number: Current round number
        parser: Custom parser function (optional)

    Returns:
        Constructed Argument instance
    """
    parse_fn = parser or parse_argument
    components = parse_fn(response)

    return construct_argument(
        position=position,
        claim=components.get("claim", response[:200]),
        evidence=components.get("evidence", ""),
        warrant=components.get("warrant", ""),
        backing=components.get("backing", ""),
        qualifier=components.get("qualifier", ""),
        rebuttal=components.get("rebuttal", ""),
        round_number=round_number,
    )


def validate_argument(argument: Argument) -> tuple[bool, list[str]]:
    """
    Validate that an argument meets minimum requirements.

    Args:
        argument: The argument to validate

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues: list[str] = []

    if not argument.claim:
        issues.append("Argument must have a claim")

    if not argument.position_id:
        issues.append("Argument must be associated with a position")

    if argument.strength < 0.0 or argument.strength > 1.0:
        issues.append("Strength must be between 0.0 and 1.0")

    return len(issues) == 0, issues
