"""System Router - Automatic routing to the best system for a task.

Uses keyword analysis and pattern matching to suggest which experimental
system is best suited for a given prompt.
"""

import re
from dataclasses import dataclass
from typing import Optional

from .config import SystemType, SYSTEM_INFO


# Keywords and patterns for routing
ROUTING_PATTERNS = {
    SystemType.REDTEAM: {
        "keywords": [
            "security", "vulnerability", "attack", "exploit", "audit",
            "pentest", "threat", "weakness", "flaw", "review code",
            "find bugs", "hack", "bypass", "injection", "xss", "csrf",
        ],
        "patterns": [
            r"find\s+(?:security\s+)?(?:issues?|problems?|bugs?|flaws?)",
            r"(?:security|code)\s+(?:audit|review)",
            r"attack\s+(?:this|the)",
            r"what(?:'s| is)\s+wrong\s+with",
        ],
    },
    SystemType.EVOLUTION: {
        "keywords": [
            "optimize", "evolve", "genetic", "fitness", "mutate",
            "generation", "breed", "selection", "algorithm", "improve code",
            "better performance", "faster", "efficient",
        ],
        "patterns": [
            r"optimize\s+(?:this|the)\s+(?:code|algorithm|function)",
            r"make\s+(?:this|it)\s+(?:faster|more efficient|better)",
            r"evolve\s+(?:a|this)\s+(?:solution|algorithm)",
            r"improve\s+(?:the\s+)?performance",
        ],
    },
    SystemType.ARGSWARM: {
        "keywords": [
            "debate", "argue", "proposition", "thesis", "for and against",
            "pros and cons", "should we", "is it true", "evaluate claim",
            "both sides", "counterargument",
        ],
        "patterns": [
            r"debate\s+(?:whether|if)",
            r"(?:pros?|cons?)\s+(?:and|&)\s+(?:pros?|cons?)",
            r"should\s+(?:we|i|they)",
            r"is\s+(?:it|this)\s+(?:true|correct|right)\s+that",
            r"argue\s+(?:for|against)",
        ],
    },
    SystemType.PERSONAS: {
        "keywords": [
            "perspective", "stakeholder", "viewpoint", "persona", "role",
            "different views", "multiple angles", "who would", "how would X see",
            "optimist", "pessimist", "expert", "novice",
        ],
        "patterns": [
            r"(?:different|multiple|various)\s+(?:perspectives?|viewpoints?|angles?)",
            r"how\s+would\s+(?:a|an|the)\s+\w+\s+(?:see|view|think)",
            r"from\s+(?:the\s+)?perspective\s+of",
            r"(?:stakeholder|persona)\s+analysis",
        ],
    },
    SystemType.TEMPORAL: {
        "keywords": [
            "future", "scenario", "timeline", "forecast", "predict",
            "what if", "projection", "years from now", "long term",
            "trend", "trajectory",
        ],
        "patterns": [
            r"what\s+(?:will|might|could)\s+happen",
            r"in\s+(?:\d+|the\s+next)\s+(?:years?|months?|decades?)",
            r"(?:future|long.?term)\s+(?:scenario|outlook|projection)",
            r"how\s+(?:will|might)\s+(?:this|things?)\s+(?:unfold|develop|evolve)",
        ],
    },
    SystemType.KNOWLEDGE: {
        "keywords": [
            "concept", "relationship", "graph", "map", "connect",
            "how does X relate to Y", "explain the connections",
            "knowledge", "ontology", "taxonomy",
        ],
        "patterns": [
            r"(?:knowledge|concept)\s+(?:graph|map)",
            r"how\s+(?:does?|do)\s+\w+\s+relate\s+to",
            r"(?:map|explain)\s+(?:the\s+)?(?:concepts?|relationships?|connections?)",
            r"what\s+are\s+the\s+(?:key\s+)?(?:concepts?|ideas?|relationships?)",
        ],
    },
    SystemType.EMBODIED: {
        "keywords": [
            "robot", "physical", "action", "move", "pick up", "navigate",
            "environment", "constraint", "workspace", "manipulate",
            "plan steps", "execute",
        ],
        "patterns": [
            r"(?:robot|physical)\s+(?:action|task|plan)",
            r"how\s+(?:to|would\s+you)\s+(?:physically|actually)\s+(?:do|perform)",
            r"step.?by.?step\s+(?:physical\s+)?actions?",
            r"(?:pick up|move|navigate|manipulate)",
        ],
    },
    SystemType.METALEARNER: {
        "keywords": [
            "learn", "adapt", "improve over time", "pattern",
            "meta", "strategy", "self-improve", "feedback",
            "what worked", "what didn't",
        ],
        "patterns": [
            r"(?:learn|adapt)\s+(?:from|over\s+time)",
            r"what\s+(?:worked|didn't work|failed)",
            r"(?:improve|optimize)\s+(?:the\s+)?(?:strategy|approach|process)",
            r"(?:meta|self).?(?:learn|improve|optimize)",
        ],
    },
    SystemType.DREAMLOGIC: {
        "keywords": [
            "creative", "dream", "imagine", "weird", "surreal",
            "brainstorm", "wild ideas", "free association", "bizarre",
            "unexpected", "lateral", "outside the box",
        ],
        "patterns": [
            r"(?:creative|wild|crazy|weird)\s+ideas?",
            r"(?:brainstorm|imagine|dream\s+up)",
            r"(?:outside|beyond)\s+the\s+box",
            r"what\s+if\s+(?:we\s+)?(?:completely|totally|radically)",
            r"free\s+(?:association|thinking)",
        ],
    },
    SystemType.NEGOTIATE: {
        "keywords": [
            "negotiate", "deal", "agreement", "compromise", "parties",
            "conflict", "mediate", "resolve", "settlement", "terms",
            "concession", "BATNA",
        ],
        "patterns": [
            r"(?:negotiate|reach)\s+(?:a|an)\s+(?:deal|agreement|settlement)",
            r"(?:resolve|mediate)\s+(?:the\s+)?(?:conflict|dispute)",
            r"(?:multiple|different)\s+(?:parties|sides|stakeholders)",
            r"what\s+(?:should|would)\s+(?:each\s+)?(?:party|side)\s+(?:offer|accept)",
        ],
    },
}


@dataclass
class RouteDecision:
    """Result of routing decision.

    Attributes:
        system_type: Recommended system
        confidence: Confidence score (0.0 to 1.0)
        reasoning: Why this system was chosen
        alternatives: Other possible systems with scores
    """

    system_type: SystemType
    confidence: float
    reasoning: str
    alternatives: list[tuple[SystemType, float]]


class SystemRouter:
    """Routes prompts to the most appropriate system."""

    def __init__(self, custom_patterns: Optional[dict] = None):
        """Initialize router.

        Args:
            custom_patterns: Additional routing patterns to merge
        """
        self.patterns = dict(ROUTING_PATTERNS)
        if custom_patterns:
            for sys_type, patterns in custom_patterns.items():
                if sys_type in self.patterns:
                    self.patterns[sys_type]["keywords"].extend(
                        patterns.get("keywords", [])
                    )
                    self.patterns[sys_type]["patterns"].extend(
                        patterns.get("patterns", [])
                    )
                else:
                    self.patterns[sys_type] = patterns

    def route(self, prompt: str) -> RouteDecision:
        """Route a prompt to the best system.

        Args:
            prompt: The user's prompt

        Returns:
            RouteDecision with recommended system
        """
        scores = self._score_all_systems(prompt)

        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        best_system, best_score = ranked[0]
        alternatives = ranked[1:4]  # Top 3 alternatives

        # Generate reasoning
        reasoning = self._generate_reasoning(prompt, best_system, best_score)

        return RouteDecision(
            system_type=best_system,
            confidence=best_score,
            reasoning=reasoning,
            alternatives=alternatives,
        )

    def _score_all_systems(self, prompt: str) -> dict[SystemType, float]:
        """Score all systems for a prompt.

        Args:
            prompt: The user's prompt

        Returns:
            Dictionary of system type to score
        """
        prompt_lower = prompt.lower()
        scores = {}

        for sys_type, patterns in self.patterns.items():
            score = 0.0

            # Keyword matching
            keywords = patterns.get("keywords", [])
            for keyword in keywords:
                if keyword.lower() in prompt_lower:
                    score += 0.15

            # Pattern matching
            regex_patterns = patterns.get("patterns", [])
            for pattern in regex_patterns:
                if re.search(pattern, prompt_lower, re.IGNORECASE):
                    score += 0.25

            # Cap at 1.0
            scores[sys_type] = min(1.0, score)

        # If no strong signals, default to personas (most general)
        if all(s < 0.2 for s in scores.values()):
            scores[SystemType.PERSONAS] = 0.3

        return scores

    def _generate_reasoning(
        self,
        prompt: str,
        system: SystemType,
        score: float
    ) -> str:
        """Generate human-readable reasoning for the route decision.

        Args:
            prompt: Original prompt
            system: Chosen system
            score: Confidence score

        Returns:
            Reasoning string
        """
        info = SYSTEM_INFO.get(system, {})
        name = info.get("name", system.value)
        description = info.get("description", "")
        best_for = info.get("best_for", [])

        if score >= 0.5:
            confidence_desc = "high confidence"
        elif score >= 0.3:
            confidence_desc = "moderate confidence"
        else:
            confidence_desc = "low confidence"

        parts = [
            f"Routing to {name} with {confidence_desc} ({score:.0%}).",
            f"This system {description.lower()}.",
        ]

        if best_for:
            parts.append(f"Best suited for: {', '.join(best_for[:3])}.")

        return " ".join(parts)

    def suggest_for_task_type(self, task_type: str) -> list[SystemType]:
        """Suggest systems for a general task type.

        Args:
            task_type: Type of task (e.g., "analysis", "creative", "planning")

        Returns:
            List of suitable system types
        """
        task_mapping = {
            "analysis": [
                SystemType.PERSONAS,
                SystemType.ARGSWARM,
                SystemType.REDTEAM,
            ],
            "creative": [
                SystemType.DREAMLOGIC,
                SystemType.EVOLUTION,
                SystemType.PERSONAS,
            ],
            "planning": [
                SystemType.TEMPORAL,
                SystemType.EMBODIED,
                SystemType.KNOWLEDGE,
            ],
            "security": [
                SystemType.REDTEAM,
                SystemType.ARGSWARM,
            ],
            "debate": [
                SystemType.ARGSWARM,
                SystemType.NEGOTIATE,
            ],
            "learning": [
                SystemType.METALEARNER,
                SystemType.KNOWLEDGE,
            ],
            "negotiation": [
                SystemType.NEGOTIATE,
                SystemType.PERSONAS,
                SystemType.ARGSWARM,
            ],
        }

        return task_mapping.get(task_type.lower(), [SystemType.PERSONAS])


def auto_route(prompt: str) -> RouteDecision:
    """Convenience function to auto-route a prompt.

    Args:
        prompt: The user's prompt

    Returns:
        RouteDecision
    """
    router = SystemRouter()
    return router.route(prompt)
