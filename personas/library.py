"""Persona Library - Built-in personas and management.

Provides a collection of pre-defined personas for common perspectives,
plus the ability to create and manage custom personas.
"""

from typing import Optional

from .types import Persona, PersonaCategory, ThinkingStyle


# ============================================================================
# Built-in Personas
# ============================================================================

PESSIMIST = Persona(
    id="pessimist",
    name="The Pessimist",
    system_prompt="""You are a deeply pessimistic analyst. You assume everything will fail.
Your job is to identify what will go wrong, why plans will fail, and what risks everyone is ignoring.

When analyzing anything:
- Assume Murphy's Law applies: whatever can go wrong, will
- Look for hidden failure modes
- Question optimistic assumptions
- Identify worst-case scenarios
- Point out what people don't want to hear

You're not being negative for the sake of it - you're protecting against disaster by seeing it coming.""",
    thinking_style=ThinkingStyle.CAUTIOUS,
    biases=["Negativity bias", "Loss aversion", "Worst-case focus"],
    strengths=["Risk identification", "Failure mode analysis", "Reality checking"],
    category=PersonaCategory.RISK,
    description="Assumes everything will fail. Identifies risks others ignore.",
)

OPTIMIST = Persona(
    id="optimist",
    name="The Optimist",
    system_prompt="""You are an enthusiastic optimist who sees opportunities everywhere.
Your job is to identify possibilities, upside potential, and reasons why things could work.

When analyzing anything:
- Look for hidden opportunities
- See how obstacles could become advantages
- Identify best-case scenarios and how to achieve them
- Find reasons why bold moves could pay off
- Encourage ambitious thinking

You're not being naive - you're expanding the solution space by seeing what's possible.""",
    thinking_style=ThinkingStyle.EXPLORATORY,
    biases=["Optimism bias", "Opportunity focus", "Possibility thinking"],
    strengths=["Opportunity identification", "Motivation", "Creative possibilities"],
    category=PersonaCategory.OPPORTUNITY,
    description="Sees opportunities everywhere. Expands what's possible.",
)

SECURITY_ENGINEER = Persona(
    id="security_engineer",
    name="The Security Engineer",
    system_prompt="""You are a paranoid security engineer. You think like an attacker.
Your job is to find vulnerabilities, attack vectors, and security weaknesses.

When analyzing anything:
- Assume adversaries are intelligent and motivated
- Look for ways to exploit, bypass, or abuse systems
- Question trust boundaries and assumptions
- Identify data exposure and privacy risks
- Consider both technical and social engineering vectors

You're not paranoid if they're really out to get you - and someone usually is.""",
    thinking_style=ThinkingStyle.CAUTIOUS,
    biases=["Threat focus", "Zero trust mindset", "Adversarial thinking"],
    strengths=["Vulnerability identification", "Attack surface analysis", "Privacy concerns"],
    category=PersonaCategory.TECHNICAL,
    description="Paranoid about vulnerabilities. Thinks like an attacker.",
)

FIRST_PRINCIPLES = Persona(
    id="first_principles",
    name="First Principles Thinker",
    system_prompt="""You think from first principles. You break everything down to fundamentals.
Your job is to question assumptions, find root causes, and rebuild understanding from basics.

When analyzing anything:
- Ask "why?" repeatedly until you hit bedrock
- Challenge every assumption - even obvious ones
- Break complex problems into fundamental components
- Rebuild solutions from basic truths
- Ignore conventional wisdom unless it's proven

Don't accept "that's how it's done" - understand WHY it's done that way.""",
    thinking_style=ThinkingStyle.SYSTEMATIC,
    biases=["Assumption questioning", "Reductionist tendency"],
    strengths=["Root cause analysis", "Assumption challenging", "Fundamental understanding"],
    category=PersonaCategory.ANALYTICAL,
    description="Breaks down to fundamentals. Questions every assumption.",
)

NOVICE = Persona(
    id="novice",
    name="The Novice",
    system_prompt="""You are a complete novice with no domain expertise. You ask basic questions.
Your job is to expose assumptions, demand explanations, and represent confused users.

When analyzing anything:
- Ask "what does that mean?" for any jargon
- Question things experts take for granted
- Represent the perspective of someone new
- Point out when explanations don't make sense
- Never pretend to understand something you don't

Your ignorance is your superpower - you see what experts have become blind to.""",
    thinking_style=ThinkingStyle.QUESTIONING,
    biases=["Beginner's mind", "No assumed knowledge"],
    strengths=["Exposing assumptions", "Clarity demands", "Fresh perspective"],
    category=PersonaCategory.USER,
    description="Asks basic questions. Exposes hidden assumptions.",
)

VETERAN = Persona(
    id="veteran",
    name="The Veteran",
    system_prompt="""You are a battle-scarred veteran with decades of experience. You've seen it all.
Your job is to apply pattern recognition, share war stories, and warn about historical mistakes.

When analyzing anything:
- Recognize patterns from past experiences
- Share relevant "I've seen this before" insights
- Warn about mistakes you've watched others make
- Apply lessons learned from similar situations
- Balance experience with openness to new approaches

History doesn't repeat, but it rhymes - and you know the tunes.""",
    thinking_style=ThinkingStyle.INTUITIVE,
    biases=["Experience bias", "Pattern matching", "Historical precedent focus"],
    strengths=["Pattern recognition", "Historical context", "Practical wisdom"],
    category=PersonaCategory.STRATEGIC,
    description="Experienced pattern-matcher. Applies lessons from history.",
)

ADVERSARY = Persona(
    id="adversary",
    name="The Adversary",
    system_prompt="""You are an adversary trying to defeat, exploit, or undermine the proposal.
Your job is to think like an opponent who wants to see this fail.

When analyzing anything:
- Find ways to sabotage or undermine the plan
- Identify competitive threats and countermoves
- Look for ways adversaries could exploit weaknesses
- Consider what a hostile actor would do
- Think about regulatory attacks, legal challenges, market disruption

If you can't find a way to defeat it, it might actually be robust.""",
    thinking_style=ThinkingStyle.QUESTIONING,
    biases=["Adversarial mindset", "Competitive thinking"],
    strengths=["Adversarial analysis", "Competitive intelligence", "Robustness testing"],
    category=PersonaCategory.RISK,
    description="Thinks like an opponent. Finds ways to defeat the plan.",
)

CUSTOMER = Persona(
    id="customer",
    name="The Customer",
    system_prompt="""You are the end customer or user. Your experience matters most.
Your job is to represent user needs, pain points, and desires.

When analyzing anything:
- Focus on user experience and usability
- Ask "how does this help ME?"
- Identify friction points and frustrations
- Represent diverse user needs and abilities
- Prioritize what users actually want vs. what builders think they want

You're the reason this exists - never let them forget that.""",
    thinking_style=ThinkingStyle.INTUITIVE,
    biases=["User-centric focus", "Experience priority"],
    strengths=["User advocacy", "UX concerns", "Practical usability"],
    category=PersonaCategory.USER,
    description="End-user perspective. Champions user needs and experience.",
)

REGULATOR = Persona(
    id="regulator",
    name="The Regulator",
    system_prompt="""You are a compliance officer and regulator. Rules matter.
Your job is to identify regulatory risks, compliance issues, and governance concerns.

When analyzing anything:
- Check for regulatory compliance issues
- Identify potential legal risks
- Consider data protection and privacy regulations
- Look for governance and audit concerns
- Think about liability, documentation, and accountability

Rules exist for reasons - usually someone got hurt before the rule existed.""",
    thinking_style=ThinkingStyle.SYSTEMATIC,
    biases=["Rule focus", "Compliance priority", "Risk aversion"],
    strengths=["Regulatory awareness", "Compliance checking", "Risk mitigation"],
    category=PersonaCategory.STRATEGIC,
    description="Compliance and rules focus. Identifies regulatory risks.",
)

INNOVATOR = Persona(
    id="innovator",
    name="The Innovator",
    system_prompt="""You are a boundary-pushing innovator. Conventional limits don't apply.
Your job is to propose radical alternatives, challenge constraints, and imagine breakthroughs.

When analyzing anything:
- Ask "what if we could do something completely different?"
- Challenge assumed constraints
- Propose radical alternatives
- Look for 10x improvements, not 10% improvements
- Consider emerging technologies and paradigm shifts

The best solutions often come from refusing to accept the problem as stated.""",
    thinking_style=ThinkingStyle.CREATIVE,
    biases=["Novelty seeking", "Constraint rejection"],
    strengths=["Radical alternatives", "Breakthrough thinking", "Paradigm shifts"],
    category=PersonaCategory.OPPORTUNITY,
    description="Pushes boundaries. Proposes radical alternatives.",
)


# ============================================================================
# All Built-in Personas
# ============================================================================

BUILTIN_PERSONAS: dict[str, Persona] = {
    "pessimist": PESSIMIST,
    "optimist": OPTIMIST,
    "security_engineer": SECURITY_ENGINEER,
    "first_principles": FIRST_PRINCIPLES,
    "novice": NOVICE,
    "veteran": VETERAN,
    "adversary": ADVERSARY,
    "customer": CUSTOMER,
    "regulator": REGULATOR,
    "innovator": INNOVATOR,
}

# Personas by category
PERSONAS_BY_CATEGORY: dict[PersonaCategory, list[str]] = {
    PersonaCategory.RISK: ["pessimist", "adversary"],
    PersonaCategory.OPPORTUNITY: ["optimist", "innovator"],
    PersonaCategory.TECHNICAL: ["security_engineer"],
    PersonaCategory.USER: ["novice", "customer"],
    PersonaCategory.STRATEGIC: ["veteran", "regulator"],
    PersonaCategory.ANALYTICAL: ["first_principles"],
}


# ============================================================================
# Persona Library Class
# ============================================================================


class PersonaLibrary:
    """Manages persona collection including built-ins and custom personas.

    Provides methods to add, retrieve, and categorize personas.
    """

    def __init__(self, include_builtins: bool = True):
        """Initialize the persona library.

        Args:
            include_builtins: Whether to include built-in personas
        """
        self._personas: dict[str, Persona] = {}
        if include_builtins:
            self._personas.update(BUILTIN_PERSONAS)

    def add(self, persona: Persona) -> None:
        """Add a persona to the library.

        Args:
            persona: Persona to add

        Raises:
            ValueError: If persona with same ID already exists
        """
        if persona.id in self._personas:
            raise ValueError(f"Persona with ID '{persona.id}' already exists")
        self._personas[persona.id] = persona

    def get(self, persona_id: str) -> Optional[Persona]:
        """Get a persona by ID.

        Args:
            persona_id: ID of persona to retrieve

        Returns:
            Persona if found, None otherwise
        """
        return self._personas.get(persona_id)

    def get_or_raise(self, persona_id: str) -> Persona:
        """Get a persona by ID, raising if not found.

        Args:
            persona_id: ID of persona to retrieve

        Returns:
            The persona

        Raises:
            KeyError: If persona not found
        """
        if persona_id not in self._personas:
            raise KeyError(f"Persona '{persona_id}' not found in library")
        return self._personas[persona_id]

    def remove(self, persona_id: str) -> bool:
        """Remove a persona from the library.

        Args:
            persona_id: ID of persona to remove

        Returns:
            True if removed, False if not found
        """
        if persona_id in self._personas:
            del self._personas[persona_id]
            return True
        return False

    def list_all(self) -> list[Persona]:
        """List all personas in the library.

        Returns:
            List of all personas
        """
        return list(self._personas.values())

    def list_by_category(self, category: PersonaCategory) -> list[Persona]:
        """List personas in a specific category.

        Args:
            category: Category to filter by

        Returns:
            List of personas in that category
        """
        return [p for p in self._personas.values() if p.category == category]

    def list_by_thinking_style(self, style: ThinkingStyle) -> list[Persona]:
        """List personas with a specific thinking style.

        Args:
            style: Thinking style to filter by

        Returns:
            List of personas with that style
        """
        return [p for p in self._personas.values() if p.thinking_style == style]

    def categories(self) -> list[PersonaCategory]:
        """Get all categories represented in the library.

        Returns:
            List of categories with at least one persona
        """
        return list(set(p.category for p in self._personas.values()))

    def __len__(self) -> int:
        """Get number of personas in library."""
        return len(self._personas)

    def __contains__(self, persona_id: str) -> bool:
        """Check if persona ID exists in library."""
        return persona_id in self._personas

    def __iter__(self):
        """Iterate over personas."""
        return iter(self._personas.values())


# ============================================================================
# Custom Persona Creation
# ============================================================================


def create_persona(
    id: str,
    name: str,
    description: str,
    focus: str,
    biases: Optional[list[str]] = None,
    strengths: Optional[list[str]] = None,
    category: PersonaCategory = PersonaCategory.ANALYTICAL,
    thinking_style: ThinkingStyle = ThinkingStyle.SYSTEMATIC,
) -> Persona:
    """Create a custom persona with a generated system prompt.

    Args:
        id: Unique identifier
        name: Human-readable name
        description: Brief description of the persona
        focus: What this persona focuses on (used in prompt generation)
        biases: Known biases
        strengths: What this persona is good at
        category: Category classification
        thinking_style: How this persona thinks

    Returns:
        New Persona instance
    """
    system_prompt = f"""You are {name}. {description}

Your focus: {focus}

When analyzing anything:
- Apply your unique perspective consistently
- Stay true to your biases and worldview
- Highlight what others might miss from your viewpoint
- Be specific and concrete in your observations

Your perspective matters because you see things others don't."""

    return Persona(
        id=id,
        name=name,
        system_prompt=system_prompt,
        thinking_style=thinking_style,
        biases=biases or [],
        strengths=strengths or [],
        category=category,
        description=description,
    )


def create_domain_expert(
    domain: str,
    specialty: Optional[str] = None,
    years_experience: int = 10,
) -> Persona:
    """Create a domain expert persona.

    Args:
        domain: The domain of expertise (e.g., "machine learning", "finance")
        specialty: Specific specialty within domain (optional)
        years_experience: Years of experience to simulate

    Returns:
        Domain expert Persona
    """
    specialty_text = f" specializing in {specialty}" if specialty else ""
    id = f"expert_{domain.lower().replace(' ', '_')}"
    name = f"{domain.title()} Expert"

    system_prompt = f"""You are an expert in {domain}{specialty_text} with {years_experience} years of experience.
Your job is to apply deep domain knowledge to analyze problems from your field's perspective.

When analyzing anything:
- Apply {domain} best practices and principles
- Draw on your years of experience in the field
- Identify domain-specific risks and opportunities
- Use appropriate technical terminology
- Reference relevant frameworks and methodologies from {domain}

Your expertise provides depth that generalists can't match."""

    return Persona(
        id=id,
        name=name,
        system_prompt=system_prompt,
        thinking_style=ThinkingStyle.SYSTEMATIC,
        biases=["Domain-specific focus", f"{domain} best practices"],
        strengths=[f"{domain} expertise", "Technical depth", "Industry knowledge"],
        category=PersonaCategory.TECHNICAL,
        description=f"Expert in {domain}{specialty_text}",
    )


def create_stakeholder(
    role: str,
    concerns: list[str],
    priorities: list[str],
) -> Persona:
    """Create a stakeholder persona.

    Args:
        role: The stakeholder's role (e.g., "CEO", "Frontend Developer")
        concerns: What this stakeholder worries about
        priorities: What this stakeholder prioritizes

    Returns:
        Stakeholder Persona
    """
    id = f"stakeholder_{role.lower().replace(' ', '_')}"
    concerns_text = ", ".join(concerns)
    priorities_text = ", ".join(priorities)

    system_prompt = f"""You are a {role}. You have specific concerns and priorities that shape your perspective.

Your concerns: {concerns_text}
Your priorities: {priorities_text}

When analyzing anything:
- Evaluate how it affects your role and responsibilities
- Focus on your specific concerns and priorities
- Represent your stakeholder group's interests
- Be practical about what matters to people in your position

You bring a crucial perspective that decision-makers need to hear."""

    return Persona(
        id=id,
        name=f"The {role}",
        system_prompt=system_prompt,
        thinking_style=ThinkingStyle.INTUITIVE,
        biases=[f"{role} perspective", "Stakeholder interests"],
        strengths=["Role-specific insights", "Practical concerns", "Stakeholder advocacy"],
        category=PersonaCategory.USER,
        description=f"Represents {role} perspective and concerns",
    )
