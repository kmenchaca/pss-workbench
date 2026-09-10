"""
Dream Logic Generator - Association Database

Manages concept associations for surprising connections and chains.
"""

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from .types import AssociativeLeap, LeapType


@dataclass
class Association:
    """A single association between concepts."""
    source: str
    target: str
    leap_type: LeapType
    weight: float = 1.0  # Higher = more common/obvious, lower = more surprising

    @property
    def surprise_factor(self) -> float:
        """How surprising is this association? (inverse of weight)."""
        return 1.0 - min(self.weight, 1.0)


# Built-in associations organized by source concept
BUILT_IN_ASSOCIATIONS: dict[str, list[Association]] = {
    "clock": [
        Association("clock", "melting cheese", LeapType.METAPHOR, 0.2),
        Association("clock", "heartbeat", LeapType.METAPHOR, 0.6),
        Association("clock", "prison", LeapType.ABSTRACTION, 0.3),
        Association("clock", "forever", LeapType.INVERSION, 0.4),
        Association("clock", "sand falling upward", LeapType.INVERSION, 0.1),
    ],
    "door": [
        Association("door", "mouth", LeapType.METAPHOR, 0.4),
        Association("door", "question", LeapType.ABSTRACTION, 0.3),
        Association("door", "wall that forgot its purpose", LeapType.INVERSION, 0.15),
        Association("door", "the space between breaths", LeapType.ABSTRACTION, 0.1),
    ],
    "mirror": [
        Association("mirror", "twin who hates you", LeapType.PERSONIFICATION, 0.2),
        Association("mirror", "lake at midnight", LeapType.METAPHOR, 0.5),
        Association("mirror", "memory", LeapType.ABSTRACTION, 0.3),
        Association("mirror", "the wrong version", LeapType.INVERSION, 0.15),
    ],
    "tree": [
        Association("tree", "lungs of a sleeping giant", LeapType.METAPHOR, 0.2),
        Association("tree", "upside-down river", LeapType.INVERSION, 0.15),
        Association("tree", "waiting", LeapType.ABSTRACTION, 0.25),
        Association("tree", "ancestor who never left", LeapType.PERSONIFICATION, 0.2),
    ],
    "water": [
        Association("water", "solid thoughts", LeapType.INVERSION, 0.1),
        Association("water", "forgetting", LeapType.ABSTRACTION, 0.3),
        Association("water", "time you can touch", LeapType.METAPHOR, 0.2),
        Association("water", "the world before language", LeapType.ABSTRACTION, 0.15),
    ],
    "fire": [
        Association("fire", "hungry flower", LeapType.PERSONIFICATION, 0.2),
        Association("fire", "cold rage", LeapType.INVERSION, 0.15),
        Association("fire", "the sun's tantrum", LeapType.SCALE_SHIFT, 0.3),
        Association("fire", "what ideas feel like", LeapType.ABSTRACTION, 0.2),
    ],
    "star": [
        Association("star", "pinhole in a dark blanket", LeapType.SCALE_SHIFT, 0.3),
        Association("star", "the dead speaking", LeapType.TIME_WARP, 0.2),
        Association("star", "wish that escaped", LeapType.PERSONIFICATION, 0.25),
        Association("star", "atom of universe", LeapType.SCALE_SHIFT, 0.2),
    ],
    "shadow": [
        Association("shadow", "soul's stain", LeapType.METAPHOR, 0.2),
        Association("shadow", "the part of you that remembers being nothing", LeapType.ABSTRACTION, 0.1),
        Association("shadow", "light's nightmare", LeapType.INVERSION, 0.15),
        Association("shadow", "faithful betrayer", LeapType.PERSONIFICATION, 0.2),
    ],
    "key": [
        Association("key", "frozen possibility", LeapType.ABSTRACTION, 0.2),
        Association("key", "metal question", LeapType.METAPHOR, 0.25),
        Association("key", "lock's secret lover", LeapType.PERSONIFICATION, 0.15),
        Association("key", "the shape of permission", LeapType.ABSTRACTION, 0.2),
    ],
    "house": [
        Association("house", "skull you chose", LeapType.METAPHOR, 0.2),
        Association("house", "memory with walls", LeapType.ABSTRACTION, 0.25),
        Association("house", "the portable prison", LeapType.INVERSION, 0.2),
        Association("house", "a creature that swallowed you gently", LeapType.PERSONIFICATION, 0.1),
    ],
}

# Generic associations for any concept
GENERIC_ASSOCIATIONS: list[tuple[str, LeapType]] = [
    ("what it feels like to forget {}", LeapType.ABSTRACTION),
    ("the opposite of {} dreaming", LeapType.INVERSION),
    ("if {} were a sound in empty space", LeapType.ABSTRACTION),
    ("{} but underwater and upside down", LeapType.INVERSION),
    ("the ghost of future {}", LeapType.TIME_WARP),
    ("{} at the scale of atoms", LeapType.SCALE_SHIFT),
    ("{} at the scale of galaxies", LeapType.SCALE_SHIFT),
    ("the moment before {} existed", LeapType.TIME_WARP),
    ("{} if it had feelings", LeapType.PERSONIFICATION),
    ("the color of {}", LeapType.ABSTRACTION),
]


class AssociationDB:
    """Database of concept associations."""

    def __init__(
        self,
        llm_generator: Optional[Callable[[str], list[str]]] = None
    ) -> None:
        """
        Initialize the association database.

        Args:
            llm_generator: Optional function to generate associations via LLM.
                          Takes a concept string, returns list of associated concepts.
        """
        self._associations: dict[str, list[Association]] = {}
        self._llm_generator = llm_generator

        # Load built-in associations
        for concept, assocs in BUILT_IN_ASSOCIATIONS.items():
            self._associations[concept.lower()] = assocs.copy()

    def get_associations(self, concept: str) -> list[Association]:
        """
        Find associations for a concept.

        Args:
            concept: The concept to find associations for.

        Returns:
            List of associations, possibly empty.
        """
        normalized = concept.lower().strip()

        # Check for direct match
        if normalized in self._associations:
            return self._associations[normalized]

        # Check for partial matches
        matches: list[Association] = []
        for key, assocs in self._associations.items():
            if key in normalized or normalized in key:
                matches.extend(assocs)

        if matches:
            return matches

        # Generate generic associations
        return self._generate_generic_associations(concept)

    def _generate_generic_associations(self, concept: str) -> list[Association]:
        """Generate generic associations for unknown concepts."""
        associations = []
        for template, leap_type in GENERIC_ASSOCIATIONS:
            target = template.format(concept)
            weight = random.uniform(0.1, 0.4)  # Random low weight = high surprise
            associations.append(Association(
                source=concept,
                target=target,
                leap_type=leap_type,
                weight=weight
            ))
        return associations

    def add_association(
        self,
        source: str,
        target: str,
        leap_type: LeapType,
        weight: float = 0.5
    ) -> Association:
        """
        Add a new association to the database.

        Args:
            source: Source concept.
            target: Target concept.
            leap_type: Type of conceptual leap.
            weight: Association weight (higher = more common).

        Returns:
            The created association.
        """
        normalized = source.lower().strip()
        assoc = Association(source=source, target=target, leap_type=leap_type, weight=weight)

        if normalized not in self._associations:
            self._associations[normalized] = []

        self._associations[normalized].append(assoc)
        return assoc

    def surprising_association(self, concept: str) -> Optional[Association]:
        """
        Get the most surprising (least obvious) association for a concept.

        Args:
            concept: The concept to find surprising association for.

        Returns:
            The most surprising association, or None if no associations exist.
        """
        associations = self.get_associations(concept)
        if not associations:
            return None

        # Sort by surprise factor (inverse of weight), return most surprising
        sorted_assocs = sorted(associations, key=lambda a: a.surprise_factor, reverse=True)
        return sorted_assocs[0]

    def chain_associations(
        self,
        start: str,
        depth: int = 3,
        visited: Optional[set[str]] = None
    ) -> list[AssociativeLeap]:
        """
        Follow a chain of associations from a starting concept.

        Args:
            start: Starting concept.
            depth: How many leaps to follow.
            visited: Set of already-visited concepts (for recursion).

        Returns:
            List of associative leaps forming the chain.
        """
        if visited is None:
            visited = set()

        if depth <= 0 or start.lower() in visited:
            return []

        visited.add(start.lower())
        chain: list[AssociativeLeap] = []

        # Get a surprising association
        assoc = self.surprising_association(start)
        if assoc is None:
            # Generate one if needed
            assocs = self._generate_generic_associations(start)
            if not assocs:
                return []
            assoc = random.choice(assocs)

        # Create leap
        leap = AssociativeLeap(
            source=assoc.source,
            target=assoc.target,
            leap_type=assoc.leap_type,
            distance=assoc.surprise_factor
        )
        chain.append(leap)

        # Continue chain from the target
        # Extract key concept from target for chaining
        next_concept = self._extract_key_concept(assoc.target)
        if next_concept and next_concept.lower() not in visited:
            chain.extend(self.chain_associations(next_concept, depth - 1, visited))

        return chain

    def _extract_key_concept(self, text: str) -> Optional[str]:
        """Extract a key concept from association text for chaining."""
        # Simple heuristic: take the last significant word
        words = text.split()
        # Filter out common words
        skip_words = {
            'the', 'a', 'an', 'of', 'in', 'at', 'to', 'for', 'with',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'that', 'this', 'it', 'its', 'if', 'but', 'and', 'or'
        }
        significant = [w for w in words if w.lower() not in skip_words]
        return significant[-1] if significant else None

    def generate_with_llm(self, concept: str) -> list[Association]:
        """
        Generate new associations using LLM if available.

        Args:
            concept: Concept to generate associations for.

        Returns:
            List of generated associations.
        """
        if self._llm_generator is None:
            return []

        try:
            generated = self._llm_generator(concept)
            associations = []
            for target in generated:
                leap_type = random.choice(list(LeapType))
                weight = random.uniform(0.1, 0.4)  # Assume LLM generates surprising ones
                assoc = self.add_association(concept, target, leap_type, weight)
                associations.append(assoc)
            return associations
        except Exception:
            return []

    @property
    def all_concepts(self) -> list[str]:
        """Get all concepts in the database."""
        return list(self._associations.keys())

    def random_concept(self) -> Optional[str]:
        """Get a random concept from the database."""
        if not self._associations:
            return None
        return random.choice(list(self._associations.keys()))


def get_associations(concept: str, db: Optional[AssociationDB] = None) -> list[Association]:
    """
    Convenience function to get associations for a concept.

    Args:
        concept: The concept to find associations for.
        db: Optional existing database (creates new if not provided).

    Returns:
        List of associations.
    """
    if db is None:
        db = AssociationDB()
    return db.get_associations(concept)


def surprising_association(concept: str, db: Optional[AssociationDB] = None) -> Optional[Association]:
    """
    Convenience function to get most surprising association.

    Args:
        concept: The concept to find surprising association for.
        db: Optional existing database.

    Returns:
        Most surprising association or None.
    """
    if db is None:
        db = AssociationDB()
    return db.surprising_association(concept)


def chain_associations(
    start: str,
    depth: int = 3,
    db: Optional[AssociationDB] = None
) -> list[AssociativeLeap]:
    """
    Convenience function to chain associations.

    Args:
        start: Starting concept.
        depth: Chain depth.
        db: Optional existing database.

    Returns:
        List of associative leaps.
    """
    if db is None:
        db = AssociationDB()
    return db.chain_associations(start, depth)


def add_association(
    source: str,
    target: str,
    leap_type: LeapType,
    db: Optional[AssociationDB] = None
) -> Association:
    """
    Convenience function to add an association.

    Args:
        source: Source concept.
        target: Target concept.
        leap_type: Type of leap.
        db: Optional existing database.

    Returns:
        Created association.
    """
    if db is None:
        db = AssociationDB()
    return db.add_association(source, target, leap_type)
