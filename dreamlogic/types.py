"""
Dream Logic Generator - Core Types

Dataclasses for dream fragments, associative leaps, interestingness scoring,
and dream collage synthesis.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import uuid


class DreamStyle(Enum):
    """Mode of dream generation."""
    SURREALIST = auto()  # Dali-esque, melting reality
    ABSURDIST = auto()   # Kafka, Beckett, logical illogic
    MYTHIC = auto()      # Archetypal, hero's journey twisted
    LIMINAL = auto()     # Threshold spaces, transitions
    COSMIC = auto()      # Vast scales, existential


class LeapType(Enum):
    """Types of associative leaps."""
    METAPHOR = auto()       # X is like Y
    INVERSION = auto()      # Opposite of X
    SCALE_SHIFT = auto()    # Microscopic/cosmic
    TIME_WARP = auto()      # Past/future shift
    PERSONIFICATION = auto() # Give life to inanimate
    ABSTRACTION = auto()    # Concrete to abstract


class ConstraintType(Enum):
    """Types of random creative constraints."""
    MUST_INCLUDE = auto()   # Must contain this element
    MUST_EXCLUDE = auto()   # Cannot contain this element
    STYLE = auto()          # Must adopt this style
    PERSPECTIVE = auto()    # Must use this viewpoint
    SCALE = auto()          # Must operate at this scale
    TIME = auto()           # Must be set in this time


@dataclass
class DreamFragment:
    """A piece of dream output."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str = ""
    associations: list[str] = field(default_factory=list)
    surprise_score: float = 0.0
    branch_id: Optional[str] = None
    style: Optional[DreamStyle] = None

    def __post_init__(self) -> None:
        """Validate fragment after initialization."""
        if self.surprise_score < 0.0:
            self.surprise_score = 0.0
        elif self.surprise_score > 1.0:
            self.surprise_score = 1.0


@dataclass
class AssociativeLeap:
    """Connection between concepts."""
    source: str
    target: str
    leap_type: LeapType
    distance: float = 0.5  # 0 = obvious, 1 = wildly unexpected

    def __post_init__(self) -> None:
        """Validate leap distance."""
        if self.distance < 0.0:
            self.distance = 0.0
        elif self.distance > 1.0:
            self.distance = 1.0


@dataclass
class InterestingnessScore:
    """Evaluation of content interestingness."""
    novelty: float = 0.0         # How new is this?
    surprise: float = 0.0        # How unexpected?
    aesthetic: float = 0.0       # How beautiful/evocative?
    coherence_penalty: float = 0.0  # Penalty for being too logical

    @property
    def total(self) -> float:
        """Calculate total interestingness score."""
        raw = (self.novelty + self.surprise + self.aesthetic) / 3.0
        return max(0.0, raw - self.coherence_penalty)

    def __post_init__(self) -> None:
        """Validate all scores are in range."""
        for attr in ['novelty', 'surprise', 'aesthetic', 'coherence_penalty']:
            value = getattr(self, attr)
            if value < 0.0:
                setattr(self, attr, 0.0)
            elif value > 1.0:
                setattr(self, attr, 1.0)


@dataclass
class Constraint:
    """Random constraint injection."""
    type: ConstraintType
    value: str
    difficulty: float = 0.5  # 0 = easy, 1 = hard

    def __post_init__(self) -> None:
        """Validate difficulty."""
        if self.difficulty < 0.0:
            self.difficulty = 0.0
        elif self.difficulty > 1.0:
            self.difficulty = 1.0


@dataclass
class DreamCollage:
    """Synthesis output - arrangement without smoothing."""
    fragments: list[DreamFragment] = field(default_factory=list)
    connections: list[AssociativeLeap] = field(default_factory=list)
    emergent_themes: list[str] = field(default_factory=list)
    style: Optional[DreamStyle] = None
    total_interestingness: float = 0.0

    def add_fragment(self, fragment: DreamFragment) -> None:
        """Add a fragment to the collage."""
        self.fragments.append(fragment)
        self._update_interestingness()

    def add_connection(self, leap: AssociativeLeap) -> None:
        """Add an associative connection."""
        self.connections.append(leap)

    def add_theme(self, theme: str) -> None:
        """Add an emergent theme."""
        if theme not in self.emergent_themes:
            self.emergent_themes.append(theme)

    def _update_interestingness(self) -> None:
        """Recalculate total interestingness from fragments."""
        if self.fragments:
            self.total_interestingness = sum(
                f.surprise_score for f in self.fragments
            ) / len(self.fragments)


@dataclass
class DreamBranch:
    """A branch in the dream exploration tree."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    fragments: list[DreamFragment] = field(default_factory=list)
    parent_id: Optional[str] = None
    style: Optional[DreamStyle] = None
    constraints: list[Constraint] = field(default_factory=list)
    boring_score: float = 0.0  # Higher = more boring, should be terminated
    alive: bool = True

    @property
    def interestingness(self) -> float:
        """Average interestingness of fragments."""
        if not self.fragments:
            return 0.0
        return sum(f.surprise_score for f in self.fragments) / len(self.fragments)


@dataclass
class DreamTree:
    """Tree of dream branches for exploration."""
    branches: dict[str, DreamBranch] = field(default_factory=dict)
    root_id: Optional[str] = None

    def add_branch(self, branch: DreamBranch) -> None:
        """Add a branch to the tree."""
        self.branches[branch.id] = branch
        if self.root_id is None:
            self.root_id = branch.id

    def get_branch(self, branch_id: str) -> Optional[DreamBranch]:
        """Get a branch by ID."""
        return self.branches.get(branch_id)

    def get_alive_branches(self) -> list[DreamBranch]:
        """Get all alive branches."""
        return [b for b in self.branches.values() if b.alive]

    def terminate_branch(self, branch_id: str) -> None:
        """Mark a branch as dead."""
        if branch_id in self.branches:
            self.branches[branch_id].alive = False

    def get_children(self, parent_id: str) -> list[DreamBranch]:
        """Get child branches of a parent."""
        return [
            b for b in self.branches.values()
            if b.parent_id == parent_id
        ]
