"""
Dream Logic Generator - Human Curation Interface

Allows humans to guide the dream exploration by rating fragments,
selecting favorites, and providing feedback that improves the
interestingness model.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
from datetime import datetime
import uuid

from .types import DreamFragment, DreamCollage, DreamStyle
from .interestingness import InterestingnessModel, score_interestingness


@dataclass
class CurationRating:
    """A human rating for a fragment."""
    fragment_id: str
    rating: float  # 0 to 1
    timestamp: datetime = field(default_factory=datetime.now)
    notes: Optional[str] = None
    tags: list[str] = field(default_factory=list)


@dataclass
class CurationChoice:
    """A selection/rejection decision."""
    fragment_id: str
    selected: bool
    timestamp: datetime = field(default_factory=datetime.now)
    reason: Optional[str] = None


@dataclass
class CurationSession:
    """
    A curation session for human guidance.

    Presents fragments to humans, collects ratings and selections,
    and uses feedback to improve the interestingness model.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    fragments: list[DreamFragment] = field(default_factory=list)
    ratings: list[CurationRating] = field(default_factory=list)
    choices: list[CurationChoice] = field(default_factory=list)
    favorites: list[str] = field(default_factory=list)  # Fragment IDs
    model: Optional[InterestingnessModel] = None

    def __post_init__(self):
        if self.model is None:
            self.model = InterestingnessModel()

    def add_fragments(self, fragments: list[DreamFragment]) -> None:
        """Add fragments to the curation session."""
        self.fragments.extend(fragments)

    def add_fragment(self, fragment: DreamFragment) -> None:
        """Add a single fragment to the session."""
        self.fragments.append(fragment)

    def get_fragment(self, fragment_id: str) -> Optional[DreamFragment]:
        """Get a fragment by ID."""
        for fragment in self.fragments:
            if fragment.id == fragment_id:
                return fragment
        return None

    def rate_fragment(
        self,
        fragment_id: str,
        rating: float,
        notes: Optional[str] = None,
        tags: Optional[list[str]] = None
    ) -> CurationRating:
        """
        Rate a fragment (human feedback).

        Args:
            fragment_id: ID of fragment to rate.
            rating: Rating from 0 to 1.
            notes: Optional notes about the rating.
            tags: Optional tags for categorization.

        Returns:
            The created rating.
        """
        # Clamp rating
        rating = max(0.0, min(1.0, rating))

        curation_rating = CurationRating(
            fragment_id=fragment_id,
            rating=rating,
            notes=notes,
            tags=tags or [],
        )
        self.ratings.append(curation_rating)

        # Add to model for learning
        fragment = self.get_fragment(fragment_id)
        if fragment and self.model:
            self.model.add_rating(fragment.content, rating)

        return curation_rating

    def select_fragment(
        self,
        fragment_id: str,
        selected: bool = True,
        reason: Optional[str] = None
    ) -> CurationChoice:
        """
        Select or reject a fragment.

        Args:
            fragment_id: ID of fragment.
            selected: True if selected, False if rejected.
            reason: Optional reason for the choice.

        Returns:
            The created choice.
        """
        choice = CurationChoice(
            fragment_id=fragment_id,
            selected=selected,
            reason=reason,
        )
        self.choices.append(choice)

        if selected and fragment_id not in self.favorites:
            self.favorites.append(fragment_id)
        elif not selected and fragment_id in self.favorites:
            self.favorites.remove(fragment_id)

        return choice

    def select_favorites(self, fragment_ids: list[str]) -> list[CurationChoice]:
        """
        Select multiple fragments as favorites.

        Args:
            fragment_ids: IDs of fragments to select.

        Returns:
            List of created choices.
        """
        choices = []
        for fid in fragment_ids:
            choice = self.select_fragment(fid, selected=True)
            choices.append(choice)
        return choices

    def get_favorites(self) -> list[DreamFragment]:
        """Get all favorite fragments."""
        return [f for f in self.fragments if f.id in self.favorites]

    def get_rated_fragments(self) -> list[tuple[DreamFragment, float]]:
        """Get fragments with their ratings."""
        fragment_ratings: dict[str, float] = {}
        for rating in self.ratings:
            # Use most recent rating for each fragment
            fragment_ratings[rating.fragment_id] = rating.rating

        result = []
        for fragment in self.fragments:
            if fragment.id in fragment_ratings:
                result.append((fragment, fragment_ratings[fragment.id]))
        return result

    def learn_from_curation(self) -> None:
        """
        Update the interestingness model based on curation feedback.

        Uses ratings to adjust model weights.
        """
        if self.model:
            self.model.learn_from_ratings()

    def get_recommendations(self, n: int = 5) -> list[DreamFragment]:
        """
        Get recommended fragments based on learned preferences.

        Args:
            n: Number of recommendations.

        Returns:
            List of recommended fragments.
        """
        if not self.fragments or not self.model:
            return []

        # Score all fragments
        scored = []
        for fragment in self.fragments:
            # Skip already rated/selected
            if fragment.id in self.favorites:
                continue

            score = score_interestingness(fragment.content, model=self.model)
            scored.append((fragment, score.total))

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        return [f for f, _ in scored[:n]]

    def get_statistics(self) -> dict:
        """Get session statistics."""
        avg_rating = 0.0
        if self.ratings:
            avg_rating = sum(r.rating for r in self.ratings) / len(self.ratings)

        return {
            "total_fragments": len(self.fragments),
            "rated_fragments": len(set(r.fragment_id for r in self.ratings)),
            "selected_fragments": len(self.favorites),
            "average_rating": avg_rating,
            "total_ratings": len(self.ratings),
        }


class InteractiveCurator:
    """
    Interactive curator for terminal-based curation.

    Presents fragments one at a time and collects ratings.
    """

    def __init__(
        self,
        session: Optional[CurationSession] = None,
        display_fn: Optional[Callable[[str], None]] = None,
        input_fn: Optional[Callable[[str], str]] = None,
    ):
        """
        Initialize the interactive curator.

        Args:
            session: Existing session or create new.
            display_fn: Function to display text (default: print).
            input_fn: Function to get input (default: input).
        """
        self.session = session or CurationSession()
        self.display = display_fn or print
        self.get_input = input_fn or input

    def present_fragment(self, fragment: DreamFragment) -> Optional[CurationRating]:
        """
        Present a fragment for rating.

        Args:
            fragment: Fragment to present.

        Returns:
            Rating if provided, None if skipped.
        """
        self.display("\n" + "=" * 50)
        self.display(f"Fragment [{fragment.id}]")
        if fragment.style:
            self.display(f"Style: {fragment.style.name}")
        self.display("-" * 50)
        self.display(fragment.content)
        self.display("-" * 50)

        # Get rating
        while True:
            response = self.get_input("Rate (0-5, or 's' to skip): ").strip().lower()

            if response == 's':
                return None

            try:
                rating = float(response)
                if 0 <= rating <= 5:
                    # Convert to 0-1 scale
                    normalized = rating / 5.0
                    return self.session.rate_fragment(fragment.id, normalized)
                else:
                    self.display("Please enter a number between 0 and 5.")
            except ValueError:
                self.display("Invalid input. Enter a number 0-5 or 's' to skip.")

    def curate_all(self) -> None:
        """Curate all fragments in the session."""
        self.display(f"\nCurating {len(self.session.fragments)} fragments...")

        for i, fragment in enumerate(self.session.fragments):
            self.display(f"\n[{i + 1}/{len(self.session.fragments)}]")
            self.present_fragment(fragment)

        # Learn from ratings
        self.session.learn_from_curation()

        # Show summary
        stats = self.session.get_statistics()
        self.display("\n" + "=" * 50)
        self.display("Curation Complete!")
        self.display(f"Rated: {stats['rated_fragments']}/{stats['total_fragments']}")
        self.display(f"Average rating: {stats['average_rating']:.2f}")
        self.display("=" * 50)

    def select_favorites(self) -> list[DreamFragment]:
        """
        Interactive favorite selection.

        Returns:
            List of selected favorite fragments.
        """
        self.display("\nSelect your favorites (enter fragment IDs, comma-separated):")

        for fragment in self.session.fragments:
            preview = fragment.content[:100].replace('\n', ' ')
            self.display(f"  [{fragment.id}] {preview}...")

        response = self.get_input("\nFavorite IDs: ").strip()
        if not response:
            return []

        ids = [id.strip() for id in response.split(',')]
        self.session.select_favorites(ids)

        return self.session.get_favorites()


def create_curation_session(
    fragments: Optional[list[DreamFragment]] = None,
    collage: Optional[DreamCollage] = None,
) -> CurationSession:
    """
    Create a curation session from fragments or a collage.

    Args:
        fragments: Optional list of fragments.
        collage: Optional collage to extract fragments from.

    Returns:
        New curation session.
    """
    session = CurationSession()

    if fragments:
        session.add_fragments(fragments)

    if collage:
        session.add_fragments(collage.fragments)

    return session


def rate_fragment(
    fragment: DreamFragment,
    rating: float,
    session: Optional[CurationSession] = None
) -> CurationRating:
    """
    Convenience function to rate a single fragment.

    Args:
        fragment: Fragment to rate.
        rating: Rating from 0 to 1.
        session: Optional existing session.

    Returns:
        The created rating.
    """
    if session is None:
        session = CurationSession()
        session.add_fragment(fragment)

    return session.rate_fragment(fragment.id, rating)


def select_favorites(
    fragments: list[DreamFragment],
    selected_ids: list[str],
) -> list[DreamFragment]:
    """
    Convenience function to select favorites from a list.

    Args:
        fragments: All fragments.
        selected_ids: IDs of selected favorites.

    Returns:
        List of selected fragments.
    """
    session = CurationSession()
    session.add_fragments(fragments)
    session.select_favorites(selected_ids)
    return session.get_favorites()


def learn_from_curation(
    ratings: list[tuple[str, float]],
    model: Optional[InterestingnessModel] = None
) -> InterestingnessModel:
    """
    Train an interestingness model from curation data.

    Args:
        ratings: List of (content, rating) tuples.
        model: Optional existing model.

    Returns:
        Trained model.
    """
    if model is None:
        model = InterestingnessModel()

    for content, rating in ratings:
        model.add_rating(content, rating)

    model.learn_from_ratings()
    return model
