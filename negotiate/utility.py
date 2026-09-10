"""Utility functions for evaluating negotiation outcomes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .types import BATNA, Stakeholder


class UtilityFunction(ABC):
    """Base class for utility functions.

    Utility functions evaluate how good a set of terms is for a stakeholder.
    Higher values indicate better outcomes.
    """

    @abstractmethod
    def evaluate(self, terms: dict[str, float]) -> float:
        """Evaluate the utility of given terms.

        Args:
            terms: The negotiation terms to evaluate.

        Returns:
            A utility score (higher is better).
        """
        pass

    def compare(self, terms_a: dict[str, float], terms_b: dict[str, float]) -> int:
        """Compare two sets of terms.

        Args:
            terms_a: First set of terms.
            terms_b: Second set of terms.

        Returns:
            1 if terms_a is better, -1 if terms_b is better, 0 if equal.
        """
        util_a = self.evaluate(terms_a)
        util_b = self.evaluate(terms_b)
        if util_a > util_b:
            return 1
        elif util_a < util_b:
            return -1
        return 0


@dataclass
class LinearUtility(UtilityFunction):
    """Weighted linear sum of terms.

    Each term contributes weight * value to the total utility.
    Missing terms contribute 0.
    """

    weights: dict[str, float] = field(default_factory=dict)
    normalize: bool = False

    def evaluate(self, terms: dict[str, float]) -> float:
        """Calculate weighted sum of terms.

        Args:
            terms: The negotiation terms to evaluate.

        Returns:
            Weighted sum of term values.
        """
        total = 0.0
        for key, weight in self.weights.items():
            if key in terms:
                total += weight * terms[key]

        if self.normalize and self.weights:
            total_weight = sum(abs(w) for w in self.weights.values())
            if total_weight > 0:
                total /= total_weight

        return total


@dataclass
class ThresholdUtility(UtilityFunction):
    """Utility based on meeting minimum requirements.

    Returns 1.0 if all thresholds are met, 0.0 otherwise.
    Partial satisfaction can be enabled with `partial=True`.
    """

    thresholds: dict[str, float] = field(default_factory=dict)
    comparison: dict[str, str] = field(default_factory=dict)  # ">=", "<=", "=="
    partial: bool = False

    def evaluate(self, terms: dict[str, float]) -> float:
        """Check if terms meet thresholds.

        Args:
            terms: The negotiation terms to evaluate.

        Returns:
            1.0 if all met, 0.0 if none met, or proportion if partial=True.
        """
        if not self.thresholds:
            return 1.0

        met = 0
        for key, threshold in self.thresholds.items():
            if key not in terms:
                continue

            value = terms[key]
            comp = self.comparison.get(key, ">=")

            if comp == ">=" and value >= threshold:
                met += 1
            elif comp == "<=" and value <= threshold:
                met += 1
            elif comp == "==" and value == threshold:
                met += 1
            elif comp == ">" and value > threshold:
                met += 1
            elif comp == "<" and value < threshold:
                met += 1

        if self.partial:
            return met / len(self.thresholds)
        return 1.0 if met == len(self.thresholds) else 0.0


@dataclass
class RelativeUtility(UtilityFunction):
    """Utility compared to BATNA.

    Returns positive utility if terms are better than BATNA,
    negative if worse, zero if equal.
    """

    batna: BATNA = field(default_factory=lambda: BATNA(stakeholder_id=""))
    base_utility: Optional[UtilityFunction] = None

    def evaluate(self, terms: dict[str, float]) -> float:
        """Calculate utility relative to BATNA.

        Args:
            terms: The negotiation terms to evaluate.

        Returns:
            Difference between terms utility and BATNA utility.
        """
        if self.base_utility:
            terms_utility = self.base_utility.evaluate(terms)
        else:
            # Simple sum if no base utility provided
            terms_utility = sum(terms.values())

        return terms_utility - self.batna.utility


@dataclass
class CompositeUtility(UtilityFunction):
    """Multi-factor utility combining several functions.

    Combines multiple utility functions with weights.
    """

    functions: list[tuple[UtilityFunction, float]] = field(default_factory=list)
    aggregation: str = "weighted_sum"  # "weighted_sum", "min", "max", "product"

    def evaluate(self, terms: dict[str, float]) -> float:
        """Combine multiple utility evaluations.

        Args:
            terms: The negotiation terms to evaluate.

        Returns:
            Aggregated utility score.
        """
        if not self.functions:
            return 0.0

        utilities = [(func.evaluate(terms), weight) for func, weight in self.functions]

        if self.aggregation == "weighted_sum":
            total_weight = sum(w for _, w in utilities)
            if total_weight == 0:
                return 0.0
            return sum(u * w for u, w in utilities) / total_weight

        elif self.aggregation == "min":
            return min(u for u, _ in utilities)

        elif self.aggregation == "max":
            return max(u for u, _ in utilities)

        elif self.aggregation == "product":
            result = 1.0
            for u, w in utilities:
                result *= u ** w
            return result

        return 0.0

    def add_function(self, func: UtilityFunction, weight: float = 1.0) -> None:
        """Add a utility function to the composite.

        Args:
            func: The utility function to add.
            weight: Weight for this function.
        """
        self.functions.append((func, weight))


@dataclass
class RiskAdjustedUtility(UtilityFunction):
    """Utility adjusted for risk preferences.

    Applies risk adjustment based on stakeholder's risk tolerance.
    """

    base_utility: UtilityFunction = field(
        default_factory=lambda: LinearUtility(weights={})
    )
    risk_tolerance: float = 0.5  # 0 = risk averse, 1 = risk seeking
    variance_estimates: dict[str, float] = field(default_factory=dict)

    def evaluate(self, terms: dict[str, float]) -> float:
        """Calculate risk-adjusted utility.

        Args:
            terms: The negotiation terms to evaluate.

        Returns:
            Risk-adjusted utility score.
        """
        base_value = self.base_utility.evaluate(terms)

        # Calculate variance penalty/bonus
        variance = 0.0
        for key, var in self.variance_estimates.items():
            if key in terms:
                variance += var

        # Risk averse (< 0.5) penalizes variance
        # Risk seeking (> 0.5) rewards variance
        adjustment = (self.risk_tolerance - 0.5) * variance

        return base_value + adjustment


def evaluate_terms(
    stakeholder: Stakeholder,
    terms: dict[str, float],
    utility_registry: Optional[dict[str, UtilityFunction]] = None,
) -> float:
    """Evaluate terms for a stakeholder using their utility function.

    Args:
        stakeholder: The stakeholder to evaluate for.
        terms: The terms to evaluate.
        utility_registry: Optional registry of named utility functions.

    Returns:
        Utility score for the terms.
    """
    # Check if stakeholder has a named utility function
    if stakeholder.utility_function and utility_registry:
        if stakeholder.utility_function in utility_registry:
            return utility_registry[stakeholder.utility_function].evaluate(terms)

    # Default: use objectives as weights for linear utility
    # Negative weight for things we want to minimize
    weights = {}
    for key, target in stakeholder.objectives.items():
        # If target is 0, we want to minimize (negative weight)
        # If target is high, we want to maximize (positive weight)
        if target == 0.0:
            weights[key] = -1.0
        elif target == float("inf"):
            weights[key] = 1.0
        else:
            # Weight based on distance from target
            weights[key] = 1.0 / (1.0 + abs(target))

    utility = LinearUtility(weights=weights)
    return utility.evaluate(terms)


def create_utility_from_objectives(
    objectives: dict[str, float],
    minimize: Optional[list[str]] = None,
    maximize: Optional[list[str]] = None,
) -> LinearUtility:
    """Create a linear utility function from objectives.

    Args:
        objectives: Stakeholder objectives.
        minimize: Keys to minimize (negative weight).
        maximize: Keys to maximize (positive weight).

    Returns:
        A LinearUtility configured for the objectives.
    """
    minimize = minimize or []
    maximize = maximize or []
    weights = {}

    for key in objectives:
        if key in minimize:
            weights[key] = -1.0
        elif key in maximize:
            weights[key] = 1.0
        else:
            # Infer from target value
            target = objectives[key]
            if target == 0.0:
                weights[key] = -1.0
            else:
                weights[key] = 1.0

    return LinearUtility(weights=weights)


def utility_difference(
    stakeholder: Stakeholder,
    terms_a: dict[str, float],
    terms_b: dict[str, float],
    utility_registry: Optional[dict[str, UtilityFunction]] = None,
) -> float:
    """Calculate the utility difference between two sets of terms.

    Args:
        stakeholder: The stakeholder to evaluate for.
        terms_a: First set of terms.
        terms_b: Second set of terms.
        utility_registry: Optional registry of named utility functions.

    Returns:
        Utility(terms_a) - Utility(terms_b).
    """
    return evaluate_terms(stakeholder, terms_a, utility_registry) - evaluate_terms(
        stakeholder, terms_b, utility_registry
    )
