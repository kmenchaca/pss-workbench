"""Scenario comparison and synthesis for the Temporal Scenario Planner.

This module provides analysis across multiple timelines to find robust
decisions, brittle assumptions, and key uncertainties.
"""

import statistics
from collections import defaultdict
from typing import Any, Callable

from .causality import build_causal_graph, find_leverage_points
from .types import (
    BrittleDecision,
    Decision,
    KeyUncertainty,
    RobustDecision,
    ScenarioComparison,
    Timeline,
    TimelineMetrics,
)


def compare_timelines(timelines: list[Timeline]) -> ScenarioComparison:
    """Analyze and compare multiple timelines.

    Args:
        timelines: Timelines to compare.

    Returns:
        ScenarioComparison with analysis results.
    """
    if not timelines:
        return ScenarioComparison()

    comparison = ScenarioComparison(timelines=timelines)

    # Find best, worst, and most likely cases
    comparison.best_case = _find_best_case(timelines)
    comparison.worst_case = _find_worst_case(timelines)
    comparison.most_likely = _find_most_likely(timelines)

    # Analyze decisions
    comparison.robust_decisions = find_robust_decisions(timelines)
    comparison.brittle_decisions = find_brittle_decisions(timelines)

    # Identify uncertainties
    comparison.key_uncertainties = identify_key_uncertainties(timelines)

    # Calculate expected value
    comparison.expected_value = expected_value(timelines, "success_score")

    # Generate recommendation
    comparison.recommendation = _generate_recommendation(comparison)

    return comparison


def _find_best_case(timelines: list[Timeline]) -> Timeline | None:
    """Find the timeline with the best outcome."""
    valid = [t for t in timelines if t.final_state]
    if not valid:
        return None

    return max(
        valid,
        key=lambda t: t.final_state.metrics.success_score  # type: ignore
        - t.final_state.metrics.risk_score  # type: ignore
    )


def _find_worst_case(timelines: list[Timeline]) -> Timeline | None:
    """Find the timeline with the worst outcome."""
    valid = [t for t in timelines if t.final_state]
    if not valid:
        return None

    return min(
        valid,
        key=lambda t: t.final_state.metrics.success_score  # type: ignore
        - t.final_state.metrics.risk_score  # type: ignore
    )


def _find_most_likely(timelines: list[Timeline]) -> Timeline | None:
    """Find the timeline with highest probability."""
    if not timelines:
        return None

    return max(timelines, key=lambda t: t.probability * t.plausibility)


def find_robust_decisions(timelines: list[Timeline]) -> list[RobustDecision]:
    """Find decisions that perform well across all scenarios.

    A robust decision is one where:
    - The same choice appears in multiple successful timelines
    - Outcomes are consistently positive regardless of other factors

    Args:
        timelines: Timelines to analyze.

    Returns:
        List of robust decisions sorted by robustness score.
    """
    # Group decisions by description (same decision point)
    decision_groups: dict[str, list[tuple[Decision, Timeline]]] = defaultdict(list)
    for timeline in timelines:
        for decision in timeline.decisions:
            key = decision.description
            decision_groups[key].append((decision, timeline))

    robust: list[RobustDecision] = []

    for description, decision_timeline_pairs in decision_groups.items():
        if len(decision_timeline_pairs) < 2:
            continue

        # Group by choice made
        choices: dict[str, list[float]] = defaultdict(list)
        for decision, timeline in decision_timeline_pairs:
            if decision.chosen and timeline.final_state:
                score = timeline.final_state.metrics.success_score
                choices[decision.chosen].append(score)

        # Find most robust choice
        for choice, scores in choices.items():
            if len(scores) < 2:
                continue

            avg = statistics.mean(scores)
            variance = statistics.variance(scores) if len(scores) > 1 else 0

            # Robustness = high average + low variance
            robustness = avg * (1 - min(1, variance))

            # Get a representative decision
            rep_decision = next(
                d for d, _ in decision_timeline_pairs
                if d.chosen == choice
            )

            robust.append(RobustDecision(
                decision=rep_decision,
                avg_outcome=avg,
                min_outcome=min(scores),
                max_outcome=max(scores),
                variance=variance,
                robustness_score=robustness,
            ))

    robust.sort(key=lambda r: r.robustness_score, reverse=True)
    return robust


def find_brittle_decisions(timelines: list[Timeline]) -> list[BrittleDecision]:
    """Find decisions that only work in specific scenarios.

    A brittle decision is one where:
    - Outcomes vary widely based on other factors
    - Success depends on specific assumptions being true

    Args:
        timelines: Timelines to analyze.

    Returns:
        List of brittle decisions sorted by brittleness score.
    """
    decision_groups: dict[str, list[tuple[Decision, Timeline]]] = defaultdict(list)
    for timeline in timelines:
        for decision in timeline.decisions:
            decision_groups[decision.description].append((decision, timeline))

    brittle: list[BrittleDecision] = []

    for description, decision_timeline_pairs in decision_groups.items():
        if len(decision_timeline_pairs) < 2:
            continue

        # Analyze outcome variance
        choices: dict[str, list[tuple[float, str]]] = defaultdict(list)
        for decision, timeline in decision_timeline_pairs:
            if decision.chosen and timeline.final_state:
                score = timeline.final_state.metrics.success_score
                choices[decision.chosen].append((score, timeline.id))

        for choice, score_timelines in choices.items():
            if len(score_timelines) < 2:
                continue

            scores = [s for s, _ in score_timelines]
            score_range = max(scores) - min(scores)

            # High range indicates brittleness
            if score_range < 0.3:
                continue

            success_ids = [tid for s, tid in score_timelines if s > 0.6]
            failure_ids = [tid for s, tid in score_timelines if s < 0.4]

            # Infer critical assumptions from successful vs failed timelines
            assumptions = _infer_critical_assumptions(
                [t for t in timelines if t.id in success_ids],
                [t for t in timelines if t.id in failure_ids],
            )

            rep_decision = next(
                d for d, _ in decision_timeline_pairs
                if d.chosen == choice
            )

            brittleness = score_range * (1 - len(success_ids) / len(score_timelines))

            brittle.append(BrittleDecision(
                decision=rep_decision,
                success_scenarios=success_ids,
                failure_scenarios=failure_ids,
                critical_assumptions=assumptions,
                brittleness_score=brittleness,
            ))

    brittle.sort(key=lambda b: b.brittleness_score, reverse=True)
    return brittle


def _infer_critical_assumptions(
    success_timelines: list[Timeline],
    failure_timelines: list[Timeline],
) -> list[str]:
    """Infer what assumptions differentiate success from failure."""
    assumptions: list[str] = []

    if not success_timelines or not failure_timelines:
        return assumptions

    # Compare event categories
    success_events = set()
    for t in success_timelines:
        for e in t.events:
            success_events.add(e.description.split()[0])  # First word as category

    failure_events = set()
    for t in failure_timelines:
        for e in t.events:
            failure_events.add(e.description.split()[0])

    unique_to_success = success_events - failure_events
    unique_to_failure = failure_events - success_events

    if unique_to_success:
        assumptions.append(f"Requires: {', '.join(list(unique_to_success)[:3])}")
    if unique_to_failure:
        assumptions.append(f"Must avoid: {', '.join(list(unique_to_failure)[:3])}")

    # Compare metrics
    if success_timelines and failure_timelines:
        success_risk = statistics.mean(
            t.final_state.metrics.risk_score
            for t in success_timelines
            if t.final_state
        )
        failure_risk = statistics.mean(
            t.final_state.metrics.risk_score
            for t in failure_timelines
            if t.final_state
        )

        if failure_risk - success_risk > 0.2:
            assumptions.append("Requires low risk environment")

    return assumptions


def expected_value(
    timelines: list[Timeline],
    metric: str,
) -> float:
    """Calculate probability-weighted outcome for a metric.

    Args:
        timelines: Timelines to analyze.
        metric: Metric name to calculate.

    Returns:
        Expected value of the metric.
    """
    if not timelines:
        return 0.0

    total_weight = 0.0
    weighted_sum = 0.0

    for timeline in timelines:
        if not timeline.final_state:
            continue

        weight = timeline.probability * timeline.plausibility
        if weight <= 0:
            continue

        # Get metric value
        if hasattr(timeline.final_state.metrics, metric):
            value = getattr(timeline.final_state.metrics, metric)
        elif metric in timeline.final_state.metrics.custom_metrics:
            value = timeline.final_state.metrics.custom_metrics[metric]
        else:
            continue

        weighted_sum += value * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0

    return weighted_sum / total_weight


def identify_key_uncertainties(timelines: list[Timeline]) -> list[KeyUncertainty]:
    """Identify uncertainties that significantly affect outcomes.

    Args:
        timelines: Timelines to analyze.

    Returns:
        List of key uncertainties sorted by impact.
    """
    uncertainties: list[KeyUncertainty] = []

    if len(timelines) < 2:
        return uncertainties

    # Analyze variance in outcomes
    if any(t.final_state for t in timelines):
        scores = [
            t.final_state.metrics.success_score
            for t in timelines
            if t.final_state
        ]
        if len(scores) > 1 and statistics.variance(scores) > 0.1:
            uncertainties.append(KeyUncertainty(
                description="Overall success outcome",
                impact_range=(min(scores), max(scores)),
                affected_timelines=[t.id for t in timelines],
                controllable=True,
            ))

    # Find events that appear in some timelines but not others
    all_event_descriptions: dict[str, list[str]] = defaultdict(list)
    for timeline in timelines:
        for event in timeline.events:
            all_event_descriptions[event.description].append(timeline.id)

    for description, timeline_ids in all_event_descriptions.items():
        # Event appears in some but not all timelines
        fraction = len(timeline_ids) / len(timelines)
        if 0.2 < fraction < 0.8:
            # Calculate impact of this event
            with_event = [t for t in timelines if t.id in timeline_ids and t.final_state]
            without_event = [t for t in timelines if t.id not in timeline_ids and t.final_state]

            if with_event and without_event:
                with_avg = statistics.mean(
                    t.final_state.metrics.success_score for t in with_event  # type: ignore
                )
                without_avg = statistics.mean(
                    t.final_state.metrics.success_score for t in without_event  # type: ignore
                )

                impact = abs(with_avg - without_avg)
                if impact > 0.1:
                    uncertainties.append(KeyUncertainty(
                        description=f"Whether '{description}' occurs",
                        impact_range=(min(with_avg, without_avg), max(with_avg, without_avg)),
                        affected_timelines=timeline_ids,
                        controllable=False,
                    ))

    # Sort by impact range
    uncertainties.sort(
        key=lambda u: u.impact_range[1] - u.impact_range[0],
        reverse=True,
    )

    return uncertainties[:10]  # Top 10 uncertainties


def _generate_recommendation(comparison: ScenarioComparison) -> str:
    """Generate a strategic recommendation based on comparison."""
    parts: list[str] = []

    # Best case insight
    if comparison.best_case:
        parts.append(
            f"Best case scenario (success: "
            f"{comparison.best_case.final_state.metrics.success_score:.0%})"  # type: ignore
            f" achievable through careful planning."
        )

    # Robust decision recommendation
    if comparison.robust_decisions:
        top = comparison.robust_decisions[0]
        parts.append(
            f"Strongly recommend: '{top.decision.chosen}' for "
            f"'{top.decision.description}' (robust score: {top.robustness_score:.2f})."
        )

    # Brittle decision warning
    if comparison.brittle_decisions:
        top = comparison.brittle_decisions[0]
        parts.append(
            f"Caution with: '{top.decision.chosen}' for "
            f"'{top.decision.description}' - only works in specific scenarios."
        )

    # Key uncertainty
    if comparison.key_uncertainties:
        top = comparison.key_uncertainties[0]
        parts.append(
            f"Key uncertainty: {top.description} "
            f"(impact range: {top.impact_range[0]:.0%} to {top.impact_range[1]:.0%})."
        )

    # Expected value
    parts.append(f"Expected success: {comparison.expected_value:.0%}.")

    return " ".join(parts)


def scenario_matrix(
    timelines: list[Timeline],
    row_metric: str = "success_score",
    col_metric: str = "risk_score",
) -> dict[str, Any]:
    """Create a 2D matrix of scenarios by metrics.

    Args:
        timelines: Timelines to plot.
        row_metric: Metric for rows.
        col_metric: Metric for columns.

    Returns:
        Dict with matrix data and labels.
    """
    matrix: dict[str, Any] = {
        "row_metric": row_metric,
        "col_metric": col_metric,
        "quadrants": {
            "high_high": [],
            "high_low": [],
            "low_high": [],
            "low_low": [],
        },
    }

    for timeline in timelines:
        if not timeline.final_state:
            continue

        metrics = timeline.final_state.metrics
        row_val = getattr(metrics, row_metric, 0.5)
        col_val = getattr(metrics, col_metric, 0.5)

        row_high = row_val > 0.5
        col_high = col_val > 0.5

        quadrant = f"{'high' if row_high else 'low'}_{'high' if col_high else 'low'}"
        matrix["quadrants"][quadrant].append({
            "id": timeline.id,
            "row_value": row_val,
            "col_value": col_val,
            "probability": timeline.probability,
        })

    return matrix


def divergence_analysis(timelines: list[Timeline]) -> dict[str, Any]:
    """Analyze where timelines diverged.

    Args:
        timelines: Timelines to analyze.

    Returns:
        Dict describing divergence points.
    """
    if len(timelines) < 2:
        return {"divergence_points": []}

    # Find decisions where choices differed
    decision_variations: dict[str, set[str]] = defaultdict(set)
    for timeline in timelines:
        for decision in timeline.decisions:
            if decision.chosen:
                decision_variations[decision.description].add(decision.chosen)

    divergence_points: list[dict[str, Any]] = []
    for description, choices in decision_variations.items():
        if len(choices) > 1:
            # Find timelines for each choice
            by_choice: dict[str, list[str]] = defaultdict(list)
            for timeline in timelines:
                for decision in timeline.decisions:
                    if decision.description == description and decision.chosen:
                        by_choice[decision.chosen].append(timeline.id)

            divergence_points.append({
                "decision": description,
                "choices": list(choices),
                "timelines_by_choice": dict(by_choice),
            })

    return {
        "divergence_points": divergence_points,
        "total_variations": sum(len(d["choices"]) for d in divergence_points),
    }


def synthesis_summary(comparison: ScenarioComparison) -> str:
    """Create a human-readable synthesis summary.

    Args:
        comparison: Scenario comparison to summarize.

    Returns:
        Summary string.
    """
    lines = [
        "Temporal Scenario Synthesis",
        "=" * 40,
        f"Timelines analyzed: {len(comparison.timelines)}",
        f"Expected value: {comparison.expected_value:.0%}",
    ]

    if comparison.best_case and comparison.best_case.final_state:
        lines.append(
            f"\nBest case: {comparison.best_case.id} "
            f"(success: {comparison.best_case.final_state.metrics.success_score:.0%})"
        )

    if comparison.worst_case and comparison.worst_case.final_state:
        lines.append(
            f"Worst case: {comparison.worst_case.id} "
            f"(success: {comparison.worst_case.final_state.metrics.success_score:.0%})"
        )

    if comparison.robust_decisions:
        lines.append(f"\nRobust decisions ({len(comparison.robust_decisions)}):")
        for rd in comparison.robust_decisions[:3]:
            lines.append(
                f"  - {rd.decision.description}: '{rd.decision.chosen}' "
                f"(score: {rd.robustness_score:.2f})"
            )

    if comparison.brittle_decisions:
        lines.append(f"\nBrittle decisions ({len(comparison.brittle_decisions)}):")
        for bd in comparison.brittle_decisions[:3]:
            lines.append(
                f"  - {bd.decision.description}: '{bd.decision.chosen}' "
                f"(brittleness: {bd.brittleness_score:.2f})"
            )

    if comparison.key_uncertainties:
        lines.append(f"\nKey uncertainties ({len(comparison.key_uncertainties)}):")
        for ku in comparison.key_uncertainties[:3]:
            lines.append(f"  - {ku.description}")

    if comparison.recommendation:
        lines.append(f"\nRecommendation:\n{comparison.recommendation}")

    return "\n".join(lines)
