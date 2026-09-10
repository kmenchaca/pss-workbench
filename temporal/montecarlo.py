"""Monte Carlo aggregation for the Temporal Scenario Planner.

This module provides Monte Carlo simulation capabilities for running
many scenarios and analyzing probability distributions of outcomes.
"""

import math
import random
import statistics
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Callable

from .actors import BaseActor
from .simulation import simulate_timeline
from .state import create_initial_state
from .types import (
    Decision,
    MonteCarloResult,
    SensitivityResult,
    Timeline,
    TimelineState,
)


def monte_carlo_simulate(
    initial_state: TimelineState,
    decisions: list[Decision],
    n_runs: int,
    horizon: timedelta,
    actors: list[BaseActor] | None = None,
    llm_provider: Callable[[str], str] | None = None,
    randomize_decisions: bool = False,
    callback: Callable[[int, Timeline], None] | None = None,
) -> list[Timeline]:
    """Run Monte Carlo simulation with many runs.

    Args:
        initial_state: Starting state.
        decisions: Decisions to include.
        n_runs: Number of simulation runs.
        horizon: Simulation horizon.
        actors: Actors in the simulation.
        llm_provider: Optional LLM.
        randomize_decisions: Whether to randomize decision choices.
        callback: Optional callback after each run.

    Returns:
        List of simulated timelines.
    """
    timelines: list[Timeline] = []

    for i in range(n_runs):
        # Optionally vary decisions
        run_decisions = decisions
        if randomize_decisions:
            run_decisions = _randomize_decisions(decisions)

        # Simulate
        timeline = simulate_timeline(
            initial_state=deepcopy(initial_state),
            decisions=run_decisions,
            horizon=horizon,
            actors=actors,
            llm_provider=llm_provider,
        )

        timelines.append(timeline)

        if callback:
            callback(i, timeline)

    return timelines


def _randomize_decisions(decisions: list[Decision]) -> list[Decision]:
    """Randomize decision choices."""
    randomized = []
    for decision in decisions:
        new_decision = deepcopy(decision)
        if new_decision.options:
            new_decision.chosen = random.choice(new_decision.options)
        randomized.append(new_decision)
    return randomized


def probability_distribution(
    timelines: list[Timeline],
    metric: str,
) -> MonteCarloResult:
    """Calculate probability distribution of a metric.

    Args:
        timelines: Timelines to analyze.
        metric: Metric name to analyze.

    Returns:
        MonteCarloResult with distribution statistics.
    """
    values: list[float] = []

    for timeline in timelines:
        if not timeline.final_state:
            continue

        if hasattr(timeline.final_state.metrics, metric):
            value = getattr(timeline.final_state.metrics, metric)
        elif metric in timeline.final_state.metrics.custom_metrics:
            value = timeline.final_state.metrics.custom_metrics[metric]
        else:
            continue

        values.append(value)

    if not values:
        return MonteCarloResult(n_runs=len(timelines), metric_name=metric)

    # Calculate statistics
    mean = statistics.mean(values)
    std_dev = statistics.stdev(values) if len(values) > 1 else 0

    # Calculate percentiles
    sorted_values = sorted(values)
    n = len(sorted_values)
    percentiles = {
        5: sorted_values[int(n * 0.05)] if n > 0 else 0,
        25: sorted_values[int(n * 0.25)] if n > 0 else 0,
        50: sorted_values[int(n * 0.50)] if n > 0 else 0,
        75: sorted_values[int(n * 0.75)] if n > 0 else 0,
        95: sorted_values[int(n * 0.95)] if n > 0 else 0,
    }

    # 95% confidence interval
    z = 1.96
    margin = z * std_dev / math.sqrt(n) if n > 0 else 0
    confidence_interval = (mean - margin, mean + margin)

    return MonteCarloResult(
        n_runs=len(timelines),
        metric_name=metric,
        mean=mean,
        std_dev=std_dev,
        percentiles=percentiles,
        distribution=values,
        confidence_interval=confidence_interval,
    )


def confidence_intervals(
    timelines: list[Timeline],
    metrics: list[str] | None = None,
    confidence_level: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Calculate confidence intervals for metrics.

    Args:
        timelines: Timelines to analyze.
        metrics: Metrics to calculate (default: all standard metrics).
        confidence_level: Confidence level (default 0.95).

    Returns:
        Dict mapping metric names to (lower, upper) intervals.
    """
    if metrics is None:
        metrics = ["success_score", "risk_score", "opportunity_score", "stability_score"]

    intervals: dict[str, tuple[float, float]] = {}

    # Z-score for confidence level
    z_scores = {
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576,
    }
    z = z_scores.get(confidence_level, 1.96)

    for metric in metrics:
        result = probability_distribution(timelines, metric)

        if result.n_runs > 0 and result.std_dev > 0:
            margin = z * result.std_dev / math.sqrt(result.n_runs)
            intervals[metric] = (result.mean - margin, result.mean + margin)
        else:
            intervals[metric] = (result.mean, result.mean)

    return intervals


def sensitivity_analysis(
    initial_state: TimelineState,
    decisions: list[Decision],
    input_name: str,
    input_range: tuple[float, float],
    n_samples: int = 10,
    horizon: timedelta | None = None,
    metric: str = "success_score",
    actors: list[BaseActor] | None = None,
    llm_provider: Callable[[str], str] | None = None,
) -> SensitivityResult:
    """Analyze sensitivity of output to an input variable.

    Args:
        initial_state: Starting state.
        decisions: Decisions to include.
        input_name: Name of input to vary (context key).
        input_range: Range of values to test.
        n_samples: Number of samples across range.
        horizon: Simulation horizon.
        metric: Output metric to measure.
        actors: Actors in simulation.
        llm_provider: Optional LLM.

    Returns:
        SensitivityResult with analysis.
    """
    horizon = horizon or timedelta(days=365)

    input_values: list[float] = []
    output_values: list[float] = []

    step = (input_range[1] - input_range[0]) / (n_samples - 1) if n_samples > 1 else 0

    for i in range(n_samples):
        input_value = input_range[0] + (i * step)
        input_values.append(input_value)

        # Modify initial state
        modified_state = deepcopy(initial_state)
        modified_state.context[input_name] = input_value

        # Simulate
        timeline = simulate_timeline(
            initial_state=modified_state,
            decisions=decisions,
            horizon=horizon,
            actors=actors,
            llm_provider=llm_provider,
        )

        # Get output metric
        if timeline.final_state:
            if hasattr(timeline.final_state.metrics, metric):
                output_values.append(getattr(timeline.final_state.metrics, metric))
            elif metric in timeline.final_state.metrics.custom_metrics:
                output_values.append(timeline.final_state.metrics.custom_metrics[metric])
            else:
                output_values.append(0.0)
        else:
            output_values.append(0.0)

    # Calculate sensitivity metrics
    correlation = _calculate_correlation(input_values, output_values)
    sensitivity_score = abs(correlation)

    # Calculate elasticity at midpoint
    elasticity = _calculate_elasticity(input_values, output_values)

    # Find breakpoints (where output changes significantly)
    breakpoints = _find_breakpoints(input_values, output_values)

    return SensitivityResult(
        input_name=input_name,
        sensitivity_score=sensitivity_score,
        correlation=correlation,
        elasticity=elasticity,
        breakpoints=breakpoints,
    )


def _calculate_correlation(x: list[float], y: list[float]) -> float:
    """Calculate Pearson correlation coefficient."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))

    sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
    sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)

    denominator = math.sqrt(sum_sq_x * sum_sq_y)

    if denominator == 0:
        return 0.0

    return numerator / denominator


def _calculate_elasticity(x: list[float], y: list[float]) -> float:
    """Calculate elasticity (% change in y per % change in x)."""
    if len(x) < 2 or len(y) < 2:
        return 0.0

    mid_idx = len(x) // 2
    if mid_idx == 0:
        mid_idx = 1

    x1, x2 = x[mid_idx - 1], x[mid_idx]
    y1, y2 = y[mid_idx - 1], y[mid_idx]

    if x1 == 0 or y1 == 0:
        return 0.0

    pct_change_x = (x2 - x1) / x1
    pct_change_y = (y2 - y1) / y1

    if pct_change_x == 0:
        return 0.0

    return pct_change_y / pct_change_x


def _find_breakpoints(x: list[float], y: list[float], threshold: float = 0.1) -> list[float]:
    """Find input values where output changes significantly."""
    if len(x) < 3:
        return []

    breakpoints: list[float] = []

    for i in range(1, len(y) - 1):
        # Check for significant change in slope
        left_slope = (y[i] - y[i - 1]) / (x[i] - x[i - 1]) if x[i] != x[i - 1] else 0
        right_slope = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) if x[i + 1] != x[i] else 0

        if abs(right_slope - left_slope) > threshold:
            breakpoints.append(x[i])

    return breakpoints


def multi_metric_analysis(
    timelines: list[Timeline],
    metrics: list[str] | None = None,
) -> dict[str, MonteCarloResult]:
    """Analyze multiple metrics across timelines.

    Args:
        timelines: Timelines to analyze.
        metrics: Metrics to analyze (default: all standard).

    Returns:
        Dict mapping metric names to MonteCarloResults.
    """
    if metrics is None:
        metrics = ["success_score", "risk_score", "opportunity_score", "stability_score"]

    results: dict[str, MonteCarloResult] = {}

    for metric in metrics:
        results[metric] = probability_distribution(timelines, metric)

    return results


def scenario_probability(
    timelines: list[Timeline],
    condition: Callable[[Timeline], bool],
) -> float:
    """Calculate probability of a scenario condition.

    Args:
        timelines: Timelines to analyze.
        condition: Function that returns True if condition is met.

    Returns:
        Probability (0.0 to 1.0).
    """
    if not timelines:
        return 0.0

    matching = sum(1 for t in timelines if condition(t))
    return matching / len(timelines)


def value_at_risk(
    timelines: list[Timeline],
    metric: str = "success_score",
    confidence: float = 0.95,
) -> float:
    """Calculate Value at Risk for a metric.

    VaR is the worst expected outcome at a given confidence level.

    Args:
        timelines: Timelines to analyze.
        metric: Metric to analyze.
        confidence: Confidence level.

    Returns:
        Value at risk.
    """
    result = probability_distribution(timelines, metric)

    if not result.distribution:
        return 0.0

    # VaR is the (1 - confidence) percentile
    sorted_values = sorted(result.distribution)
    index = int(len(sorted_values) * (1 - confidence))
    index = max(0, min(index, len(sorted_values) - 1))

    return sorted_values[index]


def conditional_value_at_risk(
    timelines: list[Timeline],
    metric: str = "success_score",
    confidence: float = 0.95,
) -> float:
    """Calculate Conditional Value at Risk (Expected Shortfall).

    CVaR is the expected value given that we're in the tail of the distribution.

    Args:
        timelines: Timelines to analyze.
        metric: Metric to analyze.
        confidence: Confidence level.

    Returns:
        Expected shortfall.
    """
    result = probability_distribution(timelines, metric)

    if not result.distribution:
        return 0.0

    # Get tail values
    sorted_values = sorted(result.distribution)
    cutoff_index = int(len(sorted_values) * (1 - confidence))
    tail_values = sorted_values[:max(1, cutoff_index)]

    return statistics.mean(tail_values) if tail_values else 0.0


def monte_carlo_summary(
    timelines: list[Timeline],
    metrics: list[str] | None = None,
) -> str:
    """Create summary of Monte Carlo analysis.

    Args:
        timelines: Timelines to summarize.
        metrics: Metrics to include.

    Returns:
        Human-readable summary.
    """
    if metrics is None:
        metrics = ["success_score", "risk_score", "opportunity_score", "stability_score"]

    lines = [
        "Monte Carlo Simulation Summary",
        "=" * 40,
        f"Number of runs: {len(timelines)}",
    ]

    results = multi_metric_analysis(timelines, metrics)

    for metric_name, result in results.items():
        lines.append(f"\n{metric_name}:")
        lines.append(f"  Mean: {result.mean:.2f}")
        lines.append(f"  Std Dev: {result.std_dev:.2f}")
        lines.append(f"  95% CI: ({result.confidence_interval[0]:.2f}, {result.confidence_interval[1]:.2f})")
        lines.append(f"  Range: [{result.percentiles.get(5, 0):.2f}, {result.percentiles.get(95, 0):.2f}]")

    # Add scenario probabilities
    success_prob = scenario_probability(
        timelines,
        lambda t: t.final_state is not None and t.final_state.metrics.success_score > 0.6
    )
    failure_prob = scenario_probability(
        timelines,
        lambda t: t.final_state is not None and t.final_state.metrics.success_score < 0.3
    )

    lines.append(f"\nScenario probabilities:")
    lines.append(f"  Success (>60%): {success_prob:.0%}")
    lines.append(f"  Failure (<30%): {failure_prob:.0%}")

    # Add VaR
    var = value_at_risk(timelines, "success_score", 0.95)
    cvar = conditional_value_at_risk(timelines, "success_score", 0.95)

    lines.append(f"\nRisk metrics:")
    lines.append(f"  Value at Risk (95%): {var:.2f}")
    lines.append(f"  Conditional VaR (95%): {cvar:.2f}")

    return "\n".join(lines)
