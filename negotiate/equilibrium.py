"""Game-theoretic analysis for negotiations."""

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Optional

from .types import GameOutcome, NegotiationState, Stakeholder
from .utility import evaluate_terms


@dataclass
class Game:
    """A game-theoretic representation of the negotiation.

    Attributes:
        players: List of stakeholder IDs.
        strategies: Available strategies per player.
        payoff_matrix: Payoffs for each strategy combination.
    """

    players: list[str] = field(default_factory=list)
    strategies: dict[str, list[str]] = field(default_factory=dict)
    payoff_matrix: dict[tuple[str, ...], dict[str, float]] = field(default_factory=dict)


def build_game(
    stakeholders: list[Stakeholder],
    strategy_options: Optional[dict[str, list[str]]] = None,
    payoff_function: Optional[callable] = None,
) -> Game:
    """Build a game from stakeholders and their strategies.

    Args:
        stakeholders: List of stakeholders.
        strategy_options: Available strategies per stakeholder.
        payoff_function: Function to compute payoffs for strategy combinations.

    Returns:
        A Game object.
    """
    players = [s.id for s in stakeholders]

    if strategy_options is None:
        # Default strategies
        default_strategies = ["cooperate", "compete", "compromise"]
        strategy_options = {s.id: default_strategies for s in stakeholders}

    game = Game(
        players=players,
        strategies=strategy_options,
    )

    # Build payoff matrix
    strategy_lists = [strategy_options[p] for p in players]
    for strategy_combo in product(*strategy_lists):
        combo_dict = dict(zip(players, strategy_combo))

        if payoff_function:
            payoffs = payoff_function(stakeholders, combo_dict)
        else:
            payoffs = _default_payoffs(stakeholders, combo_dict)

        game.payoff_matrix[strategy_combo] = payoffs

    return game


def _default_payoffs(
    stakeholders: list[Stakeholder],
    strategies: dict[str, str],
) -> dict[str, float]:
    """Calculate default payoffs for strategy combinations.

    Args:
        stakeholders: List of stakeholders.
        strategies: Strategy chosen by each player.

    Returns:
        Payoff for each player.
    """
    payoffs = {}

    # Count cooperators and competitors
    cooperators = sum(1 for s in strategies.values() if s == "cooperate")
    competitors = sum(1 for s in strategies.values() if s == "compete")
    total = len(strategies)

    for s in stakeholders:
        strategy = strategies.get(s.id, "compromise")

        if strategy == "cooperate":
            # Cooperators do well when others cooperate
            base = 60
            bonus = cooperators * 10 - competitors * 15
            payoffs[s.id] = base + bonus

        elif strategy == "compete":
            # Competitors do well against cooperators, poorly against other competitors
            if cooperators > 0 and competitors == 1:
                payoffs[s.id] = 90  # Exploit cooperators
            elif competitors > 1:
                payoffs[s.id] = 30  # Mutual competition
            else:
                payoffs[s.id] = 50

        else:  # compromise
            # Compromisers get middle payoff
            payoffs[s.id] = 50 + cooperators * 5

    return payoffs


def find_nash_equilibrium(game: Game) -> list[GameOutcome]:
    """Find Nash equilibrium outcomes.

    A Nash equilibrium is a strategy profile where no player can
    improve their payoff by unilaterally changing their strategy.

    Args:
        game: The game to analyze.

    Returns:
        List of Nash equilibrium outcomes.
    """
    equilibria = []

    for strategy_combo, payoffs in game.payoff_matrix.items():
        is_nash = True

        # Check each player's incentive to deviate
        for i, player in enumerate(game.players):
            current_strategy = strategy_combo[i]
            current_payoff = payoffs[player]

            # Try all alternative strategies
            for alt_strategy in game.strategies[player]:
                if alt_strategy == current_strategy:
                    continue

                # Build alternative strategy combination
                alt_combo = list(strategy_combo)
                alt_combo[i] = alt_strategy
                alt_combo = tuple(alt_combo)

                if alt_combo in game.payoff_matrix:
                    alt_payoff = game.payoff_matrix[alt_combo].get(player, 0)

                    # If player can improve, not Nash
                    if alt_payoff > current_payoff:
                        is_nash = False
                        break

            if not is_nash:
                break

        if is_nash:
            outcome = GameOutcome(
                strategies=dict(zip(game.players, strategy_combo)),
                payoffs=payoffs.copy(),
                is_nash=True,
            )
            equilibria.append(outcome)

    return equilibria


def pareto_frontier(outcomes: list[GameOutcome]) -> list[GameOutcome]:
    """Find Pareto-efficient outcomes.

    An outcome is Pareto efficient if no other outcome makes
    someone better off without making someone else worse off.

    Args:
        outcomes: List of outcomes to analyze.

    Returns:
        List of Pareto-efficient outcomes.
    """
    frontier = []

    for outcome in outcomes:
        is_dominated = False

        for other in outcomes:
            if other is outcome:
                continue

            # Check if other dominates outcome
            at_least_as_good = all(
                other.payoffs.get(p, 0) >= outcome.payoffs.get(p, 0)
                for p in outcome.payoffs
            )
            strictly_better = any(
                other.payoffs.get(p, 0) > outcome.payoffs.get(p, 0)
                for p in outcome.payoffs
            )

            if at_least_as_good and strictly_better:
                is_dominated = True
                break

        if not is_dominated:
            outcome.is_pareto = True
            frontier.append(outcome)

    return frontier


def dominant_strategy(
    game: Game,
    player: str,
) -> Optional[str]:
    """Find dominant strategy for a player if one exists.

    A dominant strategy gives the best payoff regardless of
    what other players do.

    Args:
        game: The game.
        player: The player ID.

    Returns:
        Dominant strategy name or None if none exists.
    """
    player_idx = game.players.index(player)
    strategies = game.strategies[player]

    for strategy in strategies:
        is_dominant = True

        # Get all strategy combos where player uses this strategy
        for combo, payoffs in game.payoff_matrix.items():
            if combo[player_idx] != strategy:
                continue

            current_payoff = payoffs.get(player, 0)

            # Check against all other strategies for this player
            for alt_strategy in strategies:
                if alt_strategy == strategy:
                    continue

                # Build alternative combo
                alt_combo = list(combo)
                alt_combo[player_idx] = alt_strategy
                alt_combo = tuple(alt_combo)

                if alt_combo in game.payoff_matrix:
                    alt_payoff = game.payoff_matrix[alt_combo].get(player, 0)

                    # Must be at least as good as all alternatives
                    if alt_payoff > current_payoff:
                        is_dominant = False
                        break

            if not is_dominant:
                break

        if is_dominant:
            return strategy

    return None


def minimax(
    game: Game,
    player: str,
) -> tuple[str, float]:
    """Find minimax strategy (minimize maximum loss).

    The minimax strategy minimizes the worst-case payoff
    the player might receive.

    Args:
        game: The game.
        player: The player ID.

    Returns:
        Tuple of (strategy, guaranteed_payoff).
    """
    player_idx = game.players.index(player)
    strategies = game.strategies[player]

    best_strategy = None
    best_worst_case = float("-inf")

    for strategy in strategies:
        # Find worst case payoff for this strategy
        worst_case = float("inf")

        for combo, payoffs in game.payoff_matrix.items():
            if combo[player_idx] == strategy:
                payoff = payoffs.get(player, 0)
                worst_case = min(worst_case, payoff)

        if worst_case > best_worst_case:
            best_worst_case = worst_case
            best_strategy = strategy

    return (best_strategy, best_worst_case)


def analyze_game(
    stakeholders: list[Stakeholder],
    strategy_options: Optional[dict[str, list[str]]] = None,
) -> dict[str, Any]:
    """Perform full game-theoretic analysis.

    Args:
        stakeholders: List of stakeholders.
        strategy_options: Available strategies per stakeholder.

    Returns:
        Comprehensive analysis results.
    """
    game = build_game(stakeholders, strategy_options)

    # Find all outcomes
    all_outcomes = []
    for combo, payoffs in game.payoff_matrix.items():
        all_outcomes.append(
            GameOutcome(
                strategies=dict(zip(game.players, combo)),
                payoffs=payoffs.copy(),
            )
        )

    # Nash equilibria
    nash = find_nash_equilibrium(game)

    # Pareto frontier
    pareto = pareto_frontier(all_outcomes)

    # Dominant strategies
    dominant = {}
    for player in game.players:
        dom = dominant_strategy(game, player)
        if dom:
            dominant[player] = dom

    # Minimax strategies
    minimax_strategies = {}
    for player in game.players:
        strategy, value = minimax(game, player)
        minimax_strategies[player] = {"strategy": strategy, "guaranteed": value}

    # Find socially optimal (maximize total payoff)
    social_optimal = max(all_outcomes, key=lambda o: sum(o.payoffs.values()))

    # Check for prisoner's dilemma structure
    prisoners_dilemma = _check_prisoners_dilemma(game)

    return {
        "game": game,
        "num_outcomes": len(all_outcomes),
        "nash_equilibria": nash,
        "pareto_frontier": pareto,
        "dominant_strategies": dominant,
        "minimax_strategies": minimax_strategies,
        "social_optimal": social_optimal,
        "prisoners_dilemma": prisoners_dilemma,
        "efficiency_of_equilibria": _efficiency_analysis(nash, pareto, social_optimal),
    }


def _check_prisoners_dilemma(game: Game) -> dict[str, Any]:
    """Check if the game has prisoner's dilemma structure.

    Args:
        game: The game to check.

    Returns:
        Analysis of prisoner's dilemma characteristics.
    """
    if len(game.players) != 2:
        return {"is_pd": False, "reason": "Not a 2-player game"}

    if "cooperate" not in game.strategies.get(game.players[0], []):
        return {"is_pd": False, "reason": "No cooperate strategy"}

    if "compete" not in game.strategies.get(game.players[0], []):
        return {"is_pd": False, "reason": "No compete strategy"}

    # Get key payoffs
    try:
        cc = game.payoff_matrix[("cooperate", "cooperate")]
        cd = game.payoff_matrix[("cooperate", "compete")]
        dc = game.payoff_matrix[("compete", "cooperate")]
        dd = game.payoff_matrix[("compete", "compete")]
    except KeyError:
        return {"is_pd": False, "reason": "Missing strategy combinations"}

    p1, p2 = game.players

    # PD conditions: T > R > P > S and 2R > T + S
    # T = temptation (compete vs cooperate)
    # R = reward (both cooperate)
    # P = punishment (both compete)
    # S = sucker (cooperate vs compete)

    t1, t2 = dc[p1], cd[p2]
    r1, r2 = cc[p1], cc[p2]
    p1_val, p2_val = dd[p1], dd[p2]
    s1, s2 = cd[p1], dc[p2]

    pd_structure = (
        t1 > r1 > p1_val > s1
        and t2 > r2 > p2_val > s2
        and 2 * r1 > t1 + s1
        and 2 * r2 > t2 + s2
    )

    return {
        "is_pd": pd_structure,
        "payoffs": {
            "mutual_cooperation": cc,
            "mutual_defection": dd,
            "temptation": {"p1": t1, "p2": t2},
            "sucker": {"p1": s1, "p2": s2},
        },
        "recommendation": "Establish trust/repeated interaction" if pd_structure else None,
    }


def _efficiency_analysis(
    nash: list[GameOutcome],
    pareto: list[GameOutcome],
    social_optimal: GameOutcome,
) -> dict[str, Any]:
    """Analyze efficiency of equilibria.

    Args:
        nash: Nash equilibria.
        pareto: Pareto frontier.
        social_optimal: Social optimum.

    Returns:
        Efficiency analysis.
    """
    social_welfare = sum(social_optimal.payoffs.values())

    nash_efficiency = []
    for eq in nash:
        eq_welfare = sum(eq.payoffs.values())
        efficiency = eq_welfare / social_welfare if social_welfare > 0 else 0
        is_on_pareto = any(
            eq.strategies == p.strategies for p in pareto
        )
        nash_efficiency.append({
            "strategies": eq.strategies,
            "welfare": eq_welfare,
            "efficiency": efficiency,
            "is_pareto_optimal": is_on_pareto,
        })

    avg_efficiency = (
        sum(e["efficiency"] for e in nash_efficiency) / len(nash_efficiency)
        if nash_efficiency
        else 0
    )

    return {
        "social_optimal_welfare": social_welfare,
        "nash_equilibria_analysis": nash_efficiency,
        "average_nash_efficiency": avg_efficiency,
        "price_of_anarchy": 1 - avg_efficiency if avg_efficiency < 1 else 0,
    }


def compute_bargaining_solution(
    stakeholders: list[Stakeholder],
    disagreement_point: dict[str, float],
    feasible_outcomes: list[dict[str, float]],
    solution_type: str = "nash",
) -> Optional[dict[str, Any]]:
    """Compute bargaining solution.

    Args:
        stakeholders: List of stakeholders.
        disagreement_point: Payoffs if no agreement (BATNA utilities).
        feasible_outcomes: List of feasible payoff distributions.
        solution_type: "nash" or "kalai-smorodinsky".

    Returns:
        The bargaining solution.
    """
    if not feasible_outcomes:
        return None

    players = [s.id for s in stakeholders]

    # Filter to individually rational outcomes
    ir_outcomes = []
    for outcome in feasible_outcomes:
        is_ir = all(
            outcome.get(p, 0) >= disagreement_point.get(p, 0) for p in players
        )
        if is_ir:
            ir_outcomes.append(outcome)

    if not ir_outcomes:
        return {
            "solution": None,
            "reason": "No individually rational outcomes",
        }

    if solution_type == "nash":
        # Maximize product of gains
        best = None
        best_product = float("-inf")

        for outcome in ir_outcomes:
            product = 1.0
            for p in players:
                gain = outcome.get(p, 0) - disagreement_point.get(p, 0)
                product *= max(gain, 0.001)  # Avoid zero

            if product > best_product:
                best_product = product
                best = outcome

        return {
            "solution": best,
            "nash_product": best_product,
            "gains": {p: best.get(p, 0) - disagreement_point.get(p, 0) for p in players}
            if best
            else None,
        }

    elif solution_type == "kalai-smorodinsky":
        # Find ideal point
        ideal = {}
        for p in players:
            ideal[p] = max(o.get(p, 0) for o in ir_outcomes)

        # Find solution on line from disagreement to ideal
        # that is Pareto optimal
        best = None
        best_ratio = float("-inf")

        for outcome in ir_outcomes:
            # Calculate ratio of gains
            ratios = []
            for p in players:
                max_gain = ideal[p] - disagreement_point.get(p, 0)
                actual_gain = outcome.get(p, 0) - disagreement_point.get(p, 0)
                if max_gain > 0:
                    ratios.append(actual_gain / max_gain)

            if ratios:
                min_ratio = min(ratios)
                if min_ratio > best_ratio:
                    best_ratio = min_ratio
                    best = outcome

        return {
            "solution": best,
            "ideal_point": ideal,
            "ratio": best_ratio,
        }

    return None
