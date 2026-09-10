"""Actor modeling for the Temporal Scenario Planner.

This module provides actor classes that model different types of agents
in a scenario (competitors, markets, users, regulators) and their
decision-making behavior patterns.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .types import (
    Actor,
    ActorState,
    BehaviorModel,
    Decision,
    DecisionStatus,
    Event,
    TimelineState,
)


class Situation:
    """Context for an actor's decision.

    Attributes:
        state: Current timeline state.
        decision: Decision to be made.
        options: Available options.
        other_actors: States of other actors.
        recent_events: Recent events affecting the situation.
    """

    def __init__(
        self,
        state: TimelineState,
        decision: Decision,
        other_actors: dict[str, ActorState] | None = None,
        recent_events: list[Event] | None = None,
    ):
        self.state = state
        self.decision = decision
        self.options = decision.options
        self.other_actors = other_actors or {}
        self.recent_events = recent_events or []


class ActorDecider(Protocol):
    """Protocol for actor decision-making."""

    def decide(self, situation: Situation) -> str:
        """Make a decision given a situation.

        Args:
            situation: Context for the decision.

        Returns:
            The chosen option.
        """
        ...


class BaseActor(ABC):
    """Base class for actor implementations.

    Attributes:
        actor: The underlying Actor data.
        llm_provider: Optional LLM for complex decisions.
    """

    def __init__(
        self,
        actor: Actor,
        llm_provider: Callable[[str], str] | None = None,
    ):
        self.actor = actor
        self.llm_provider = llm_provider

    @property
    def id(self) -> str:
        return self.actor.id

    @property
    def name(self) -> str:
        return self.actor.name

    @property
    def behavior_model(self) -> BehaviorModel:
        return self.actor.behavior_model

    @property
    def state(self) -> ActorState:
        return self.actor.current_state

    @abstractmethod
    def decide(self, situation: Situation) -> str:
        """Make a decision given a situation.

        Args:
            situation: Context for the decision.

        Returns:
            The chosen option.
        """
        pass

    def _evaluate_option(self, option: str, situation: Situation) -> float:
        """Score an option based on the actor's behavior model.

        Args:
            option: Option to evaluate.
            situation: Decision context.

        Returns:
            Score for this option (higher is better).
        """
        # Base implementation uses behavior model heuristics
        score = 0.5

        if self.behavior_model == BehaviorModel.AGGRESSIVE:
            # Prefer bold, high-reward options
            if any(word in option.lower() for word in ["attack", "expand", "increase", "launch"]):
                score += 0.3
            if any(word in option.lower() for word in ["wait", "defend", "reduce", "cautious"]):
                score -= 0.2

        elif self.behavior_model == BehaviorModel.CONSERVATIVE:
            # Prefer safe, low-risk options
            if any(word in option.lower() for word in ["protect", "maintain", "stable", "defend"]):
                score += 0.3
            if any(word in option.lower() for word in ["risk", "aggressive", "gamble", "expand"]):
                score -= 0.2

        elif self.behavior_model == BehaviorModel.REACTIVE:
            # Score based on recent events
            for event in situation.recent_events:
                if event.magnitude > 0:
                    if "respond" in option.lower() or "adapt" in option.lower():
                        score += 0.2
                elif event.magnitude < 0:
                    if "counter" in option.lower() or "defend" in option.lower():
                        score += 0.2

        elif self.behavior_model == BehaviorModel.PROACTIVE:
            # Prefer options that anticipate future needs
            if any(word in option.lower() for word in ["prepare", "invest", "build", "anticipate"]):
                score += 0.3
            if any(word in option.lower() for word in ["react", "respond", "wait"]):
                score -= 0.2

        return max(0.0, min(1.0, score))

    def _select_best_option(self, situation: Situation) -> str:
        """Select the best option based on scoring.

        Args:
            situation: Decision context.

        Returns:
            The best option.
        """
        if not situation.options:
            return ""

        scores = [(opt, self._evaluate_option(opt, situation)) for opt in situation.options]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]

    def record_decision(self, decision: Decision, choice: str, rationale: str) -> None:
        """Record a decision in the actor's history.

        Args:
            decision: The decision made.
            choice: The chosen option.
            rationale: Why this choice was made.
        """
        self.actor.decision_history.append(
            f"{decision.date.isoformat()}: {decision.description} -> {choice}"
        )


class CompetitorActor(BaseActor):
    """Models competitor behavior in the market.

    Competitors are typically trying to maximize their market share
    and respond to the player's actions.
    """

    def __init__(
        self,
        actor: Actor,
        market_share: float = 0.2,
        aggressiveness: float = 0.5,
        llm_provider: Callable[[str], str] | None = None,
    ):
        super().__init__(actor, llm_provider)
        self.market_share = market_share
        self.aggressiveness = aggressiveness

    def decide(self, situation: Situation) -> str:
        """Make a competitive decision.

        Competitors consider market dynamics and player actions.
        """
        # Check if we should use LLM for complex decisions
        if self.llm_provider and len(situation.options) > 3:
            return self._llm_decide(situation)

        # Adjust scoring based on market position
        base_choice = self._select_best_option(situation)

        # If market share is low, be more aggressive
        if self.market_share < 0.15:
            for opt in situation.options:
                if "aggressive" in opt.lower() or "attack" in opt.lower():
                    return opt

        # If market share is high, be more defensive
        if self.market_share > 0.35:
            for opt in situation.options:
                if "defend" in opt.lower() or "maintain" in opt.lower():
                    return opt

        return base_choice

    def _llm_decide(self, situation: Situation) -> str:
        """Use LLM for complex competitive decisions."""
        prompt = f"""You are a competitor in a market simulation.
Your market share: {self.market_share:.1%}
Your behavior style: {self.behavior_model.value}
Your objectives: {', '.join(self.actor.objectives)}

Decision: {situation.decision.description}
Options: {', '.join(situation.options)}

Recent events:
{chr(10).join(f'- {e.description}' for e in situation.recent_events[-5:])}

Choose the option that best serves your competitive interests.
Respond with just the option text."""

        if self.llm_provider:
            response = self.llm_provider(prompt)
            # Match response to closest option
            for opt in situation.options:
                if opt.lower() in response.lower() or response.lower() in opt.lower():
                    return opt
        return self._select_best_option(situation)


class MarketActor(BaseActor):
    """Models market dynamics.

    Markets are reactive by default, responding to aggregate behavior
    of participants and external events.
    """

    def __init__(
        self,
        actor: Actor,
        volatility: float = 0.3,
        trend: float = 0.0,
        llm_provider: Callable[[str], str] | None = None,
    ):
        super().__init__(actor, llm_provider)
        self.volatility = volatility
        self.trend = trend  # -1.0 (bearish) to 1.0 (bullish)

    def decide(self, situation: Situation) -> str:
        """Make a market-based decision.

        Markets aggregate behavior and respond to trends.
        """
        # Markets are inherently reactive
        if situation.recent_events:
            # Calculate aggregate sentiment from recent events
            sentiment = sum(e.magnitude for e in situation.recent_events) / len(situation.recent_events)

            # Adjust based on trend
            sentiment += self.trend * 2

            # Find option matching sentiment
            if sentiment > 2:
                for opt in situation.options:
                    if any(w in opt.lower() for w in ["grow", "expand", "bull", "increase"]):
                        return opt
            elif sentiment < -2:
                for opt in situation.options:
                    if any(w in opt.lower() for w in ["contract", "decline", "bear", "decrease"]):
                        return opt

        return self._select_best_option(situation)

    def adjust_volatility(self, events: list[Event]) -> None:
        """Adjust market volatility based on events.

        Args:
            events: Recent events to consider.
        """
        if events:
            # High-magnitude events increase volatility
            avg_magnitude = sum(abs(e.magnitude) for e in events) / len(events)
            self.volatility = min(1.0, self.volatility + avg_magnitude * 0.1)

            # Volatility naturally decreases over time
            self.volatility = max(0.1, self.volatility * 0.95)


class UserActor(BaseActor):
    """Models user/customer behavior.

    Users make decisions based on value, convenience, and sentiment.
    """

    def __init__(
        self,
        actor: Actor,
        price_sensitivity: float = 0.5,
        brand_loyalty: float = 0.3,
        adoption_rate: float = 0.5,
        llm_provider: Callable[[str], str] | None = None,
    ):
        super().__init__(actor, llm_provider)
        self.price_sensitivity = price_sensitivity
        self.brand_loyalty = brand_loyalty
        self.adoption_rate = adoption_rate

    def decide(self, situation: Situation) -> str:
        """Make a user-perspective decision.

        Users balance price, quality, and familiarity.
        """
        best_score = -1.0
        best_option = situation.options[0] if situation.options else ""

        for opt in situation.options:
            score = 0.5

            # Price sensitivity
            if self.price_sensitivity > 0.5:
                if any(w in opt.lower() for w in ["cheap", "discount", "free", "save"]):
                    score += 0.3
                if any(w in opt.lower() for w in ["premium", "expensive", "luxury"]):
                    score -= 0.2

            # Brand loyalty
            if self.brand_loyalty > 0.5:
                if any(w in opt.lower() for w in ["familiar", "trusted", "known", "same"]):
                    score += 0.3
                if any(w in opt.lower() for w in ["new", "switch", "alternative"]):
                    score -= 0.2

            # Adoption rate for new things
            if self.adoption_rate > 0.5:
                if any(w in opt.lower() for w in ["new", "innovative", "latest", "cutting-edge"]):
                    score += 0.3

            if score > best_score:
                best_score = score
                best_option = opt

        return best_option


class RegulatorActor(BaseActor):
    """Models regulatory response.

    Regulators respond to market conditions, public sentiment,
    and policy objectives with rules and enforcement.
    """

    def __init__(
        self,
        actor: Actor,
        strictness: float = 0.5,
        enforcement_capacity: float = 0.5,
        policy_priorities: list[str] | None = None,
        llm_provider: Callable[[str], str] | None = None,
    ):
        super().__init__(actor, llm_provider)
        self.strictness = strictness
        self.enforcement_capacity = enforcement_capacity
        self.policy_priorities = policy_priorities or ["consumer_protection", "market_stability"]

    def decide(self, situation: Situation) -> str:
        """Make a regulatory decision.

        Regulators balance enforcement with practical constraints.
        """
        # Check if recent events demand response
        urgent_response_needed = False
        for event in situation.recent_events:
            if event.category.value in ["regulatory", "social"] and event.magnitude < -3:
                urgent_response_needed = True
                break

        if urgent_response_needed and self.strictness > 0.5:
            for opt in situation.options:
                if any(w in opt.lower() for w in ["enforce", "ban", "restrict", "penalty"]):
                    return opt

        # Consider enforcement capacity
        if self.enforcement_capacity < 0.3:
            for opt in situation.options:
                if any(w in opt.lower() for w in ["warn", "monitor", "study", "review"]):
                    return opt

        # Default to policy-aligned decisions
        return self._select_best_option(situation)

    def update_strictness(self, public_pressure: float) -> None:
        """Update strictness based on public pressure.

        Args:
            public_pressure: Level of public demand for regulation (-1 to 1).
        """
        self.strictness = max(0.0, min(1.0, self.strictness + public_pressure * 0.2))


def create_actor(
    actor_type: str,
    actor_id: str,
    name: str,
    behavior: BehaviorModel = BehaviorModel.REACTIVE,
    llm_provider: Callable[[str], str] | None = None,
    **kwargs: Any,
) -> BaseActor:
    """Factory function to create actors.

    Args:
        actor_type: Type of actor ('competitor', 'market', 'user', 'regulator').
        actor_id: Unique ID for the actor.
        name: Human-readable name.
        behavior: Behavior model.
        llm_provider: Optional LLM for decisions.
        **kwargs: Additional arguments for specific actor types.

    Returns:
        Appropriate actor instance.
    """
    base_actor = Actor(
        id=actor_id,
        name=name,
        behavior_model=behavior,
        current_state=ActorState(),
        objectives=kwargs.get("objectives", []),
    )

    if actor_type == "competitor":
        return CompetitorActor(
            base_actor,
            market_share=kwargs.get("market_share", 0.2),
            aggressiveness=kwargs.get("aggressiveness", 0.5),
            llm_provider=llm_provider,
        )
    elif actor_type == "market":
        return MarketActor(
            base_actor,
            volatility=kwargs.get("volatility", 0.3),
            trend=kwargs.get("trend", 0.0),
            llm_provider=llm_provider,
        )
    elif actor_type == "user":
        return UserActor(
            base_actor,
            price_sensitivity=kwargs.get("price_sensitivity", 0.5),
            brand_loyalty=kwargs.get("brand_loyalty", 0.3),
            adoption_rate=kwargs.get("adoption_rate", 0.5),
            llm_provider=llm_provider,
        )
    elif actor_type == "regulator":
        return RegulatorActor(
            base_actor,
            strictness=kwargs.get("strictness", 0.5),
            enforcement_capacity=kwargs.get("enforcement_capacity", 0.5),
            policy_priorities=kwargs.get("policy_priorities"),
            llm_provider=llm_provider,
        )
    else:
        raise ValueError(f"Unknown actor type: {actor_type}")


def simulate_actor_interaction(
    actors: list[BaseActor],
    situation: Situation,
) -> dict[str, str]:
    """Simulate interactions between multiple actors.

    Args:
        actors: List of actors to simulate.
        situation: Shared situation context.

    Returns:
        Dict mapping actor IDs to their decisions.
    """
    decisions: dict[str, str] = {}

    for actor in actors:
        # Create actor-specific situation with awareness of others
        actor_situation = Situation(
            state=situation.state,
            decision=Decision(
                id=f"{situation.decision.id}_{actor.id}",
                date=situation.decision.date,
                description=situation.decision.description,
                options=situation.options,
                actor_id=actor.id,
            ),
            other_actors={
                a.id: a.state
                for a in actors
                if a.id != actor.id
            },
            recent_events=situation.recent_events,
        )

        decisions[actor.id] = actor.decide(actor_situation)

    return decisions
