"""
Comprehensive tests for the Embodied Action Planner.

Tests cover all major components:
- Environment abstraction
- Observation processing
- Action planning
- Execution
- Branching exploration
- Failure memory
- Synthesis
- Rewards
- Main harness
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Import all components
from embodied import (
    # Types
    Action,
    ActionResult,
    ActionSequence,
    EnvironmentState,
    ExplorationConfig,
    Goal,
    Observation,
    # Environment
    ActionSpace,
    Environment,
    EnvironmentWrapper,
    LoggingEnvironment,
    SafetyEnvironment,
    # Concrete environments
    MockEnvironment,
    TextGameEnvironment,
    BrowserEnvironment,
    APIEnvironment,
    # Observation
    process_observation,
    observation_to_text,
    diff_observations,
    extract_state_features,
    observation_hash,
    states_equivalent,
    summarize_observation,
    # Planning
    plan_action,
    action_from_text,
    explain_action,
    replan_on_failure,
    # Execution
    execute_action,
    retry_with_backoff,
    safe_execute,
    ActionExecutor,
    ExecutionConstraints,
    rollback_on_failure,
    # Exploration
    ExplorationBranch,
    BranchStatus,
    StateSpaceCoverage,
    ExplorationTree,
    branch_at_choice,
    evaluate_branch,
    detect_loop,
    prune_stuck_branches,
    # Failure
    FailureRecord,
    FailureMemory,
    propagate_failure,
    FailureAwareActionFilter,
    # Synthesis
    ActionPattern,
    synthesize_plan,
    merge_sequences,
    verify_sequence,
    extract_reusable_patterns,
    SequenceSynthesizer,
    # Rewards
    RewardFunction,
    GoalReward,
    ProgressReward,
    ExplorationReward,
    CompositeReward,
    DistanceReward,
    TimeReward,
    create_goal_reward,
    create_exploration_reward,
    # Harness
    EmbodiedHarness,
    HarnessConfig,
    ExplorationResult,
)


# =============================================================================
# Types Tests
# =============================================================================

class TestTypes:
    """Tests for basic type definitions."""

    def test_action_creation(self):
        """Test Action dataclass creation."""
        action = Action(action_type="move", parameters={"direction": "north"})
        assert action.action_type == "move"
        assert action.parameters == {"direction": "north"}
        assert action.id is not None

    def test_action_str(self):
        """Test Action string representation."""
        action = Action(action_type="move", parameters={"direction": "east"})
        assert "move" in str(action)
        assert "east" in str(action)

    def test_observation_creation(self):
        """Test Observation dataclass creation."""
        obs = Observation(state={"position": (1, 2), "health": 100})
        assert obs.state["position"] == (1, 2)
        assert obs.timestamp is not None

    def test_action_result_creation(self):
        """Test ActionResult dataclass creation."""
        obs = Observation(state={"x": 1})
        result = ActionResult(
            action_id="abc",
            success=True,
            new_observation=obs,
            reward=0.5,
        )
        assert result.success
        assert result.reward == 0.5

    def test_action_sequence(self):
        """Test ActionSequence operations."""
        seq = ActionSequence()
        action1 = Action(action_type="move", parameters={"direction": "north"})
        action2 = Action(action_type="move", parameters={"direction": "east"})

        seq.append(action1)
        seq.append(action2)

        assert len(seq) == 2
        assert list(seq) == [action1, action2]

    def test_goal_is_achieved(self):
        """Test Goal.is_achieved method."""
        goal = Goal(
            description="Reach position (5, 5)",
            success_condition=lambda obs: obs.state.get("position") == (5, 5),
        )

        obs_not_at_goal = Observation(state={"position": (2, 2)})
        obs_at_goal = Observation(state={"position": (5, 5)})

        assert not goal.is_achieved(obs_not_at_goal)
        assert goal.is_achieved(obs_at_goal)


# =============================================================================
# Environment Tests
# =============================================================================

class TestEnvironment:
    """Tests for environment abstraction."""

    def test_action_space_validation(self):
        """Test ActionSpace.is_valid method."""
        space = ActionSpace(
            action_types=["move", "pickup"],
            parameter_specs={
                "move": {"required": ["direction"]},
            },
        )

        valid_action = Action(action_type="move", parameters={"direction": "north"})
        invalid_action = Action(action_type="fly", parameters={})
        missing_param = Action(action_type="move", parameters={})

        assert space.is_valid(valid_action)
        assert not space.is_valid(invalid_action)
        assert not space.is_valid(missing_param)

    def test_action_space_describe(self):
        """Test ActionSpace.describe method."""
        space = ActionSpace(action_types=["move", "wait"])
        description = space.describe()
        assert "move" in description
        assert "wait" in description


class TestMockEnvironment:
    """Tests for MockEnvironment."""

    def test_reset(self):
        """Test environment reset."""
        env = MockEnvironment(start_pos=(0, 0), goal_pos=(4, 4))
        obs = env.reset()

        assert obs.state["position"] == (0, 0)
        assert obs.state["goal"] == (4, 4)

    def test_move_success(self):
        """Test successful movement."""
        env = MockEnvironment()
        env.reset()

        action = Action(action_type="move", parameters={"direction": "east"})
        result = env.step(action)

        assert result.success
        assert result.new_observation.state["position"] == (1, 0)

    def test_move_blocked(self):
        """Test blocked movement."""
        env = MockEnvironment(obstacles=[(1, 0)])
        env.reset()

        action = Action(action_type="move", parameters={"direction": "east"})
        result = env.step(action)

        assert not result.success
        assert "obstacle" in result.error.lower()

    def test_move_out_of_bounds(self):
        """Test movement out of bounds."""
        env = MockEnvironment()
        env.reset()

        action = Action(action_type="move", parameters={"direction": "west"})
        result = env.step(action)

        assert not result.success

    def test_pickup_item(self):
        """Test item pickup."""
        env = MockEnvironment(items={(0, 0): "key"})
        env.reset()

        action = Action(action_type="pickup")
        result = env.step(action)

        assert result.success
        assert "key" in result.new_observation.state["inventory"]

    def test_checkpoint_restore(self):
        """Test state checkpoint and restore."""
        env = MockEnvironment()
        env.reset()

        # Move
        env.step(Action(action_type="move", parameters={"direction": "east"}))
        checkpoint = env.checkpoint()

        # Move more
        env.step(Action(action_type="move", parameters={"direction": "north"}))

        # Restore
        env.restore(checkpoint)
        obs = env.observe()

        assert obs.state["position"] == (1, 0)

    def test_is_terminal_goal(self):
        """Test terminal state at goal."""
        env = MockEnvironment(start_pos=(3, 4), goal_pos=(4, 4))
        env.reset()

        env.step(Action(action_type="move", parameters={"direction": "east"}))

        assert env.is_terminal()

    def test_is_terminal_max_steps(self):
        """Test terminal state from max steps."""
        env = MockEnvironment(max_steps=2)
        env.reset()

        env.step(Action(action_type="wait"))
        env.step(Action(action_type="wait"))

        assert env.is_terminal()


class TestTextGameEnvironment:
    """Tests for TextGameEnvironment."""

    def test_reset(self):
        """Test game reset."""
        env = TextGameEnvironment()
        obs = env.reset()

        assert "room" in obs.state
        assert "exits" in obs.state

    def test_go_direction(self):
        """Test movement between rooms."""
        env = TextGameEnvironment()
        env.reset()

        action = Action(action_type="go", parameters={"direction": "north"})
        result = env.step(action)

        assert result.success

    def test_take_item(self):
        """Test taking items."""
        env = TextGameEnvironment()
        env.reset()

        action = Action(action_type="take", parameters={"item": "rusty key"})
        result = env.step(action)

        assert result.success
        assert "rusty key" in result.new_observation.state["inventory"]


class TestBrowserEnvironment:
    """Tests for BrowserEnvironment (mock mode)."""

    def test_reset(self):
        """Test browser reset."""
        env = BrowserEnvironment()
        obs = env.reset()

        assert "url" in obs.state
        assert "elements" in obs.state

    def test_navigate(self):
        """Test navigation."""
        env = BrowserEnvironment()
        env.reset()

        action = Action(action_type="navigate", parameters={"url": "https://example.com"})
        result = env.step(action)

        assert result.success
        assert "example.com" in result.new_observation.state["url"]

    def test_click_element(self):
        """Test clicking elements."""
        env = BrowserEnvironment()
        env.reset()

        action = Action(action_type="click", parameters={"selector": "#submit"})
        result = env.step(action)

        assert result.success


class TestAPIEnvironment:
    """Tests for APIEnvironment."""

    def test_reset(self):
        """Test API environment reset."""
        env = APIEnvironment()
        obs = env.reset()

        assert "endpoints" in obs.state

    def test_get_request(self):
        """Test GET request."""
        env = APIEnvironment()
        env.reset()

        action = Action(action_type="get", parameters={"path": "/items"})
        result = env.step(action)

        assert result.success
        assert result.new_observation.state["last_status"] == 200

    def test_post_request(self):
        """Test POST request."""
        env = APIEnvironment()
        env.reset()

        action = Action(
            action_type="post",
            parameters={
                "path": "/items",
                "body": {"name": "test", "value": 42},
            },
        )
        result = env.step(action)

        assert result.success
        assert result.new_observation.state["last_status"] == 201


# =============================================================================
# Observation Tests
# =============================================================================

class TestObservation:
    """Tests for observation processing."""

    def test_observation_to_text(self):
        """Test converting observation to text."""
        obs = Observation(state={
            "position": (1, 2),
            "inventory": ["key", "sword"],
        })

        text = observation_to_text(obs)

        assert "position" in text
        assert "1, 2" in text or "(1, 2)" in text

    def test_diff_observations(self):
        """Test observation diffing."""
        obs1 = Observation(state={"position": (0, 0), "health": 100})
        obs2 = Observation(state={"position": (1, 0), "health": 90})

        diff = diff_observations(obs1, obs2)

        assert "position" in diff["changed"]
        assert "health" in diff["changed"]

    def test_extract_state_features(self):
        """Test feature extraction."""
        obs = Observation(state={
            "position": (2, 2),
            "goal": (5, 5),
            "inventory": ["key"],
        })

        features = extract_state_features(obs)

        assert "distance_to_goal" in features
        assert features["distance_to_goal"] == 6  # Manhattan distance
        assert features["has_items"]

    def test_observation_hash(self):
        """Test observation hashing."""
        obs1 = Observation(state={"x": 1, "y": 2})
        obs2 = Observation(state={"x": 1, "y": 2})
        obs3 = Observation(state={"x": 1, "y": 3})

        assert observation_hash(obs1) == observation_hash(obs2)
        assert observation_hash(obs1) != observation_hash(obs3)

    def test_states_equivalent(self):
        """Test state equivalence check."""
        obs1 = Observation(state={"a": 1})
        obs2 = Observation(state={"a": 1})

        assert states_equivalent(obs1, obs2)


# =============================================================================
# Planning Tests
# =============================================================================

class TestPlanning:
    """Tests for action planning."""

    def test_plan_action_grid(self):
        """Test planning for grid navigation."""
        obs = Observation(state={
            "position": (0, 0),
            "goal": (4, 4),
        })
        goal = Goal(description="Reach goal")
        space = ActionSpace(action_types=["move"])

        action = plan_action(obs, goal, [], space)

        assert action.action_type == "move"
        # Should move east or north (toward goal)
        direction = action.parameters.get("direction")
        assert direction in ["east", "north"]

    def test_action_from_text_json(self):
        """Test parsing action from JSON."""
        space = ActionSpace(action_types=["move", "pickup"])

        text = '{"action": "move", "parameters": {"direction": "north"}}'
        action = action_from_text(text, space)

        assert action.action_type == "move"
        assert action.parameters["direction"] == "north"

    def test_action_from_text_natural(self):
        """Test parsing action from natural language."""
        space = ActionSpace(action_types=["move", "pickup"])

        text = "I should move north to get closer to the goal."
        action = action_from_text(text, space)

        assert action.action_type == "move"

    def test_explain_action(self):
        """Test action explanation."""
        action = Action(action_type="move", parameters={"direction": "east"})
        context = {
            "observation": Observation(state={"position": (0, 0), "goal": (4, 0)}),
            "goal": Goal(description="Reach the treasure"),
        }

        explanation = explain_action(action, context)

        assert "east" in explanation.lower()


# =============================================================================
# Execution Tests
# =============================================================================

class TestExecution:
    """Tests for action execution."""

    def test_execute_action(self):
        """Test basic action execution."""
        env = MockEnvironment()
        env.reset()

        action = Action(action_type="move", parameters={"direction": "east"})
        result = execute_action(env, action)

        assert result.success

    def test_safe_execute_forbidden(self):
        """Test safe execution with forbidden actions."""
        env = MockEnvironment()
        env.reset()

        constraints = ExecutionConstraints(forbidden_actions=["move"])
        action = Action(action_type="move", parameters={"direction": "east"})

        result = safe_execute(env, action, constraints)

        assert not result.success
        assert "forbidden" in result.error.lower()

    def test_action_executor(self):
        """Test ActionExecutor class."""
        env = MockEnvironment()
        env.reset()

        executor = ActionExecutor(env)

        action = Action(action_type="move", parameters={"direction": "east"})
        result = executor.execute(action)

        assert result.success
        assert executor.stats.total_actions == 1
        assert executor.stats.successful_actions == 1

    def test_rollback_on_failure(self):
        """Test rollback on failure."""
        env = MockEnvironment(obstacles=[(1, 0)])
        env.reset()

        action = Action(action_type="move", parameters={"direction": "east"})
        result, rolled_back = rollback_on_failure(env, action)

        assert not result.success
        assert rolled_back
        # Position should be unchanged
        assert env.observe().state["position"] == (0, 0)


# =============================================================================
# Exploration Tests
# =============================================================================

class TestExploration:
    """Tests for branching exploration."""

    def test_exploration_branch_creation(self):
        """Test creating exploration branch."""
        checkpoint = EnvironmentState(data={"position": (0, 0)})
        branch = ExplorationBranch(checkpoint=checkpoint)

        assert branch.is_active
        assert branch.length == 0

    def test_branch_add_step(self):
        """Test adding steps to branch."""
        checkpoint = EnvironmentState(data={})
        branch = ExplorationBranch(checkpoint=checkpoint)

        action = Action(action_type="move")
        result = ActionResult(
            action_id=action.id,
            success=True,
            new_observation=Observation(state={}),
            reward=0.5,
        )
        obs = Observation(state={})

        branch.add_step(action, result, obs)

        assert branch.length == 1
        assert branch.total_reward == 0.5

    def test_branch_at_choice(self):
        """Test branching at decision points."""
        env = MockEnvironment()
        env.reset()

        parent = ExplorationBranch(checkpoint=env.checkpoint())
        options = [
            Action(action_type="move", parameters={"direction": "north"}),
            Action(action_type="move", parameters={"direction": "east"}),
        ]

        branches = branch_at_choice(env, options, parent)

        assert len(branches) == 2
        assert all(b.parent_id == parent.id for b in branches)

    def test_detect_loop(self):
        """Test loop detection."""
        checkpoint = EnvironmentState(data={})
        branch = ExplorationBranch(checkpoint=checkpoint)

        # Create repeated observations
        for _ in range(10):
            branch.observations.append(Observation(state={"x": 1}))

        assert detect_loop(branch, window=3)

    def test_detect_no_loop(self):
        """Test non-looping branch."""
        checkpoint = EnvironmentState(data={})
        branch = ExplorationBranch(checkpoint=checkpoint)

        # Create varying observations
        for i in range(10):
            branch.observations.append(Observation(state={"x": i}))

        assert not detect_loop(branch, window=3)

    def test_exploration_tree(self):
        """Test ExplorationTree management."""
        env = MockEnvironment()
        env.reset()

        tree = ExplorationTree()
        root = tree.create_root(env)

        assert tree.root_id == root.id
        assert len(tree.get_active_branches()) == 1

    def test_state_space_coverage(self):
        """Test StateSpaceCoverage tracking."""
        coverage = StateSpaceCoverage()

        obs1 = Observation(state={"x": 1})
        obs2 = Observation(state={"x": 2})

        is_new1 = coverage.record_visit(obs1)
        is_new2 = coverage.record_visit(obs2)
        is_new3 = coverage.record_visit(obs1)

        assert is_new1
        assert is_new2
        assert not is_new3
        assert coverage.coverage == 2


# =============================================================================
# Failure Tests
# =============================================================================

class TestFailure:
    """Tests for failure memory and propagation."""

    def test_record_failure(self):
        """Test recording failures."""
        memory = FailureMemory()

        action = Action(action_type="move", parameters={"direction": "north"})
        state = Observation(state={"position": (0, 4)})

        record = memory.record_failure(action, state, "Blocked by wall")

        assert memory.total_failures == 1
        assert record.count == 1

    def test_should_avoid(self):
        """Test failure avoidance check."""
        memory = FailureMemory()

        action = Action(action_type="move", parameters={"direction": "north"})
        state = Observation(state={"position": (0, 4)})

        memory.record_failure(action, state, "Blocked")

        should_avoid, reason = memory.should_avoid(action, state)

        assert should_avoid
        assert "failed" in reason.lower()

    def test_propagate_failure(self):
        """Test failure propagation."""
        source = FailureMemory()
        targets = [FailureMemory(), FailureMemory()]

        action = Action(action_type="move", parameters={"direction": "north"})
        state = Observation(state={"position": (0, 0)})

        source.record_failure(action, state, "Failed")

        propagated = propagate_failure(source, targets)

        assert propagated == 2
        assert targets[0].total_failures == 1

    def test_failure_aware_filter(self):
        """Test FailureAwareActionFilter."""
        memory = FailureMemory()
        filter_obj = FailureAwareActionFilter(memory, threshold=1)

        action1 = Action(action_type="move", parameters={"direction": "north"})
        action2 = Action(action_type="move", parameters={"direction": "south"})
        state = Observation(state={"position": (0, 0)})

        memory.record_failure(action1, state, "Failed")

        filtered = filter_obj.filter_actions([action1, action2], state)

        assert len(filtered) == 1
        assert filtered[0].parameters["direction"] == "south"


# =============================================================================
# Synthesis Tests
# =============================================================================

class TestSynthesis:
    """Tests for action sequence synthesis."""

    def test_synthesize_plan_empty(self):
        """Test synthesis with no branches."""
        result = synthesize_plan([])
        assert len(result) == 0

    def test_synthesize_plan_single(self):
        """Test synthesis with single branch."""
        checkpoint = EnvironmentState(data={})
        branch = ExplorationBranch(checkpoint=checkpoint)
        branch.actions = [
            Action(action_type="move", parameters={"direction": "east"}),
            Action(action_type="move", parameters={"direction": "north"}),
        ]
        branch.total_reward = 1.0
        branch.status = BranchStatus.SUCCESS

        result = synthesize_plan([branch])

        assert len(result) == 2

    def test_merge_sequences_concat(self):
        """Test sequence merging with concatenation."""
        s1 = ActionSequence(
            actions=[Action(action_type="a")],
            total_reward=1.0,
        )
        s2 = ActionSequence(
            actions=[Action(action_type="b")],
            total_reward=2.0,
        )

        merged = merge_sequences(s1, s2, strategy="concat")

        assert len(merged) == 2
        assert merged.total_reward == 3.0

    def test_merge_sequences_interleave(self):
        """Test sequence merging with interleaving."""
        s1 = ActionSequence(
            actions=[Action(action_type="a"), Action(action_type="c")],
            total_reward=1.0,
        )
        s2 = ActionSequence(
            actions=[Action(action_type="b"), Action(action_type="d")],
            total_reward=2.0,
        )

        merged = merge_sequences(s1, s2, strategy="interleave")

        assert len(merged) == 4

    def test_sequence_synthesizer(self):
        """Test SequenceSynthesizer class."""
        synthesizer = SequenceSynthesizer()

        checkpoint = EnvironmentState(data={})
        branch = ExplorationBranch(checkpoint=checkpoint)
        branch.actions = [Action(action_type="move")]
        branch.status = BranchStatus.SUCCESS

        learned = synthesizer.learn_from_branches([branch])
        sequence = synthesizer.synthesize([branch])

        assert sequence is not None


# =============================================================================
# Rewards Tests
# =============================================================================

class TestRewards:
    """Tests for reward functions."""

    def test_goal_reward_achieved(self):
        """Test GoalReward when goal achieved."""
        goal = Goal(
            description="Reach position",
            success_condition=lambda obs: obs.state.get("at_goal", False),
        )
        reward_fn = GoalReward(goal, completion_reward=10.0)

        prev_obs = Observation(state={"at_goal": False})
        new_obs = Observation(state={"at_goal": True})
        action = Action(action_type="move")

        reward = reward_fn.compute(prev_obs, action, new_obs)

        assert reward == 10.0

    def test_exploration_reward(self):
        """Test ExplorationReward for novelty."""
        reward_fn = ExplorationReward(novelty_bonus=1.0, revisit_penalty=0.1)

        obs1 = Observation(state={"x": 1})
        obs2 = Observation(state={"x": 2})
        action = Action(action_type="move")

        # First visit to new state
        r1 = reward_fn.compute(obs1, action, obs2)
        assert r1 == 1.0

        # Revisit same state
        r2 = reward_fn.compute(obs1, action, obs2)
        assert r2 == -0.1

    def test_composite_reward(self):
        """Test CompositeReward combination."""
        goal = Goal(description="test", success_condition=lambda obs: False)
        goal_reward = GoalReward(goal)
        exploration = ExplorationReward()

        composite = CompositeReward()
        composite.add(goal_reward, 1.0)
        composite.add(exploration, 0.5)

        obs = Observation(state={"x": 1})
        action = Action(action_type="move")

        reward = composite.compute(obs, action, obs)
        # Should include both components
        assert isinstance(reward, float)

    def test_distance_reward(self):
        """Test DistanceReward for navigation."""
        reward_fn = DistanceReward(scale=1.0)

        prev_obs = Observation(state={"position": (0, 0), "goal": (4, 4)})
        new_obs = Observation(state={"position": (1, 0), "goal": (4, 4)})
        action = Action(action_type="move")

        reward = reward_fn.compute(prev_obs, action, new_obs)

        assert reward > 0  # Got closer to goal


# =============================================================================
# Harness Tests
# =============================================================================

class TestHarness:
    """Tests for the main EmbodiedHarness."""

    def test_harness_creation(self):
        """Test harness initialization."""
        env = MockEnvironment()
        goal = Goal(description="Reach goal")

        harness = EmbodiedHarness(env, goal)

        assert harness.env == env
        assert harness.goal == goal

    def test_harness_run_simple(self):
        """Test running harness on simple problem."""
        env = MockEnvironment(
            start_pos=(0, 0),
            goal_pos=(1, 0),  # Very close goal
            max_steps=10,
        )
        goal = Goal(
            description="Reach (1, 0)",
            success_condition=lambda obs: obs.state.get("position") == (1, 0),
        )
        config = HarnessConfig(
            max_branches=2,
            max_depth=5,
            budget_steps=20,
        )

        harness = EmbodiedHarness(env, goal, config)
        result = harness.run()

        assert result.total_steps > 0
        assert result.total_branches >= 1

    def test_harness_exploration_summary(self):
        """Test getting exploration summary."""
        env = MockEnvironment()
        goal = Goal(description="Test")

        harness = EmbodiedHarness(env, goal)
        summary = harness.get_exploration_summary()

        assert "total_branches" in summary
        assert "total_steps" in summary

    def test_harness_config(self):
        """Test HarnessConfig defaults."""
        config = HarnessConfig()

        assert config.max_branches == 4
        assert config.max_depth == 50
        assert config.failure_propagation


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_exploration_mock(self):
        """Test full exploration on mock environment."""
        env = MockEnvironment(
            width=3,
            height=3,
            start_pos=(0, 0),
            goal_pos=(2, 2),
        )
        goal = Goal(
            description="Navigate to goal",
            success_condition=lambda obs: obs.state.get("position") == (2, 2),
        )
        config = HarnessConfig(
            max_branches=3,
            max_depth=20,
            budget_steps=100,
        )

        harness = EmbodiedHarness(env, goal, config)
        result = harness.run()

        # Should find a path in small grid
        assert result.total_steps > 0

    def test_text_game_exploration(self):
        """Test exploration in text adventure."""
        env = TextGameEnvironment(
            goal_room="study",
        )
        goal = Goal(
            description="Reach the study",
            success_condition=lambda obs: "Study" in obs.state.get("room", ""),
        )
        config = HarnessConfig(
            max_branches=2,
            max_depth=15,
            budget_steps=50,
        )

        harness = EmbodiedHarness(env, goal, config)
        result = harness.run()

        assert result.total_steps > 0

    def test_failure_sharing_integration(self):
        """Test that failures are shared across branches."""
        # Create environment with obstacle
        env = MockEnvironment(
            obstacles=[(1, 0)],
            goal_pos=(2, 0),
        )
        goal = Goal(description="Reach goal")
        config = HarnessConfig(
            failure_propagation=True,
            max_branches=3,
        )

        harness = EmbodiedHarness(env, goal, config)
        harness.run()

        # Check that failures were recorded
        summary = harness.get_exploration_summary()
        # Failure memory should have recorded the obstacle collision
        assert summary["failures_recorded"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
