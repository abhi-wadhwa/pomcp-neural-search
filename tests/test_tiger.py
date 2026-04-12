"""Tests for the Tiger POMDP environment."""

import numpy as np
import pytest

from src.environments.tiger import (
    TigerAction,
    TigerObservation,
    TigerPOMDP,
    TigerState,
)


class TestTigerEnvironment:
    """Test Tiger POMDP dynamics."""

    def setup_method(self) -> None:
        self.env = TigerPOMDP()
        self.env.seed(42)

    def test_states(self) -> None:
        states = self.env.get_states()
        assert len(states) == 2
        assert TigerState.TIGER_LEFT in states
        assert TigerState.TIGER_RIGHT in states

    def test_actions(self) -> None:
        actions = self.env.get_actions()
        assert len(actions) == 3
        assert TigerAction.LISTEN in actions
        assert TigerAction.OPEN_LEFT in actions
        assert TigerAction.OPEN_RIGHT in actions

    def test_observations(self) -> None:
        obs = self.env.get_observations()
        assert len(obs) == 2

    def test_initial_belief_uniform(self) -> None:
        belief = self.env.get_initial_belief()
        assert len(belief) == 2
        np.testing.assert_almost_equal(belief[0], 0.5)
        np.testing.assert_almost_equal(belief[1], 0.5)

    def test_listen_does_not_change_state(self) -> None:
        state = TigerState.TIGER_LEFT
        for _ in range(100):
            result = self.env.step(state, TigerAction.LISTEN)
            assert result.next_state == state
            assert result.done is False

    def test_listen_reward(self) -> None:
        state = TigerState.TIGER_LEFT
        result = self.env.step(state, TigerAction.LISTEN)
        assert result.reward == -1.0

    def test_listen_observation_is_noisy(self) -> None:
        """Verify observation accuracy is approximately 85%."""
        state = TigerState.TIGER_LEFT
        correct_count = 0
        n_trials = 5000

        for _ in range(n_trials):
            result = self.env.step(state, TigerAction.LISTEN)
            if result.observation == TigerObservation.HEAR_LEFT:
                correct_count += 1

        accuracy = correct_count / n_trials
        assert abs(accuracy - 0.85) < 0.03, f"Expected ~0.85, got {accuracy}"

    def test_open_correct_door(self) -> None:
        state = TigerState.TIGER_LEFT
        result = self.env.step(state, TigerAction.OPEN_RIGHT)
        assert result.reward == 10.0
        assert result.done is True

    def test_open_tiger_door(self) -> None:
        state = TigerState.TIGER_LEFT
        result = self.env.step(state, TigerAction.OPEN_LEFT)
        assert result.reward == -100.0
        assert result.done is True

    def test_transition_probability_listen(self) -> None:
        p = self.env.get_transition_probability(
            TigerState.TIGER_LEFT, TigerAction.LISTEN, TigerState.TIGER_LEFT
        )
        assert p == 1.0

        p = self.env.get_transition_probability(
            TigerState.TIGER_LEFT, TigerAction.LISTEN, TigerState.TIGER_RIGHT
        )
        assert p == 0.0

    def test_transition_probability_open(self) -> None:
        p = self.env.get_transition_probability(
            TigerState.TIGER_LEFT, TigerAction.OPEN_LEFT, TigerState.TIGER_LEFT
        )
        assert p == 0.5

    def test_observation_probability(self) -> None:
        p = self.env.get_observation_probability(
            TigerState.TIGER_LEFT, TigerAction.LISTEN, TigerObservation.HEAR_LEFT
        )
        assert p == 0.85

        p = self.env.get_observation_probability(
            TigerState.TIGER_LEFT, TigerAction.LISTEN, TigerObservation.HEAR_RIGHT
        )
        assert abs(p - 0.15) < 1e-10

    def test_observation_probability_open(self) -> None:
        p = self.env.get_observation_probability(
            TigerState.TIGER_LEFT, TigerAction.OPEN_LEFT, TigerObservation.HEAR_LEFT
        )
        assert p == 0.5

    def test_reward_function(self) -> None:
        assert self.env.get_reward(TigerState.TIGER_LEFT, TigerAction.LISTEN) == -1.0
        assert self.env.get_reward(TigerState.TIGER_LEFT, TigerAction.OPEN_LEFT) == -100.0
        assert self.env.get_reward(TigerState.TIGER_LEFT, TigerAction.OPEN_RIGHT) == 10.0
        assert self.env.get_reward(TigerState.TIGER_RIGHT, TigerAction.OPEN_LEFT) == 10.0
        assert self.env.get_reward(TigerState.TIGER_RIGHT, TigerAction.OPEN_RIGHT) == -100.0

    def test_belief_features(self) -> None:
        particles = [TigerState.TIGER_LEFT] * 7 + [TigerState.TIGER_RIGHT] * 3
        features = self.env.belief_features(particles)
        assert len(features) == 2
        np.testing.assert_almost_equal(features[0], 0.7)
        np.testing.assert_almost_equal(features[1], 0.3)

    def test_belief_features_empty(self) -> None:
        features = self.env.belief_features([])
        np.testing.assert_array_almost_equal(features, [0.5, 0.5])

    def test_optimal_action_uncertain(self) -> None:
        """Uniform belief -> should listen."""
        action = self.env.optimal_action_for_belief(np.array([0.5, 0.5]))
        assert action == TigerAction.LISTEN

    def test_optimal_action_confident_left(self) -> None:
        """High confidence tiger is left -> open right."""
        action = self.env.optimal_action_for_belief(np.array([0.99, 0.01]))
        assert action == TigerAction.OPEN_RIGHT

    def test_optimal_action_confident_right(self) -> None:
        """High confidence tiger is right -> open left."""
        action = self.env.optimal_action_for_belief(np.array([0.01, 0.99]))
        assert action == TigerAction.OPEN_LEFT

    def test_name(self) -> None:
        assert self.env.name == "Tiger"
