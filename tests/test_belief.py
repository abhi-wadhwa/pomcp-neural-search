"""Tests for the particle filter belief state."""

import numpy as np
import pytest

from src.core.belief import ParticleFilter
from src.environments.tiger import TigerAction, TigerObservation, TigerPOMDP, TigerState


class TestParticleFilter:
    """Test particle filter operations."""

    def setup_method(self) -> None:
        self.env = TigerPOMDP()
        self.env.seed(42)

    def test_initialization(self) -> None:
        belief = ParticleFilter(self.env, num_particles=100)
        assert len(belief) == 100

    def test_initial_distribution_approximately_uniform(self) -> None:
        """Initial particles should be roughly 50/50."""
        belief = ParticleFilter(self.env, num_particles=1000)
        dist = belief.get_state_distribution()

        p_left = dist.get(TigerState.TIGER_LEFT, 0.0)
        assert abs(p_left - 0.5) < 0.1, f"Expected ~0.5, got {p_left}"

    def test_update_shifts_belief_correctly(self) -> None:
        """After hearing left multiple times, belief should shift toward tiger-left."""
        belief = ParticleFilter(self.env, num_particles=2000)

        # Simulate hearing LEFT 5 times (tiger is likely on the left)
        for _ in range(5):
            belief.update(TigerAction.LISTEN, TigerObservation.HEAR_LEFT)

        dist = belief.get_state_distribution()
        p_left = dist.get(TigerState.TIGER_LEFT, 0.0)

        # After 5 correct observations, P(left) should be high
        assert p_left > 0.7, f"Expected P(left) > 0.7, got {p_left}"

    def test_bayesian_consistency(self) -> None:
        """Verify particle filter converges to the correct Bayesian posterior.

        After hearing LEFT once with accuracy 0.85:
        P(TL | HL) = P(HL|TL) * P(TL) / P(HL)
                   = 0.85 * 0.5 / (0.85*0.5 + 0.15*0.5)
                   = 0.85
        """
        # Use many particles for accuracy
        belief = ParticleFilter(self.env, num_particles=5000)
        belief.update(TigerAction.LISTEN, TigerObservation.HEAR_LEFT)

        dist = belief.get_state_distribution()
        p_left = dist.get(TigerState.TIGER_LEFT, 0.0)

        # Should be approximately 0.85
        assert abs(p_left - 0.85) < 0.05, f"Expected ~0.85, got {p_left}"

    def test_update_with_hear_right(self) -> None:
        """After hearing RIGHT, belief should shift toward tiger-right."""
        belief = ParticleFilter(self.env, num_particles=5000)
        belief.update(TigerAction.LISTEN, TigerObservation.HEAR_RIGHT)

        dist = belief.get_state_distribution()
        p_right = dist.get(TigerState.TIGER_RIGHT, 0.0)

        assert abs(p_right - 0.85) < 0.05, f"Expected ~0.85, got {p_right}"

    def test_two_consistent_observations(self) -> None:
        """Two HEAR_LEFT observations in a row should increase P(TL) more.

        After 2 HL observations:
        P(TL | HL, HL) = 0.85^2 / (0.85^2 + 0.15^2) = 0.7225 / 0.745 ≈ 0.9697
        """
        belief = ParticleFilter(self.env, num_particles=5000)
        belief.update(TigerAction.LISTEN, TigerObservation.HEAR_LEFT)
        belief.update(TigerAction.LISTEN, TigerObservation.HEAR_LEFT)

        dist = belief.get_state_distribution()
        p_left = dist.get(TigerState.TIGER_LEFT, 0.0)

        expected = 0.85**2 / (0.85**2 + 0.15**2)
        assert abs(p_left - expected) < 0.07, f"Expected ~{expected:.4f}, got {p_left}"

    def test_reset(self) -> None:
        belief = ParticleFilter(self.env, num_particles=100)
        belief.update(TigerAction.LISTEN, TigerObservation.HEAR_LEFT)
        belief.reset()

        dist = belief.get_state_distribution()
        p_left = dist.get(TigerState.TIGER_LEFT, 0.0)
        assert abs(p_left - 0.5) < 0.15

    def test_sample(self) -> None:
        belief = ParticleFilter(self.env, num_particles=100)
        state = belief.sample()
        assert state in [TigerState.TIGER_LEFT, TigerState.TIGER_RIGHT]

    def test_get_features(self) -> None:
        belief = ParticleFilter(self.env, num_particles=100)
        features = belief.get_features()
        assert len(features) == 2
        np.testing.assert_almost_equal(features.sum(), 1.0)

    def test_inject_particles(self) -> None:
        belief = ParticleFilter(self.env, num_particles=100)
        new_particles = [TigerState.TIGER_LEFT] * 10
        belief.inject_particles(new_particles)
        assert len(belief) == 10

        dist = belief.get_state_distribution()
        assert dist[TigerState.TIGER_LEFT] == 1.0

    def test_reinvigoration(self) -> None:
        """Reinvigoration should prevent particle depletion."""
        belief = ParticleFilter(self.env, num_particles=100, max_rejections=100)

        # Even with few rejection attempts, we should still have particles
        belief.update(TigerAction.LISTEN, TigerObservation.HEAR_LEFT)
        assert len(belief) > 0
