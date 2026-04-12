"""Particle filter for belief state representation in POMDPs.

A belief state b(s) is a probability distribution over states. In POMCP,
beliefs are represented as unweighted particle sets: a collection of
state samples that approximate the posterior distribution.

Key operations:
- Update: given action a and observation o, update particles via rejection sampling
- Reinvigorate: when particles are depleted, add new consistent particles
- Features: extract a fixed-size feature vector from the particle set
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.environments.pomdp_base import POMDPEnv


class ParticleFilter:
    """Particle filter belief representation for POMCP.

    Maintains an unweighted set of state particles representing
    the current belief distribution. Particles are updated via
    rejection sampling from the generative model.

    Parameters
    ----------
    env : POMDPEnv
        The POMDP environment (provides the generative model).
    num_particles : int
        Target number of particles to maintain.
    reinvigoration_count : int
        Number of fresh particles to add during reinvigoration.
    max_rejections : int
        Maximum rejection sampling attempts before reinvigoration.
    """

    def __init__(
        self,
        env: POMDPEnv,
        num_particles: int = 1000,
        reinvigoration_count: int = 50,
        max_rejections: int = 5000,
    ):
        self.env = env
        self.num_particles = num_particles
        self.reinvigoration_count = reinvigoration_count
        self.max_rejections = max_rejections
        self.particles: list[Any] = []
        self._initialize()

    def _initialize(self) -> None:
        """Initialize particles from the prior distribution."""
        self.particles = [
            self.env.sample_initial_state() for _ in range(self.num_particles)
        ]

    def reset(self) -> None:
        """Reset to the prior belief."""
        self._initialize()

    def update(self, action: Any, observation: Any) -> None:
        """Update belief given action and observation.

        Uses rejection sampling: repeatedly simulate (s, a) -> (s', o)
        from existing particles, keeping only those where o matches
        the actual observation. This implements exact Bayesian filtering
        for the particle approximation.

        If too few particles survive, reinvigoration adds fresh ones.
        """
        new_particles: list[Any] = []
        attempts = 0

        while len(new_particles) < self.num_particles and attempts < self.max_rejections:
            # Sample a state from current particles
            idx = self.env.rng.integers(0, len(self.particles))
            state = self.particles[idx]

            # Simulate forward
            result = self.env.step(state, action)

            # Accept if observation matches
            if result.observation == observation:
                new_particles.append(result.next_state)

            attempts += 1

        # Reinvigorate if particles are depleted
        if len(new_particles) < self.num_particles:
            new_particles = self._reinvigorate(
                new_particles, action, observation
            )

        self.particles = new_particles

    def _reinvigorate(
        self,
        current_particles: list[Any],
        action: Any,
        observation: Any,
    ) -> list[Any]:
        """Add new particles consistent with the observation.

        Reinvigoration is critical for POMCP's particle filter.
        When particles are depleted (e.g., after unlikely observations),
        we sample new states from the prior and filter for consistency
        with the observation history.

        Strategy: sample from prior, simulate with action, keep if
        observation matches. Also duplicate existing particles.
        """
        # First, pad with random consistent particles
        attempts = 0
        while (
            len(current_particles) < self.num_particles
            and attempts < self.max_rejections
        ):
            # Sample fresh state from prior
            state = self.env.sample_initial_state()
            result = self.env.step(state, action)

            if result.observation == observation:
                current_particles.append(result.next_state)

            attempts += 1

        # If still not enough, duplicate existing particles
        if current_particles and len(current_particles) < self.num_particles:
            while len(current_particles) < self.num_particles:
                idx = self.env.rng.integers(0, len(current_particles))
                current_particles.append(current_particles[idx])

        # Fallback: if we have nothing, sample from prior
        if not current_particles:
            current_particles = [
                self.env.sample_initial_state()
                for _ in range(self.num_particles)
            ]

        return current_particles

    def get_features(self) -> np.ndarray:
        """Extract feature vector from current particle set.

        Delegates to the environment's belief_features method.
        """
        return self.env.belief_features(self.particles)

    def get_state_distribution(self) -> dict[Any, float]:
        """Compute empirical distribution over states.

        Returns a dict mapping states to their estimated probabilities.
        """
        counts: dict[Any, int] = {}
        for p in self.particles:
            key = p
            counts[key] = counts.get(key, 0) + 1

        total = len(self.particles)
        return {s: c / total for s, c in counts.items()}

    def sample(self) -> Any:
        """Sample a single state from the current belief."""
        idx = self.env.rng.integers(0, len(self.particles))
        return self.particles[idx]

    def inject_particles(self, new_particles: list[Any]) -> None:
        """Inject particles directly (used by POMCP tree reuse)."""
        self.particles = list(new_particles)

    def __len__(self) -> int:
        return len(self.particles)

    def __repr__(self) -> str:
        dist = self.get_state_distribution()
        top_states = sorted(dist.items(), key=lambda x: -x[1])[:5]
        desc = ", ".join(f"{s}: {p:.2f}" for s, p in top_states)
        return f"ParticleFilter(n={len(self.particles)}, top=[{desc}])"
