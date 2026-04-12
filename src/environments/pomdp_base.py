"""Abstract POMDP environment interface.

A POMDP is defined by the tuple (S, A, O, T, Z, R, gamma):
- S: state space
- A: action space
- O: observation space
- T(s, a, s'): transition probability P(s' | s, a)
- Z(s', a, o): observation probability P(o | s', a)
- R(s, a): reward function
- gamma: discount factor
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class StepResult:
    """Result of taking an action in a POMDP."""

    next_state: Any
    observation: Any
    reward: float
    done: bool


class POMDPEnv(ABC):
    """Abstract base class for POMDP environments.

    All environments must implement the generative model interface,
    which allows sampling transitions (s, a) -> (s', o, r).
    This is sufficient for POMCP (no explicit T, Z matrices needed).
    """

    def __init__(self, discount: float = 0.95):
        self.discount = discount
        self.rng = np.random.default_rng()

    def seed(self, seed: int) -> None:
        """Set the random seed for reproducibility."""
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def get_actions(self) -> list[Any]:
        """Return the list of available actions."""
        ...

    @abstractmethod
    def get_states(self) -> list[Any]:
        """Return the list of all states (for tabular environments)."""
        ...

    @abstractmethod
    def get_observations(self) -> list[Any]:
        """Return the list of all possible observations."""
        ...

    @abstractmethod
    def sample_initial_state(self) -> Any:
        """Sample an initial state from the prior distribution."""
        ...

    @abstractmethod
    def step(self, state: Any, action: Any) -> StepResult:
        """Generative model: sample (s', o, r) given (s, a).

        This is the core interface used by POMCP. It must be a
        black-box simulator that can be called with any state.
        """
        ...

    @abstractmethod
    def get_initial_belief(self) -> np.ndarray:
        """Return the initial belief distribution over states.

        Returns a probability vector over get_states().
        """
        ...

    @abstractmethod
    def belief_features(self, particles: list[Any]) -> np.ndarray:
        """Convert a particle set (belief approximation) to a feature vector.

        Used as input to neural networks. Should produce a fixed-size
        vector regardless of the number of particles.
        """
        ...

    def get_reward_range(self) -> tuple[float, float]:
        """Return (min_reward, max_reward) for normalization."""
        return (-100.0, 100.0)

    def get_transition_probability(
        self, state: Any, action: Any, next_state: Any
    ) -> float:
        """Return T(s, a, s') = P(s' | s, a).

        Optional: only needed for exact methods like QMDP.
        Default raises NotImplementedError.
        """
        raise NotImplementedError("Explicit transition model not available")

    def get_observation_probability(
        self, next_state: Any, action: Any, observation: Any
    ) -> float:
        """Return Z(s', a, o) = P(o | s', a).

        Optional: only needed for exact belief updates.
        Default raises NotImplementedError.
        """
        raise NotImplementedError("Explicit observation model not available")

    def get_reward(self, state: Any, action: Any) -> float:
        """Return R(s, a).

        Optional: only needed for exact methods like QMDP.
        Default raises NotImplementedError.
        """
        raise NotImplementedError("Explicit reward model not available")

    def is_terminal(self, state: Any) -> bool:
        """Check if a state is terminal. Default: never terminal."""
        return False

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the environment name."""
        ...
