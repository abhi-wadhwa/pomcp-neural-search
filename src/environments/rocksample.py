"""RockSample[n, k] -- grid world POMDP (Smith & Simmons, 2004).

An agent navigates an n x n grid containing k rocks. Each rock is either
good (+10 reward for sampling) or bad (-10 for sampling). The agent can:
  - Move North/South/East/West
  - Sample: collect the rock at current position
  - Check(i): sense rock i (noisy, accuracy decreases with distance)

The agent exits by moving east off the grid (+10 reward).

State: (agent_x, agent_y, rock_0_good, ..., rock_{k-1}_good)
Observations: {GOOD, BAD, NONE} (only meaningful after Check)

This implements RockSample[4, 4].
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

from src.environments.pomdp_base import POMDPEnv, StepResult


class RSAction(IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3
    SAMPLE = 4
    # CHECK_0 through CHECK_{k-1} are dynamically created
    # but we use integers >= 5


class RSObservation(IntEnum):
    NONE = 0
    GOOD = 1
    BAD = 2


@dataclass(frozen=True)
class RSState:
    """RockSample state: agent position + rock goodness."""

    x: int
    y: int
    rocks: tuple[bool, ...]  # True = good rock

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.rocks))


class RockSamplePOMDP(POMDPEnv):
    """RockSample[n, k] POMDP.

    Parameters
    ----------
    grid_size : int
        Size of the grid (n x n).
    rock_positions : list of (x, y) tuples
        Positions of the k rocks on the grid.
    half_efficiency_distance : float
        Distance at which check sensor has 50% accuracy.
    discount : float
        Discount factor.
    """

    def __init__(
        self,
        grid_size: int = 4,
        rock_positions: list[tuple[int, int]] | None = None,
        half_efficiency_distance: float = 2.0,
        discount: float = 0.95,
    ):
        super().__init__(discount=discount)
        self.grid_size = grid_size

        if rock_positions is None:
            # Default RockSample[4,4] configuration
            self.rock_positions = [(0, 1), (1, 3), (2, 0), (3, 2)]
        else:
            self.rock_positions = rock_positions

        self.num_rocks = len(self.rock_positions)
        self.half_eff_dist = half_efficiency_distance

        # Actions: 4 moves + sample + k checks
        self.num_actions = 5 + self.num_rocks

        # Exit reward
        self.exit_reward = 10.0
        self.good_rock_reward = 10.0
        self.bad_rock_penalty = -10.0
        self.move_cost = 0.0

    @property
    def name(self) -> str:
        return f"RockSample[{self.grid_size},{self.num_rocks}]"

    def get_actions(self) -> list[int]:
        return list(range(self.num_actions))

    def get_states(self) -> list[RSState]:
        """Enumerate all states (exponential in k, only for small problems)."""
        states = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                for rock_bits in range(2**self.num_rocks):
                    rocks = tuple(
                        bool((rock_bits >> i) & 1) for i in range(self.num_rocks)
                    )
                    states.append(RSState(x, y, rocks))
        # Add terminal state
        states.append(RSState(self.grid_size, 0, tuple(False for _ in range(self.num_rocks))))
        return states

    def get_observations(self) -> list[RSObservation]:
        return list(RSObservation)

    def sample_initial_state(self) -> RSState:
        """Agent starts at (0, y_mid), rocks are randomly good/bad."""
        rocks = tuple(bool(self.rng.integers(0, 2)) for _ in range(self.num_rocks))
        return RSState(x=0, y=self.grid_size // 2, rocks=rocks)

    def get_initial_belief(self) -> np.ndarray:
        """Uniform belief over rock goodness, known agent position."""
        # For simplicity, return uniform over rock configurations
        n_configs = 2**self.num_rocks
        return np.ones(n_configs) / n_configs

    def _check_accuracy(self, agent_x: int, agent_y: int, rock_idx: int) -> float:
        """Sensor accuracy as a function of distance to rock.

        accuracy = 2^{-d / half_efficiency_distance}
        where d is the Euclidean distance.
        """
        rx, ry = self.rock_positions[rock_idx]
        dist = np.sqrt((agent_x - rx) ** 2 + (agent_y - ry) ** 2)
        # Accuracy = 0.5 + 0.5 * 2^(-d/d_half)
        # At d=0: accuracy = 1.0; at d=d_half: accuracy = 0.75; at d=inf: accuracy = 0.5
        efficiency = 2.0 ** (-dist / self.half_eff_dist)
        return 0.5 + 0.5 * efficiency

    def step(self, state: RSState, action: int) -> StepResult:
        """Generative model for RockSample."""
        if self._is_terminal(state):
            return StepResult(
                next_state=state,
                observation=RSObservation.NONE,
                reward=0.0,
                done=True,
            )

        x, y = state.x, state.y
        rocks = list(state.rocks)
        reward = self.move_cost
        obs = RSObservation.NONE
        done = False

        if action == RSAction.NORTH:
            y = min(y + 1, self.grid_size - 1)
        elif action == RSAction.SOUTH:
            y = max(y - 1, 0)
        elif action == RSAction.EAST:
            x = x + 1
            if x >= self.grid_size:
                # Exit the grid
                return StepResult(
                    next_state=RSState(self.grid_size, 0, tuple(rocks)),
                    observation=RSObservation.NONE,
                    reward=self.exit_reward,
                    done=True,
                )
        elif action == RSAction.WEST:
            x = max(x - 1, 0)
        elif action == RSAction.SAMPLE:
            # Check if on a rock
            for i, (rx, ry) in enumerate(self.rock_positions):
                if x == rx and y == ry:
                    if rocks[i]:
                        reward = self.good_rock_reward
                    else:
                        reward = self.bad_rock_penalty
                    # Rock becomes bad after sampling
                    rocks[i] = False
                    break
        else:
            # Check action: action - 5 = rock index
            rock_idx = action - 5
            if 0 <= rock_idx < self.num_rocks:
                accuracy = self._check_accuracy(x, y, rock_idx)
                if self.rng.random() < accuracy:
                    obs = RSObservation.GOOD if rocks[rock_idx] else RSObservation.BAD
                else:
                    obs = RSObservation.BAD if rocks[rock_idx] else RSObservation.GOOD

        next_state = RSState(x, y, tuple(rocks))
        return StepResult(
            next_state=next_state,
            observation=obs,
            reward=reward,
            done=done,
        )

    def _is_terminal(self, state: RSState) -> bool:
        return state.x >= self.grid_size

    def is_terminal(self, state: Any) -> bool:
        return self._is_terminal(state)

    def get_reward_range(self) -> tuple[float, float]:
        return (self.bad_rock_penalty, max(self.exit_reward, self.good_rock_reward))

    def belief_features(self, particles: list[RSState]) -> np.ndarray:
        """Convert particles to feature vector.

        Features:
        - Agent position (normalized): 2 values
        - P(rock_i is good) for each rock: k values
        Total: 2 + k features
        """
        if not particles:
            features = np.zeros(2 + self.num_rocks)
            features[:2] = 0.5
            features[2:] = 0.5
            return features

        n = len(particles)
        avg_x = sum(p.x for p in particles) / n / max(self.grid_size - 1, 1)
        avg_y = sum(p.y for p in particles) / n / max(self.grid_size - 1, 1)

        rock_probs = np.zeros(self.num_rocks)
        for p in particles:
            for i in range(self.num_rocks):
                if p.rocks[i]:
                    rock_probs[i] += 1
        rock_probs /= n

        return np.concatenate([[avg_x, avg_y], rock_probs])

    def render_grid(self, state: RSState) -> str:
        """Render the grid as a string for visualization."""
        lines = []
        for y in range(self.grid_size - 1, -1, -1):
            row = []
            for x in range(self.grid_size):
                cell = "."
                # Check for rocks
                for i, (rx, ry) in enumerate(self.rock_positions):
                    if x == rx and y == ry:
                        cell = "G" if state.rocks[i] else "B"
                if x == state.x and y == state.y:
                    cell = "A"
                row.append(cell)
            lines.append(" ".join(row))
        return "\n".join(lines)
