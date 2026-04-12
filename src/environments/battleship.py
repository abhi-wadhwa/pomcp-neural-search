"""Battleship POMDP -- grid-based search game.

The agent must find and sink ships hidden on a grid by firing at cells.
Each cell is either empty or occupied by a ship segment. Firing at a cell
reveals whether it was a hit or miss.

State: the full grid of ship placements (hidden from agent)
Actions: fire at cell (r, c)
Observations: {HIT, MISS}
Rewards: +1 for hit, -1 for miss (already fired), 0 for miss (new cell)

The game ends when all ship segments are hit.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

from src.environments.pomdp_base import POMDPEnv, StepResult


class BSObservation(IntEnum):
    MISS = 0
    HIT = 1


@dataclass(frozen=True)
class BSState:
    """Battleship state: ship grid + fired cells."""

    grid: tuple[tuple[bool, ...], ...]  # True = ship present
    fired: tuple[tuple[bool, ...], ...]  # True = already fired at

    def __hash__(self) -> int:
        return hash((self.grid, self.fired))


class BattleshipPOMDP(POMDPEnv):
    """Battleship POMDP on a small grid.

    Parameters
    ----------
    grid_size : int
        Size of the grid (grid_size x grid_size).
    ship_lengths : list of int
        Length of each ship to place.
    discount : float
        Discount factor.
    """

    def __init__(
        self,
        grid_size: int = 5,
        ship_lengths: list[int] | None = None,
        discount: float = 0.99,
    ):
        super().__init__(discount=discount)
        self.grid_size = grid_size
        self.ship_lengths = ship_lengths if ship_lengths is not None else [3, 2]
        self.total_ship_cells = sum(self.ship_lengths)

    @property
    def name(self) -> str:
        return f"Battleship[{self.grid_size}x{self.grid_size}]"

    def get_actions(self) -> list[int]:
        """Actions are cell indices: 0 to grid_size^2 - 1."""
        return list(range(self.grid_size * self.grid_size))

    def action_to_coord(self, action: int) -> tuple[int, int]:
        """Convert flat action index to (row, col)."""
        return divmod(action, self.grid_size)

    def coord_to_action(self, row: int, col: int) -> int:
        """Convert (row, col) to flat action index."""
        return row * self.grid_size + col

    def get_states(self) -> list[BSState]:
        """Not enumerable for reasonable grid sizes."""
        raise NotImplementedError("Battleship state space is too large to enumerate")

    def get_observations(self) -> list[BSObservation]:
        return list(BSObservation)

    def _place_ships(self) -> tuple[tuple[bool, ...], ...]:
        """Randomly place ships on the grid."""
        grid = [[False] * self.grid_size for _ in range(self.grid_size)]

        for length in self.ship_lengths:
            placed = False
            attempts = 0
            while not placed and attempts < 1000:
                attempts += 1
                horizontal = bool(self.rng.integers(0, 2))
                if horizontal:
                    r = int(self.rng.integers(0, self.grid_size))
                    c = int(self.rng.integers(0, self.grid_size - length + 1))
                    cells = [(r, c + i) for i in range(length)]
                else:
                    r = int(self.rng.integers(0, self.grid_size - length + 1))
                    c = int(self.rng.integers(0, self.grid_size))
                    cells = [(r + i, c) for i in range(length)]

                # Check no overlap
                if all(not grid[cr][cc] for cr, cc in cells):
                    for cr, cc in cells:
                        grid[cr][cc] = True
                    placed = True

        return tuple(tuple(row) for row in grid)

    def sample_initial_state(self) -> BSState:
        """Sample a random ship configuration."""
        grid = self._place_ships()
        fired = tuple(tuple(False for _ in range(self.grid_size)) for _ in range(self.grid_size))
        return BSState(grid=grid, fired=fired)

    def get_initial_belief(self) -> np.ndarray:
        """Uniform prior -- not representable explicitly."""
        return np.ones(1)  # Placeholder; we use particles

    def step(self, state: BSState, action: int) -> StepResult:
        """Fire at a cell."""
        r, c = self.action_to_coord(action)
        fired = [list(row) for row in state.fired]

        if fired[r][c]:
            # Already fired here -- penalty
            return StepResult(
                next_state=state,
                observation=BSObservation.MISS,
                reward=-1.0,
                done=False,
            )

        fired[r][c] = True
        new_fired = tuple(tuple(row) for row in fired)
        next_state = BSState(grid=state.grid, fired=new_fired)

        if state.grid[r][c]:
            obs = BSObservation.HIT
            reward = 1.0
        else:
            obs = BSObservation.MISS
            reward = 0.0

        # Check if all ship cells have been hit
        done = self._all_sunk(state.grid, new_fired)

        return StepResult(
            next_state=next_state,
            observation=obs,
            reward=reward,
            done=done,
        )

    def _all_sunk(
        self,
        grid: tuple[tuple[bool, ...], ...],
        fired: tuple[tuple[bool, ...], ...],
    ) -> bool:
        """Check if all ship cells have been hit."""
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if grid[r][c] and not fired[r][c]:
                    return False
        return True

    def is_terminal(self, state: Any) -> bool:
        if not isinstance(state, BSState):
            return False
        return self._all_sunk(state.grid, state.fired)

    def get_reward_range(self) -> tuple[float, float]:
        return (-1.0, 1.0)

    def belief_features(self, particles: list[BSState]) -> np.ndarray:
        """Convert particles to a feature vector.

        Features: for each cell, P(ship present | particles) and whether fired.
        Total: 2 * grid_size^2 features.
        """
        n_cells = self.grid_size * self.grid_size
        if not particles:
            features = np.full(2 * n_cells, 0.5)
            return features

        n = len(particles)
        ship_probs = np.zeros(n_cells)
        fired_flags = np.zeros(n_cells)

        for p in particles:
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    idx = r * self.grid_size + c
                    if p.grid[r][c]:
                        ship_probs[idx] += 1
                    if p.fired[r][c]:
                        fired_flags[idx] = 1.0  # Same across consistent particles

        ship_probs /= n
        return np.concatenate([ship_probs, fired_flags])

    def render_grid(self, state: BSState, show_ships: bool = False) -> str:
        """Render the battleship grid as a string."""
        lines = []
        header = "  " + " ".join(str(c) for c in range(self.grid_size))
        lines.append(header)

        for r in range(self.grid_size):
            row = [str(r)]
            for c in range(self.grid_size):
                if state.fired[r][c]:
                    if state.grid[r][c]:
                        row.append("X")  # Hit
                    else:
                        row.append("o")  # Miss
                elif show_ships and state.grid[r][c]:
                    row.append("S")  # Ship (hidden)
                else:
                    row.append("~")  # Unknown water
            lines.append(" ".join(row))

        return "\n".join(lines)
