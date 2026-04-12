"""QMDP baseline for POMDP planning.

QMDP (Littman et al., 1995) approximates the POMDP value function by
solving the underlying fully observable MDP and using MDP Q-values
weighted by the belief state:

    Q_QMDP(b, a) = sum_s b(s) * Q_MDP(s, a)

This ignores the value of information -- it assumes the state will
become fully observable after one step. Despite this limitation,
QMDP is a strong baseline for problems where information gathering
is not critical, and it's very fast since it precomputes Q_MDP offline.

Algorithm:
1. Solve MDP via value iteration: V*(s) = max_a [R(s,a) + gamma * sum_s' T(s,a,s') V*(s')]
2. Compute Q*(s,a) = R(s,a) + gamma * sum_s' T(s,a,s') V*(s')
3. At planning time: a* = argmax_a sum_s b(s) Q*(s,a)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.environments.pomdp_base import POMDPEnv


class QMDP:
    """QMDP planner for POMDPs.

    Solves the underlying MDP via value iteration, then uses
    belief-weighted Q-values for action selection.

    Parameters
    ----------
    env : POMDPEnv
        The POMDP environment (must provide explicit T, R).
    tolerance : float
        Convergence tolerance for value iteration.
    max_iterations : int
        Maximum number of value iteration sweeps.
    discount : float or None
        Override the environment's discount factor.
    """

    def __init__(
        self,
        env: POMDPEnv,
        tolerance: float = 1e-6,
        max_iterations: int = 1000,
        discount: float | None = None,
    ):
        self.env = env
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.discount = discount if discount is not None else env.discount

        self.states = env.get_states()
        self.actions = env.get_actions()
        self.n_states = len(self.states)
        self.n_actions = len(self.actions)

        # Map states to indices
        self.state_to_idx: dict[Any, int] = {
            s: i for i, s in enumerate(self.states)
        }

        # Value function and Q-values
        self.V = np.zeros(self.n_states)
        self.Q = np.zeros((self.n_states, self.n_actions))

        # Precompute transition and reward matrices
        self._build_matrices()

    def _build_matrices(self) -> None:
        """Build transition and reward matrices from the environment.

        T[s, a, s'] = P(s' | s, a)
        R[s, a] = expected reward
        """
        self.T = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.R = np.zeros((self.n_states, self.n_actions))

        for si, s in enumerate(self.states):
            for ai, a in enumerate(self.actions):
                try:
                    self.R[si, ai] = self.env.get_reward(s, a)
                except NotImplementedError:
                    # If explicit reward not available, estimate via sampling
                    rewards = []
                    for _ in range(100):
                        result = self.env.step(s, a)
                        rewards.append(result.reward)
                    self.R[si, ai] = np.mean(rewards)

                for sj, s_next in enumerate(self.states):
                    try:
                        self.T[si, ai, sj] = self.env.get_transition_probability(
                            s, a, s_next
                        )
                    except NotImplementedError:
                        pass  # Will be estimated if needed

        # Normalize transition probabilities
        for si in range(self.n_states):
            for ai in range(self.n_actions):
                row_sum = self.T[si, ai].sum()
                if row_sum > 0:
                    self.T[si, ai] /= row_sum

    def solve(self, verbose: bool = False) -> int:
        """Run value iteration to compute Q_MDP(s, a).

        Returns the number of iterations until convergence.
        """
        for iteration in range(self.max_iterations):
            V_old = self.V.copy()

            # Compute Q-values: Q(s,a) = R(s,a) + gamma * sum_s' T(s,a,s') V(s')
            for ai in range(self.n_actions):
                self.Q[:, ai] = self.R[:, ai] + self.discount * (
                    self.T[:, ai, :] @ self.V
                )

            # Value function: V(s) = max_a Q(s,a)
            self.V = self.Q.max(axis=1)

            # Check convergence
            delta = np.max(np.abs(self.V - V_old))
            if verbose and (iteration + 1) % 50 == 0:
                print(f"Value iteration {iteration+1}: delta={delta:.8f}")

            if delta < self.tolerance:
                if verbose:
                    print(f"Converged after {iteration+1} iterations")
                return iteration + 1

        if verbose:
            print(f"Did not converge after {self.max_iterations} iterations")
        return self.max_iterations

    def select_action(self, belief: np.ndarray) -> Any:
        """Select action using QMDP: argmax_a sum_s b(s) Q(s,a).

        Parameters
        ----------
        belief : np.ndarray
            Belief distribution over states (probability vector).

        Returns
        -------
        action : Any
            The action with highest belief-weighted Q-value.
        """
        # Q_QMDP(b, a) = b^T Q(:, a)
        q_values = belief @ self.Q
        best_action_idx = int(np.argmax(q_values))
        return self.actions[best_action_idx]

    def select_action_from_particles(self, particles: list[Any]) -> Any:
        """Select action from a particle set (belief approximation).

        Converts particles to a belief vector and calls select_action.
        """
        belief = self._particles_to_belief(particles)
        return self.select_action(belief)

    def get_q_values(self, belief: np.ndarray) -> np.ndarray:
        """Compute Q_QMDP(b, a) for all actions.

        Parameters
        ----------
        belief : np.ndarray
            Belief distribution over states.

        Returns
        -------
        q_values : np.ndarray
            Array of Q-values for each action.
        """
        return belief @ self.Q

    def get_value(self, belief: np.ndarray) -> float:
        """Compute V_QMDP(b) = max_a Q_QMDP(b, a)."""
        q_values = self.get_q_values(belief)
        return float(np.max(q_values))

    def _particles_to_belief(self, particles: list[Any]) -> np.ndarray:
        """Convert a particle set to a belief vector."""
        belief = np.zeros(self.n_states)
        for p in particles:
            if p in self.state_to_idx:
                belief[self.state_to_idx[p]] += 1
        total = belief.sum()
        if total > 0:
            belief /= total
        else:
            belief = np.ones(self.n_states) / self.n_states
        return belief

    def get_mdp_policy(self) -> dict[Any, Any]:
        """Return the optimal MDP policy: state -> action."""
        policy = {}
        for si, s in enumerate(self.states):
            best_ai = int(np.argmax(self.Q[si]))
            policy[s] = self.actions[best_ai]
        return policy

    def __repr__(self) -> str:
        return (
            f"QMDP(states={self.n_states}, actions={self.n_actions}, "
            f"discount={self.discount})"
        )
