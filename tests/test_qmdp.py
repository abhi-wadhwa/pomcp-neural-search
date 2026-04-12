"""Tests for the QMDP baseline planner."""

import numpy as np
import pytest

from src.core.qmdp import QMDP
from src.environments.tiger import TigerAction, TigerPOMDP, TigerState


class TestQMDP:
    """Test QMDP planner."""

    def setup_method(self) -> None:
        self.env = TigerPOMDP()
        self.env.seed(42)
        self.qmdp = QMDP(self.env)

    def test_solve_converges(self) -> None:
        iterations = self.qmdp.solve()
        assert iterations < 1000, "Value iteration should converge"

    def test_value_function_shape(self) -> None:
        self.qmdp.solve()
        assert self.qmdp.V.shape == (2,)  # 2 states
        assert self.qmdp.Q.shape == (2, 3)  # 2 states, 3 actions

    def test_mdp_optimal_policy(self) -> None:
        """In the fully observable MDP, optimal policy opens the safe door.

        If we know the state:
        - TIGER_LEFT -> open right (+10 always better than listen -1)
        - TIGER_RIGHT -> open left (+10 always better than listen -1)

        But due to the reset dynamics, the MDP solution considers
        the discounted future as well.
        """
        self.qmdp.solve()
        policy = self.qmdp.get_mdp_policy()

        # When tiger is left, should open right
        assert policy[TigerState.TIGER_LEFT] == TigerAction.OPEN_RIGHT
        # When tiger is right, should open left
        assert policy[TigerState.TIGER_RIGHT] == TigerAction.OPEN_LEFT

    def test_qmdp_listen_when_uncertain(self) -> None:
        """QMDP with uniform belief should not commit to a door.

        With b = [0.5, 0.5]:
        Q(b, listen) = 0.5*(-1) + 0.5*(-1) = -1
        Q(b, open_left) = 0.5*(-100) + 0.5*(10) = -45
        Q(b, open_right) = 0.5*(10) + 0.5*(-100) = -45

        So QMDP selects LISTEN (the one-step Q is higher).
        """
        self.qmdp.solve()
        belief = np.array([0.5, 0.5])
        action = self.qmdp.select_action(belief)

        assert action == TigerAction.LISTEN

    def test_qmdp_open_when_confident(self) -> None:
        """With high confidence, QMDP should open the safe door."""
        self.qmdp.solve()

        # 99% confident tiger is left -> open right
        belief = np.array([0.99, 0.01])
        action = self.qmdp.select_action(belief)
        assert action == TigerAction.OPEN_RIGHT

        # 99% confident tiger is right -> open left
        belief = np.array([0.01, 0.99])
        action = self.qmdp.select_action(belief)
        assert action == TigerAction.OPEN_LEFT

    def test_q_values_shape(self) -> None:
        self.qmdp.solve()
        belief = np.array([0.5, 0.5])
        q_values = self.qmdp.get_q_values(belief)
        assert len(q_values) == 3  # 3 actions

    def test_value_positive_for_known_state(self) -> None:
        """Value of a known state should be positive (can always win)."""
        self.qmdp.solve()
        belief_known = np.array([1.0, 0.0])  # Know tiger is left
        value = self.qmdp.get_value(belief_known)
        assert value > 0, f"Value of known state should be positive, got {value}"

    def test_select_from_particles(self) -> None:
        """With 99% confidence, QMDP should open the safe door."""
        self.qmdp.solve()
        # Need very high confidence due to Tiger's asymmetric rewards (-100 vs +10)
        # At 90% confidence: EV(open_right) = 0.9*10 + 0.1*(-100) = -1.0 ~ EV(listen) = -1.0
        # At 99%: EV(open_right) = 0.99*10 + 0.01*(-100) = 8.9 >> -1
        particles = [TigerState.TIGER_LEFT] * 99 + [TigerState.TIGER_RIGHT] * 1
        action = self.qmdp.select_action_from_particles(particles)
        assert action == TigerAction.OPEN_RIGHT

    def test_value_iteration_values_reasonable(self) -> None:
        """MDP values should be in a reasonable range."""
        self.qmdp.solve()

        # Value of fully observable state (know tiger location)
        # Should be positive -- we can always open the right door for +10
        for v in self.qmdp.V:
            assert v > 0, f"MDP value should be positive, got {v}"

    def test_transition_matrix_normalized(self) -> None:
        """Transition matrix rows should sum to 1."""
        for si in range(self.qmdp.n_states):
            for ai in range(self.qmdp.n_actions):
                row_sum = self.qmdp.T[si, ai].sum()
                np.testing.assert_almost_equal(
                    row_sum, 1.0, decimal=5,
                    err_msg=f"T[{si},{ai}] sums to {row_sum}"
                )

    def test_repr(self) -> None:
        s = repr(self.qmdp)
        assert "QMDP" in s
        assert "states=2" in s
        assert "actions=3" in s

    def test_qmdp_solves_fully_observable_optimally(self) -> None:
        """When belief is a delta (fully observable), QMDP is optimal.

        QMDP is exact for fully observable POMDPs, so with a point belief
        it should give the same answer as the MDP optimal policy.
        """
        self.qmdp.solve()

        # Delta belief: tiger is left
        belief_left = np.array([1.0, 0.0])
        action = self.qmdp.select_action(belief_left)
        assert action == TigerAction.OPEN_RIGHT, "Should open right when tiger is left"

        # Delta belief: tiger is right
        belief_right = np.array([0.0, 1.0])
        action = self.qmdp.select_action(belief_right)
        assert action == TigerAction.OPEN_LEFT, "Should open left when tiger is right"
