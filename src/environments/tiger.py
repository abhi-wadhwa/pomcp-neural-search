"""Tiger Problem -- classic POMDP benchmark (Kaelbling et al., 1998).

The agent stands in front of two doors. Behind one door is a tiger,
behind the other is a reward. The agent can:
  - Listen: receive a noisy observation about the tiger's location
  - Open-Left: open the left door
  - Open-Right: open the right door

States: {TIGER_LEFT, TIGER_RIGHT}
Actions: {LISTEN, OPEN_LEFT, OPEN_RIGHT}
Observations: {HEAR_LEFT, HEAR_RIGHT}

Rewards:
  - Listen: -1 (cost of listening)
  - Open correct door: +10
  - Open door with tiger: -100

Observation model:
  - After listen: hear correct side with probability 0.85
  - After opening a door: uniform observation (reset)
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from src.environments.pomdp_base import POMDPEnv, StepResult


class TigerState(IntEnum):
    TIGER_LEFT = 0
    TIGER_RIGHT = 1


class TigerAction(IntEnum):
    LISTEN = 0
    OPEN_LEFT = 1
    OPEN_RIGHT = 2


class TigerObservation(IntEnum):
    HEAR_LEFT = 0
    HEAR_RIGHT = 1


class TigerPOMDP(POMDPEnv):
    """Tiger Problem POMDP.

    Parameters
    ----------
    listen_accuracy : float
        Probability of hearing the correct side when listening.
    listen_cost : float
        Cost (negative reward) of the listen action.
    tiger_penalty : float
        Penalty for opening the door with the tiger.
    escape_reward : float
        Reward for opening the correct (safe) door.
    discount : float
        Discount factor.
    """

    def __init__(
        self,
        listen_accuracy: float = 0.85,
        listen_cost: float = -1.0,
        tiger_penalty: float = -100.0,
        escape_reward: float = 10.0,
        discount: float = 0.95,
    ):
        super().__init__(discount=discount)
        self.listen_accuracy = listen_accuracy
        self.listen_cost = listen_cost
        self.tiger_penalty = tiger_penalty
        self.escape_reward = escape_reward

    @property
    def name(self) -> str:
        return "Tiger"

    def get_actions(self) -> list[TigerAction]:
        return list(TigerAction)

    def get_states(self) -> list[TigerState]:
        return list(TigerState)

    def get_observations(self) -> list[TigerObservation]:
        return list(TigerObservation)

    def sample_initial_state(self) -> TigerState:
        return TigerState(self.rng.integers(0, 2))

    def get_initial_belief(self) -> np.ndarray:
        return np.array([0.5, 0.5])

    def step(self, state: TigerState, action: TigerAction) -> StepResult:
        """Generative model for the Tiger problem."""
        if action == TigerAction.LISTEN:
            # State doesn't change
            next_state = state
            reward = self.listen_cost

            # Noisy observation
            if self.rng.random() < self.listen_accuracy:
                # Correct observation
                if state == TigerState.TIGER_LEFT:
                    obs = TigerObservation.HEAR_LEFT
                else:
                    obs = TigerObservation.HEAR_RIGHT
            else:
                # Incorrect observation
                if state == TigerState.TIGER_LEFT:
                    obs = TigerObservation.HEAR_RIGHT
                else:
                    obs = TigerObservation.HEAR_LEFT

            done = False
        else:
            # Opening a door
            if action == TigerAction.OPEN_LEFT:
                opened_tiger = state == TigerState.TIGER_LEFT
            else:
                opened_tiger = state == TigerState.TIGER_RIGHT

            reward = self.tiger_penalty if opened_tiger else self.escape_reward

            # After opening, the problem resets: new random tiger location
            next_state = TigerState(self.rng.integers(0, 2))

            # Observation is uniform after opening
            obs = TigerObservation(self.rng.integers(0, 2))

            done = True

        return StepResult(
            next_state=next_state,
            observation=obs,
            reward=reward,
            done=done,
        )

    def get_transition_probability(
        self, state: TigerState, action: TigerAction, next_state: TigerState
    ) -> float:
        """T(s, a, s') for Tiger."""
        if action == TigerAction.LISTEN:
            # State doesn't change when listening
            return 1.0 if state == next_state else 0.0
        else:
            # After opening, tiger position is uniformly random
            return 0.5

    def get_observation_probability(
        self, next_state: TigerState, action: TigerAction, observation: TigerObservation
    ) -> float:
        """Z(s', a, o) for Tiger."""
        if action == TigerAction.LISTEN:
            if next_state == TigerState.TIGER_LEFT:
                if observation == TigerObservation.HEAR_LEFT:
                    return self.listen_accuracy
                else:
                    return 1.0 - self.listen_accuracy
            else:
                if observation == TigerObservation.HEAR_RIGHT:
                    return self.listen_accuracy
                else:
                    return 1.0 - self.listen_accuracy
        else:
            # Uniform observation after opening
            return 0.5

    def get_reward(self, state: TigerState, action: TigerAction) -> float:
        """R(s, a) for Tiger."""
        if action == TigerAction.LISTEN:
            return self.listen_cost
        elif action == TigerAction.OPEN_LEFT:
            return self.tiger_penalty if state == TigerState.TIGER_LEFT else self.escape_reward
        else:
            return self.tiger_penalty if state == TigerState.TIGER_RIGHT else self.escape_reward

    def get_reward_range(self) -> tuple[float, float]:
        return (self.tiger_penalty, self.escape_reward)

    def belief_features(self, particles: list[TigerState]) -> np.ndarray:
        """Convert particle set to belief feature vector.

        For Tiger, this is simply [P(tiger_left), P(tiger_right)].
        """
        if not particles:
            return np.array([0.5, 0.5])
        counts = np.zeros(2)
        for p in particles:
            counts[int(p)] += 1
        return counts / counts.sum()

    def optimal_action_for_belief(self, belief: np.ndarray) -> TigerAction:
        """Return the optimal action for a given belief state.

        For Tiger:
        - If belief is uncertain, listen
        - If confident tiger is left, open right
        - If confident tiger is right, open left
        """
        p_left = belief[0]
        # Expected value of each action
        ev_listen = self.listen_cost
        ev_open_left = p_left * self.tiger_penalty + (1 - p_left) * self.escape_reward
        ev_open_right = (1 - p_left) * self.tiger_penalty + p_left * self.escape_reward

        evs = [ev_listen, ev_open_left, ev_open_right]
        return TigerAction(int(np.argmax(evs)))
