"""POMCP -- Partially Observable Monte-Carlo Planning (Silver & Veness, 2010).

POMCP extends UCT (Upper Confidence bounds applied to Trees) to partially
observable domains by maintaining a tree of action-observation histories.
Each belief node stores a set of particles representing the belief state.

Algorithm outline:
1. SEARCH: from root belief node, repeat for N simulations:
   a. Sample a state s ~ B(root)
   b. SIMULATE(s, root, depth=0)
2. SIMULATE(s, node, depth):
   a. If depth > max_depth or terminal: return 0
   b. If leaf node: expand, then ROLLOUT(s, depth)
   c. Otherwise: SELECT action via UCB1, simulate forward:
      - (s', o, r) = G(s, a)
      - If observation child doesn't exist, create it
      - q = r + gamma * SIMULATE(s', child[o], depth+1)
      - Update N(node,a), Q(node,a), add s to B(child[o])
      - Return q
3. SELECT: argmax_a Q(node,a) + c * sqrt(ln N(node) / N(node,a))
4. ROLLOUT: simulate with random/neural policy to max depth

Key features:
- Belief is implicit in particles at each node
- Tree reuse: after real action+observation, reuse subtree
- Particle reinvigoration at belief nodes
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from src.environments.pomdp_base import POMDPEnv


class BeliefNode:
    """A belief node in the POMCP search tree.

    Corresponds to an action-observation history h.
    Stores particles representing the belief B(h).
    Children are ActionNodes indexed by action.
    """

    def __init__(self) -> None:
        self.particles: list[Any] = []
        self.children: dict[Any, ActionNode] = {}
        self.visit_count: int = 0

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def add_particle(self, state: Any) -> None:
        self.particles.append(state)

    def __repr__(self) -> str:
        return f"BeliefNode(visits={self.visit_count}, particles={len(self.particles)}, children={len(self.children)})"


class ActionNode:
    """An action node in the POMCP search tree.

    Corresponds to history h followed by action a.
    Children are BeliefNodes indexed by observation.
    """

    def __init__(self, action: Any) -> None:
        self.action = action
        self.children: dict[Any, BeliefNode] = {}
        self.visit_count: int = 0
        self.value: float = 0.0

    def __repr__(self) -> str:
        return (
            f"ActionNode(a={self.action}, visits={self.visit_count}, "
            f"Q={self.value:.3f}, obs_children={len(self.children)})"
        )


class POMCP:
    """POMCP planning algorithm.

    Parameters
    ----------
    env : POMDPEnv
        The POMDP environment.
    num_simulations : int
        Number of Monte Carlo simulations per planning step.
    max_depth : int
        Maximum search depth.
    exploration_constant : float
        UCB1 exploration constant c.
    rollout_policy : callable or None
        Custom rollout policy: (env, state, depth) -> action.
        If None, uses uniform random.
    value_estimator : callable or None
        Neural value estimator: (env, particles) -> float.
        If None, uses rollout to estimate value.
    discount : float or None
        Override environment discount factor.
    """

    def __init__(
        self,
        env: POMDPEnv,
        num_simulations: int = 1000,
        max_depth: int = 50,
        exploration_constant: float = 3.0,
        rollout_policy: Callable[..., Any] | None = None,
        value_estimator: Callable[..., float] | None = None,
        discount: float | None = None,
    ):
        self.env = env
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.c = exploration_constant
        self.rollout_policy = rollout_policy
        self.value_estimator = value_estimator
        self.discount = discount if discount is not None else env.discount
        self.root = BeliefNode()
        self.actions = env.get_actions()

        # Statistics
        self.total_simulations = 0
        self.tree_depth_sum = 0

    def search(self, particles: list[Any] | None = None) -> Any:
        """Run POMCP search and return the best action.

        Parameters
        ----------
        particles : list of states, optional
            Initial particle set for the root belief.
            If None, samples from prior.

        Returns
        -------
        best_action : Any
            The action with highest value at the root.
        """
        # Initialize root particles
        if particles is not None:
            self.root.particles = list(particles)
        elif not self.root.particles:
            self.root.particles = [
                self.env.sample_initial_state()
                for _ in range(max(self.num_simulations, 500))
            ]

        # Run simulations
        for _ in range(self.num_simulations):
            # Sample state from root belief
            if self.root.particles:
                idx = self.env.rng.integers(0, len(self.root.particles))
                state = self.root.particles[idx]
            else:
                state = self.env.sample_initial_state()

            self._simulate(state, self.root, 0)
            self.total_simulations += 1

        # Return best action (most visited or highest value)
        return self._best_action(self.root)

    def _simulate(self, state: Any, node: BeliefNode, depth: int) -> float:
        """Recursive POMCP simulation.

        Returns the discounted cumulative reward from this point.
        """
        if depth >= self.max_depth or self.env.is_terminal(state):
            return 0.0

        # Expand leaf node
        if node.is_leaf():
            for a in self.actions:
                node.children[a] = ActionNode(a)
            node.visit_count += 1
            self.tree_depth_sum += depth
            return self._evaluate_leaf(state, depth)

        # Selection: UCB1
        action = self._select_action(node)
        action_node = node.children[action]

        # Simulate step
        result = self.env.step(state, action)
        obs = result.observation
        reward = result.reward

        # Get or create observation child
        if obs not in action_node.children:
            action_node.children[obs] = BeliefNode()

        child_belief = action_node.children[obs]
        child_belief.add_particle(result.next_state)

        # Recurse
        if result.done:
            future_value = 0.0
        else:
            future_value = self._simulate(result.next_state, child_belief, depth + 1)

        q = reward + self.discount * future_value

        # Backup: incremental mean update
        node.visit_count += 1
        action_node.visit_count += 1
        action_node.value += (q - action_node.value) / action_node.visit_count

        return q

    def _select_action(self, node: BeliefNode) -> Any:
        """Select action using UCB1.

        UCB1(a) = Q(a) + c * sqrt(ln(N) / N(a))

        Unvisited actions get priority (infinite UCB).
        """
        log_total = math.log(max(node.visit_count, 1))
        best_value = -float("inf")
        best_actions: list[Any] = []

        for action, action_node in node.children.items():
            if action_node.visit_count == 0:
                # Unvisited: highest priority
                return action

            ucb = action_node.value + self.c * math.sqrt(
                log_total / action_node.visit_count
            )

            if ucb > best_value:
                best_value = ucb
                best_actions = [action]
            elif ucb == best_value:
                best_actions.append(action)

        # Break ties randomly
        idx = self.env.rng.integers(0, len(best_actions))
        return best_actions[idx]

    def _evaluate_leaf(self, state: Any, depth: int) -> float:
        """Evaluate a leaf node using rollout or neural value estimate."""
        if self.value_estimator is not None:
            # Use neural value network
            return self.value_estimator(self.env, [state])

        # Default: random rollout
        return self._rollout(state, depth)

    def _rollout(self, state: Any, depth: int) -> float:
        """Simulate with rollout policy from current state to max depth.

        Uses either the provided rollout policy or uniform random.
        Returns discounted cumulative reward.
        """
        cumulative_reward = 0.0
        current_discount = 1.0
        current_state = state

        for d in range(depth, self.max_depth):
            if self.env.is_terminal(current_state):
                break

            if self.rollout_policy is not None:
                action = self.rollout_policy(self.env, current_state, d)
            else:
                # Uniform random
                action = self.actions[self.env.rng.integers(0, len(self.actions))]

            result = self.env.step(current_state, action)
            cumulative_reward += current_discount * result.reward
            current_discount *= self.discount
            current_state = result.next_state

            if result.done:
                break

        return cumulative_reward

    def _best_action(self, node: BeliefNode) -> Any:
        """Return the best action at a node.

        Uses visit count as the selection criterion (more robust than value).
        """
        best_count = -1
        best_actions: list[Any] = []

        for action, action_node in node.children.items():
            if action_node.visit_count > best_count:
                best_count = action_node.visit_count
                best_actions = [action]
            elif action_node.visit_count == best_count:
                best_actions.append(action)

        idx = self.env.rng.integers(0, len(best_actions))
        return best_actions[idx]

    def update(self, action: Any, observation: Any) -> None:
        """Update the tree after taking an action and receiving an observation.

        Reuses the subtree rooted at the matching action-observation node.
        If the subtree doesn't exist, creates a new root.
        """
        if action in self.root.children:
            action_node = self.root.children[action]
            if observation in action_node.children:
                self.root = action_node.children[observation]
                return

        # Subtree not found -- create fresh root
        self.root = BeliefNode()

    def get_action_values(self) -> dict[Any, tuple[float, int]]:
        """Return Q-values and visit counts for all actions at root.

        Returns
        -------
        dict mapping action -> (Q-value, visit_count)
        """
        result = {}
        for action, action_node in self.root.children.items():
            result[action] = (action_node.value, action_node.visit_count)
        return result

    def get_tree_statistics(self) -> dict[str, Any]:
        """Return statistics about the search tree."""
        total_nodes = self._count_nodes(self.root)
        max_depth = self._tree_depth(self.root)

        return {
            "total_simulations": self.total_simulations,
            "total_nodes": total_nodes,
            "max_depth": max_depth,
            "root_visits": self.root.visit_count,
            "root_particles": len(self.root.particles),
        }

    def _count_nodes(self, node: BeliefNode, depth: int = 0) -> int:
        """Count total nodes in the tree (DFS)."""
        if depth > 20:  # Prevent stack overflow
            return 1
        count = 1
        for action_node in node.children.values():
            for obs_node in action_node.children.values():
                count += 1 + self._count_nodes(obs_node, depth + 1)
        return count

    def _tree_depth(self, node: BeliefNode, depth: int = 0) -> int:
        """Compute maximum depth of the tree."""
        if depth > 20 or node.is_leaf():
            return depth
        max_d = depth
        for action_node in node.children.values():
            for obs_node in action_node.children.values():
                max_d = max(max_d, self._tree_depth(obs_node, depth + 1))
        return max_d

    def reset(self) -> None:
        """Reset the search tree."""
        self.root = BeliefNode()
        self.total_simulations = 0
        self.tree_depth_sum = 0
