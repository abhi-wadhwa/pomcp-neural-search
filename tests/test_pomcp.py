"""Tests for the POMCP planning algorithm."""

import numpy as np
import pytest

from src.core.pomcp import POMCP, ActionNode, BeliefNode
from src.environments.tiger import TigerAction, TigerPOMDP, TigerState


class TestPOMCPNodes:
    """Test POMCP tree node structures."""

    def test_belief_node_creation(self) -> None:
        node = BeliefNode()
        assert node.visit_count == 0
        assert len(node.particles) == 0
        assert node.is_leaf()

    def test_belief_node_add_particle(self) -> None:
        node = BeliefNode()
        node.add_particle(TigerState.TIGER_LEFT)
        assert len(node.particles) == 1

    def test_action_node_creation(self) -> None:
        node = ActionNode(TigerAction.LISTEN)
        assert node.action == TigerAction.LISTEN
        assert node.visit_count == 0
        assert node.value == 0.0
        assert len(node.children) == 0

    def test_belief_node_not_leaf_after_expansion(self) -> None:
        node = BeliefNode()
        node.children[TigerAction.LISTEN] = ActionNode(TigerAction.LISTEN)
        assert not node.is_leaf()


class TestPOMCP:
    """Test POMCP planning algorithm."""

    def setup_method(self) -> None:
        self.env = TigerPOMDP()
        self.env.seed(42)

    def test_search_returns_valid_action(self) -> None:
        planner = POMCP(self.env, num_simulations=100, max_depth=10)
        action = planner.search()
        assert action in self.env.get_actions()

    def test_search_with_particles(self) -> None:
        planner = POMCP(self.env, num_simulations=100, max_depth=10)
        particles = [TigerState.TIGER_LEFT] * 50 + [TigerState.TIGER_RIGHT] * 50
        action = planner.search(particles=particles)
        assert action in self.env.get_actions()

    def test_listen_when_uncertain(self) -> None:
        """With uniform belief, POMCP should prefer listening."""
        planner = POMCP(self.env, num_simulations=1000, max_depth=20)
        particles = [TigerState.TIGER_LEFT] * 500 + [TigerState.TIGER_RIGHT] * 500

        # Run multiple trials to check tendency
        listen_count = 0
        for _ in range(20):
            planner.reset()
            action = planner.search(particles=particles)
            if action == TigerAction.LISTEN:
                listen_count += 1

        # POMCP should listen at least sometimes with uncertain belief
        assert listen_count >= 5, f"Expected mostly LISTEN, got {listen_count}/20"

    def test_open_when_confident(self) -> None:
        """When very confident tiger is left, should open right."""
        planner = POMCP(self.env, num_simulations=500, max_depth=20)
        # 95% confident tiger is left
        particles = [TigerState.TIGER_LEFT] * 950 + [TigerState.TIGER_RIGHT] * 50

        open_right_count = 0
        for _ in range(10):
            planner.reset()
            action = planner.search(particles=particles)
            if action == TigerAction.OPEN_RIGHT:
                open_right_count += 1

        assert open_right_count >= 5, f"Expected mostly OPEN_RIGHT, got {open_right_count}/10"

    def test_action_values_populated(self) -> None:
        planner = POMCP(self.env, num_simulations=100, max_depth=10)
        planner.search()
        values = planner.get_action_values()

        assert len(values) == 3  # 3 actions in Tiger
        for action, (q, n) in values.items():
            assert n > 0, f"Action {action} should have visits"

    def test_tree_update(self) -> None:
        planner = POMCP(self.env, num_simulations=100, max_depth=10)
        planner.search()

        # Update tree after action and observation
        planner.update(TigerAction.LISTEN, 0)

        # Root should change or be fresh
        assert planner.root is not None

    def test_tree_reset(self) -> None:
        planner = POMCP(self.env, num_simulations=100, max_depth=10)
        planner.search()
        planner.reset()
        assert planner.root.is_leaf()
        assert planner.total_simulations == 0

    def test_tree_statistics(self) -> None:
        planner = POMCP(self.env, num_simulations=200, max_depth=10)
        planner.search()
        stats = planner.get_tree_statistics()

        assert stats["total_simulations"] == 200
        assert stats["total_nodes"] > 0
        assert stats["root_visits"] > 0

    def test_pomcp_achieves_reasonable_reward_on_tiger(self) -> None:
        """POMCP should outperform purely random action selection on Tiger.

        A random policy that immediately opens a door gets:
        E[reward] = 0.5*10 + 0.5*(-100) = -45

        POMCP, by listening before acting, should do significantly better.
        """
        n_episodes = 50
        total_rewards = []

        for _ in range(n_episodes):
            planner = POMCP(
                self.env, num_simulations=500, max_depth=20, exploration_constant=3.0
            )
            state = self.env.sample_initial_state()
            particles = [self.env.sample_initial_state() for _ in range(500)]
            episode_reward = 0.0
            discount_acc = 1.0

            for step in range(20):
                action = planner.search(particles=particles)
                result = self.env.step(state, action)
                episode_reward += discount_acc * result.reward
                discount_acc *= self.env.discount

                # Particle update via rejection sampling
                new_particles = []
                for p in particles:
                    r = self.env.step(p, action)
                    if r.observation == result.observation:
                        new_particles.append(r.next_state)
                if new_particles:
                    while len(new_particles) < 500:
                        idx = self.env.rng.integers(0, len(new_particles))
                        new_particles.append(new_particles[idx])
                    particles = new_particles
                else:
                    particles = [self.env.sample_initial_state() for _ in range(500)]

                planner.update(action, result.observation)
                state = result.next_state

                if result.done:
                    break

            total_rewards.append(episode_reward)

        mean_reward = np.mean(total_rewards)
        # POMCP should do significantly better than random (-45)
        assert mean_reward > -40, f"POMCP mean reward {mean_reward:.2f} too low (random = -45)"

    def test_custom_rollout_policy(self) -> None:
        """Test POMCP with a custom rollout policy."""

        def always_listen(env, state, depth):
            return TigerAction.LISTEN

        planner = POMCP(
            self.env,
            num_simulations=100,
            max_depth=10,
            rollout_policy=always_listen,
        )
        action = planner.search()
        assert action in self.env.get_actions()

    def test_custom_value_estimator(self) -> None:
        """Test POMCP with a custom value estimator."""

        def constant_value(env, particles):
            return 0.0

        planner = POMCP(
            self.env,
            num_simulations=100,
            max_depth=10,
            value_estimator=constant_value,
        )
        action = planner.search()
        assert action in self.env.get_actions()
