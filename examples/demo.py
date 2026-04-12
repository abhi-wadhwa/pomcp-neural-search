"""Demo: POMCP with neural heuristics on the Tiger problem.

This script demonstrates:
1. Running POMCP with random rollouts on Tiger
2. Training a neural rollout policy from POMCP data
3. Running POMCP with the neural rollout policy
4. Comparing with the QMDP baseline
"""

from __future__ import annotations

import time

import numpy as np

from src.core.belief import ParticleFilter
from src.core.pomcp import POMCP
from src.core.neural_rollout import NeuralRolloutPolicy
from src.core.neural_value import NeuralValueNetwork
from src.core.qmdp import QMDP
from src.environments.tiger import TigerAction, TigerPOMDP, TigerState


def run_pomcp_episode(
    env: TigerPOMDP,
    planner: POMCP,
    max_steps: int = 20,
    verbose: bool = False,
) -> tuple[float, list[tuple]]:
    """Run a single POMCP episode.

    Returns (total_reward, history) where history is a list of
    (belief_features, action, observation, reward) tuples.
    """
    state = env.sample_initial_state()
    particles = [env.sample_initial_state() for _ in range(500)]
    planner.reset()

    total_reward = 0.0
    discount = 1.0
    history = []

    for step in range(max_steps):
        # Get belief features
        features = env.belief_features(particles)

        # Plan
        action = planner.search(particles=particles)

        # Execute
        result = env.step(state, action)
        total_reward += discount * result.reward
        discount *= env.discount

        if verbose:
            action_names = {
                TigerAction.LISTEN: "Listen",
                TigerAction.OPEN_LEFT: "Open Left",
                TigerAction.OPEN_RIGHT: "Open Right",
            }
            print(
                f"  Step {step}: {action_names[action]} -> "
                f"obs={result.observation}, r={result.reward:.0f}"
            )

        history.append((features, action, result.observation, result.reward))

        # Update particles
        new_particles = []
        for p in particles:
            r = env.step(p, action)
            if r.observation == result.observation:
                new_particles.append(r.next_state)

        if new_particles:
            while len(new_particles) < 500:
                idx = env.rng.integers(0, len(new_particles))
                new_particles.append(new_particles[idx])
            particles = new_particles
        else:
            particles = [env.sample_initial_state() for _ in range(500)]

        planner.update(action, result.observation)
        state = result.next_state

        if result.done:
            break

    return total_reward, history


def main() -> None:
    print("=" * 60)
    print("POMCP Neural Search -- Tiger Problem Demo")
    print("=" * 60)

    env = TigerPOMDP()
    env.seed(42)

    n_episodes = 50
    num_sims = 300

    # --------------------------------------------------------
    # Phase 1: POMCP with random rollouts
    # --------------------------------------------------------
    print("\n--- Phase 1: POMCP (Random Rollouts) ---")
    start = time.time()

    pomcp_rewards = []
    all_histories = []

    for ep in range(n_episodes):
        planner = POMCP(env=env, num_simulations=num_sims, max_depth=20)
        reward, history = run_pomcp_episode(env, planner, verbose=(ep == 0))
        pomcp_rewards.append(reward)
        all_histories.extend(history)

    elapsed = time.time() - start
    print(f"  Episodes: {n_episodes}, Time: {elapsed:.1f}s")
    print(f"  Mean reward: {np.mean(pomcp_rewards):.3f} +/- {np.std(pomcp_rewards):.3f}")

    # --------------------------------------------------------
    # Phase 2: Train neural rollout policy
    # --------------------------------------------------------
    print("\n--- Phase 2: Training Neural Rollout Policy ---")
    neural_rollout = NeuralRolloutPolicy(env, hidden_dim=32, learning_rate=1e-3)

    # Add training data from POMCP runs
    for features, action, obs, reward in all_histories:
        action_idx = list(TigerAction).index(action)
        neural_rollout.add_training_data(features, action_idx)

    print(f"  Training data: {neural_rollout.buffer_size} examples")
    train_history = neural_rollout.train(epochs=50, batch_size=32, verbose=True)

    if train_history["accuracy"]:
        print(f"  Final accuracy: {train_history['accuracy'][-1]:.3f}")

    # --------------------------------------------------------
    # Phase 3: POMCP with neural rollout
    # --------------------------------------------------------
    print("\n--- Phase 3: POMCP + Neural Rollout ---")
    start = time.time()

    neural_rewards = []
    for ep in range(n_episodes):
        planner = POMCP(
            env=env, num_simulations=num_sims, max_depth=20,
            rollout_policy=neural_rollout,
        )
        reward, _ = run_pomcp_episode(env, planner, verbose=(ep == 0))
        neural_rewards.append(reward)

    elapsed = time.time() - start
    print(f"  Episodes: {n_episodes}, Time: {elapsed:.1f}s")
    print(f"  Mean reward: {np.mean(neural_rewards):.3f} +/- {np.std(neural_rewards):.3f}")

    # --------------------------------------------------------
    # Phase 4: Train neural value network
    # --------------------------------------------------------
    print("\n--- Phase 4: Training Neural Value Network ---")
    value_net = NeuralValueNetwork(env, hidden_dim=64, learning_rate=1e-3)

    # Collect value training data: (belief_features, discounted_return)
    for ep in range(n_episodes):
        planner = POMCP(env=env, num_simulations=num_sims, max_depth=20)
        reward, history = run_pomcp_episode(env, planner)

        # Compute discounted returns for each step
        returns = []
        g = 0.0
        for _, _, _, r in reversed(history):
            g = r + env.discount * g
            returns.insert(0, g)

        for (features, _, _, _), ret in zip(history, returns):
            value_net.add_training_data(features, ret)

    print(f"  Training data: {value_net.buffer_size} examples")
    val_history = value_net.train(epochs=50, batch_size=32, verbose=True)

    # --------------------------------------------------------
    # Phase 5: POMCP with neural value estimation
    # --------------------------------------------------------
    print("\n--- Phase 5: POMCP + Neural Value ---")
    start = time.time()

    value_rewards = []
    for ep in range(n_episodes):
        planner = POMCP(
            env=env, num_simulations=num_sims, max_depth=20,
            value_estimator=value_net,
        )
        reward, _ = run_pomcp_episode(env, planner, verbose=(ep == 0))
        value_rewards.append(reward)

    elapsed = time.time() - start
    print(f"  Episodes: {n_episodes}, Time: {elapsed:.1f}s")
    print(f"  Mean reward: {np.mean(value_rewards):.3f} +/- {np.std(value_rewards):.3f}")

    # --------------------------------------------------------
    # Phase 6: QMDP baseline
    # --------------------------------------------------------
    print("\n--- Phase 6: QMDP Baseline ---")
    qmdp = QMDP(env)
    iterations = qmdp.solve(verbose=False)
    print(f"  Value iteration converged in {iterations} iterations")

    qmdp_rewards = []
    for ep in range(n_episodes):
        state = env.sample_initial_state()
        belief = ParticleFilter(env, num_particles=500)
        total_r = 0.0
        discount_acc = 1.0

        for step in range(20):
            # Convert particles to belief vector
            dist = belief.get_state_distribution()
            b = np.zeros(2)
            b[0] = dist.get(TigerState.TIGER_LEFT, 0.0)
            b[1] = dist.get(TigerState.TIGER_RIGHT, 0.0)
            if b.sum() > 0:
                b /= b.sum()
            else:
                b = np.array([0.5, 0.5])

            action = qmdp.select_action(b)
            result = env.step(state, action)
            total_r += discount_acc * result.reward
            discount_acc *= env.discount
            belief.update(action, result.observation)
            state = result.next_state
            if result.done:
                break

        qmdp_rewards.append(total_r)

    print(f"  Mean reward: {np.mean(qmdp_rewards):.3f} +/- {np.std(qmdp_rewards):.3f}")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    methods = {
        "POMCP (Random)": pomcp_rewards,
        "POMCP + Neural Rollout": neural_rewards,
        "POMCP + Neural Value": value_rewards,
        "QMDP": qmdp_rewards,
    }

    for name, rewards in methods.items():
        mean = np.mean(rewards)
        std = np.std(rewards)
        print(f"  {name:25s}: {mean:8.3f} +/- {std:.3f}")


if __name__ == "__main__":
    main()
