"""Command-line interface for POMCP Neural Search.

Provides commands to:
- Run POMCP on different environments
- Train neural heuristics
- Compare planners (POMCP, POMCP+Neural, QMDP)
- Launch the Streamlit visualization
"""

from __future__ import annotations

import time
from typing import Any

import click
import numpy as np

from src.core.belief import ParticleFilter
from src.core.pomcp import POMCP
from src.core.neural_rollout import NeuralRolloutPolicy
from src.core.neural_value import NeuralValueNetwork
from src.core.qmdp import QMDP
from src.environments.tiger import TigerPOMDP, TigerAction, TigerState
from src.environments.rocksample import RockSamplePOMDP
from src.environments.battleship import BattleshipPOMDP


def get_env(name: str) -> Any:
    """Create environment by name."""
    if name == "tiger":
        return TigerPOMDP()
    elif name == "rocksample":
        return RockSamplePOMDP()
    elif name == "battleship":
        return BattleshipPOMDP()
    else:
        raise ValueError(f"Unknown environment: {name}")


def run_episode(
    env: Any,
    planner: POMCP,
    belief: ParticleFilter,
    max_steps: int = 50,
    verbose: bool = False,
) -> tuple[float, int]:
    """Run a single episode using POMCP planning.

    Returns (total_reward, num_steps).
    """
    state = env.sample_initial_state()
    belief.reset()
    planner.reset()

    total_reward = 0.0
    discount = 1.0

    for step in range(max_steps):
        # Plan
        action = planner.search(particles=belief.particles)

        if verbose:
            action_values = planner.get_action_values()
            print(f"  Step {step}: state={state}")
            for a, (q, n) in sorted(action_values.items()):
                print(f"    action={a}: Q={q:.3f}, visits={n}")
            print(f"  -> Selected action: {action}")

        # Execute
        result = env.step(state, action)

        if verbose:
            print(f"  -> Observation: {result.observation}, Reward: {result.reward}")

        total_reward += discount * result.reward
        discount *= env.discount

        # Update belief
        belief.update(action, result.observation)

        # Update POMCP tree
        planner.update(action, result.observation)

        state = result.next_state

        if result.done:
            break

    return total_reward, step + 1


@click.group()
def main() -> None:
    """POMCP Neural Search -- POMDP planning with neural heuristics."""
    pass


@main.command()
@click.option("--env", "env_name", default="tiger", help="Environment name")
@click.option("--episodes", default=100, help="Number of episodes")
@click.option("--simulations", default=500, help="POMCP simulations per step")
@click.option("--max-depth", default=30, help="Max search depth")
@click.option("--particles", default=500, help="Number of particles")
@click.option("--seed", default=42, help="Random seed")
@click.option("--verbose", is_flag=True, help="Print detailed output")
def run(
    env_name: str,
    episodes: int,
    simulations: int,
    max_depth: int,
    particles: int,
    seed: int,
    verbose: bool,
) -> None:
    """Run POMCP on a POMDP environment."""
    env = get_env(env_name)
    env.seed(seed)

    planner = POMCP(
        env=env,
        num_simulations=simulations,
        max_depth=max_depth,
    )

    belief = ParticleFilter(env, num_particles=particles)

    rewards = []
    steps_list = []

    click.echo(f"Running POMCP on {env.name} for {episodes} episodes...")
    start_time = time.time()

    for ep in range(episodes):
        reward, steps = run_episode(env, planner, belief, verbose=verbose)
        rewards.append(reward)
        steps_list.append(steps)

        if (ep + 1) % 10 == 0:
            avg = np.mean(rewards[-10:])
            click.echo(f"  Episode {ep+1}/{episodes}: avg_reward={avg:.2f}")

    elapsed = time.time() - start_time
    click.echo(f"\nResults ({elapsed:.1f}s):")
    click.echo(f"  Mean reward: {np.mean(rewards):.3f} +/- {np.std(rewards):.3f}")
    click.echo(f"  Mean steps:  {np.mean(steps_list):.1f}")


@main.command()
@click.option("--env", "env_name", default="tiger", help="Environment name")
@click.option("--episodes", default=50, help="Episodes per method")
@click.option("--simulations", default=500, help="POMCP simulations per step")
@click.option("--seed", default=42, help="Random seed")
def compare(env_name: str, episodes: int, simulations: int, seed: int) -> None:
    """Compare POMCP, POMCP+Neural, and QMDP."""
    env = get_env(env_name)
    env.seed(seed)

    results: dict[str, list[float]] = {}

    # 1. POMCP (random rollouts)
    click.echo("Running POMCP (random rollouts)...")
    planner = POMCP(env=env, num_simulations=simulations)
    belief = ParticleFilter(env)
    pomcp_rewards = []
    for _ in range(episodes):
        r, _ = run_episode(env, planner, belief)
        pomcp_rewards.append(r)
    results["POMCP"] = pomcp_rewards

    # 2. Collect data and train neural rollout
    click.echo("Training neural rollout policy...")
    neural_rollout = NeuralRolloutPolicy(env)

    # Collect training data from POMCP runs
    for _ in range(episodes):
        state = env.sample_initial_state()
        belief.reset()
        planner.reset()

        for step in range(30):
            features = belief.get_features()
            action = planner.search(particles=belief.particles)
            action_idx = env.get_actions().index(action)
            neural_rollout.add_training_data(features, action_idx)

            result = env.step(state, action)
            belief.update(action, result.observation)
            planner.update(action, result.observation)
            state = result.next_state
            if result.done:
                break

    neural_rollout.train(epochs=50, verbose=False)

    # 3. POMCP + Neural rollout
    click.echo("Running POMCP + Neural Rollout...")
    planner_neural = POMCP(
        env=env, num_simulations=simulations, rollout_policy=neural_rollout
    )
    neural_rewards = []
    for _ in range(episodes):
        r, _ = run_episode(env, planner_neural, belief)
        neural_rewards.append(r)
    results["POMCP+Neural"] = neural_rewards

    # 4. QMDP baseline (if environment supports it)
    try:
        click.echo("Running QMDP...")
        qmdp = QMDP(env)
        qmdp.solve()

        qmdp_rewards = []
        for _ in range(episodes):
            state = env.sample_initial_state()
            belief_filter = ParticleFilter(env)
            total_reward = 0.0
            discount_factor = 1.0

            for step in range(50):
                belief_vec = env.belief_features(belief_filter.particles)
                # For QMDP, use proper belief vector
                if hasattr(env, "get_states"):
                    states = env.get_states()
                    b = np.zeros(len(states))
                    dist = belief_filter.get_state_distribution()
                    state_to_idx = {s: i for i, s in enumerate(states)}
                    for s, p in dist.items():
                        if s in state_to_idx:
                            b[state_to_idx[s]] = p
                    if b.sum() > 0:
                        b /= b.sum()
                    else:
                        b = np.ones(len(states)) / len(states)
                    action = qmdp.select_action(b)
                else:
                    action = qmdp.select_action_from_particles(belief_filter.particles)

                result = env.step(state, action)
                total_reward += discount_factor * result.reward
                discount_factor *= env.discount
                belief_filter.update(action, result.observation)
                state = result.next_state
                if result.done:
                    break

            qmdp_rewards.append(total_reward)
        results["QMDP"] = qmdp_rewards
    except (NotImplementedError, Exception) as e:
        click.echo(f"  QMDP not available: {e}")

    # Print comparison
    click.echo("\n--- Comparison ---")
    for method, rews in results.items():
        mean = np.mean(rews)
        std = np.std(rews)
        click.echo(f"  {method:20s}: {mean:8.3f} +/- {std:.3f}")


@main.command()
def ui() -> None:
    """Launch the Streamlit visualization."""
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/viz/app.py"])


if __name__ == "__main__":
    main()
