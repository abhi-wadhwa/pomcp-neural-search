"""Streamlit visualization for POMCP Neural Search.

Features:
- Belief particle visualizer on grid world
- MCTS tree inspector with visit counts
- Performance comparison: POMCP vs POMCP+Neural vs QMDP
- Interactive play mode for Tiger problem
"""

from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.core.belief import ParticleFilter
from src.core.pomcp import POMCP
from src.core.neural_rollout import NeuralRolloutPolicy
from src.core.neural_value import NeuralValueNetwork
from src.core.qmdp import QMDP
from src.environments.tiger import TigerPOMDP, TigerAction, TigerObservation, TigerState
from src.environments.rocksample import RockSamplePOMDP, RSObservation


def main() -> None:
    st.set_page_config(
        page_title="POMCP Neural Search",
        page_icon="🔍",
        layout="wide",
    )

    st.title("POMCP Neural Search")
    st.markdown(
        "Interactive visualization of **Partially Observable Monte-Carlo Planning** "
        "with neural heuristics."
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Tiger Play Mode",
        "Belief Visualizer",
        "MCTS Tree Inspector",
        "Performance Comparison",
    ])

    with tab1:
        tiger_play_mode()

    with tab2:
        belief_visualizer()

    with tab3:
        mcts_tree_inspector()

    with tab4:
        performance_comparison()


def tiger_play_mode() -> None:
    """Interactive Tiger problem -- human vs POMCP."""
    st.header("Tiger Problem -- Play Mode")
    st.markdown("""
    A tiger is behind one of two doors. Listen to gather noisy information,
    then choose a door. Opening the tiger's door gives **-100**, the safe door gives **+10**.
    Listening costs **-1** but gives a clue (85% accurate).
    """)

    # Initialize session state
    if "tiger_env" not in st.session_state:
        st.session_state.tiger_env = TigerPOMDP()
        st.session_state.tiger_env.seed(42)
        st.session_state.tiger_state = st.session_state.tiger_env.sample_initial_state()
        st.session_state.tiger_belief = np.array([0.5, 0.5])
        st.session_state.tiger_history = []
        st.session_state.tiger_total_reward = 0.0
        st.session_state.tiger_done = False

    env = st.session_state.tiger_env

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Your Actions")

        # Display belief
        fig = go.Figure(data=[
            go.Bar(
                x=["Tiger Left", "Tiger Right"],
                y=st.session_state.tiger_belief,
                marker_color=["#e74c3c", "#3498db"],
            )
        ])
        fig.update_layout(
            title="Current Belief",
            yaxis_title="Probability",
            yaxis_range=[0, 1],
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

        if not st.session_state.tiger_done:
            action_cols = st.columns(3)
            with action_cols[0]:
                listen = st.button("Listen (-1)")
            with action_cols[1]:
                open_left = st.button("Open Left")
            with action_cols[2]:
                open_right = st.button("Open Right")

            action = None
            if listen:
                action = TigerAction.LISTEN
            elif open_left:
                action = TigerAction.OPEN_LEFT
            elif open_right:
                action = TigerAction.OPEN_RIGHT

            if action is not None:
                result = env.step(st.session_state.tiger_state, action)
                st.session_state.tiger_total_reward += result.reward

                # Update belief
                if action == TigerAction.LISTEN:
                    obs = result.observation
                    b = st.session_state.tiger_belief.copy()
                    for s_idx in range(2):
                        s = TigerState(s_idx)
                        b[s_idx] *= env.get_observation_probability(s, action, obs)
                    if b.sum() > 0:
                        b /= b.sum()
                    st.session_state.tiger_belief = b

                    obs_str = "Heard LEFT" if obs == TigerObservation.HEAR_LEFT else "Heard RIGHT"
                    st.session_state.tiger_history.append(
                        f"Listen -> {obs_str} (reward: {result.reward})"
                    )
                else:
                    action_str = "Opened LEFT" if action == TigerAction.OPEN_LEFT else "Opened RIGHT"
                    tiger_str = "LEFT" if st.session_state.tiger_state == TigerState.TIGER_LEFT else "RIGHT"
                    st.session_state.tiger_history.append(
                        f"{action_str} -> Tiger was {tiger_str} (reward: {result.reward})"
                    )
                    st.session_state.tiger_done = True

                st.session_state.tiger_state = result.next_state
                st.rerun()
        else:
            st.success(f"Episode complete! Total reward: {st.session_state.tiger_total_reward:.0f}")
            if st.button("New Game"):
                st.session_state.tiger_state = env.sample_initial_state()
                st.session_state.tiger_belief = np.array([0.5, 0.5])
                st.session_state.tiger_history = []
                st.session_state.tiger_total_reward = 0.0
                st.session_state.tiger_done = False
                st.rerun()

    with col2:
        st.subheader("History")
        for i, h in enumerate(st.session_state.tiger_history):
            st.text(f"{i+1}. {h}")

        # POMCP recommendation
        st.subheader("POMCP Recommendation")
        if not st.session_state.tiger_done:
            with st.spinner("Planning..."):
                planner = POMCP(env=env, num_simulations=200, max_depth=20)
                # Create particles matching current belief
                n_particles = 200
                particles = []
                for _ in range(n_particles):
                    if env.rng.random() < st.session_state.tiger_belief[0]:
                        particles.append(TigerState.TIGER_LEFT)
                    else:
                        particles.append(TigerState.TIGER_RIGHT)

                best_action = planner.search(particles=particles)
                action_values = planner.get_action_values()

                action_names = {
                    TigerAction.LISTEN: "Listen",
                    TigerAction.OPEN_LEFT: "Open Left",
                    TigerAction.OPEN_RIGHT: "Open Right",
                }

                st.write(f"Recommended: **{action_names[best_action]}**")
                for a, (q, n) in sorted(action_values.items()):
                    st.write(f"  {action_names[a]}: Q={q:.2f}, visits={n}")


def belief_visualizer() -> None:
    """Visualize belief particles on a grid world (RockSample)."""
    st.header("Belief Particle Visualizer")
    st.markdown("Visualize the particle filter belief state on the RockSample grid.")

    num_particles = st.slider("Number of particles", 50, 2000, 500)
    num_sims = st.slider("POMCP simulations", 100, 2000, 500, key="bv_sims")

    if st.button("Run Episode Step", key="bv_run"):
        env = RockSamplePOMDP(grid_size=4)
        env.seed(int(time.time()) % 10000)

        belief = ParticleFilter(env, num_particles=num_particles)
        planner = POMCP(env=env, num_simulations=num_sims, max_depth=20)
        state = env.sample_initial_state()

        # Run a few steps
        steps_data = []
        for step in range(5):
            action = planner.search(particles=belief.particles)
            result = env.step(state, action)
            belief.update(action, result.observation)
            planner.update(action, result.observation)

            # Record particle positions
            x_positions = [p.x for p in belief.particles]
            y_positions = [p.y for p in belief.particles]
            rock_beliefs = []
            for i in range(env.num_rocks):
                p_good = sum(1 for p in belief.particles if p.rocks[i]) / len(belief.particles)
                rock_beliefs.append(p_good)

            steps_data.append({
                "step": step,
                "action": action,
                "obs": result.observation,
                "reward": result.reward,
                "x": x_positions,
                "y": y_positions,
                "rock_beliefs": rock_beliefs,
                "state": state,
            })

            state = result.next_state
            if result.done:
                break

        # Visualize the last step
        if steps_data:
            data = steps_data[-1]

            fig = go.Figure()

            # Plot particles as scatter
            fig.add_trace(go.Histogram2d(
                x=data["x"],
                y=data["y"],
                nbinsx=env.grid_size,
                nbinsy=env.grid_size,
                colorscale="Blues",
                name="Particle density",
            ))

            # Plot rock positions
            for i, (rx, ry) in enumerate(env.rock_positions):
                color = "green" if data["rock_beliefs"][i] > 0.5 else "red"
                fig.add_trace(go.Scatter(
                    x=[rx], y=[ry],
                    mode="markers+text",
                    marker=dict(size=20, color=color, symbol="diamond"),
                    text=[f"R{i}: {data['rock_beliefs'][i]:.0%}"],
                    textposition="top center",
                    name=f"Rock {i}",
                ))

            # Plot true agent position
            fig.add_trace(go.Scatter(
                x=[data["state"].x],
                y=[data["state"].y],
                mode="markers",
                marker=dict(size=15, color="yellow", symbol="star", line=dict(width=2, color="black")),
                name="True position",
            ))

            fig.update_layout(
                title=f"Belief after step {data['step']+1}",
                xaxis_title="X",
                yaxis_title="Y",
                xaxis=dict(range=[-0.5, env.grid_size - 0.5]),
                yaxis=dict(range=[-0.5, env.grid_size - 0.5]),
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Show step history
            st.subheader("Step History")
            for d in steps_data:
                action_names = {0: "North", 1: "South", 2: "East", 3: "West", 4: "Sample"}
                a_name = action_names.get(d["action"], f"Check({d['action'] - 5})")
                obs_names = {RSObservation.NONE: "None", RSObservation.GOOD: "Good", RSObservation.BAD: "Bad"}
                o_name = obs_names.get(d["obs"], str(d["obs"]))
                st.text(f"Step {d['step']+1}: action={a_name}, obs={o_name}, reward={d['reward']:.1f}")


def mcts_tree_inspector() -> None:
    """Inspect the POMCP search tree."""
    st.header("MCTS Tree Inspector")
    st.markdown("Explore the POMCP search tree structure, visit counts, and Q-values.")

    env_name = st.selectbox("Environment", ["Tiger", "RockSample[4,4]"], key="tree_env")
    num_sims = st.slider("Simulations", 100, 5000, 1000, key="tree_sims")

    if st.button("Build Tree", key="tree_build"):
        if env_name == "Tiger":
            env = TigerPOMDP()
        else:
            env = RockSamplePOMDP()

        env.seed(42)
        planner = POMCP(env=env, num_simulations=num_sims, max_depth=20)
        particles = [env.sample_initial_state() for _ in range(500)]
        best_action = planner.search(particles=particles)

        # Display tree statistics
        stats = planner.get_tree_statistics()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Nodes", stats["total_nodes"])
        col2.metric("Max Depth", stats["max_depth"])
        col3.metric("Root Visits", stats["root_visits"])

        # Display root action values
        action_values = planner.get_action_values()

        if env_name == "Tiger":
            action_names = {
                TigerAction.LISTEN: "Listen",
                TigerAction.OPEN_LEFT: "Open Left",
                TigerAction.OPEN_RIGHT: "Open Right",
            }
        else:
            action_names_rs = {0: "North", 1: "South", 2: "East", 3: "West", 4: "Sample"}
            action_names = {}
            for a in action_values:
                action_names[a] = action_names_rs.get(a, f"Check({a - 5})")

        names = []
        q_vals = []
        visits = []
        for a, (q, n) in sorted(action_values.items()):
            names.append(str(action_names.get(a, a)))
            q_vals.append(q)
            visits.append(n)

        # Q-values bar chart
        fig_q = go.Figure(data=[
            go.Bar(x=names, y=q_vals, marker_color="#2ecc71")
        ])
        fig_q.update_layout(title="Q-Values at Root", yaxis_title="Q-value", height=350)
        st.plotly_chart(fig_q, use_container_width=True)

        # Visit counts bar chart
        fig_v = go.Figure(data=[
            go.Bar(x=names, y=visits, marker_color="#3498db")
        ])
        fig_v.update_layout(title="Visit Counts at Root", yaxis_title="Visits", height=350)
        st.plotly_chart(fig_v, use_container_width=True)

        # Tree structure (first two levels)
        st.subheader("Tree Structure (depth 2)")
        tree_data = []
        for a, action_node in planner.root.children.items():
            a_name = str(action_names.get(a, a))
            for obs, obs_node in action_node.children.items():
                tree_data.append({
                    "Action": a_name,
                    "Observation": str(obs),
                    "Q-value": f"{action_node.value:.3f}",
                    "Action Visits": action_node.visit_count,
                    "Belief Particles": len(obs_node.particles),
                    "Belief Visits": obs_node.visit_count,
                })

        if tree_data:
            st.dataframe(tree_data)


def performance_comparison() -> None:
    """Compare POMCP, POMCP+Neural, and QMDP performance."""
    st.header("Performance Comparison")

    env_name = st.selectbox("Environment", ["Tiger"], key="perf_env")
    episodes = st.slider("Episodes", 10, 200, 50, key="perf_eps")
    num_sims = st.slider("Simulations per step", 50, 1000, 200, key="perf_sims")

    if st.button("Run Comparison", key="perf_run"):
        env = TigerPOMDP()
        env.seed(42)

        progress = st.progress(0)
        status = st.empty()

        results = {}
        total_methods = 3

        # 1. POMCP
        status.text("Running POMCP (random rollouts)...")
        pomcp_rewards = []
        for ep in range(episodes):
            planner = POMCP(env=env, num_simulations=num_sims, max_depth=20)
            belief = ParticleFilter(env, num_particles=300)
            state = env.sample_initial_state()
            total_r = 0.0
            discount_acc = 1.0

            for step in range(30):
                action = planner.search(particles=belief.particles)
                result = env.step(state, action)
                total_r += discount_acc * result.reward
                discount_acc *= env.discount
                belief.update(action, result.observation)
                planner.update(action, result.observation)
                state = result.next_state
                if result.done:
                    break

            pomcp_rewards.append(total_r)
            progress.progress((ep + 1) / (episodes * total_methods))

        results["POMCP"] = pomcp_rewards

        # 2. Train neural policy and run POMCP + Neural
        status.text("Training neural rollout policy...")
        neural_rollout = NeuralRolloutPolicy(env, hidden_dim=32)

        # Collect training data
        for ep in range(min(episodes, 30)):
            planner = POMCP(env=env, num_simulations=num_sims, max_depth=20)
            belief = ParticleFilter(env, num_particles=300)
            state = env.sample_initial_state()

            for step in range(20):
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

        neural_rollout.train(epochs=30)

        status.text("Running POMCP + Neural Rollout...")
        neural_rewards = []
        for ep in range(episodes):
            planner = POMCP(
                env=env, num_simulations=num_sims, max_depth=20,
                rollout_policy=neural_rollout,
            )
            belief = ParticleFilter(env, num_particles=300)
            state = env.sample_initial_state()
            total_r = 0.0
            discount_acc = 1.0

            for step in range(30):
                action = planner.search(particles=belief.particles)
                result = env.step(state, action)
                total_r += discount_acc * result.reward
                discount_acc *= env.discount
                belief.update(action, result.observation)
                planner.update(action, result.observation)
                state = result.next_state
                if result.done:
                    break

            neural_rewards.append(total_r)
            progress.progress((episodes + ep + 1) / (episodes * total_methods))

        results["POMCP+Neural"] = neural_rewards

        # 3. QMDP
        status.text("Running QMDP...")
        qmdp = QMDP(env)
        qmdp.solve()
        qmdp_rewards = []

        for ep in range(episodes):
            state = env.sample_initial_state()
            belief_filter = ParticleFilter(env, num_particles=300)
            total_r = 0.0
            discount_acc = 1.0

            for step in range(30):
                dist = belief_filter.get_state_distribution()
                b = np.zeros(len(env.get_states()))
                state_to_idx = {s: i for i, s in enumerate(env.get_states())}
                for s, p in dist.items():
                    if s in state_to_idx:
                        b[state_to_idx[s]] = p
                if b.sum() > 0:
                    b /= b.sum()
                else:
                    b = np.ones(len(env.get_states())) / len(env.get_states())

                action = qmdp.select_action(b)
                result = env.step(state, action)
                total_r += discount_acc * result.reward
                discount_acc *= env.discount
                belief_filter.update(action, result.observation)
                state = result.next_state
                if result.done:
                    break

            qmdp_rewards.append(total_r)
            progress.progress((2 * episodes + ep + 1) / (episodes * total_methods))

        results["QMDP"] = qmdp_rewards

        progress.progress(1.0)
        status.text("Complete!")

        # Plot results
        fig = go.Figure()
        colors = {"POMCP": "#e74c3c", "POMCP+Neural": "#2ecc71", "QMDP": "#3498db"}

        for method, rewards in results.items():
            # Compute running average
            cumulative = np.cumsum(rewards)
            running_avg = cumulative / (np.arange(len(rewards)) + 1)

            fig.add_trace(go.Scatter(
                x=list(range(1, len(rewards) + 1)),
                y=running_avg,
                mode="lines",
                name=method,
                line=dict(color=colors.get(method, "#999")),
            ))

        fig.update_layout(
            title="Cumulative Average Reward",
            xaxis_title="Episode",
            yaxis_title="Average Reward",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary statistics
        st.subheader("Summary")
        summary_data = []
        for method, rewards in results.items():
            summary_data.append({
                "Method": method,
                "Mean Reward": f"{np.mean(rewards):.3f}",
                "Std Dev": f"{np.std(rewards):.3f}",
                "Min": f"{np.min(rewards):.3f}",
                "Max": f"{np.max(rewards):.3f}",
            })
        st.dataframe(summary_data)


if __name__ == "__main__":
    main()
