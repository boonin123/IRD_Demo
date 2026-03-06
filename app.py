"""
IRL Explorer — Interactive Inverse Reward Design Visualizer
Tab 1: Grid World RL Baseline
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

from gridworld import GridWorld
from agents import QLearningAgent, ValueIterationAgent
from irl import (
    collect_demos, maxent_irl, ird_reward, ird_reward_multi,
    plan_with_reward, run_policy_episode, run_batch_episodes,
)


# ======================================================================
# Page setup
# ======================================================================

st.set_page_config(
    page_title="IRL Explorer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stMetric > div { background: #1e1e2e; border-radius: 8px; padding: 8px 12px; }
    .block-container { padding-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ======================================================================
# Visualisation helpers
# ======================================================================

ARROW_DX = {0: 0.0, 1: 0.0, 2: -0.32, 3: 0.32}   # col offset per action
ARROW_DY = {0: -0.32, 1: 0.32, 2: 0.0, 3: 0.0}    # row offset per action


def _make_grid_figure(
    env: GridWorld,
    value_fn: np.ndarray,
    policy: np.ndarray,
    title: str = "",
    trajectory=None,
) -> plt.Figure:
    size = env.size
    fig, ax = plt.subplots(figsize=(max(5, size), max(5, size)))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    # Value heatmap — mask obstacles
    display_V = value_fn.astype(float).copy()
    mask = np.zeros((size, size), dtype=bool)
    for r, c in env.obstacles:
        mask[r, c] = True
        display_V[r, c] = np.nan

    vmin = float(np.nanmin(display_V)) if not np.all(np.isnan(display_V)) else 0
    vmax = float(np.nanmax(display_V)) if not np.all(np.isnan(display_V)) else 1

    cmap = plt.cm.RdYlGn
    cmap.set_bad(color="#444444")
    im = ax.imshow(
        display_V, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest"
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("State Value  V(s)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")

    # Grid lines
    for i in range(size + 1):
        ax.axhline(i - 0.5, color="#555555", linewidth=0.6, zorder=2)
        ax.axvline(i - 0.5, color="#555555", linewidth=0.6, zorder=2)

    # Cell annotations: obstacles, lava, goal, start, policy arrows
    for r in range(size):
        for c in range(size):
            if (r, c) in env.obstacles:
                ax.add_patch(
                    plt.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        color="#444444", zorder=3,
                    )
                )
                ax.text(
                    c, r, "■", ha="center", va="center",
                    fontsize=16, color="#222222", zorder=4,
                )
                continue

            if (r, c) in getattr(env, "lava_cells", set()):
                ax.add_patch(
                    plt.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        color="#7a1a00", zorder=3, alpha=0.85,
                    )
                )
                ax.text(
                    c, r, "LAVA", ha="center", va="center",
                    fontsize=max(4, 8 - size // 3), color="#ff6b35",
                    fontweight="bold", zorder=4,
                )
                a = int(policy[r, c])
                dx, dy = ARROW_DX[a], ARROW_DY[a]
                ax.annotate(
                    "",
                    xy=(c + dx, r + dy),
                    xytext=(c, r),
                    arrowprops=dict(
                        arrowstyle="-|>", color="#ff6b35",
                        lw=1.2, mutation_scale=9,
                    ),
                    zorder=5,
                )
                continue

            if (r, c) in getattr(env, "mud_cells", set()):
                ax.add_patch(
                    plt.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        color="#3d1f00", zorder=3, alpha=0.9,
                    )
                )
                ax.text(
                    c, r, "MUD", ha="center", va="center",
                    fontsize=max(4, 8 - size // 3), color="#c07830",
                    fontweight="bold", zorder=4,
                )
                a = int(policy[r, c])
                dx, dy = ARROW_DX[a], ARROW_DY[a]
                ax.annotate(
                    "",
                    xy=(c + dx, r + dy),
                    xytext=(c, r),
                    arrowprops=dict(
                        arrowstyle="-|>", color="#c07830",
                        lw=1.2, mutation_scale=9,
                    ),
                    zorder=5,
                )
                continue

            if (r, c) in getattr(env, "ice_cells", set()):
                ax.add_patch(
                    plt.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        color="#0a3d5c", zorder=3, alpha=0.85,
                    )
                )
                ax.text(
                    c, r, "ICE", ha="center", va="center",
                    fontsize=max(4, 8 - size // 3), color="#7ed4f0",
                    fontweight="bold", zorder=4,
                )
                a = int(policy[r, c])
                dx, dy = ARROW_DX[a], ARROW_DY[a]
                ax.annotate(
                    "",
                    xy=(c + dx, r + dy),
                    xytext=(c, r),
                    arrowprops=dict(
                        arrowstyle="-|>", color="#7ed4f0",
                        lw=1.2, mutation_scale=9,
                    ),
                    zorder=5,
                )
                continue

            if (r, c) == env.goal:
                ax.text(
                    c, r, "G", ha="center", va="center",
                    fontsize=13, fontweight="bold", color="white",
                    bbox=dict(
                        boxstyle="round,pad=0.25", facecolor="#27ae60",
                        edgecolor="white", linewidth=1.5,
                    ),
                    zorder=5,
                )
                continue

            if (r, c) == env.start:
                ax.text(
                    c + 0.35, r - 0.35, "S",
                    ha="center", va="center",
                    fontsize=7, color="white", alpha=0.6, zorder=4,
                )

            # Policy arrow
            a = int(policy[r, c])
            dx, dy = ARROW_DX[a], ARROW_DY[a]
            ax.annotate(
                "",
                xy=(c + dx, r + dy),
                xytext=(c, r),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="white",
                    lw=1.4,
                    mutation_scale=10,
                ),
                zorder=5,
            )

    # Trajectory overlay
    if trajectory and len(trajectory) > 1:
        traj_r = [pos[0] for pos in trajectory]
        traj_c = [pos[1] for pos in trajectory]
        ax.plot(traj_c, traj_r, "-o", color="#3498db", linewidth=2,
                markersize=4, alpha=0.75, zorder=6)
        # Mark start of trajectory
        ax.plot(traj_c[0], traj_r[0], "o", color="#2ecc71",
                markersize=8, zorder=7, label="Start")
        # Mark end
        ax.plot(traj_c[-1], traj_r[-1], "o", color="#e74c3c",
                markersize=8, zorder=7, label="End")

    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)   # row 0 at top
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")

    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
    fig.tight_layout()
    return fig


def _training_curve(rewards: list, window: int = 50) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=rewards,
        mode="lines",
        name="Episode reward",
        line=dict(color="rgba(52,152,219,0.4)", width=1),
    ))
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        fig.add_trace(go.Scatter(
            x=list(range(window - 1, len(rewards))),
            y=smoothed.tolist(),
            mode="lines",
            name=f"{window}-ep avg",
            line=dict(color="#3498db", width=2.5),
        ))
    fig.update_layout(
        title="Training Progress",
        xaxis_title="Episode",
        yaxis_title="Total Reward",
        height=270,
        margin=dict(l=0, r=0, t=36, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font_size=11),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="#333"),
        yaxis=dict(gridcolor="#333"),
    )
    return fig


def _convergence_curve(deltas: list) -> go.Figure:
    fig = go.Figure(go.Scatter(
        y=deltas,
        mode="lines",
        line=dict(color="#2ecc71", width=2.5),
        name="max |ΔV|",
    ))
    fig.update_layout(
        title="Value Iteration — Convergence",
        xaxis_title="Sweep",
        yaxis_title="max |ΔV|",
        yaxis_type="log",
        height=270,
        margin=dict(l=0, r=0, t=36, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="#333"),
        yaxis=dict(gridcolor="#333"),
    )
    return fig


# ======================================================================
# World configuration presets
# ======================================================================

def build_world_config(preset: str, size: int) -> dict:
    """
    Returns {"obstacles": [...], "lava": [...]}.
    Lava cells are traversable during training (lava_active=False) but
    lethal during test-time episode replay (lava_active=True).
    """
    mid = size // 2
    protected = {(0, 0), (size - 1, size - 1)}

    if preset == "Horizontal Wall":
        obs = [(mid, c) for c in range(size - 1)]
        return {"obstacles": obs, "lava": []}

    if preset == "Vertical Wall":
        obs = [(r, mid) for r in range(size - 1)]
        return {"obstacles": obs, "lava": []}

    if preset == "Complex Maze":
        # Horizontal barriers at every other row with alternating gaps.
        # Even-indexed barrier: gap on the RIGHT (wall spans cols 0..N-2).
        # Odd-indexed  barrier: gap on the LEFT  (wall spans cols 1..N-1).
        # This forces the agent to snake across the grid.
        obs = []
        barrier_rows = list(range(2, size - 1, 2))
        for i, row in enumerate(barrier_rows):
            wall_cols = range(0, size - 1) if i % 2 == 0 else range(1, size)
            obs += [(row, c) for c in wall_cols if (row, c) not in protected]
        return {"obstacles": obs, "lava": []}

    if preset == "Zigzag Walls":
        # Vertical barriers at every other column with alternating gaps.
        obs = []
        for i, col in enumerate(range(1, size, 2)):
            row_range = range(0, size - 1) if i % 2 == 0 else range(1, size)
            obs += [(r, col) for r in row_range if (r, col) not in protected]
        return {"obstacles": obs, "lava": []}

    if preset == "Lava Field":
        # A horizontal lava strip across the middle row, with a gap at the
        # LAST column (col N-1).  The last row, last column, and goal area
        # are completely lava-free, so the goal is always reachable.
        #
        # Why this catches the RL agent: VI on a clean grid goes DOWN first
        # (tie-breaking favours action 1 = Down).  The agent walks straight
        # into the lava strip after size//2 steps.
        #
        # Safe path (IRD discovers it): go RIGHT along row 0 to (0, N-1),
        # then DOWN col N-1 through the gap, then to the goal.
        mid = size // 2
        lava = [(mid, c) for c in range(0, size - 1)   # full row except last col
                if (mid, c) not in protected]
        return {"obstacles": [], "lava": lava}

    return {"obstacles": [], "lava": []}   # "None"


# ======================================================================
# Tab 1 — Grid World RL Baseline (self-contained controls)
# ======================================================================

def gridworld_tab():
    st.markdown("### Grid World: Reinforcement Learning Baseline")
    st.markdown(
        "Inspired by the canonical grid world in "
        "[Ng & Russell (2000) — *Algorithms for Inverse Reinforcement Learning*](https://ai.stanford.edu/~ang/papers/icml00-irl.pdf). "
        "The agent learns to navigate from **S** (top-left) to **G** (bottom-right) "
        "using Q-Learning or Value Iteration."
    )

    # ── Inline configuration panel ────────────────────────────────────
    trained = st.session_state.get("trained", False)
    with st.expander("Configuration", expanded=not trained):
        c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
        with c1:
            st.markdown("**Environment**")
            grid_size = st.slider("Grid size  (N × N)", min_value=3, max_value=10, value=5,
                                  help="Capped at 10×10 — MaxEnt IRL (Tab 2/3) is O(N⁴) in pure Python",
                                  key="t1_grid_size")
            step_cost = st.number_input(
                "Step cost", value=-0.04, min_value=-1.0, max_value=0.0, step=0.01,
                help="Reward at every non-goal step", key="t1_step_cost",
            )
            goal_reward = st.number_input(
                "Goal reward", value=1.0, min_value=0.1, max_value=10.0, step=0.1,
                key="t1_goal_reward",
            )
            noise = st.slider("Transition noise", 0.0, 0.5, 0.0, step=0.05,
                              help="Prob of random action instead of intended",
                              key="t1_noise")
        with c2:
            st.markdown("**Layout**")
            preset = st.selectbox(
                "Preset",
                ["None", "Horizontal Wall", "Vertical Wall",
                 "Complex Maze", "Zigzag Walls", "Lava Field"],
                key="t1_preset",
            )
            st.markdown("")
            st.markdown(
                "*Lava Field*: lava strip at mid-row, gap on the right. "
                "RL agent walks into lava; IRD agent routes through the gap."
                if preset == "Lava Field" else ""
            )
        with c3:
            st.markdown("**Algorithm**")
            algo = st.radio("Algo", ["Q-Learning", "Value Iteration"], horizontal=True,
                            label_visibility="collapsed", key="t1_algo")
            gamma = st.slider("Discount  γ", 0.5, 1.0, 0.99, step=0.01, key="t1_gamma")
        with c4:
            st.markdown("**Q-Learning hyper-params**")
            alpha = st.slider("Learning rate  α", 0.01, 1.0, 0.1, step=0.01,
                              disabled=(algo != "Q-Learning"), key="t1_alpha")
            eps_start = st.slider("Initial ε", 0.1, 1.0, 1.0, step=0.05,
                                  disabled=(algo != "Q-Learning"), key="t1_eps")
            eps_decay = st.slider("ε decay", 0.980, 1.000, 0.995, step=0.001,
                                  format="%.3f", disabled=(algo != "Q-Learning"),
                                  key="t1_eps_decay")
            n_episodes = st.slider("Episodes", 100, 5000, 1000, step=100,
                                   disabled=(algo != "Q-Learning"), key="t1_n_ep")

        st.divider()
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            train_btn = st.button("Train Agent", type="primary", width="stretch",
                                  key="t1_train")
        with btn_col2:
            reset_btn = st.button("Reset", width="stretch", key="t1_reset")

    if reset_btn:
        for key in ["trained", "agent", "env", "trajectory", "algo_type", "ird", "challenge",
                    "ch_lava_grid", "ch_grid_size", "ch_base_lava_key"]:
            st.session_state.pop(key, None)
        st.rerun()

    # ── Training ────────────────────────────────────────────────────────
    if train_btn:
        world = build_world_config(preset, grid_size)
        env = GridWorld(
            size=grid_size,
            step_cost=step_cost,
            goal_reward=goal_reward,
            noise=noise,
            obstacles=world["obstacles"],
            lava_cells=world["lava"],
        )
        st.session_state.env = env
        st.session_state.trajectory = None
        st.session_state.pop("ird", None)   # invalidate any prior IRD result

        progress = st.progress(0.0, text="Initialising…")

        if algo == "Value Iteration":
            agent = ValueIterationAgent(env, gamma=gamma)
            agent.run_to_convergence()
            progress.progress(1.0, text=f"Converged in {agent.iterations} sweeps!")
            st.session_state.algo_type = "vi"
        else:
            agent = QLearningAgent(env, alpha=alpha, gamma=gamma,
                                   epsilon=eps_start, epsilon_decay=eps_decay)
            n = n_episodes
            update_every = max(1, n // 200)
            for ep in range(n):
                agent.train_episode()
                if ep % update_every == 0:
                    progress.progress((ep + 1) / n,
                                      text=f"Episode {ep+1}/{n}  |  ε = {agent.epsilon:.3f}")
            progress.progress(1.0, text="Training complete!")
            st.session_state.algo_type = "ql"

        st.session_state.agent = agent
        st.session_state.trained = True
        time.sleep(0.3)
        progress.empty()

    # Lava callout
    if st.session_state.get("env") and getattr(st.session_state.env, "lava_cells", set()):
        st.warning(
            "**Lava Field active.** The agent trained without knowing lava is dangerous. "
            "Click **Run Greedy Episode** to see it walk into the lava — then go to the "
            "**Inverse Reward Design** tab to fix it."
        )

    # ── Display ─────────────────────────────────────────────────────────
    if not st.session_state.get("trained"):
        # Preview the chosen layout before training
        size = st.session_state.get("t1_grid_size", 5)
        cur_preset = st.session_state.get("t1_preset", "None")
        cur_step_cost = st.session_state.get("t1_step_cost", -0.04)
        cur_goal_reward = st.session_state.get("t1_goal_reward", 1.0)
        world = build_world_config(cur_preset, size)
        env_preview = GridWorld(
            size=size,
            obstacles=world["obstacles"],
            lava_cells=world["lava"],
            step_cost=cur_step_cost,
            goal_reward=cur_goal_reward,
        )
        dummy_V = np.zeros((size, size))
        dummy_pol = np.zeros((size, size), dtype=int)
        fig = _make_grid_figure(
            env_preview, dummy_V, dummy_pol,
            title="Layout preview — click Train Agent to learn a policy",
        )
        col_g, col_r = st.columns([3, 2])
        with col_g:
            st.pyplot(fig, width="stretch")
        plt.close(fig)
        with col_r:
            st.info(
                "**How to use**\n\n"
                "1. Adjust grid size, costs, and noise above.\n"
                "2. Choose an obstacle layout preset.\n"
                "3. Select Q-Learning or Value Iteration.\n"
                "4. Click **Train Agent** — the grid will update with the learned policy and value function."
            )
        return

    env: GridWorld = st.session_state.env
    agent = st.session_state.agent
    algo_type: str = st.session_state.algo_type
    trajectory = st.session_state.get("trajectory")

    policy = agent.get_policy()
    value_fn = agent.get_value_function()
    algo_label = "Q-Learning" if algo_type == "ql" else "Value Iteration"

    # Main columns
    col_grid, col_right = st.columns([3, 2])

    with col_grid:
        fig = _make_grid_figure(
            env, value_fn, policy,
            title=f"{algo_label} — Learned Policy & Value Function",
            trajectory=trajectory,
        )
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with col_right:
        # Metrics
        if algo_type == "vi":
            st.metric("Sweeps to converge", agent.iterations)
            st.metric("Max  V*(s)", f"{np.max(value_fn):.4f}")
            non_obs = [
                value_fn[r, c]
                for r in range(env.size) for c in range(env.size)
                if (r, c) not in env.obstacles
            ]
            st.metric("Min  V*(s)", f"{min(non_obs):.4f}")
        else:
            rewards = agent.episode_rewards
            last_n = min(50, len(rewards))
            st.metric("Episodes trained", len(rewards))
            st.metric(f"Avg reward (last {last_n})", f"{np.mean(rewards[-last_n:]):.3f}")
            st.metric("Final ε", f"{agent.epsilon:.4f}")

        # Convergence / training chart
        if algo_type == "vi":
            st.plotly_chart(
                _convergence_curve(agent.delta_history),
                width="stretch",
            )
        else:
            st.plotly_chart(
                _training_curve(agent.episode_rewards),
                width="stretch",
            )

    # ── Greedy episode playback ────────────────────────────────────────
    st.divider()
    col_btn, col_traj = st.columns([1, 3])
    with col_btn:
        if st.button("Run Greedy Episode", width="stretch"):
            # Activate lava for test-time evaluation — agent was trained without it
            if hasattr(env, "lava_active"):
                env.lava_active = True
            traj, total_r = agent.run_greedy_episode()
            if hasattr(env, "lava_active"):
                env.lava_active = False
            st.session_state.trajectory = traj
            st.session_state.traj_reward = total_r
            st.rerun()

        if trajectory:
            if st.button("Clear Trajectory", width="stretch"):
                st.session_state.trajectory = None
                st.rerun()

    with col_traj:
        if trajectory:
            final_pos = trajectory[-1]
            goal_reached = final_pos == env.goal
            lava_hit = final_pos in getattr(env, "lava_cells", set())
            traj_reward = st.session_state.get("traj_reward", 0.0)

            if lava_hit:
                status = "Terminated — agent stepped on lava"
                st.error(
                    f"**{status}**  |  Steps: **{len(trajectory) - 1}**  |  "
                    f"Total reward: **{traj_reward:.3f}**  \n"
                    "The agent had no concept of lava in its reward function — "
                    "it walked into a tile it was never taught to avoid."
                )
            elif goal_reached:
                status = "Goal reached"
                st.success(
                    f"**{status}**  |  Steps: **{len(trajectory) - 1}**  |  "
                    f"Total reward: **{traj_reward:.3f}**"
                )
            else:
                status = "Did not reach goal"
                st.info(
                    f"**{status}**  |  Steps: **{len(trajectory) - 1}**  |  "
                    f"Total reward: **{traj_reward:.3f}**"
                )


# ======================================================================
# Main entry point
# ======================================================================

# ======================================================================
# IRD reward heatmap helper
# ======================================================================

def _reward_heatmap(R: np.ndarray, env: GridWorld, title: str) -> plt.Figure:
    size = env.size
    fig, ax = plt.subplots(figsize=(max(4, size), max(4, size)))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    display_R = R.astype(float).copy()
    for r, c in env.obstacles:
        display_R[r, c] = np.nan

    vabs = max(abs(float(np.nanmin(display_R))), abs(float(np.nanmax(display_R))), 0.01)
    cmap = plt.cm.RdBu
    cmap.set_bad(color="#444444")
    im = ax.imshow(display_R, cmap=cmap, vmin=-vabs, vmax=vabs,
                   aspect="equal", interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")

    for i in range(size + 1):
        ax.axhline(i - 0.5, color="#555555", linewidth=0.5, zorder=2)
        ax.axvline(i - 0.5, color="#555555", linewidth=0.5, zorder=2)

    for r in range(size):
        for c in range(size):
            if (r, c) in env.obstacles:
                continue
            if (r, c) in getattr(env, "lava_cells", set()):
                ax.add_patch(plt.Rectangle((c-.5, r-.5), 1, 1,
                                           color="#7a1a00", alpha=0.6, zorder=3))
                ax.text(c, r, "L", ha="center", va="center",
                        fontsize=7, color="#ff6b35", zorder=4)
            elif (r, c) == env.goal:
                ax.text(c, r, "G", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="#27ae60",
                                  edgecolor="white", linewidth=1), zorder=5)
            else:
                ax.text(c, r, f"{R[r,c]:.2f}", ha="center", va="center",
                        fontsize=max(4, 7 - size // 4), color="white", alpha=0.8, zorder=4)

    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.tick_params(colors="white", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#555555")
    ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=6)
    fig.tight_layout()
    return fig


# ======================================================================
# Tab 2 — IRD
# ======================================================================

def ird_tab():
    st.markdown("### Inverse Reward Design")
    st.markdown(
        "Based on [Hadfield-Menell et al. (2017) — *Inverse Reward Design*](https://arxiv.org/abs/1711.02827).  \n"
        "**Workflow:** (1) collect demonstrations from the RL agent trained in Tab 1, "
        "(2) run **MaxEnt IRL** to recover the reward function, "
        "(3) apply a **pessimistic prior** to unseen cell types (lava) — the IRD step — "
        "and re-plan.  The IRD agent routes around lava; the vanilla RL agent does not."
    )

    env = st.session_state.get("env")
    agent = st.session_state.get("agent")

    if env is None or agent is None:
        st.warning("Train an RL agent in the **Grid World RL** tab first, then return here.")
        return

    has_lava = bool(getattr(env, "lava_cells", set()))
    if not has_lava:
        st.info(
            "No lava cells in the current environment.  \n"
            "Switch to the **Lava Field** preset in the **Grid World RL** tab "
            "and retrain. IRD is most instructive when lava is present."
        )

    # ── IRD parameters ────────────────────────────────────────────────
    with st.expander("IRD Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            n_demos = st.slider("Demonstrations", 10, 200, 50, step=10,
                                help="Greedy rollouts collected from the RL agent")
            n_iters = st.slider("IRL iterations", 50, 400, 150, step=50)
        with col2:
            lr_irl = st.slider("IRL learning rate", 0.005, 0.2, 0.05, step=0.005,
                               format="%.3f")
            gamma_irl = st.slider("IRL discount  γ", 0.8, 1.0, 0.99, step=0.01)
        with col3:
            unseen_penalty = st.slider(
                "Unseen-tile penalty", -5.0, -0.1, -1.0, step=0.1,
                help="Pessimistic reward IRD assigns to lava — controls how strongly the agent avoids it",
            )
        run_ird_btn = st.button("Run IRD", type="primary", width="stretch")

    if run_ird_btn:
        prog = st.progress(0.0, text="Collecting demonstrations…")
        demos = collect_demos(agent, env, n=n_demos)
        prog.progress(0.1, text=f"Collected {n_demos} demos. Running MaxEnt IRL…")

        def _cb(frac, grad_norm):
            prog.progress(0.1 + 0.85 * frac,
                          text=f"IRL iter {int(frac * n_iters)}/{n_iters}  |  grad norm = {grad_norm:.4f}")

        R_irl, history = maxent_irl(env, demos, gamma=gamma_irl,
                                    n_iters=n_iters, lr=lr_irl, progress_cb=_cb)
        prog.progress(0.95, text="Planning IRD policy…")

        R_ird = ird_reward(R_irl, env, unseen_penalty=unseen_penalty)
        V_ird, policy_ird = plan_with_reward(env, R_ird, gamma=gamma_irl)

        # Collect comparison trajectories
        rl_policy = agent.get_policy()
        traj_rl, r_rl = run_policy_episode(env, rl_policy, lava_active=True)
        traj_ird, r_ird = run_policy_episode(env, policy_ird, lava_active=True)

        st.session_state.ird = dict(
            R_irl=R_irl, R_ird=R_ird, history=history,
            V_ird=V_ird, policy_ird=policy_ird,
            traj_rl=traj_rl, r_rl=r_rl,
            traj_ird=traj_ird, r_ird=r_ird,
        )
        prog.progress(1.0, text="Done!")
        time.sleep(0.3)
        prog.empty()
        st.rerun()

    ird = st.session_state.get("ird")
    if ird is None:
        st.info("Configure parameters above and click **Run IRD**.")
        return

    R_irl = ird["R_irl"]
    R_ird = ird["R_ird"]
    history = ird["history"]
    V_ird = ird["V_ird"]
    policy_ird = ird["policy_ird"]
    traj_rl = ird["traj_rl"]
    r_rl = ird["r_rl"]
    traj_ird = ird["traj_ird"]
    r_ird = ird["r_ird"]
    rl_policy = agent.get_policy()

    # ── Three-column comparison ────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("#### RL Agent — Test Env")
        st.caption("Policy learned without lava. Deployed into lava field.")
        fig = _make_grid_figure(env, agent.get_value_function(), rl_policy,
                                title="RL policy (trained without lava)",
                                trajectory=traj_rl)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        lava_hit = traj_rl[-1] in getattr(env, "lava_cells", set())
        goal_reached = traj_rl[-1] == env.goal
        if lava_hit:
            st.error(f"Terminated by lava  |  steps: {len(traj_rl)-1}  |  reward: {r_rl:.3f}")
        elif goal_reached:
            st.success(f"Goal reached  |  steps: {len(traj_rl)-1}  |  reward: {r_rl:.3f}")
        else:
            st.warning(f"Did not reach goal  |  steps: {len(traj_rl)-1}  |  reward: {r_rl:.3f}")

    with col_b:
        st.markdown("#### Recovered Reward (IRL)")
        st.caption("MaxEnt IRL estimate from demonstrations — lava cells get the pessimistic prior (IRD step).")
        fig_irl = _reward_heatmap(R_irl, env, "R_IRL (from demos)")
        st.pyplot(fig_irl, width="stretch")
        plt.close(fig_irl)
        fig_ird = _reward_heatmap(R_ird, env, "R_IRD (lava penalised)")
        st.pyplot(fig_ird, width="stretch")
        plt.close(fig_ird)

        # IRL convergence
        conv_fig = go.Figure(go.Scatter(
            y=history, mode="lines",
            line=dict(color="#f39c12", width=2), name="grad norm",
        ))
        conv_fig.update_layout(
            title="IRL convergence", xaxis_title="Iteration",
            yaxis_title="|grad|", height=220,
            margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"), xaxis=dict(gridcolor="#333"),
            yaxis=dict(gridcolor="#333"),
        )
        st.plotly_chart(conv_fig, width="stretch")

    with col_c:
        st.markdown("#### IRD Agent — Test Env")
        st.caption("Re-planned with pessimistic lava reward. Knows to treat novel tiles with caution.")
        fig = _make_grid_figure(env, V_ird, policy_ird,
                                title="IRD policy (lava-aware)",
                                trajectory=traj_ird)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        lava_hit_ird = traj_ird[-1] in getattr(env, "lava_cells", set())
        goal_reached_ird = traj_ird[-1] == env.goal
        if lava_hit_ird:
            st.error(f"Terminated by lava  |  steps: {len(traj_ird)-1}  |  reward: {r_ird:.3f}")
        elif goal_reached_ird:
            st.success(f"Goal reached  |  steps: {len(traj_ird)-1}  |  reward: {r_ird:.3f}")
        else:
            st.warning(f"Did not reach goal  |  steps: {len(traj_ird)-1}  |  reward: {r_ird:.3f}")

    # ── Summary table ─────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Comparison Summary")
    def _outcome(traj, env):
        final = traj[-1]
        if final == env.goal:
            return "Goal reached"
        if final in getattr(env, "lava_cells", set()):
            return "Lava — terminated"
        return "Max steps reached"

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("RL outcome", _outcome(traj_rl, env))
    col_m2.metric("RL total reward", f"{r_rl:.3f}")
    col_m3.metric("IRD outcome", _outcome(traj_ird, env))
    col_m4.metric("IRD total reward", f"{r_ird:.3f}")


# ======================================================================
# Tab 3 helpers
# ======================================================================


def _outcome_bar_chart(rl_stats: dict, ird_stats: dict) -> go.Figure:
    """Grouped bar chart comparing RL vs IRD outcomes."""
    agents = ["RL Agent", "IRD Agent"]
    success = [rl_stats["success_rate"] * 100, ird_stats["success_rate"] * 100]
    lava = [rl_stats["lava_rate"] * 100, ird_stats["lava_rate"] * 100]
    timeout = [rl_stats["timeout_rate"] * 100, ird_stats["timeout_rate"] * 100]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Goal reached", x=agents, y=success,
                         marker_color="#27ae60"))
    fig.add_trace(go.Bar(name="Hazard death", x=agents, y=lava,
                         marker_color="#e74c3c"))
    fig.add_trace(go.Bar(name="Timeout", x=agents, y=timeout,
                         marker_color="#95a5a6"))
    fig.update_layout(
        barmode="stack",
        title="Episode Outcomes (%)",
        yaxis_title="% of episodes",
        yaxis=dict(range=[0, 105], gridcolor="#333"),
        height=300,
        margin=dict(l=0, r=0, t=36, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font_size=11),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="#333"),
    )
    return fig


def _reward_comparison_chart(rl_stats: dict, ird_stats: dict) -> go.Figure:
    """Bar chart comparing avg reward and avg steps."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Avg reward",
        x=["RL Agent", "IRD Agent"],
        y=[rl_stats["avg_reward"], ird_stats["avg_reward"]],
        marker_color=["#3498db", "#9b59b6"],
        yaxis="y",
    ))
    fig.update_layout(
        title="Average Episode Reward",
        yaxis_title="Reward",
        yaxis=dict(gridcolor="#333"),
        height=260,
        margin=dict(l=0, r=0, t=36, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="#333"),
    )
    return fig


# ======================================================================
# Tab 3 — Robustness Challenge (user-placed lava)
# ======================================================================

def challenge_tab():
    st.markdown("### Robustness Challenge")
    st.markdown(
        "Design your own lava layout and compare how the RL agent (oblivious to lava) "
        "performs against the IRD agent (pessimistic prior on unseen tiles).  \n"
        "**Check cells** in the grid to place lava, then click **Run Challenge**."
    )

    env = st.session_state.get("env")
    agent = st.session_state.get("agent")

    if env is None or agent is None:
        st.warning("Train an RL agent in the **Grid World RL** tab first, then return here.")
        return

    size = env.size
    protected = {env.start, env.goal} | env.obstacles

    # ── Initialise lava grid in session state ─────────────────────────
    base_lava = frozenset(getattr(env, "lava_cells", set()))
    if (
        "ch_lava_grid" not in st.session_state
        or st.session_state.get("ch_grid_size") != size
        or st.session_state.get("ch_base_lava_key") != base_lava
    ):
        st.session_state.ch_lava_grid = [
            [bool((r, c) in base_lava) for c in range(size)]
            for r in range(size)
        ]
        st.session_state.ch_grid_size = size
        st.session_state.ch_base_lava_key = base_lava

    # ── Preset buttons ────────────────────────────────────────────────
    mid = size // 2

    def _apply_preset(lava_set):
        st.session_state.ch_lava_grid = [
            [bool((r, c) in lava_set and (r, c) not in protected) for c in range(size)]
            for r in range(size)
        ]

    bp1, bp2, bp3, bp4, bp5 = st.columns(5)
    with bp1:
        if st.button("From Tab 1", key="ch_p_tab1", width="stretch",
                     help="Load the lava layout from the Tab 1 environment"):
            _apply_preset(set(base_lava))
            st.rerun()
    with bp2:
        if st.button("H-Wall", key="ch_p_hwall", width="stretch",
                     help="Horizontal lava strip at mid-row with gap on right"):
            _apply_preset({(mid, c) for c in range(size - 1) if (mid, c) not in protected})
            st.rerun()
    with bp3:
        if st.button("Diagonal", key="ch_p_diag", width="stretch",
                     help="Lava along the anti-diagonal"):
            _apply_preset({(i, size - 1 - i) for i in range(1, size - 1)
                           if (i, size - 1 - i) not in protected})
            st.rerun()
    with bp4:
        if st.button("V-Wall", key="ch_p_vwall", width="stretch",
                     help="Vertical lava strip at mid-col with gap at bottom"):
            _apply_preset({(r, mid) for r in range(size - 1) if (r, mid) not in protected})
            st.rerun()
    with bp5:
        if st.button("Clear", key="ch_p_clear", width="stretch"):
            _apply_preset(set())
            st.rerun()

    # ── Lava grid editor ──────────────────────────────────────────────
    st.caption(
        "Check a cell = lava.  Start (0,0) and goal (N-1,N-1) are protected. "
        "Columns = grid columns, rows = grid rows."
    )
    df_in = pd.DataFrame(
        st.session_state.ch_lava_grid,
        index=[str(r) for r in range(size)],
        columns=[str(c) for c in range(size)],
        dtype=bool,
    )
    for r, c in protected:
        if r < size and c < size:
            df_in.iloc[r, c] = False

    edited = st.data_editor(
        df_in,
        key="ch_lava_editor",
        height=min(420, 40 + 36 * size),
        use_container_width=True,
    )

    # Sync and extract new lava set
    st.session_state.ch_lava_grid = [
        [bool(edited.iloc[r, c]) for c in range(size)]
        for r in range(size)
    ]
    new_lava = {
        (r, c)
        for r in range(size)
        for c in range(size)
        if edited.iloc[r, c] and (r, c) not in protected
    }

    if not new_lava:
        st.info("No lava placed — check cells above or use a preset, then Run Challenge.")

    # ── IRD parameters ────────────────────────────────────────────────
    with st.expander("IRD Parameters", expanded=True):
        cp1, cp2 = st.columns(2)
        with cp1:
            ch_unseen_pen = st.slider("Lava penalty (IRD prior)", -5.0, -0.1, -1.0, step=0.1,
                                      key="ch_pen",
                                      help="How pessimistic IRD is about unseen lava cells")
            ch_n_demos = st.slider("Demonstrations", 10, 100, 50, step=10, key="ch_demos")
        with cp2:
            ch_n_iters = st.slider("IRL iterations", 50, 300, 150, step=50, key="ch_iters")
            ch_n_ep = st.slider("Evaluation episodes", 20, 200, 100, step=20, key="ch_n_ep")
        ch_gamma = st.slider("Discount  γ", 0.8, 1.0, 0.99, step=0.01, key="ch_gamma")
        run_ch_btn = st.button(
            "Run Challenge", type="primary", width="stretch", key="ch_run",
            disabled=not bool(new_lava),
        )

    if run_ch_btn:
        ch_env = GridWorld(
            size=env.size,
            start=env.start,
            goal=env.goal,
            obstacles=list(env.obstacles),
            lava_cells=list(new_lava),
            step_cost=env.step_cost,
            goal_reward=env.goal_reward,
            noise=env.noise,
        )

        prog = st.progress(0.0, text="Collecting demonstrations on clean environment…")
        demos = collect_demos(agent, env, n=ch_n_demos)
        prog.progress(0.1, text=f"Collected {ch_n_demos} demos — running MaxEnt IRL…")

        def _cb(frac, gn):
            prog.progress(0.1 + 0.7 * frac,
                          text=f"IRL iter {int(frac * ch_n_iters)}/{ch_n_iters}  |  grad = {gn:.4f}")

        R_irl, irl_history = maxent_irl(
            env, demos, gamma=ch_gamma, n_iters=ch_n_iters, lr=0.05, progress_cb=_cb,
        )
        prog.progress(0.8, text="Applying IRD lava penalty and re-planning…")

        R_ird = ird_reward(R_irl, ch_env, unseen_penalty=ch_unseen_pen)
        V_ird, policy_ird = plan_with_reward(ch_env, R_ird, gamma=ch_gamma)

        rl_policy = agent.get_policy()

        prog.progress(0.88, text="Running batch evaluation — RL agent…")
        rl_stats = run_batch_episodes(
            ch_env, rl_policy, n=ch_n_ep,
            lava_active=True, mud_active=False, ice_active=False,
        )
        prog.progress(0.94, text="Running batch evaluation — IRD agent…")
        ird_stats = run_batch_episodes(
            ch_env, policy_ird, n=ch_n_ep,
            lava_active=True, mud_active=False, ice_active=False,
        )

        rl_traj, rl_r = run_policy_episode(ch_env, rl_policy, lava_active=True)
        ird_traj, ird_r = run_policy_episode(ch_env, policy_ird, lava_active=True)

        prog.progress(1.0, text="Done!")
        time.sleep(0.3)
        prog.empty()

        st.session_state.challenge = dict(
            ch_env=ch_env,
            R_irl=R_irl, R_ird=R_ird,
            V_ird=V_ird, policy_ird=policy_ird,
            rl_stats=rl_stats, ird_stats=ird_stats,
            rl_traj=rl_traj, rl_r=rl_r,
            ird_traj=ird_traj, ird_r=ird_r,
            irl_history=irl_history,
        )
        st.rerun()

    ch = st.session_state.get("challenge")
    if ch is None:
        return

    ch_env: GridWorld = ch["ch_env"]
    rl_policy = agent.get_policy()
    policy_ird = ch["policy_ird"]
    rl_stats = ch["rl_stats"]
    ird_stats = ch["ird_stats"]
    rl_traj = ch["rl_traj"]
    ird_traj = ch["ird_traj"]

    st.markdown("#### Results")

    # ── Side-by-side policy grids ─────────────────────────────────────
    col_rl, col_ird = st.columns(2)

    with col_rl:
        st.markdown("**RL Agent** — trained without lava, deployed with lava active")
        fig = _make_grid_figure(
            ch_env, agent.get_value_function(), rl_policy,
            title="RL Policy", trajectory=rl_traj,
        )
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        final = rl_traj[-1]
        if final == ch_env.goal:
            st.success(f"Goal reached  |  steps: {len(rl_traj)-1}  |  reward: {ch['rl_r']:.3f}")
        elif final in ch_env.lava_cells:
            st.error(f"Lava — terminated  |  steps: {len(rl_traj)-1}  |  reward: {ch['rl_r']:.3f}")
        else:
            st.warning(f"Did not reach goal  |  steps: {len(rl_traj)-1}  |  reward: {ch['rl_r']:.3f}")

    with col_ird:
        st.markdown("**IRD Agent** — pessimistic prior on unseen lava, re-planned")
        fig = _make_grid_figure(
            ch_env, ch["V_ird"], policy_ird,
            title="IRD Policy", trajectory=ird_traj,
        )
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        final = ird_traj[-1]
        if final == ch_env.goal:
            st.success(f"Goal reached  |  steps: {len(ird_traj)-1}  |  reward: {ch['ird_r']:.3f}")
        elif final in ch_env.lava_cells:
            st.error(f"Lava — terminated  |  steps: {len(ird_traj)-1}  |  reward: {ch['ird_r']:.3f}")
        else:
            st.warning(f"Did not reach goal  |  steps: {len(ird_traj)-1}  |  reward: {ch['ird_r']:.3f}")

    # ── Batch statistics ─────────────────────────────────────────────
    st.divider()
    st.markdown(f"#### Batch Evaluation — {rl_stats['n']} episodes each")

    col_bar, col_reward = st.columns(2)
    with col_bar:
        st.plotly_chart(_outcome_bar_chart(rl_stats, ird_stats), width="stretch")
    with col_reward:
        st.plotly_chart(_reward_comparison_chart(rl_stats, ird_stats), width="stretch")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("RL success rate", f"{rl_stats['success_rate']*100:.1f}%")
    m2.metric("RL avg reward", f"{rl_stats['avg_reward']:.3f}")
    m3.metric("RL avg steps", f"{rl_stats['avg_steps']:.1f}")
    m4.metric("IRD success rate", f"{ird_stats['success_rate']*100:.1f}%")
    m5.metric("IRD avg reward", f"{ird_stats['avg_reward']:.3f}")
    m6.metric("IRD avg steps", f"{ird_stats['avg_steps']:.1f}")

    # ── IRL convergence ────────────────────────────────────────────────
    st.divider()
    conv_fig = go.Figure(go.Scatter(
        y=ch["irl_history"], mode="lines",
        line=dict(color="#f39c12", width=2), name="grad norm",
    ))
    conv_fig.update_layout(
        title="IRL convergence (challenge run)",
        xaxis_title="Iteration", yaxis_title="|grad|",
        height=220, margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
    )
    st.plotly_chart(conv_fig, width="stretch")

    if st.button("Clear Results", key="ch_clear"):
        st.session_state.pop("challenge", None)
        st.rerun()


# ======================================================================
# Main entry point
# ======================================================================

def main():
    st.title("IRL Explorer")
    st.caption("An interactive journey from Reinforcement Learning to Inverse Reward Design")

    tab_rl, tab_ird, tab_challenge = st.tabs([
        "Grid World RL",
        "Inverse Reward Design",
        "Robustness Challenge",
    ])

    with tab_rl:
        gridworld_tab()

    with tab_ird:
        ird_tab()

    with tab_challenge:
        challenge_tab()


if __name__ == "__main__":
    main()
