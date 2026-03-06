"""
Inverse Reward Design utilities.

  maxent_irl     — Maximum Entropy IRL (Ziebart et al. 2008)
  ird_reward     — Pessimistic reward for unseen cell types (lava)
  plan_with_reward — Value Iteration given an explicit reward matrix
  collect_demos  — Run the trained RL agent to gather demonstrations
"""

import numpy as np
from typing import List, Tuple
from gridworld import GridWorld


# ======================================================================
# Helpers
# ======================================================================

def _logsumexp(arr):
    a = np.max(arr)
    return a + np.log(np.sum(np.exp(np.array(arr) - a)))


def collect_demos(agent, env: GridWorld, n: int = 50) -> List[List[Tuple[int, int]]]:
    """Run n greedy episodes (lava inactive) and return trajectories."""
    demos = []
    for _ in range(n):
        traj, _ = agent.run_greedy_episode()
        demos.append(traj)
    return demos


# ======================================================================
# Soft value iteration (MaxEnt backbone)
# ======================================================================

def soft_value_iteration(
    env: GridWorld,
    R: np.ndarray,
    gamma: float = 0.99,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Computes the soft (log-sum-exp) value function for reward matrix R.
    Used internally by MaxEnt IRL.
    """
    V = np.zeros((env.size, env.size))
    for _ in range(max_iter):
        V_new = np.zeros((env.size, env.size))
        for r in range(env.size):
            for c in range(env.size):
                if (r, c) in env.obstacles:
                    continue
                if (r, c) == env.goal:
                    V_new[r, c] = R[r, c]
                    continue
                Q = []
                for a in range(env.n_actions):
                    nr, nc, _, done = env.deterministic_transition(r, c, a)
                    Q.append(R[r, c] + (0.0 if done else gamma * V[nr, nc]))
                V_new[r, c] = _logsumexp(Q)
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new
    return V


def _soft_policy(env: GridWorld, V: np.ndarray, R: np.ndarray, gamma: float) -> np.ndarray:
    """Boltzmann policy: pi(a|s) proportional to exp(Q(s,a))."""
    pi = np.ones((env.size, env.size, env.n_actions)) / env.n_actions
    for r in range(env.size):
        for c in range(env.size):
            if (r, c) in env.obstacles or (r, c) == env.goal:
                continue
            Q = np.array([
                R[r, c] + (0.0 if env.deterministic_transition(r, c, a)[3]
                           else gamma * V[env.deterministic_transition(r, c, a)[0],
                                          env.deterministic_transition(r, c, a)[1]])
                for a in range(env.n_actions)
            ])
            exp_Q = np.exp(Q - np.max(Q))
            pi[r, c] = exp_Q / exp_Q.sum()
    return pi


# ======================================================================
# State visitation
# ======================================================================

def state_visitation_from_demos(
    env: GridWorld, demos: List[List[Tuple[int, int]]]
) -> np.ndarray:
    """Empirical state visitation counts normalised to a distribution."""
    visit = np.zeros((env.size, env.size))
    total = 0
    for traj in demos:
        for pos in traj:
            visit[pos] += 1
            total += 1
    return visit / max(total, 1)


def expected_state_visitation(
    env: GridWorld, pi: np.ndarray, n_steps: int = 80
) -> np.ndarray:
    """Forward-pass occupancy measure under Boltzmann policy pi."""
    D = np.zeros((env.size, env.size))
    D_t = np.zeros((env.size, env.size))
    r0, c0 = env.start
    D_t[r0, c0] = 1.0
    D += D_t

    for _ in range(n_steps - 1):
        D_next = np.zeros((env.size, env.size))
        for r in range(env.size):
            for c in range(env.size):
                if D_t[r, c] < 1e-12:
                    continue
                if (r, c) == env.goal or (r, c) in env.obstacles:
                    continue
                for a in range(env.n_actions):
                    nr, nc, _, _ = env.deterministic_transition(r, c, a)
                    D_next[nr, nc] += pi[r, c, a] * D_t[r, c]
        D_t = D_next
        D += D_t

    return D / n_steps


# ======================================================================
# MaxEnt IRL
# ======================================================================

def maxent_irl(
    env: GridWorld,
    demos: List[List[Tuple[int, int]]],
    gamma: float = 0.99,
    n_iters: int = 150,
    lr: float = 0.05,
    progress_cb=None,
) -> Tuple[np.ndarray, List[float]]:
    """
    Maximum Entropy IRL with one-hot (per-state) features.
    Returns (reward_matrix, grad_norm_history).

    The gradient is: mu_demo - mu_theta (feature expectation matching).
    """
    mu_demo = state_visitation_from_demos(env, demos)
    theta = np.zeros((env.size, env.size))
    history = []

    for i in range(n_iters):
        V = soft_value_iteration(env, theta, gamma)
        pi = _soft_policy(env, V, theta, gamma)
        mu_pi = expected_state_visitation(env, pi)
        grad = mu_demo - mu_pi
        theta += lr * grad
        grad_norm = float(np.sum(np.abs(grad)))
        history.append(grad_norm)
        if progress_cb:
            progress_cb((i + 1) / n_iters, grad_norm)

    return theta, history


# ======================================================================
# IRD-aware reward and planning
# ======================================================================

def ird_reward(
    R_irl: np.ndarray,
    env: GridWorld,
    unseen_penalty: float = -1.0,
) -> np.ndarray:
    """
    Construct the IRD reward matrix.

    Known-cell rewards come from the IRL estimate.
    Unseen cell types (lava) receive a pessimistic penalty reflecting
    the agent's uncertainty — the core of Inverse Reward Design.
    """
    R_ird = R_irl.copy()
    for r, c in getattr(env, "lava_cells", set()):
        R_ird[r, c] = unseen_penalty
    return R_ird


def plan_with_reward(
    env: GridWorld,
    R: np.ndarray,
    gamma: float = 0.99,
    lava_terminal: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Value Iteration given an explicit per-state reward matrix R.

    When lava_terminal=True (IRD mode), lava cells are treated as
    terminal states whose value equals R[lava] — the pessimistic estimate.
    This causes the agent to route around them.

    Returns (V, greedy_policy).
    """
    size = env.size
    # Treat ALL unseen tile types as terminal pseudo-absorbing states.
    # This is the core IRD insight: uncertainty about novel tiles → avoid them.
    if lava_terminal:
        lava: set = set(getattr(env, "lava_cells", set()))
        lava.update(getattr(env, "mud_cells", set()))
        lava.update(getattr(env, "ice_cells", set()))
    else:
        lava = set()

    # Initialise terminal values
    V = np.zeros((size, size))
    for r, c in lava:
        V[r, c] = R[r, c]

    for _ in range(1000):
        V_new = V.copy()
        delta = 0.0
        for r in range(size):
            for c in range(size):
                if (r, c) in env.obstacles:
                    continue
                if (r, c) == env.goal or (r, c) in lava:
                    continue   # terminal — value fixed
                vals = []
                for a in range(env.n_actions):
                    nr, nc, _, done = env.deterministic_transition(r, c, a)
                    if (nr, nc) in lava:
                        # agent believes: step to lava = step_cost + gamma*V(lava)
                        v = env.step_cost + gamma * V[nr, nc]
                    elif done:
                        v = env.step_cost + env.goal_reward
                    else:
                        v = env.step_cost + gamma * V[nr, nc]
                    vals.append(v)
                V_new[r, c] = max(vals)
                delta = max(delta, abs(V_new[r, c] - V[r, c]))
        V = V_new
        if delta < 1e-6:
            break

    # Greedy policy
    policy = np.zeros((size, size), dtype=int)
    for r in range(size):
        for c in range(size):
            if (r, c) in env.obstacles or (r, c) == env.goal or (r, c) in lava:
                continue
            vals = []
            for a in range(env.n_actions):
                nr, nc, _, done = env.deterministic_transition(r, c, a)
                if (nr, nc) in lava:
                    v = env.step_cost + gamma * V[nr, nc]
                elif done:
                    v = env.step_cost + env.goal_reward
                else:
                    v = env.step_cost + gamma * V[nr, nc]
                vals.append(v)
            policy[r, c] = int(np.argmax(vals))

    return V, policy


def run_policy_episode(
    env: GridWorld,
    policy: np.ndarray,
    lava_active: bool = True,
    max_steps: int = 300,
) -> Tuple[List[Tuple[int, int]], float]:
    """Execute a deterministic policy and return (trajectory, total_reward)."""
    prev = env.lava_active
    env.lava_active = lava_active
    r, c = env.reset()
    traj = [(r, c)]
    total = 0.0
    for _ in range(max_steps):
        a = int(policy[r, c])
        nr, nc, reward, done = env.transition(r, c, a)
        r, c = nr, nc
        traj.append((r, c))
        total += reward
        if done:
            break
    env.lava_active = prev
    return traj, total


def ird_reward_multi(
    R_irl: np.ndarray,
    env: GridWorld,
    penalties: dict,
) -> np.ndarray:
    """
    Apply pessimistic IRD penalties for multiple unseen tile types.

    penalties: mapping from GridWorld attribute name to penalty value.
    Example: {"lava_cells": -5.0, "mud_cells": -0.5, "ice_cells": -0.3}

    Cells of each type receive the given fixed reward, overriding the IRL estimate.
    """
    R = R_irl.copy()
    for attr, penalty in penalties.items():
        cells = getattr(env, attr, set())
        for r, c in cells:
            R[r, c] = penalty
    return R


def run_batch_episodes(
    env: GridWorld,
    policy: np.ndarray,
    n: int = 100,
    lava_active: bool = True,
    mud_active: bool = True,
    ice_active: bool = True,
    max_steps: int = 300,
) -> dict:
    """
    Run n stochastic episodes under the given policy and return aggregate stats.

    Tile activation flags control which hazard types are live during evaluation.
    Returns a dict with keys:
      success_rate, lava_rate, timeout_rate, avg_reward, avg_steps
    """
    # Save and set tile-activation flags
    saved = {}
    for attr in ("lava_active", "mud_active", "ice_active"):
        if hasattr(env, attr):
            saved[attr] = getattr(env, attr)
    env.lava_active = lava_active
    if hasattr(env, "mud_active"):
        env.mud_active = mud_active
    if hasattr(env, "ice_active"):
        env.ice_active = ice_active

    lava_cells = getattr(env, "lava_cells", set())
    successes = 0
    lava_deaths = 0
    timeouts = 0
    total_rewards: List[float] = []
    steps_list: List[int] = []

    for _ in range(n):
        r, c = env.reset()
        total = 0.0
        steps = 0
        done = False
        for _ in range(max_steps):
            a = int(policy[r, c])
            r, c, reward, done = env.transition(r, c, a)
            total += reward
            steps += 1
            if done:
                break
        total_rewards.append(total)
        steps_list.append(steps)
        if (r, c) == env.goal:
            successes += 1
        elif lava_active and (r, c) in lava_cells:
            lava_deaths += 1
        else:
            timeouts += 1

    # Restore flags
    for attr, val in saved.items():
        setattr(env, attr, val)

    return {
        "success_rate": successes / n,
        "lava_rate": lava_deaths / n,
        "timeout_rate": timeouts / n,
        "avg_reward": float(np.mean(total_rewards)),
        "avg_steps": float(np.mean(steps_list)),
        "n": n,
    }
