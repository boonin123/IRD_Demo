"""
RL agents for the IRL Explorer.
  - QLearningAgent  : model-free, learns from experience
  - ValueIterationAgent : model-based, solves the Bellman equations exactly
"""

import numpy as np
from typing import List, Tuple
from gridworld import GridWorld


# ======================================================================
# Q-Learning
# ======================================================================

class QLearningAgent:
    """
    Tabular Q-Learning with epsilon-greedy exploration and epsilon decay.
    """

    def __init__(
        self,
        env: GridWorld,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.Q = np.zeros((env.n_states, env.n_actions))
        self.episode_rewards: List[float] = []
        self.episode_steps: List[int] = []

    # ------------------------------------------------------------------

    def select_action(self, state: int, greedy: bool = False) -> int:
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.env.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(
        self, s: int, a: int, r: float, s2: int, done: bool
    ) -> None:
        target = r if done else r + self.gamma * np.max(self.Q[s2])
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def train_episode(self, max_steps: int = 500) -> Tuple[float, int]:
        r, c = self.env.reset()
        total_reward = 0.0
        steps = 0

        for _ in range(max_steps):
            s = self.env.pos_to_state(r, c)
            a = self.select_action(s)
            nr, nc, reward, done = self.env.transition(r, c, a)
            s2 = self.env.pos_to_state(nr, nc)
            self.update(s, a, reward, s2, done)
            r, c = nr, nc
            total_reward += reward
            steps += 1
            if done:
                break

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode_rewards.append(total_reward)
        self.episode_steps.append(steps)
        return total_reward, steps

    # ------------------------------------------------------------------
    # Policy / value extraction
    # ------------------------------------------------------------------

    def get_policy(self) -> np.ndarray:
        return np.argmax(self.Q, axis=1).reshape(self.env.size, self.env.size)

    def get_value_function(self) -> np.ndarray:
        return np.max(self.Q, axis=1).reshape(self.env.size, self.env.size)

    def run_greedy_episode(self, max_steps: int = 300) -> Tuple[List[Tuple[int, int]], float]:
        """Execute the greedy policy and return the trajectory."""
        r, c = self.env.reset()
        trajectory = [(r, c)]
        total_reward = 0.0

        for _ in range(max_steps):
            s = self.env.pos_to_state(r, c)
            a = self.select_action(s, greedy=True)
            nr, nc, reward, done = self.env.transition(r, c, a)
            r, c = nr, nc
            trajectory.append((r, c))
            total_reward += reward
            if done:
                break

        return trajectory, total_reward


# ======================================================================
# Value Iteration
# ======================================================================

class ValueIterationAgent:
    """
    Exact value iteration — solves V*(s) via Bellman optimality updates.
    Uses the deterministic transition model (no noise applied).
    """

    def __init__(
        self,
        env: GridWorld,
        gamma: float = 0.99,
        theta: float = 1e-6,
    ):
        self.env = env
        self.gamma = gamma
        self.theta = theta

        self.V = np.zeros((env.size, env.size))
        self.iterations = 0
        self.delta_history: List[float] = []

    # ------------------------------------------------------------------

    def _bellman_value(self, r: int, c: int) -> float:
        """Max over actions of expected value."""
        if (r, c) == self.env.goal:
            return self.env.goal_reward
        if (r, c) in self.env.obstacles:
            return 0.0

        values = []
        for a in range(self.env.n_actions):
            nr, nc, reward, done = self.env.deterministic_transition(r, c, a)
            v = reward + (0.0 if done else self.gamma * self.V[nr, nc])
            values.append(v)
        return max(values)

    def sweep(self) -> float:
        """One full Bellman sweep.  Returns max |delta V|."""
        new_V = self.V.copy()
        delta = 0.0

        for r in range(self.env.size):
            for c in range(self.env.size):
                v_new = self._bellman_value(r, c)
                delta = max(delta, abs(v_new - self.V[r, c]))
                new_V[r, c] = v_new

        self.V = new_V
        self.iterations += 1
        self.delta_history.append(delta)
        return delta

    def run_to_convergence(self, max_iter: int = 1000) -> None:
        for _ in range(max_iter):
            delta = self.sweep()
            if delta < self.theta:
                break

    # ------------------------------------------------------------------
    # Policy / value extraction
    # ------------------------------------------------------------------

    def get_policy(self) -> np.ndarray:
        policy = np.zeros((self.env.size, self.env.size), dtype=int)
        for r in range(self.env.size):
            for c in range(self.env.size):
                if (r, c) in self.env.obstacles or (r, c) == self.env.goal:
                    continue
                values = []
                for a in range(self.env.n_actions):
                    nr, nc, reward, done = self.env.deterministic_transition(r, c, a)
                    v = reward + (0.0 if done else self.gamma * self.V[nr, nc])
                    values.append(v)
                policy[r, c] = int(np.argmax(values))
        return policy

    def get_value_function(self) -> np.ndarray:
        return self.V.copy()

    def run_greedy_episode(self, max_steps: int = 300) -> Tuple[List[Tuple[int, int]], float]:
        policy = self.get_policy()
        r, c = self.env.reset()
        trajectory = [(r, c)]
        total_reward = 0.0

        for _ in range(max_steps):
            a = int(policy[r, c])
            nr, nc, reward, done = self.env.transition(r, c, a)
            r, c = nr, nc
            trajectory.append((r, c))
            total_reward += reward
            if done:
                break

        return trajectory, total_reward
