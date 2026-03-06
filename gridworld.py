"""
GridWorld environment for the IRL Explorer.
Inspired by the grid world used in Ng & Russell (2000),
"Algorithms for Inverse Reinforcement Learning".
"""

import numpy as np
from typing import List, Tuple, Set, Optional


class GridWorld:
    """
    A stochastic grid world environment.

    States:  N x N grid cells, indexed (row, col) with (0,0) at top-left.
    Actions: 0=Up, 1=Down, 2=Left, 3=Right
    Reward:  goal_reward at goal cell, step_cost elsewhere, 0 at obstacles.
    """

    ACTIONS: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    ACTION_SYMBOLS: List[str] = ["↑", "↓", "←", "→"]
    ACTION_NAMES: List[str] = ["Up", "Down", "Left", "Right"]

    def __init__(
        self,
        size: int = 5,
        goal: Optional[Tuple[int, int]] = None,
        start: Optional[Tuple[int, int]] = None,
        obstacles: Optional[List[Tuple[int, int]]] = None,
        lava_cells: Optional[List[Tuple[int, int]]] = None,
        lava_penalty: float = -1.0,
        mud_cells: Optional[List[Tuple[int, int]]] = None,
        mud_multiplier: float = 3.0,
        ice_cells: Optional[List[Tuple[int, int]]] = None,
        ice_slip_prob: float = 0.5,
        step_cost: float = -0.04,
        goal_reward: float = 1.0,
        noise: float = 0.0,
    ):
        self.size = size
        self.goal = goal if goal is not None else (size - 1, size - 1)
        self.start = start if start is not None else (0, 0)
        self.obstacles: Set[Tuple[int, int]] = set(
            map(tuple, obstacles) if obstacles else []
        )
        self.lava_cells: Set[Tuple[int, int]] = set(
            map(tuple, lava_cells) if lava_cells else []
        )
        self.lava_penalty = lava_penalty

        # Mud: high-friction tiles with multiplied step cost (agent trained without this)
        self.mud_cells: Set[Tuple[int, int]] = set(
            map(tuple, mud_cells) if mud_cells else []
        )
        self.mud_multiplier: float = mud_multiplier
        self.mud_active: bool = False

        # Ice: slippery tiles — with probability ice_slip_prob, agent slides one extra step
        self.ice_cells: Set[Tuple[int, int]] = set(
            map(tuple, ice_cells) if ice_cells else []
        )
        self.ice_slip_prob: float = ice_slip_prob
        self.ice_active: bool = False

        self.step_cost = step_cost
        self.goal_reward = goal_reward
        self.noise = noise  # probability of taking a uniformly random action

        # When True, stepping on lava ends the episode with lava_penalty.
        # Set to False during training so the agent has no concept of lava.
        self.lava_active: bool = False

        self.n_states = size * size
        self.n_actions = 4

    # ------------------------------------------------------------------
    # State / position helpers
    # ------------------------------------------------------------------

    def pos_to_state(self, r: int, c: int) -> int:
        return r * self.size + c

    def state_to_pos(self, s: int) -> Tuple[int, int]:
        return divmod(s, self.size)

    def is_valid(self, r: int, c: int) -> bool:
        return (
            0 <= r < self.size
            and 0 <= c < self.size
            and (r, c) not in self.obstacles
        )

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def transition(
        self, r: int, c: int, action: int
    ) -> Tuple[int, int, float, bool]:
        """
        Take one step.  Returns (next_r, next_c, reward, done).
        If noise > 0, the actual action is random with that probability.
        Ice slip: if departing from an ice cell, agent may slide one extra step.
        Mud: landing on a mud cell multiplies the step cost.
        """
        if self.noise > 0 and np.random.random() < self.noise:
            action = np.random.randint(self.n_actions)

        # Ice slip: departing from an ice cell may cause an extra slide
        slip = (
            self.ice_active
            and (r, c) in self.ice_cells
            and np.random.random() < self.ice_slip_prob
        )

        dr, dc = self.ACTIONS[action]
        nr, nc = r + dr, c + dc
        if not self.is_valid(nr, nc):
            nr, nc = r, c  # bump — stay in place

        # Apply extra slide if slipping on ice
        if slip:
            nr2, nc2 = nr + dr, nc + dc
            if self.is_valid(nr2, nc2):
                nr, nc = nr2, nc2

        # Lava check — only active during test/evaluation, not during training
        if self.lava_active and (nr, nc) in self.lava_cells:
            return nr, nc, self.lava_penalty, True

        done = (nr, nc) == self.goal
        if done:
            reward = self.goal_reward
        elif self.mud_active and (nr, nc) in self.mud_cells:
            reward = self.step_cost * self.mud_multiplier
        else:
            reward = self.step_cost
        return nr, nc, reward, done

    def deterministic_transition(
        self, r: int, c: int, action: int
    ) -> Tuple[int, int, float, bool]:
        """Noise-free transition — used by Value Iteration."""
        dr, dc = self.ACTIONS[action]
        nr, nc = r + dr, c + dc
        if not self.is_valid(nr, nc):
            nr, nc = r, c
        done = (nr, nc) == self.goal
        reward = self.goal_reward if done else self.step_cost
        return nr, nc, reward, done

    def reset(self) -> Tuple[int, int]:
        return self.start

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def reward_matrix(self) -> np.ndarray:
        """Ground-truth reward for each cell (for display only)."""
        R = np.full((self.size, self.size), self.step_cost)
        R[self.goal] = self.goal_reward
        for r, c in self.obstacles:
            R[r, c] = 0.0
        return R
