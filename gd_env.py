"""Fast numerical Geometry Dash-like simulator; no screen, input or wall clock.

Units: cube width = 1. Coordinates: x right, y up; cube position is bottom-left.
Obstacle rows: [x, width, height, kind], kind 0 = triangular spike, 1 = block.
This is deliberately simplified physics, not a replica of the commercial game.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class GeometryDashEnv(gym.Env):
    metadata = {'render_modes': []}
    SPEED = 6.0
    GRAVITY = 32.0
    JUMP_SPEED = 12.0
    MAX_FALL_SPEED = 20.0
    SIZE = 1.0
    DT = 1.0 / 30.0
    SUBSTEPS = 4
    LOOKAHEAD = 3

    def __init__(self, level_length=200.0, max_episode_steps=3000, obstacles=None):
        super().__init__()
        if not np.isfinite(level_length) or level_length < 20:
            raise ValueError('level_length must be finite and >= 20.')
        if not isinstance(max_episode_steps, int) or max_episode_steps < 1:
            raise ValueError('max_episode_steps must be a positive integer.')
        self.level_length = float(level_length)
        self.max_episode_steps = max_episode_steps
        self._fixed_obstacles = None if obstacles is None else self._validate(obstacles)
        self.action_space = spaces.Discrete(2)  # 0 release, 1 hold jump
        # y, vy, vx, grounded, remaining distance + 3*(dx,w,h,kind,present).
        self.observation_space = spaces.Box(-1.0, 1.0, (20,), np.float32)
        self._done = True
        self.obstacles = np.empty((0, 4), dtype=np.float64)

    def _validate(self, obstacles):
        result = np.asarray(obstacles, dtype=np.float64)
        if result.size == 0:
            return np.empty((0, 4), dtype=np.float64)
        if result.ndim != 2 or result.shape[1] != 4 or not np.isfinite(result).all():
            raise ValueError('Obstacles must be finite rows [x,width,height,kind].')
        result = result[np.argsort(result[:, 0])].copy()
        if (np.any(result[:, 0] < 2) or np.any(result[:, 1:3] <= 0)
                or np.any(result[:, 0] + result[:, 1] >= self.level_length)
                or not np.isin(result[:, 3], [0, 1]).all()):
            raise ValueError('Invalid obstacle dimensions, location or kind.')
        if np.any(result[1:, 0] < result[:-1, 0] + result[:-1, 1]):
            raise ValueError('Obstacles must not overlap horizontally.')
        return result

    def _generate_level(self):
        # Isolated hazards, long safe gaps, blocks lower than the jump apex.
        rows = []
        x = 10.0
        while x < self.level_length - 10:
            kind = int(self.np_random.random() < 0.3)
            width = float(self.np_random.uniform(0.8, 1.4)) if kind else 0.65
            height = 1.0 if kind else 0.8
            rows.append((x, width, height, kind))
            x += width + float(self.np_random.uniform(6.0, 10.0))
        return np.asarray(rows, dtype=np.float64).reshape(-1, 4)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.obstacles = (self._generate_level() if self._fixed_obstacles is None
                          else self._fixed_obstacles.copy())
        self._rights = self.obstacles[:, 0] + self.obstacles[:, 1]
        self.x = self.y = self.vy = 0.0
        self.grounded = True
        self.steps = 0
        self._done = False
        return self._observation(), self._info('')

    def _observation(self):
        obs = np.zeros(20, dtype=np.float32)
        obs[:5] = (self.y / 10, self.vy / self.MAX_FALL_SPEED, self.SPEED / 10,
                   float(self.grounded), max(0, 1-self.x/self.level_length))
        start = np.searchsorted(self._rights, self.x, side='right')
        for i, (x, width, height, kind) in enumerate(self.obstacles[start:start+3]):
            obs[5+5*i:10+5*i] = ((x-self.x-self.SIZE)/20, width/4,
                                 height/4, 1 if kind == 0 else -1, 1)
        return np.clip(obs, -1, 1)

    def _info(self, reason):
        return {'progress': min(self.x/self.level_length, 1.0),
                'is_success': reason == 'complete', 'death_reason': reason,
                'x': self.x, 'y': self.y}

    def step(self, action):
        if self._done:
            raise RuntimeError('Call reset() before stepping a finished episode.')
        if not self.action_space.contains(action):
            raise ValueError(f'Invalid action: {action!r}')
        old_x = self.x
        reason = ''
        dt = self.DT / self.SUBSTEPS
        for _ in range(self.SUBSTEPS):
            if action == 1 and self.grounded:
                self.vy = self.JUMP_SPEED
                self.grounded = False
            old_y = self.y
            self.vy = max(self.vy-self.GRAVITY*dt, -self.MAX_FALL_SPEED)
            self.x += self.SPEED*dt
            self.y += self.vy*dt
            self.grounded = False
            start = np.searchsorted(self._rights, self.x, side='right')
            for x, width, height, kind in self.obstacles[start:]:
                if x >= self.x+self.SIZE:
                    break
                if kind == 0:
                    # Triangle/AABB test: maximum spike height beneath cube span.
                    peak_x = x + width/2
                    nearest = min(max(peak_x, self.x), self.x+self.SIZE)
                    spike_top = height * max(0.0, 1-abs(nearest-peak_x)/(width/2))
                    if self.y < spike_top and self.y+self.SIZE > 0:
                        reason = 'spike'
                        break
                elif self.y < height and self.y+self.SIZE > 0:
                    if self.vy <= 0 and old_y >= height-1e-9:
                        self.y = height
                        self.vy = 0.0
                        self.grounded = True
                    else:
                        reason = 'block'
                        break
            if self.y <= 0:
                self.y = self.vy = 0.0
                self.grounded = True
            if reason:
                break
            if self.x >= self.level_length:
                reason = 'complete'
                break
        self.steps += 1
        terminated = bool(reason)
        truncated = self.steps >= self.max_episode_steps and not terminated
        self._done = terminated or truncated
        reward = (self.x-old_x)/self.level_length
        if reason:
            reward += 1.0 if reason == 'complete' else -1.0
        return self._observation(), float(reward), terminated, truncated, self._info(reason)
