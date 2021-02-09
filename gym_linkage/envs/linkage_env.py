import csv

import gym
from gym import spaces
import numpy as np
from sympy import lambdify
from scipy.integrate import odeint
from n_linkage import kane

N_LINKS = 5

GOAL_POS = np.array(
    [np.pi / 4, 3 * np.pi / 4, np.pi / 2, -np.pi / 4, -np.pi / 4, 0, 0, 0, 0, 0]
)

PARAM_VALS = np.array([9.81, 0.4, 1, 0.4, 1, 0.6, 1, 0.4, 1, 0.4, 1])
INIT_STATE = [np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0, 0, 0, 0, 0]
OBS_LOW = [
    0,
    3 * np.pi / 8,
    -np.pi / 2,
    -5 * np.pi / 8,
    -5 * np.pi / 8,
    -np.pi,
    -np.pi,
    -np.pi,
    -np.pi,
    -np.pi,
]
OBS_HIGH = [
    5 * np.pi / 8,
    3 * np.pi / 2,
    5 * np.pi / 8,
    np.pi / 2,
    np.pi / 2,
    np.pi,
    np.pi,
    np.pi,
    np.pi,
    np.pi,
]

ACT_LOW = -1
ACT_HIGH = 1

PATH = "/home/mans/Documents/skeleton_angles.csv"
TIME_STEP = 0.1
VIDEO_FPS = 30


def get_coordinates(path):
    coordinates = []
    with open(path) as fr:
        reader = csv.reader(fr)
        for idx, row in enumerate(reader):
            coordinates.append([float(i) for i in row])
    return coordinates


class LinkageEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        M, F, params = kane(n=N_LINKS)
        self.M_func = lambdify(params, M)
        self.F_func = lambdify(params, F)
        low = np.array(OBS_LOW, dtype=np.float32)
        high = np.array(OBS_HIGH, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.action_space = spaces.Box(
            low=ACT_LOW, high=ACT_HIGH, shape=(N_LINKS,), dtype=np.float32
        )
        self.frame = 0
        self.coordinates = get_coordinates(PATH)
        self.u = None
        self.reset()

    def reset(self):
        self.state = np.array(INIT_STATE, dtype=np.float32)
        self.frame = 0
        return self.state

    def step(self, u):
        self.u = u
        self.frame += int(TIME_STEP * VIDEO_FPS)
        t = np.linspace(0, TIME_STEP, 10)
        state0 = self.state
        self.state = odeint(self._rhs, state0, t, args=(PARAM_VALS,))[-1]
        terminate = self._terminate()
        reward = -np.exp(sum(abs(self.state[:5] - self.coordinates[self.frame])))
        if terminate:
            reward -= 1000
        return (self.state, reward, terminate, {})

    def _terminate(self):
        return not self.observation_space.contains(self.state)

    def render(self):
        pass

    def _rhs(self, x, t, args):
        arguments = np.hstack((x, self.u, args))
        dx = np.array(
            np.linalg.solve(self.M_func(*arguments), self.F_func(*arguments))
        ).T[0]
        return dx
