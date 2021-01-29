import sympy
from gym import spaces
from sympy import symbols
from sympy.physics.mechanics import *
import numpy as np
from scipy.integrate import odeint

LINK_1_LEN = 0.4
LINK_1_M = 1

LINK_2_LEN = 0.4
LINK_2_M = 1

LINK_3_LEN = 0.6
LINK_3_M = 1

LINK_4_LEN = 0.4
LINK_4_M = 1

LINK_5_LEN = 0.4
LINK_5_M = 1


class ActionSpace:
    def __init__(self):
        pass


class LinkageEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        low = np.array(
            [
                0.0,
                np.pi / 2,
                -np.pi / 4,
                -np.pi / 2,
                -np.pi / 2,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        high = np.array(
            [np.pi / 2, 3 * np.pi / 2, np.pi / 2, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        low = np.full(10, -1.0, dtype=np.float32)
        high = -low
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.state = None

    def reset(self):
        self.state = np.array(
            [
                0.0,
                np.pi / 2,
                -np.pi / 4,
                -np.pi / 2,
                -np.pi / 2,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

    def step(self):
        state0 = self.state
        self.state = odeint(rhs, state0, 1, args=(parameter_vals,))
        terminate = self._terminate()
        rewrad = 1
        return (self.state, reward, terminate, {})

    def _terminate(self):
        return self.observation_space.contains()

    def render(self):
        pass

    def _rhs(x, t, args):
        u = [0, 0, 0, 0, 0]
        arguments = np.hstack((x, u, args))
        dx = np.array(np.linalg.solve(M_func(*arguments), F_func(*arguments))).T[0]
        return dx
