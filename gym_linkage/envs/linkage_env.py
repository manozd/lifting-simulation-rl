import gym
from gym import spaces
import numpy as np
from sympy import Dummy, lambdify
from scipy.integrate import odeint
from n_linkage import kane

N_LINKS = 5

PARAM_VALS = np.array([9.81, 0.4, 1, 0.4, 1, 0.6, 1, 0.4, 1, 0.4, 1])

INIT_STATE = [np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0, 0, 0, 0, 0]

OBS_LOW = [
    0,
    np.pi / 2,
    -np.pi / 4,
    -np.pi / 2,
    -np.pi / 2,
    -np.pi,
    -np.pi,
    -np.pi,
    -np.pi,
    -np.pi,
]
OBS_HIGH = [
    np.pi / 2,
    3 * np.pi / 2,
    np.pi / 2,
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


class LinkageEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        M, F, params = kane(n=N_LINKS)

        # q, u, and f so 3*N_LINKS symbols
        self.M = lambdify(params, M)
        self.F = lambdify(params, F)

        low = np.array(OBS_LOW, dtype=np.float32)
        high = np.array(OBS_HIGH, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.action_space = spaces.Box(
            low=ACT_LOW, high=ACT_HIGH, shape=(N_LINKS,), dtype=np.float32
        )
        self.reset()

    def reset(self):
        self.state = np.array(INIT_STATE, dtype=np.float32)
        return self.state

    def step(self, action):
        t = np.linspace(0, 0.5, 15)
        state0 = self.state
        print(state0)
        print(t)
        print(PARAM_VALS)
        self.state = odeint(self._rhs, state0, t, args=(PARAM_VALS,))
        terminate = self._terminate()
        reward = 1
        return (self.state, reward, terminate, {})

    def _terminate(self):
        return self.observation_space.contains(self.state)

    def render(self):
        pass

    def _rhs(self, x, t, args):
        u = [1, 1, 1, 1, 1]
        arguments = np.hstack((x, u, args))
        dx = np.array(np.linalg.solve(self.M(*arguments), self.F(*arguments))).T[0]
        return dx
