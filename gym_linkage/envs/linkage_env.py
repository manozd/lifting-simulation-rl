import gym
from gym import spaces
import numpy as np
from sympy import Dummy, lambdify
from scipy.integrate import odeint


class ActionSpace:
    def __init__(self):
        pass


class LinkageEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, M, F, params, param_vals):
        self.param_vals = np.column_stack(
            (param_vals.links.lengths, param_vals.links.masses)
        ).reshape(-1)
        self.param_vals.insert(0, param_vals.g)

        dummy_symbols = [Dummy() for i in range(15)]
        self.M = lambdify(dummy_symbols + params, M)
        self.F = lambdify(dummy_symbols + params, F)
        self.init_state = param_vals.init

        low = param_vals.obs_limits.low * np.pi
        high = param_vals.obs_limits.high * np.pi
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        low = param_vals.act_limits.low
        high = param_vals.act_limits.high
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.state = None

    def reset(self):
        self.state = np.array(self.init * np.pi, dtype=np.float32)

    def step(self, u):
        t = linspace(0, 0.5, 15)
        state0 = self.state
        self.state = odeint(self._rhs, state0, t, args=(self.parameter_vals,))
        terminate = self._terminate()
        reward = 1
        return (self.state, reward, terminate, {})

    def _terminate(self):
        return self.observation_space.contains()

    def render(self):
        pass

    def _rhs(x, t, args):
        arguments = np.hstack((x, u, args))
        dx = np.array(np.linalg.solve(self.M(*arguments), self.F(*arguments))).T[0]
        return dx
