import csv

import gym
from gym import spaces
import numpy as np
from numpy import linspace, cos, sin
from scipy import interpolate
from sympy import lambdify
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from n_linkage import kane

N_LINKS = 2

# GOAL_POS = np.array(
#     [np.pi / 4, 3 * np.pi / 4, np.pi / 2, -np.pi / 4, -np.pi / 4, 0, 0, 0, 0, 0], dtype=np.float32
# )

# PARAM_VALS = np.array([9.81, 0.4, 1, 0.4, 1, 0.6, 1, 0.4, 1, 0.4, 1], dtype=np.float32)
# INIT_STATE = np.array([np.pi / 4, 3 * np.pi / 4, np.pi / 2, -np.pi / 4, -np.pi / 4, 0, 0, 0, 0, 0], dtype=np.float32)
# OBS_LOW = [
#     0,
#     3 * np.pi / 8,
#     -np.pi / 2,
#     -5 * np.pi / 8,
#     -5 * np.pi / 8,
#     -5*np.pi,
#     -5*np.pi,
#     -5*np.pi,
#     -5*np.pi,
#     -5*np.pi,
# ]
# OBS_HIGH = [
#     5 * np.pi / 8,
#     3 * np.pi / 2,
#     5 * np.pi / 8,
#     np.pi / 2,
#     np.pi / 2,
#     5*np.pi,
#     5*np.pi,
#     5*np.pi,
#     5*np.pi,
#     5*np.pi,
# ]

GOAL_POS = np.array(
    [np.pi / 4, 3 * np.pi / 4, 0, 0, np.pi / 4, 3 * np.pi / 4, 0, 0], dtype=np.float32
)

PARAM_VALS = np.array([9.81, 0.4, 1, 0.4, 1], dtype=np.float32)
INIT_STATE = np.array(
    [np.pi / 4, 3 * np.pi / 4, 0, 0, np.pi / 4, 3 * np.pi / 4, 0, 0], dtype=np.float32
)
OBS_LOW = [
    0,
    3 * np.pi / 8,
    -5 * np.pi,
    -5 * np.pi,
    0,
    3 * np.pi / 8,
    -5 * np.pi,
    -5 * np.pi,
]

OBS_HIGH = [
    5 * np.pi / 8,
    3 * np.pi / 2,
    5 * np.pi,
    5 * np.pi,
    5 * np.pi / 8,
    3 * np.pi / 2,
    5 * np.pi,
    5 * np.pi,
]


ACT_LOW = -20
ACT_HIGH = 20

PATH = "/home/mans/Documents/skeleton_angles.csv"
TIME_STEP = 0.01
VIDEO_FPS = 30
H_LOAD_STEP = 16

T = np.linspace(0, TIME_STEP, 2)
LENGTHS = [0.4, 0.4, 0.6, 0.4, 0.4]

# plt.ion()
# # the number of pendulum bobs
# numpoints = 2
# # first set up the figure, the axis, and the plot elements we want to animate
# plt.ion()
# fig = plt.figure()

# axes = fig.add_subplot(111)

# # set the limits based on the motion

# xmin = -2
# xmax = 2

# # create the axes
# ax = plt.axes(xlim=(-0.5, 1), ylim=(-0.1, 2), aspect='equal')

# # display the current time
# time_text = ax.text(0.04, 0.9, '', transform=ax.transAxes)

# plt.show()


class Coordinates:
    def __init__(self, path, fps=30):
        self.path = path
        self.fps = fps
        self.tcks = self._get_tcks()
        self.n = len(self.tcks)
        with open(self.path) as fr:
            self.t_stop = round(len(fr.readlines()) / self.fps, 1)

    def _get_tcks(self):
        with open(self.path) as fr:
            reader = csv.reader(fr)
            yy = np.array([[float(i) for i in row] for row in reader])
        x = np.array([i / self.fps for i in range(len(yy))])
        tcks = [interpolate.splrep(x, yy[:, i]) for i in range(5)]
        return tcks

    def q(self, t):
        return np.array([interpolate.splev(t, self.tcks[i]) for i in range(self.n)])

    def q_expanded(self, step):
        num = int((self.t_stop) / step)
        return np.array(
            [
                [interpolate.splev(t, self.tcks[i]) for i in range(self.n)]
                for t in linspace(0, self.t_stop, num)
            ]
        )

    def xy_expanded(self, step, l_params):
        num = int((self.t_stop) / step)
        return np.array(
            [
                [
                    [
                        l_params[i] * cos(interpolate.splev(t, self.tcks[i])),
                        l_params[i] * sin(interpolate.splev(t, self.tcks[i])),
                    ]
                    for i in range(self.n)
                ]
                for t in linspace(0, self.t_stop, num)
            ]
        )


def get_joint_coordinates(qs, ls):
    return np.array(
        [[l * cos(q), l * sin(q)] for q, l in zip(qs, ls)], dtype=np.float32
    )


class LinkageEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        M, F, params = kane(n=N_LINKS)
        self.M_func = lambdify(params, M)
        self.F_func = lambdify(params, F)
        print(M, F)

        M, F, params = kane(n=N_LINKS, hands_load=True)
        self.M_func_hl = lambdify(params, M)
        self.F_func_hl = lambdify(params, F)

        self.low = np.array(OBS_LOW, dtype=np.float32)
        self.high = np.array(OBS_HIGH, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=self.low, high=self.high, dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=ACT_LOW, high=ACT_HIGH, shape=(N_LINKS,), dtype=np.float32
        )
        self.t_step = 0
        C = Coordinates(PATH)
        self.q = C.q_expanded(TIME_STEP)
        self.reset()

    def reset(self):
        self.state = np.hstack([self.q[0][:2], [0, 0], self.q[0][:2], [0, 0]])
        self.t_step = 0
        return self.state

    def step(self, u):
        self.u = np.clip(u, ACT_LOW, ACT_HIGH)
        print(self.u)
        x0 = self.state[4:]
        x1 = odeint(self._rhs, x0, T, args=(PARAM_VALS,))[-1]
        self.t_step += 1
        self.state = np.hstack([x0, x1])
        state = np.hstack([self.state[:2], self.state[4:6]])
        ref_state = np.hstack([self.q[self.t_step - 1][:2], self.q[self.t_step][:2]])
        reward = -sum(np.power(state - ref_state, 2)) - pow(sum(abs(u)), 2)
        terminate = self._terminate()
        return (self.state, reward, False, {})

    def _terminate(self):
        pass

    def render(self, mode="human"):
        pass

    def _normalize(self, q):
        return (q - self.low) / (self.high - self.low)

    def _rhs(self, x, t, args):
        arguments = np.hstack((x, self.u, args))
        dx = np.array(
            np.linalg.solve(self.M_func(*arguments), self.F_func(*arguments))
        ).T[0]
        return dx

    def render(self, filename=None, mode="human"):
        pass
