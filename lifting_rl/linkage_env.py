import csv

import gym
from gym import spaces
import numpy as np
from sympy import lambdify
from scipy.integrate import odeint
from lifting_rl.n_linkage import kane
from scipy import interpolate

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
TIME_STEP = 0.01
VIDEO_FPS = 30


def get_coordinates(path):
    coordinates = []
    with open(path) as fr:
        reader = csv.reader(fr)
        for idx, row in enumerate(reader):
            coordinates.append([float(i) for i in row])
    return np.array(coordinates)


def get_interpolated(coords, timestamps):
    interpolated_coords = []
    for i in range(coords.shape[1]):
        y = coords[:, i]
        f = interpolate.interp1d(timestamps, y)
        interpolated_coords.append(f)
    return interpolated_coords


class LinkageEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, path: str, verbose: bool = False):
        M, F, params = kane(n=N_LINKS)
        self.M_func = lambdify(params, M)
        self.F_func = lambdify(params, F)
        low = np.array(OBS_LOW, dtype=np.float32)
        high = np.array(OBS_HIGH, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        print('observation_space: ', self.observation_space )

        self.action_space = spaces.Box(
            low=ACT_LOW, high=ACT_HIGH, shape=(N_LINKS,), dtype=np.float32
        )
        print('action_space: ', self.action_space)
        self.cur_time = 0
        self.trajectory_points = get_coordinates(path)
        num_frames = self.trajectory_points.shape[0]
        self.trajectory_timestamps = np.array([i * 1.0 / VIDEO_FPS for i in range(num_frames)])
        self.end_time = self.trajectory_timestamps.max()

        self.interpolated_trajectory = get_interpolated(self.trajectory_points, self.trajectory_timestamps)
        self.u = None
        self.verbose = verbose
        self.reset()

    def reset(self):
        init_coords = self.trajectory_points[0]
        init_vel = np.array([0] * init_coords.shape[0])
        init_state = np.concatenate((init_coords, init_vel))
        self.state = init_state
        self.cur_time = 0
        return self.state

    def step(self, u):
        self.u = u
        # self.frame += int(TIME_STEP * VIDEO_FPS
        t = np.linspace(0, TIME_STEP, 10)
        next_t = self.cur_time + TIME_STEP
        state0 = self.state

        trj_state = np.array([
            self.interpolated_trajectory[i](self.cur_time) for i in range(len(self.interpolated_trajectory))
        ])

        if self.verbose:
            print('='*50 + '\n')
            print(f'START STEP AT t = {self.cur_time}')
            print(f'\t before state: {self.state}')
            print(f'\t before trj: {trj_state}')
            print(f'\t control = {u}')

        next_coordinates = np.array([
            self.interpolated_trajectory[i](next_t) for i in range(len(self.interpolated_trajectory))
        ])
        self.state = odeint(self._rhs, state0, t, args=(PARAM_VALS,))[-1]    
        self.cur_time += TIME_STEP
        is_out_of_bounds = self._is_out_of_bounds()
        is_end = next_t >= self.end_time

        reward = -np.exp(sum(abs(self.state[:5] - next_coordinates)))
        if is_out_of_bounds:
            reward -= 1000
        
        if self.verbose:
            print(f'\t after state = {self.state}')
            print(f'\t after trj = {next_coordinates}')
            print(f'\t is_out_of_bounds = {is_out_of_bounds}, is_end = {is_end}')
            print(f'\t reward = {reward}')
            print('='*50 + '\n')

        terminate = is_out_of_bounds or is_end

        if terminate and self.verbose:
            print('*'*50)
            print('*****' + 'TERMINATE' + '*****')
            print('*'*50)

        return (self.state, reward, terminate, {})

    def _is_out_of_bounds(self):
        return not self.observation_space.contains(self.state)

    def render(self):
        pass

    def _rhs(self, x, t, args):
        arguments = np.hstack((x, self.u, args))
        dx = np.array(
            np.linalg.solve(self.M_func(*arguments), self.F_func(*arguments))
        ).T[0]
        return dx
