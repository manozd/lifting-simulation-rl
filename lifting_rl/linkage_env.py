import csv

import gym
from gym import spaces
import numpy as np
from sympy import lambdify
from scipy.integrate import odeint
from lifting_rl.n_linkage import kane
from scipy import interpolate


def get_coordinates(path):
    coordinates = []
    with open(path) as fr:
        reader = csv.reader(fr)
        for idx, row in enumerate(reader):
            coordinates.append([float(i) for i in row])
    return np.array(coordinates)


def get_interpolated(coords, timestamps, mode="spline"):
    interpolated_coords = []
    for i in range(coords.shape[1]):
        if mode == "default":
            y = coords[:, i]
            f = interpolate.interp1d(timestamps, y)
        if mode == "spline":
            y = coords[:, i]

            def f(x):
                tck = interpolate.splrep(timestamps, y)
                return interpolate.splev(timestamps, tck)

        interpolated_coords.append(f)
    return interpolated_coords


class LinkageEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, path: str, w_params: dict, verbose: bool = False):
        self.n_links = w_params["N_LINKS"]
        M, F, m_params = kane(n=w_params["N_LINKS"])
        self.M_func = lambdify(m_params, M)
        self.F_func = lambdify(m_params, F)
        self.observation_space = spaces.Box(
            low=w_params["OBS_LOW"], high=w_params["OBS_HIGH"], dtype=np.float32
        )
        print("observation_space: ", self.observation_space)

        self.action_space = spaces.Box(
            low=w_params["ACT_LOW"],
            high=w_params["ACT_HIGH"],
            shape=(w_params["N_LINKS"],),
            dtype=np.float32,
        )
        print("action_space: ", self.action_space)
        self.cur_time = 0
        self.trajectory_points = get_coordinates(path)
        num_frames = self.trajectory_points.shape[0]
        self.trajectory_timestamps = np.array(
            [i * 1.0 / w_params["VIDEO_FPS"] for i in range(num_frames)]
        )
        self.end_time = self.trajectory_timestamps.max()

        self.interpolated_trajectory = get_interpolated(
            self.trajectory_points, self.trajectory_timestamps
        )

        self.time_step = w_params["TIME_STEP"]
        self.param_vals = w_params["PARAM_VALS"]

        self.u = None
        self.verbose = verbose
        self.reset()

    def reset(self):
        init_coords = self.trajectory_points[0][: self.n_links]
        init_vel = np.array([0] * init_coords.shape[0])
        init_state = np.concatenate((init_coords, init_vel))
        self.state = init_state
        self.cur_time = 0
        return self.state

    def step(self, u):
        self.u = u
        # self.frame += int(TIME_STEP * VIDEO_FPS
        t = np.linspace(0, self.time_step, 10)
        next_t = self.cur_time + self.time_step
        state0 = self.state

        trj_state = np.array(
            [
                self.interpolated_trajectory[i](self.cur_time)
                for i in range(len(self.interpolated_trajectory))
            ]
        )
        if self.verbose:
            print("=" * 50 + "\n")
            print(f"START STEP AT t = {self.cur_time}")
            print(f"\t before state: {self.state}")
            print(f"\t before trj: {trj_state}")
            print(f"\t control = {u}")

        next_coordinates = np.array(
            [
                self.interpolated_trajectory[i](next_t)
                for i in range(len(self.interpolated_trajectory))
            ]
        )
        self.state = odeint(self._rhs, state0, t, args=(self.param_vals,))[-1]
        self.cur_time += self.time_step
        is_out_of_bounds = self._is_out_of_bounds()
        is_end = next_t >= self.end_time

        cost = sum(abs(self.state[: self.n_links])) + sum(abs(u))
        reward = (
            -sum(abs(self.state[: self.n_links] - next_coordinates[: self.n_links]))
            ** 2
            - 0.1 * cost ** 2
        )

        if self.verbose:
            print(f"\t after state = {self.state}")
            print(f"\t after trj = {next_coordinates}")
            print(f"\t is_out_of_bounds = {is_out_of_bounds}, is_end = {is_end}")
            print(f"\t reward = {reward}")
            print("=" * 50 + "\n")

        terminate = is_out_of_bounds or is_end

        if terminate and self.verbose:
            print("*" * 50)
            print("*****" + "TERMINATE" + "*****")
            print("*" * 50)

        return (self.state, reward, is_end, {})

    def _is_out_of_bounds(self):
        return not self.observation_space.contains(self.state)

    def _normalize_angles(self, q_coords):
        return [((q_coord + np.pi) % (2 * np.pi)) - np.pi for q_coord in q_coords]

    def render(self):
        pass

    def _xy_coords(self, q):
        lengths = [self.param_vals[i] for i in range(1, self.n_links, 2)]

    def _rhs(self, x, t, args):
        arguments = np.hstack((x, self.u, args))
        dx = np.array(
            np.linalg.solve(self.M_func(*arguments), self.F_func(*arguments))
        ).T[0]
        return dx
