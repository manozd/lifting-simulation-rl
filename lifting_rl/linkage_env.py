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

            def f(x, y=y):
                tck = interpolate.splrep(timestamps, y)
                return np.float32(interpolate.splev(x, tck))

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

        high = np.array([2, 2, 2, 2, 5 * np.pi, 5 * np.pi], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        print("observation_space: ", self.observation_space)

        self.act_low = w_params["ACT_LOW"]
        self.act_high = w_params["ACT_HIGH"]
        self.action_space = spaces.Box(
            low=w_params["ACT_LOW"],
            high=w_params["ACT_HIGH"],
            shape=(w_params["N_LINKS"],),
            dtype=np.float32,
        )
        print("action_space: ", self.action_space)
        self.cur_step = 0
        self.trajectory_points = get_coordinates(path)
        num_frames = self.trajectory_points.shape[0]
        self.trajectory_timestamps = np.array(
            [i * 1.0 / w_params["VIDEO_FPS"] for i in range(num_frames)]
        )

        self.time_step = w_params["TIME_STEP"]
        end_time = round(num_frames / w_params["VIDEO_FPS"], 2)

        self.interpolated_trajectory = get_interpolated(
            self.trajectory_points, self.trajectory_timestamps
        )
        timestamps = [
            np.float32(i * self.time_step)
            for i in range(int(end_time // self.time_step))
        ]
        # self.coordinates = np.array(
        #     [
        #         [
        #             self.interpolated_trajectory[i](t)
        #             for i in range(len(self.interpolated_trajectory))
        #         ]
        #         for t in timestamps
        #     ],
        #     dtype=np.float32,
        # )
        self.coordinates = []
        for t in timestamps:
            t_coords = []
            for i in range(len(self.interpolated_trajectory)):
                x = np.cos(self.interpolated_trajectory[i](t))
                y = np.sin(self.interpolated_trajectory[i](t))
                t_coords.extend([x, y])
            self.coordinates.append(t_coords)

        self.param_vals = w_params["PARAM_VALS"]

        self.u = None
        self.verbose = verbose
        self.reset()

    def reset(self):
        init_coords = self.trajectory_points[0][: self.n_links]
        init_vel = np.array([0] * init_coords.shape[0])
        init_state = np.concatenate((init_coords, init_vel))
        self.state = self._xy(init_state)
        self.angle_state = init_state
        self.cur_step = 0
        return self.state

    def step(self, u):
        self.u = np.clip(u, self.act_low, self.act_high)
        # self.frame += int(TIME_STEP * VIDEO_FPS
        t = np.linspace(0, self.time_step, 2)
        next_step = self.cur_step + 1
        angle_state0 = self.angle_state

        if self.verbose:
            print("=" * 50 + "\n")
            print(f"STEP = {self.cur_step}")
            print(f"\t before state: {self.state}")
            print(f"\t before trj: {self.coordinates[self.cur_step]}")
            print(f"\t control = {u}")

        self.angle_state = odeint(self._rhs, angle_state0, t, args=(self.param_vals,))[
            -1
        ]
        self.state = self._xy(self.angle_state)
        self.cur_step += 1
        is_out_of_bounds = self._is_out_of_bounds()
        # is_end = next_t >= self.end_time

        cost = sum(abs(self.state[2 * self.n_links :])) ** 2 + sum(abs(self.u)) ** 2
        reward = (
            -sum(
                abs(
                    self.state[: 2 * self.n_links]
                    - self.coordinates[next_step][: 2 * self.n_links]
                )
                ** 2
            )
            - 0.1 * cost
        )

        if self.verbose:
            print(f"\t after state = {self.state}")
            print(f"\t after trj = {self.coordinates[next_step]}")
            print(f"\t is_out_of_bounds = {is_out_of_bounds}")
            print(f"\t reward = {reward}")
            print("=" * 50 + "\n")

        terminate = is_out_of_bounds
        if terminate and self.verbose:
            print("*" * 50)
            print("*****" + "TERMINATE" + "*****")
            print("*" * 50)

        return (self.state, reward, False, {})

    def _is_out_of_bounds(self):
        return not self.observation_space.contains(self.state)

    def _normalize_angles(self, q_coords):
        return [((q_coord + np.pi) % (2 * np.pi)) - np.pi for q_coord in q_coords]

    def render(self, mode="human"):
        pass

    def _xy(self, state):
        lengths = [self.param_vals[i] for i in range(1, self.n_links, 2)]
        new_state = []
        for q in state[: self.n_links]:
            new_state.extend([np.cos(q), np.sin(q)])
        new_state.extend(state[self.n_links :])
        return np.array(new_state, dtype=np.float32)

    def _rhs(self, x, t, args):
        arguments = np.hstack((x, self.u, args))
        dx = np.array(
            np.linalg.solve(self.M_func(*arguments), self.F_func(*arguments))
        ).T[0]
        return dx
