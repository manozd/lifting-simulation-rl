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
        self.viewer = None
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
        self.cur_step = 0

        # self.trajectory_points = get_coordinates(path)
        # num_frames = self.trajectory_points.shape[0]
        # self.trajectory_timestamps = np.array(
        #     [i * 1.0 / w_params["VIDEO_FPS"] for i in range(num_frames)]
        # )

        self.time_step = w_params["TIME_STEP"]
        # end_time = round(num_frames / w_params["VIDEO_FPS"], 2)

        # self.interpolated_trajectory = get_interpolated(
        #     self.trajectory_points, self.trajectory_timestamps
        # )
        # timestamps = [
        #     np.float32(i * self.time_step)
        #     for i in range(int(end_time // self.time_step))
        # ]

        # coordinates = [
        #     [
        #         self.interpolated_trajectory[i](t)
        #         for i in range(len(self.interpolated_trajectory))
        #     ]
        #     for t in timestamps
        # ]

        # self.coordinates = np.array(
        #     [self._transform_state(step_coord) for step_coord in coordinates],
        #     dtype=np.float32,
        # )

        # self.coordinates = []
        # for t in timestamps:
        #     t_coords = []
        #     for i in range(len(self.interpolated_trajectory)):
        #         x = np.cos(self.interpolated_trajectory[i](t))
        #         y = np.sin(self.interpolated_trajectory[i](t))
        #         t_coords.extend([x, y])
        #     self.coordinates.append(t_coords)

        self.param_vals = w_params["PARAM_VALS"]
        self.goal_pos = np.random.uniform(low=np.pi/10, high=4*np.pi/7, size=(2,))
        self.act_max = w_params["ACT_HIGH"]
        self.u = None
        self.prev_pos_reward = 0
        self.verbose = verbose
        self.speed_max = w_params["OBS_HIGH"][self.n_links]
        self.reset()

    def reset(self):
        # init_coords = self.trajectory_points[0][: self.n_links]
        init_coords = np.random.uniform(low=np.pi / 10, high=4 * np.pi / 7, size=(2,))
        init_vel = np.random.uniform(low=-8, high=8, size=(2,))
        init_state = np.concatenate((init_coords, init_vel))
        self.state = init_state
        self.prev_pos_reward = 0
        self.goal_pos = np.random.uniform(low=np.pi/10, high=4*np.pi/7, size=(2,))
        self.cur_step = 0
        return self._get_obs()

    def step(self, u):
        print(u)
        self.u = u * self.act_max
        print(u)
        t = np.linspace(0, self.time_step, 2)
        next_step = self.cur_step + 1
        state0 = self.state

        if self.verbose:
            print("=" * 50 + "\n")
            print(f"STEP = {self.cur_step}")
            print(f"\t before state: {self.state}")
            print(f"\t before trj: {self.coordinates[self.cur_step]}")
            print(f"\t control = {u}")

        self.state = odeint(self._rhs, state0, t, args=(self.param_vals,))[-1]
        w = np.clip(self.state[self.n_links:], -self.speed_max, self.speed_max)
        self.state[self.n_links:] = w
        self.cur_step += 1

        pos_reward = - sum(abs(self.state[: self.n_links] - self.goal_pos[: self.n_links])) ** 2
        speed_reward = -sum(abs(self.state[self.n_links :])) ** 2
        control_reward = -sum(abs(self.u)) ** 2
        reward = pos_reward
        self.prev_pos_reward = pos_reward
        if self.verbose:
            print(f"\t after state = {self.state}")
            print(f"\t after trj = {self.coordinates[next_step]}")
            print(f"\t is_out_of_bounds = {is_out_of_bounds}")
            print(f"\t reward = {reward}")
            print("=" * 50 + "\n")

        is_end = self.cur_step == 288

        terminate = self._is_out_of_bounds() or is_end or self._goal_reached()
        if self._is_out_of_bounds():
            reward = -100
        
        if self._goal_reached():
            reward += 100

        if terminate and self.verbose:
            print("*" * 50)
            print("*****" + "TERMINATE" + "*****")
            print("*" * 50)
        return self._get_obs(), reward, terminate, {}

    def _is_out_of_bounds(self):
        return not self.observation_space.contains(self.state)

    def _normalize(self, vector, max):
        return np.array(
            [coord / np.sqrt(sum(vector ** 2)) for coord in vector], dtype=np.float32
        )

    def _goal_reached(self):
        return sum(abs(self.state[: self.n_links] - self.goal_pos[: self.n_links])) < 0.04

    def _get_obs(self):
        theta1, theta2, thetadot1, thetadot2 = self.state
        return np.array([np.cos(theta1), np.sin(theta1), np.cos(theta2), np.sin(theta2), np.cos(self.goal_pos[0]), np.sin(self.goal_pos[0]), np.cos(self.goal_pos[1]), np.sin(self.goal_pos[1]),   thetadot1, thetadot2])

    def _rhs(self, x, t, args):
        arguments = np.hstack((x, self.u, args))
        dx = np.array(
            np.linalg.solve(self.M_func(*arguments), self.F_func(*arguments))
        ).T[0]
        return dx

    def render(self, mode="human"):
        from gym.envs.classic_control import rendering

        s = self.state
        g = self.goal_pos
        if self.viewer is None:
            self.viewer = rendering.Viewer(400, 400)
            bound = 0.4 + 0.4 + 0.2  # 2.2 for default
            self.viewer.set_bounds(-bound, bound, -bound, bound)

        if s is None:
            return None

        p1_1 = [0.4 * np.cos(s[0]), 0.4 * np.sin(s[0])]
        p2_1 = [p1_1[0] + 0.4 * np.cos(s[0]), p1_1[1] + 0.4 * np.sin(s[0])]


        p1_2 = [0.4 * np.cos(g[0]), 0.4 * np.sin(g[0])]
        p2_2 = [p1_2[0] + 0.4 * np.cos(g[0]), p1_2[1] + 0.4 * np.sin(g[0])]


        xys_1 = np.array([[0, 0], p1_1, p2_1])
        thetas_1 = [s[0] % (2 * np.pi),  (s[0]+ s[1])%(2*np.pi)]
        link_lengths_1 = [0.4, 0.4]

        xys_2 = np.array([[0, 0], p1_2, p2_2])
        thetas_2 = [g[0] % (2 * np.pi),  (g[0]+ g[1])%(2*np.pi)]
        link_lengths_2 = [0.4, 0.4]


        self.viewer.draw_line((-2.2, 1), (2.2, 1))

        for ((x, y), th, llen) in zip(xys_1, thetas_1, link_lengths_1):
            l, r, t, b = 0, llen, 0.05, -0.05
            jtransform = rendering.Transform(rotation=th, translation=(x, y))
            link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
            link.add_attr(jtransform)
            link.set_color(0.8, 0.3, 0.3)
            circ = self.viewer.draw_circle(0.05)
            circ.set_color(0.0, 0.0, 0)
            circ.add_attr(jtransform)

        for ((x, y), th, llen) in zip(xys_2, thetas_2, link_lengths_2):
            l, r, t, b = 0, llen, 0.05, -0.05
            jtransform = rendering.Transform(rotation=th, translation=(x, y))
            link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
            link.add_attr(jtransform)
            link.set_color(0.6, 0.6, 0.2)
            circ = self.viewer.draw_circle(0.05)
            circ.set_color(0.0, 0.0, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
