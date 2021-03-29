import argparse
from ddpg_torch import Agent
import gym
import numpy as np
from lifting_rl.linkage_env import LinkageEnv

from livelossplot import PlotLosses

parser = argparse.ArgumentParser(description="Learn model")
parser.add_argument(
    "--angles",
    type=str,
    required=False,
    default="/home/mans/git/human-body-model-dynamics/data/skeleton_angles.csv",
)
args = parser.parse_args()

angles_file = args.angles

params = {
    "N_LINKS": 2,
    "INIT_STATE": np.array([np.pi / 2, np.pi / 2, 0, 0], dtype=np.float32),
    "PARAM_VALS": np.array([9.81, 0.4, 1, 0.4, 1], dtype=np.float32),
    "OBS_LOW": np.array([0, 0, -10 * np.pi, -10 * np.pi], dtype=np.float32),
    "OBS_HIGH": np.array(
        [5 * np.pi / 8, 3 * np.pi / 2, 10 * np.pi, 10 * np.pi], dtype=np.float32
    ),
    "ACT_LOW": -100,
    "ACT_HIGH": 100,
    "TIME_STEP": 0.01,
    "VIDEO_FPS": 30,
}

# env = LinkageEnv(angles_file, params, verbose=0)

env = gym.make("LunarLanderContinuous-v2")

agent = Agent(
    lr_actor=0.000025,
    lr_critic=0.00025,
    input_dims=[8],
    tau=0.001,
    env=env,
    batch_size=64,
    layer1_size=400,
    layer2_size=300,
    n_actions=2,
)

np.random.seed(0)

score_history = []

liveloss = PlotLosses()

for i in range(100000):
    done = False
    score = 0
    obs = env.reset()
    agent.noise.reset()
    while not done:
        # env.render()
        act = agent.choose_action(obs)
        new_state, reward, done, info = env.step(act)
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state

    score_history.append(score)

    metrics = {"score": score_history}

    print(
        "episode",
        i,
        "score %.2f" % score,
        "100 game average %.2f" % np.mean(score_history[-100:]),
    )

    liveloss.update(metrics)
    liveloss.send()
env.close()
