import argparse
import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from lifting_rl.linkage_env import LinkageEnv
import tensorflow as tf

parser = argparse.ArgumentParser(description="Learn model")
parser.add_argument(
    "--angles",
    type=str,
    required=False,
    default="/home/mans/git/human-body-model-dynamics/data/skeleton_angles.csv",
)
args = parser.parse_args()

angles_file = args.angles

ENV_NAME = "Linkage-v0"
DEVICE = None


params = {
    "N_LINKS": 2,
    "INIT_STATE": np.array([np.pi / 2, np.pi / 2, 0, 0], dtype=np.float32),
    "PARAM_VALS": np.array([9.81, 0.4, 1, 0.4, 1], dtype=np.float32),
    "OBS_LOW": np.array([0, 3 * np.pi / 8, -5 * np.pi, -5 * np.pi], dtype=np.float32),
    "OBS_HIGH": np.array(
        [5 * np.pi / 8, 3 * np.pi / 2, 5 * np.pi, 5 * np.pi], dtype=np.float32
    ),
    "ACT_LOW": -20,
    "ACT_HIGH": 20,
    "TIME_STEP": 0.01,
    "VIDEO_FPS": 30,
}


with tf.device(DEVICE):
    # Get the environment and extract the number of actions.
    env = LinkageEnv(angles_file, params, verbose=0)
    assert len(env.action_space.shape) == 1
    nb_actions = env.action_space.shape[0]
    # Next, we build a very simple model.
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.obs_space_shape))
    actor.add(Dense(400))
    actor.add(Activation("relu"))
    actor.add(Dense(300))
    actor.add(Activation("relu"))
    actor.add(Dense(nb_actions))
    actor.add(Activation("tanh"))
    print(actor.summary())

    action_input = Input(shape=(nb_actions,), name="action_input")
    observation_input = Input(
        shape=(1,) + env.obs_space_shape, name="observation_input"
    )
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(400)(x)
    x = Activation("relu")(x)
    x = Dense(300)(x)
    x = Activation("relu")(x)
    x = Dense(1)(x)
    x = Activation("tanh")(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print(critic.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=1000000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(
        size=nb_actions, theta=0.15, mu=0.0, sigma=0.3
    )
    agent = DDPGAgent(
        nb_actions=nb_actions,
        actor=actor,
        critic=critic,
        critic_action_input=action_input,
        memory=memory,
        nb_steps_warmup_critic=1000,
        nb_steps_warmup_actor=1000,
        random_process=random_process,
        gamma=0.99,
        target_model_update=1e-3,
    )
    agent.compile(Adam(lr=0.001, clipnorm=1.0), metrics=["mae"])

    # # Okay, now it's time to learn something! We visualize the training here for show, but this
    # # slows down training quite a lot. You can always safely abort the training prematurely using
    # # Ctrl + C.
    agent.fit(
        env, nb_steps=100000, visualize=False, verbose=1, nb_max_episode_steps=288
    )

    # # After training is done, we save the final weights.
    agent.save_weights("ddpg_{}_weights.h5f".format(ENV_NAME), overwrite=True)

    # # Finally, evaluate our algorithm for 5 episodes.
    agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=288)
