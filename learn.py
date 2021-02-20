import numpy as np
import gym
import gym_linkage

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
import tensorflow as tf
from rl.callbacks import WandbLogger

ENV_NAME = "Linkage-v0"


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(16))
actor.add(Activation("relu"))
actor.add(Dense(16))
actor.add(Activation("relu"))
actor.add(Dense(nb_actions))
actor.add(Activation("linear"))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name="action_input")
observation_input = Input(
    shape=(1,) + env.observation_space.shape, name="observation_input"
)
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(16)(x)
x = Activation("relu")(x)
x = Dense(16)(x)
x = Activation("relu")(x)
x = Dense(1)(x)
x = Activation("linear")(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=0.4, mu=0.0, sigma=0.5)
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
    target_model_update=0.0001,
)
agent.compile(Adam(lr=0.01), metrics=["mae"])

# # Okay, now it's time to learn something! We visualize the training here for show, but this
# # slows down training quite a lot. You can always safely abort the training prematurely using
# # Ctrl + C.
agent.fit(env, nb_steps=300000, visualize=False, verbose=1, nb_max_episode_steps=500)

# # After training is done, we save the final weights.
agent.save_weights("ddpg_{}_weights.h5f".format(ENV_NAME), overwrite=True)

# # Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=500)
