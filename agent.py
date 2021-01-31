import random
import numpy as np


class NNAgent:
    def __init__(self, action_space, nn):
        self.nn = nn
        self.action_space = action_space

    def act(self, observation, reward, epsilon, done):
        if random.random() < epsilon:
            action = self.action_space.sample()
        else:
            qval = self.nn.predict(np.array([observation]))
            action = np.argmax(qval)
        return action
