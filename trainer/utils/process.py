import random
import numpy as np
from keras_models.model import LossHistory


def fit_model(model, replay, batch_size, loss_log, states_size, gamma=0.85):
    # Randomly sample our experience replay memory
    minibatch = random.sample(replay, batch_size)

    # Get training values.
    X_train, y_train = process_minibatch(minibatch, model, states_size, gamma=gamma)

    # Train the model on this batch.
    history = LossHistory()
    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        nb_epoch=1,
        verbose=0,
        callbacks=[history],
    )
    loss_log.append(history.losses)
    return model, loss_log


def process_minibatch(minibatch, model, states_size, gamma=0.85):
    mb_len = len(minibatch)
    old_states = np.zeros(shape=(mb_len, states_size))
    actions = np.zeros(shape=(mb_len,))
    rewards = np.zeros(shape=(mb_len,))
    new_states = np.zeros(shape=(mb_len, states_size))
    for i, m in enumerate(minibatch):
        old_state_m, action_m, reward_m, new_state_m = m

        old_states[i, :] = old_state_m[...]
        actions[i] = action_m
        rewards[i] = reward_m
        new_states[i, :] = new_state_m[...]

    old_qvals = model.predict(old_states, batch_size=mb_len)
    new_qvals = model.predict(new_states, batch_size=mb_len)
    maxQs = np.max(new_qvals, axis=1)
    y = old_qvals

    non_term_inds = np.where(rewards != -2)[0]
    term_inds = np.where(rewards == -2)[0]
    y[non_term_inds, actions[non_term_inds].astype(int)] = rewards[non_term_inds] + (
        gamma * maxQs[non_term_inds]
    )
    y[term_inds, actions[term_inds].astype(int)] = rewards[term_inds]

    X_train = old_states
    y_train = y
    return X_train, y_train
