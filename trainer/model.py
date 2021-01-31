from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import RMSprop
from keras.layers.recurrent import LSTM
from keras.callbacks import Callback


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get("loss"))


def get_model(num_inputs, nn_param, num_outputs, load=""):
    model = Sequential()

    # First layer.
    model.add(Dense(nn_param[0], init="lecun_uniform", input_shape=(num_inputs,)))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    # Second layer.
    model.add(Dense(nn_param[1], init="lecun_uniform"))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    # Output layer.
    model.add(Dense(num_outputs, init="lecun_uniform"))
    model.add(Activation("linear"))

    rms = RMSprop()
    model.compile(loss="mse", optimizer=rms)

    if load:
        model.load_weights(load)

    return model
