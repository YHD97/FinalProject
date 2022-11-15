import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import (Add, Conv2D, Dense, Flatten, Input,
                                     Lambda, Subtract, TimeDistributed, LSTM)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop


def DRQN(n_actions, learning_rate=0.00001, input_shape=(84, 84), history_length=4, input_depth=1):
    # input_depth: Defining the size of the input's third dimension
    # param  history_size : int (Default: 4)Number of sequential frames that will be stacked together to form the input volume to the NN.
    input = Input(shape=(history_length, input_shape[0], input_shape[1], input_depth))
    lamb = Lambda(lambda x: (2 * x - 255) / 255.0, )(input)

    conv_1 = TimeDistributed(Conv2D(64, (16, 16), strides=(8, 8), activation='relu'))(lamb)
    conv_2 = TimeDistributed(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))(conv_1)
    conv_3 = TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))(conv_2)
    conv_flattened = TimeDistributed(Flatten())(conv_3)
    hidden = LSTM(512, activation='tanh')(conv_flattened)
    output = Dense(n_actions)(hidden)
    model = Model(inputs=input, outputs=output)
    model.compile(Adam(learning_rate), loss=tf.keras.losses.Huber())
    return model
