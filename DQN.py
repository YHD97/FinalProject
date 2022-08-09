import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import (Add, Conv2D, Dense, Flatten, Input,
                                     Lambda, Subtract)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop


def DQN(n_actions, learning_rate=0.00001, input_shape=(640, 640), history_length=1):
    model_input = Input(shape=(input_shape[0], input_shape[1], history_length))
    x = Lambda(lambda layer: layer / 255)(model_input)  # normalize by 255

    x = Conv2D(32, (8, 8), strides=4, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(
        x)
    x = Conv2D(64, (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(
        x)
    x = Conv2D(64, (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(
        x)
    x = Conv2D(1024, (7, 7), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu',
               use_bias=False)(x)

    # Split into value and advantage streams
    val_stream, adv_stream = Lambda(lambda w: tf.split(w, 2, 3))(x)  # custom splitting layer

    val_stream = Flatten()(val_stream)
    val = Dense(1, kernel_initializer=VarianceScaling(scale=2.))(val_stream)

    adv_stream = Flatten()(adv_stream)
    adv = Dense(n_actions, kernel_initializer=VarianceScaling(scale=2.))(adv_stream)

    # Combine streams into Q-Values
    reduce_mean = Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))  # custom layer for reduce mean

    q_vals = Add()([val, Subtract()([adv, reduce_mean(adv)])])

    # Build model
    model = Model(model_input, q_vals)
    model.compile(Adam(learning_rate), loss=tf.keras.losses.Huber())

    return model


if __name__ == '__main__':
    from mss import mss
    import numpy as np

    from gym import spaces
    from PIL import Image
    import cv2
    import tensorflow as tf
    import cv2
    import numpy as np

    # This function can resize to any shape, but was built to resize to 84x84
    import cv2
    import numpy as np


    # This function can resize to any shape, but was built to resize to 84x84
    def process_frame(frame, shape=(640, 640)):
        """Preprocesses a 210x160x3 frame to 84x84x1 grayscale
        Arguments:
            frame: The frame to process.  Must have values ranging from 0-255
        Returns:
            The processed frame
        """
        frame = frame.astype(np.uint8)  # cv2 requires np.uint8, other dtypes will not work

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = frame[34:34 + 160, :160]  # crop image
        frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
        frame = frame.reshape((*shape, 1))

        return frame


    sapce_sapce = spaces.Box(low=-np.inf, high=np.inf, shape=(640, 640), dtype=float)
    model = DQN(14)
    while True:
        sct = mss().grab(mss().monitors[1])
        sct_img = np.repeat(process_frame(np.array(sct)), 4, axis=2)
        q_vals = model.predict(sct_img.reshape((-1, 640, 640, 1)))[0]
        print(q_vals)

        # state_a = np.array(sct_img, copy=False)
        # state_v = tf.constant(state_a)
        # q_vals_v = model(state_v)
        #
        # act_v = tf.math.reduce_max(q_vals_v)
        # action = tf.cast(act_v, tf.int32)
        # print(action)
