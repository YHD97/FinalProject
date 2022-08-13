import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import (Add, Conv2D, Dense, Flatten, Input,
                                     Lambda, Subtract)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop


def DQN(n_actions, learning_rate=0.00001, input_shape=(84, 84), history_length=4):
    input = Input(shape=(input_shape[0], input_shape[1], history_length))

    lamb = Lambda(lambda x: (2 * x - 255) / 255.0, )(input)

    conv_1 = Conv2D(64, (16, 16), strides=(8, 8), activation='relu')(lamb)
    conv_2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv_1)
    conv_3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv_2)
    conv_flattened = (Flatten())(conv_3)
    hidden = Dense(512, activation='relu')(conv_flattened)
    output = Dense(n_actions)(hidden)
    model = Model(inputs=input, outputs=output)
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
