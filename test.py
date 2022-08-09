
import numpy as np
x = np.load('D:\FinalProject\data\episode_reward_history_10.npy')
print(x)

import tensorflow as tf
# from tensorflow import keras
# import gym
# model = keras.models.load_model('D:\FinalProject\data2\model.h5')
# #model = keras.models.load_model('/content/drive/MyDrive/game_ai2/model/model.h5')
# # Check its architecture
# model.summary()

# import tensorflow as tf
# from tensorflow.keras.initializers import VarianceScaling
# from tensorflow.keras.layers import (Add, Conv2D, Dense, Flatten, Input,
#                                      Lambda, Subtract)
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam, RMSprop
# def build_q_network(n_actions, learning_rate=0.00001, input_shape=(84, 84), history_length=4):
#     """Builds a dueling DQN as a Keras model
#     Arguments:
#         n_actions: Number of possible action the agent can take
#         learning_rate: Learning rate
#         input_shape: Shape of the preprocessed frame the model sees
#         history_length: Number of historical frames the agent can see
#     Returns:
#         A compiled Keras model
#     """
#     model_input = Input(shape=(input_shape[0], input_shape[1], history_length))
#     x = Lambda(lambda layer: layer / 255)(model_input)  # normalize by 255
#
#     x = Conv2D(32, (8, 8), strides=4, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
#     x = Conv2D(64, (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
#     x = Conv2D(64, (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
#     x = Conv2D(1024, (7, 7), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
#
#     # Split into value and advantage streams
#     val_stream, adv_stream = Lambda(lambda w: tf.split(w, 2, 3))(x)  # custom splitting layer
#
#     val_stream = Flatten()(val_stream)
#     val = Dense(1, kernel_initializer=VarianceScaling(scale=2.))(val_stream)
#
#     adv_stream = Flatten()(adv_stream)
#     adv = Dense(n_actions, kernel_initializer=VarianceScaling(scale=2.))(adv_stream)
#
#     # Combine streams into Q-Values
#     reduce_mean = Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))  # custom layer for reduce mean
#
#     q_vals = Add()([val, Subtract()([adv, reduce_mean(adv)])])
#
#     # Build model
#     model = Model(model_input, q_vals)
#     model.compile(Adam(learning_rate), loss=tf.keras.losses.Huber())
#
#     return model
#
# import cv2
# import numpy as np
#
# # This function can resize to any shape, but was built to resize to 84x84
# def process_frame(frame, shape=(84, 84)):
#     """Preprocesses a 210x160x3 frame to 84x84x1 grayscale
#     Arguments:
#         frame: The frame to process.  Must have values ranging from 0-255
#     Returns:
#         The processed frame
#     """
#     frame = frame.astype(np.uint8)  # cv2 requires np.uint8, other dtypes will not work
#
#     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#     frame = frame[34:34+160, :160]  # crop image
#     frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
#     frame = frame.reshape((*shape, 1))
#
#     return frame
#
# from mss import mss
# import cv2
# import numpy as np
# model = build_q_network(n_actions=13)
# model.summary()
# sct = mss().grab(mss().monitors[1])
# sct_img = process_frame(np.array(sct))
# model(sct_img)
