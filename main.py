# import cv2
# from DQN.DQN import DQN
#
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# import matplotlib.pyplot as plt
# import gym
# import gym_LoL
# import tensorflow as tf
# import cv2
# import time
# from DQN.experience import ReplayBuffer
# from DQN.Agent import Agent
#
# tf.config.experimental.enable_tensor_float_32_execution(False)
# ENV_NAME = 'gymLoL-v0'
#
#
# LOAD_FROM = 'dqnModel'
# SAVE_PATH = '../LeagueAIDQN'
# LOAD_REPLAY_BUFFER = True
#
# WRITE_TENSORBOARD = True
# TENSORBOARD_DIR = '../tensorboardDQN/'
#
#
#
# TOTAL_FRAMES = 3000  # 3000  # Total number of frames to train for
# FRAMES_BETWEEN_EVAL = 100  # 100  # Number of frames between evaluations
# EVAL_LENGTH = 10  # 10
#
# DISCOUNT_FACTOR = 0.99  # Gamma, how much to discount future rewards
#
# MIN_REPLAY_BUFFER_SIZE = 50  # The minimum size the replay buffer must be before we start to update the agent
# MEM_SIZE = 1000  # The maximum size of the replay buffer
#
# MAX_NOOP_STEPS = 20  # Randomly perform this number of actions before every evaluation to give it an element of randomness
# UPDATE_FREQ = 4  # Number of actions between gradient descent steps
# TARGET_UPDATE_FREQ = 10  # Number of actions between when the target network is updated
#
# BATCH_SIZE = 32  # Number of samples the agent learns from at once
# LEARNING_RATE = 0.01
# env = gym.make(ENV_NAME)
# INPUT_SHAPE = env.observation_space.shape
# ACTION_SPACE = env.action_space.n
# print("The environment has the following actions: {},{}".format(ACTION_SPACE, INPUT_SHAPE))
#
#
#
# # Build main and target networks
# MAIN_DQN = DQN(ACTION_SPACE, LEARNING_RATE, input_shape=INPUT_SHAPE)
# TARGET_DQN = DQN(ACTION_SPACE, input_shape=INPUT_SHAPE)
# replay_buffer = ReplayBuffer(size=MEM_SIZE, input_shape=INPUT_SHAPE)
# agent = Agent(MAIN_DQN, TARGET_DQN, replay_buffer, ACTION_SPACE, input_shape=INPUT_SHAPE,
#               batch_size=BATCH_SIZE)
# if LOAD_FROM is None:
#     frame_number = 0
#     rewards = []
#     loss_list = []
#     player_list = []
# else:
#     print('Loading from', LOAD_FROM)
#     meta = agent.load(LOAD_FROM, LOAD_REPLAY_BUFFER)
#
#     # Apply information loaded from meta
#     frame_number = meta['frame_number']
#     rewards = meta['rewards']
#     loss_list = meta['loss_list']
#     print('Loaded')
# loss,_= agent.learn(BATCH_SIZE, gamma=DISCOUNT_FACTOR)
# #double_q = future_q_vals[range(BATCH_SIZE),arg_q_max]
# print(loss)
slope = -(1 - 0.01) / 100
intercept = 1 - slope * 50
e = slope * 51 + intercept
print(slope,intercept,e)
# def process_frame(frame, shape=(640, 640)):
#
#
#     frame = frame.astype(np.uint8)  # cv2 requires np.uint8, other dtypes will not work
#
#     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#     frame = frame[270:, 720:]  # crop image
#     frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
#     frame = frame.reshape((*shape, 3))
#
#     return frame
#
#
# img0 = cv2.imread(r'D:\FinalProject\Image\test1\TestImges1080\172_jpg.rf.139b1c1c83d146d6ba125734332dafd4.jpg')
#
# #state = np.repeat(process_frame(frame), 4, axis=2)
# while True:
#     frame = process_frame(img0)
#     state = np.repeat(frame, 4, axis=2)
#     print(state.shape)
#     cv2.imshow('LOL', frame)
#
#     if cv2.waitKey(25) & 0xFF == ord('0'):  # 按q退出，记得输入切成英语再按q
#         cv2.destroyAllWindows()
#         break
