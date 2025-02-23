# Name of the Gym environment for the agent to learn & play

import gym
import gym_LoL
import tensorflow as tf
import cv2
import numpy as np
import time
from DRQN.DRQN import DRQN
from DRQN.DRQNexperience import ReplayBuffer
from DRQN.DRQNAgent import DRQNAgent

ENV_NAME = 'gymLoL-v0'

LOAD_FROM = 'save-00002773'
SAVE_PATH = 'LeagueAIDRQN'
LOAD_REPLAY_BUFFER = True

WRITE_TENSORBOARD = True
TENSORBOARD_DIR = '../tensorboard5/'

PRIORITY_SCALE = 0.7

TOTAL_FRAMES = 3000  # Total number of frames to train for
FRAMES_BETWEEN_EVAL = 100  # 100  # Number of frames between evaluations
EVAL_LENGTH = 10  # 10

DISCOUNT_FACTOR = 0.99  # Gamma, how much to discount future rewards

MIN_REPLAY_BUFFER_SIZE = 50  # The minimum size the replay buffer must be before we start to update the agent
MEM_SIZE = 1000  # The maximum size of the replay buffer

MAX_NOOP_STEPS = 20  # Randomly perform this number of actions before every evaluation to give it an element of randomness
UPDATE_FREQ = 4  # Number of actions between gradient descent steps
TARGET_UPDATE_FREQ = 10  # Number of actions between when the target network is updated

BATCH_SIZE = 32  # Number of samples the agent learns from at once
LEARNING_RATE = 0.01

env = gym.make(ENV_NAME)
INPUT_SHAPE = env.observation_space.shape
ACTION_SPACE = env.action_space.n
print("The environment has the following actions: {},{}".format(ACTION_SPACE, INPUT_SHAPE))

# Build main and target networks
MAIN_DQN = DRQN(ACTION_SPACE, LEARNING_RATE, input_shape=INPUT_SHAPE)
TARGET_DQN = DRQN(ACTION_SPACE, input_shape=INPUT_SHAPE)

replay_buffer = ReplayBuffer(size=MEM_SIZE, input_shape=INPUT_SHAPE)
agent = DRQNAgent(MAIN_DQN, TARGET_DQN, replay_buffer, ACTION_SPACE, input_shape=INPUT_SHAPE,
                  batch_size=BATCH_SIZE)


def process_frame(frame, shape=(84, 84)):
    frame = frame.astype(np.uint8)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = frame[270:, 720:]
    frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
    frame = frame.reshape((*shape, 1))

    return frame


# Training and evaluation
if LOAD_FROM is None:
    frame_number = 0
    rewards = []
    loss_list = []
    player_list = []
else:
    print('Loading from', LOAD_FROM)
    meta = agent.load(LOAD_FROM, LOAD_REPLAY_BUFFER)

    frame_number = meta['frame_number']
    rewards = meta['rewards']
    loss_list = meta['loss_list']
    print('Loaded')

n_episodes = 1
returns = []
rewards_list = []
statelist = []
action_list = []
# Main loop
for _ in range(n_episodes):
    start_time = time.time()
    state = env.reset()
    state = [np.expand_dims(process_frame(state), axis=0) for _ in range(4)]
    state = np.concatenate(state, axis=0)
    episode_reward_sum = 0
    frame_number_ = 0
    while True:
        # Get action
        action = agent.get_action(frame_number, state)
        print('action is', action)
        # Take step
        processed_frame, reward, terminal, playerState = env.step(action)

        episode_reward_sum += reward
        processed_frame = process_frame(processed_frame)
        state[:-1] = state[1:]
        state[-1] = processed_frame

        rewards_list.append(reward)
        action_list.append(action)
        statelist.append(dict(zip([frame_number_], [playerState])))

        if terminal:
            terminal = False
            break

    rewards.append(episode_reward_sum)
    returns.append(episode_reward_sum)
    print('rewards is:', episode_reward_sum)
    print('rewards is:', frame_number)
env.close()

np.save('../DRQNTESTDATA/totoalReward.npy', returns, allow_pickle=True, fix_imports=True)
np.save('../DRQNTESTDATA/playerState.npy', statelist, allow_pickle=True, fix_imports=True)
np.save('../DRQNTESTDATA/allRewards.npy', rewards_list, allow_pickle=True, fix_imports=True)
np.save('../DRQNTESTDATA/action.npy', action_list, allow_pickle=True, fix_imports=True)
