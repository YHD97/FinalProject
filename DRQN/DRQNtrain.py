
# Name of the Gym environment for the agent to learn & play

import gym
import gym_LoL
import tensorflow as tf
import cv2
import numpy as np
import time
from DRQN import DRQN
from DRQNexperience import ReplayBuffer
from DRQNAgent import DRQNAgent

ENV_NAME = 'gymLoL-v0'

LOAD_FROM = None
SAVE_PATH = 'LeagueAIDRQN'
LOAD_REPLAY_BUFFER = True

WRITE_TENSORBOARD = True
TENSORBOARD_DIR = 'tensorboard5/'



TOTAL_FRAMES = 3000  # 3000  # Total number of frames to train for
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

# TensorBoard writer
writer = tf.summary.create_file_writer(TENSORBOARD_DIR)

# Build main and target networks
MAIN_DQN = DRQN(ACTION_SPACE, LEARNING_RATE, input_shape=INPUT_SHAPE)
TARGET_DQN = DRQN(ACTION_SPACE, input_shape=INPUT_SHAPE)

replay_buffer = ReplayBuffer(size=MEM_SIZE, input_shape=INPUT_SHAPE)
agent = DRQNAgent(MAIN_DQN, TARGET_DQN, replay_buffer, ACTION_SPACE, input_shape=INPUT_SHAPE,
                  batch_size=BATCH_SIZE)


# This function can resize to any shape, but was built to resize to 84x84
def process_frame(frame, shape=(84, 84)):

    frame = frame.astype(np.uint8)  # cv2 requires np.uint8, other dtypes will not work

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = frame[270:, 720:]  # crop image
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

    # Apply information loaded from meta
    frame_number = meta['frame_number']
    rewards = meta['rewards']
    loss_list = meta['loss_list']
    player_list = [np.load("D:\FinalProject\data\playerState.npy", allow_pickle=True)]
    print('Loaded')

# Main loop
try:
    with writer.as_default():
        total_start_time = time.time()
        while frame_number < TOTAL_FRAMES:
            # Training

            epoch_frame = 0
            while epoch_frame < FRAMES_BETWEEN_EVAL:
                start_time = time.time()
                state = env.reset()
                state = [np.expand_dims(process_frame(state), axis=0) for _ in range(4)]
                state = np.concatenate(state, axis=0)
                episode_reward_sum = 0
                while True:
                    # Get action
                    action = agent.get_action(frame_number, state)
                    print('action is', action)
                    # Take step
                    processed_frame, reward, terminal, playerState = env.step(action)
                    frame_number += 1
                    epoch_frame += 1
                    episode_reward_sum += reward
                    processed_frame = process_frame(processed_frame)
                    state[:-1] = state[1:]
                    state[-1] = processed_frame

                    # Add experience to replay memory
                    agent.add_experience(action=action,
                                         frame=processed_frame[:, :, 0],
                                         reward=reward,
                                         terminal=terminal)

                    # Update agent
                    if frame_number % UPDATE_FREQ == 0 and agent.replay_buffer.count > MIN_REPLAY_BUFFER_SIZE:
                        loss, _ = agent.learn(BATCH_SIZE, gamma=DISCOUNT_FACTOR)
                        loss_list.append(loss)
                        print('loss is:', loss)

                    # Update target network
                    if frame_number % TARGET_UPDATE_FREQ == 0 and frame_number > MIN_REPLAY_BUFFER_SIZE:
                        agent.update_target_network()

                    # Break the loop when the game is over
                    if terminal:
                        player_list.append(dict(zip([frame_number], [playerState])))
                        terminal = False
                        break

                rewards.append(episode_reward_sum)
                print('rewards is:', episode_reward_sum)
                print('rewards is:', frame_number)
                # Output the progress every 10 games
                if len(rewards) % 10 == 0:
                    # Write to TensorBoard
                    if WRITE_TENSORBOARD:
                        tf.summary.scalar('Reward', np.mean(rewards[-10:]), frame_number)
                        tf.summary.scalar('Loss', np.mean(loss_list[-100:]), frame_number)
                        writer.flush()

                    print(
                        f'Game number: {str(len(rewards)).zfill(6)}  Frame number: {str(frame_number).zfill(8)}  Average reward: {np.mean(rewards[-10:]):0.1f}  Time taken: {(time.time() - start_time):.1f}s')

            MAIN_DQN.save('D:\FinalProject\data\DQN.h5')
            TARGET_DQN.save('D:\FinalProject\data\Target_DQN.h5')
            np.save('D:\FinalProject\data\episode_reward_history_10.npy', rewards, allow_pickle=True,
                    fix_imports=True)
            np.save('D:\FinalProject\data\playerState.npy', player_list, allow_pickle=True, fix_imports=True)
            np.save('D:\FinalProject\data\loss.npy', loss_list, allow_pickle=True, fix_imports=True)

            # Evaluation every `FRAMES_BETWEEN_EVAL` frames
            terminal = True
            eval_rewards = []
            evaluate_frame_number = 0
            for _ in range(EVAL_LENGTH):
                if terminal:
                    state = env.reset()
                    state = [np.expand_dims(process_frame(state), axis=0) for _ in range(4)]
                    state = np.concatenate(state, axis=0)
                    isDeaded = True
                    episode_reward_sum = 0
                    terminal = False

                action = 1 if isDeaded else agent.get_action(frame_number, state, evaluation=True)

                # Step action
                _, reward, terminal, playerState = env.step(action)
                evaluate_frame_number += 1
                episode_reward_sum += reward
                isDeaded = terminal

                # On game-over
                if terminal:
                    eval_rewards.append(episode_reward_sum)

            if len(eval_rewards) > 0:
                final_score = np.mean(eval_rewards)
            else:
                # In case the game is longer than the number of frames allowed
                final_score = episode_reward_sum
            # Print score and write to tensorboard
            print('Evaluation score:', final_score)
            env.close()
            if WRITE_TENSORBOARD:
                tf.summary.scalar('Evaluation score', final_score, frame_number)
                writer.flush()

            # Save model
            if len(rewards) > 20 and SAVE_PATH is not None:
                agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', frame_number=frame_number, rewards=rewards,
                           loss_list=loss_list)
except KeyboardInterrupt:
    print('\nTraining exited early.')
    writer.close()

    if SAVE_PATH is None:
        try:
            SAVE_PATH = input(
                'Would you like to save the trained model? If so, type in a save path, otherwise, interrupt with ctrl+c. ')
        except KeyboardInterrupt:
            print('\nExiting...')

    if SAVE_PATH is not None:
        print('Saving...')
        agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', frame_number=frame_number, rewards=rewards,
                   loss_list=loss_list)
        print('Saved.')
