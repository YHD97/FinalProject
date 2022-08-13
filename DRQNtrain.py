# # import numpy as np
# # file_content = np.load("D:\FinalProject\data\episode_reward_history_10.npy")
# # print()
# # # Print the above file content
# # print("The above file content:")
# # print(file_content)
#
# from experience import ExperienceReplay, Experience
# from Agent import Agent
# from DQN import DQN
#
# import tensorflow as tf
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.losses import MeanSquaredError
# from tensorflow import Tensor, summary, constant, gather, gather_nd, expand_dims, squeeze, reduce_max
# import numpy as np
# import collections
# import datetime
# import time
# import gym
#
# DEVICE = "GPU"
# FPS = 30
# DEFAULT_ENV_NAME = "LOL"  # Name of environment to train
# MEAN_REWARD_BOUND = 19.0  # The boundary of reward to stop training
#
# gamma = 0.99  # Discount factor
# batch_size = 32  # Minibatch size
# replay_size = 10000  # Replay buffer size
# learning_rate = 1e-4  # Learning rate
# sync_target_frames = 100  # How frequently sync model weights from main DQN to the target DQN.
# replay_start_size = 10000  # Count of frames to add to replay buffer before start training.
#
# eps_start = 1.0  # Hyperparameters related to the epsilon decay schedule
# eps_decay = .999985  # Hyperparameters related to the epsilon decay schedule
# eps_min = 0.02  # Hyperparameters related to the epsilon decay schedule
#
# print(">>>Training starts at ", datetime.datetime.now())
#
# env = gym.make('gymLoL-v0')
# input_shape = env.observation_space.shape
# n_output = env.action_space.n
#
# net = DQN(n_actions=n_output, input_shape=input_shape)
# target_net = DQN(n_actions=n_output, input_shape=input_shape)
# writer = summary.create_file_writer(DEFAULT_ENV_NAME)
#
# state_action_values = 0
# expected_state_action_values = 0
#
# buffer = ExperienceReplay(replay_size)
# agent = Agent(env, buffer)
#
# epsilon = eps_start
#
# optimizer = Adam(learning_rate=learning_rate)
# mse_loss_fn = MeanSquaredError()
# total_rewards = []
# loss_data = []
# frame_idx = 0
# play_state = []
# best_mean_reward = None
#
# while True:
#     frame_idx += 1
#     epsilon = max(epsilon * eps_decay, eps_min)
#
#     reward, playerState = agent.play_step(net, epsilon)
#     if reward is not None:
#         mean_reward = np.mean(total_rewards[-100:])
#
#         play_state.append(dict(zip([frame_idx],[playerState])))
#
#         print("%d:  %d games, mean reward %.3f, (epsilon %.2f)" % (frame_idx, len(total_rewards), mean_reward, epsilon))
#         total_rewards.append(reward)
#         net.save('D:\FinalProject\data\model.h5')
#         np.save('D:\FinalProject\data\episode_reward_history_10.npy', total_rewards, allow_pickle=True,
#                 fix_imports=True)
#         np.save('D:\FinalProject\data\episode.npy', epsilon, allow_pickle=True, fix_imports=True)
#         np.save('D:\FinalProject\data\loss.npy', loss_data, allow_pickle=True, fix_imports=True)
#         print(play_state)
#         print(total_rewards, '\n', loss_data)
#         print("data saved")
#         with writer.as_default():
#             summary.scalar("epsilon", epsilon, step=frame_idx)
#             summary.scalar("reward_100", mean_reward, step=frame_idx)
#             summary.scalar("reward", reward, step=frame_idx)
#             writer.flush()
#
#         if best_mean_reward is None or best_mean_reward < mean_reward:
#             net.save(DEFAULT_ENV_NAME + "-best.h5")
#             best_mean_reward = mean_reward
#             net.save('D:\FinalProject\data2\model.h5')
#             np.save('D:\FinalProject\data2\episode_reward_history_10.npy', total_rewards, allow_pickle=True,
#                     fix_imports=True)
#             np.save('D:\FinalProject\data2\episode.npy', epsilon, allow_pickle=True, fix_imports=True)
#             if best_mean_reward is not None:
#                 print("Best mean reward updated %.3f" % best_mean_reward)
#
#         if mean_reward > MEAN_REWARD_BOUND:
#             print("Solved in %d frames!" % frame_idx)
#             break
#
#     if len(buffer) < replay_start_size:
#         continue
#
#     batch = buffer.sample(batch_size)
#     states, actions, rewards, dones, next_states = batch
#
#     states_v = constant(states)
#     actions_v = constant(actions)
#     rewards_v = constant(rewards)
#     done_mask = constant(dones, dtype='uint8')
#     next_states_v = constant(next_states)
#
#     # state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
#     state_action_values = squeeze(gather(net(states_v), expand_dims(actions_v, -1)))
#
#     next_state_values = reduce_max(target_net(next_states_v))[0]
#
#     # next_state_values[done_mask] = 0.0
#     tf.stop_gradient(next_state_values)
#
#     expected_state_action_values = next_state_values * gamma + rewards_v
#
#     loss_t = mse_loss_fn(state_action_values, expected_state_action_values)
#     loss_data.append(loss_t)
#
#     optimizer.zero_grad()
#     loss_t.backward()
#     optimizer.step()
#
#     if frame_idx % sync_target_frames == 0:
#         net.save('D:\FinalProject\data\model.h5')
#         np.save('D:\FinalProject\data\episode_reward_history_10.npy', total_rewards, allow_pickle=True,
#                 fix_imports=True)
#         np.save('D:\FinalProject\data\episode.npy', epsilon, allow_pickle=True, fix_imports=True)
#         target_net.load_state_dict(net.state_dict())
#
# writer.close()
# print(">>>Training ends at ", datetime.datetime.now())


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
# Loading and saving information.
# If LOAD_FROM is None, it will train a new agent.
# If SAVE_PATH is None, it will not save the agent
LOAD_FROM = None
SAVE_PATH = 'LeagueAIDRQN'
LOAD_REPLAY_BUFFER = True

WRITE_TENSORBOARD = True
TENSORBOARD_DIR = 'tensorboard5/'

# If True, use the prioritized experience replay algorithm, instead of regular experience replay
# This is much more computationally expensive, but will also allow for better results. Implementing
# a binary heap, as recommended in the PER paper, would make this less expensive.
# Since Breakout is a simple game, I wouldn't recommend using it here.
USE_PER = True

PRIORITY_SCALE = 0.7  # How much the replay buffer should sample based on priorities. 0 = complete random samples, 1 = completely aligned with priorities
CLIP_REWARD = True  # Any positive reward is +1, and negative reward is -1, 0 is unchanged

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

replay_buffer = ReplayBuffer(size=MEM_SIZE, input_shape=INPUT_SHAPE, use_per=USE_PER)
agent = DRQNAgent(MAIN_DQN, TARGET_DQN, replay_buffer, ACTION_SPACE, input_shape=INPUT_SHAPE,
                  batch_size=BATCH_SIZE, use_per=USE_PER)


# This function can resize to any shape, but was built to resize to 84x84
def process_frame(frame, shape=(84, 84)):
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
                                         reward=reward, clip_reward=CLIP_REWARD,
                                         terminal=terminal)

                    # Update agent
                    if frame_number % UPDATE_FREQ == 0 and agent.replay_buffer.count > MIN_REPLAY_BUFFER_SIZE:
                        loss, _ = agent.learn(BATCH_SIZE, gamma=DISCOUNT_FACTOR, frame_number=frame_number,
                                              priority_scale=PRIORITY_SCALE)
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

                # Breakout requires a "fire" action (action #1) to start the
                # game each time a life is lost.
                # Otherwise, the agent would sit around doing nothing.
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
