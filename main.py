

import time
import gym
import gym_LoL
# from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Configuration paramaters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
        epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 10000

env = gym.make("gymLoL-v0")


num_actions = env.action_space.n


def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(84, 84, 4,))

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)


# The first model makes the predictions for Q-values which are used to
# make a action.
model = create_q_model()
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_target = create_q_model()
# model = keras.models.load_model('/content/drive/MyDrive/game_ai/model/model.h5')
# model_target = keras.models.load_model('/content/drive/MyDrive/game_ai/model/model.h5')


"""
## Train
"""
# In the Deepmind paper they use RMSProp however then Adam optimizer
# improves training time
# optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
# 3.1
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001, rho=0.99)

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
episode_reward_running_history = []
running_reward = 0
episode_count = 0
frame_count = 0
# Number of frames to take random action and observe output
# 3.2
# epsilon_random_frames = 50000
# Number of frames for exploration
epsilon_greedy_frames = 1000000.0
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
# 3.3
max_memory_length = 10000
# Train the model after 4 actions
update_after_actions = 4
# How often to update the target network
update_target_network = 10000
# Using huber loss for stability
loss_function = keras.losses.Huber()
EPISODES = 200
while True:  # Run until solved
    state = np.array(env.reset())
    episode_reward = 0

    while True:
        # env.render(); Adding this line would show the attempts
        # of the agent in a pop up window.
        frame_count += 1

        # Use epsilon-greedy for exploration
        if epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(num_actions)
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()

        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # Apply the sampled action in our environment
        state_next, reward, done, _ = env.step(action)
        state_next = np.array(state_next)

        episode_reward += reward

        # Save actions and states in replay buffer
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward)
        state = state_next

        # Update every fourth frame and once batch size is over 32
        # 3.5
        if frame_count % update_after_actions == 0 and len(done_history) >= max_memory_length:
            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(done_history)), size=batch_size)

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(done_history[i]) for i in indices]
            )

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = model_target.predict(state_next_sample)
            # Q value = reward + discount factor * expected future reward
            # updated_q_values = rewards_sample + gamma * tf.reduce_max(
            #     future_rewards, axis=1
            # )

            # If final frame set the last value to -1
            # updated_q_values = updated_q_values * (1 - done_sample) - done_sample
            # 3.7
            updated_q_values = rewards_sample + (1 - done_sample) * gamma * tf.reduce_max(future_rewards, axis=1)

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if frame_count % update_target_network == 0:
            # update the the target network with new weights
            model_target.set_weights(model.get_weights())
            # Log details
            template = "running reward: {:.2f} at episode {}, frame count {}"
            print(template.format(running_reward, episode_count, frame_count))
            # complete 10000 frame train, save the model
            # and the episode_reward_history
            model.save('data/model.h5')
            np.save('data/episode_reward_history_10.npy', episode_reward_history, allow_pickle=True,
                    fix_imports=True)
            np.save('data/episode.npy', epsilon, allow_pickle=True, fix_imports=True)

        # Limit the state and reward history
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if done:
            print("episode: score: {}, e: {:.2}".format(time, epsilon))
            break

    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    # 3.6
    episode_reward_running_history.append(episode_reward)
    if len(episode_reward_running_history) > 100:
        del episode_reward_running_history[:1]
    running_reward = np.mean(episode_reward_running_history)

    episode_count += 1

    # if running_reward > 40:  # Condition to consider the task solved
    if frame_count > 2000000:
        print("Solved at episode {}!".format(episode_count))
        break

# complete the while loop for 2000000 frame train, save the model
# and the episode_reward_history
model.save('data/model.h5')
np.save('data/episode_reward_history_10.npy', episode_reward_history, allow_pickle=True,
        fix_imports=True)
np.save('data/episode.npy', epsilon, allow_pickle=True, fix_imports=True)

# import gym
# import gym_LoL
# # from stable_baselines.common.policies import MlpLstmPolicy
# # from stable_baselines.common.vec_env import DummyVecEnv
# # from stable_baselines import PPO2
#
#
# if __name__ == "__main__":
#     # model = PPO2.load("ppo_lol")
#
#     env = gym.make('gymLoL-v0')
#     obs = env.reset()
#     sum_rewards = 0
#     while True:
#         action, _states = model.predict(obs)
#         obs, rewards, done, info = env.step(action)
#
#         sum_rewards += 0
#         if done :
#             print("Final Reward = " + sum_rewards)
#             break


# pyautogui.screenshot()
# for pos in pyautogui.locateAllOnScreen('minion.png'):
#     print(pos)
# if pyautogui.locateOnScreen('Version/Image/SetGame/Play.png', confidence=0.9):
#     print(1)
# while True:
#     # startGame()
#     # getGameData.in_game()
#     pyautogui.locateOnScreen('Version/Image/SetGame/Play.png', confidence=0.9)
#     for pos in pyautogui.locateAllOnScreen('someButton.png')
#         print(pos)
#
#
#     start_time = time.time()
#
#     # Convert RGB to BGR
#     open_cv_image = np.array(pyautogui.screenshot())[:, :, ::-1].copy()
#     cycle_time = time.time() - start_time
#     print(cycle_time)
#
#     cv2.imshow('image', open_cv_image)
#
#     if cv2.waitKey(1) & 0Xff == ord('q'):
#         break
#
# import torch
# from mss import mss
# import time
# import numpy as np
# import cv2
# from yolov7.hubconf import custom
# model = custom(path_or_model='yolov7/best.pt')
# #model = torch.hub.load('ultralytics/yolov5', 'custom', path="gymLoL/gym_LoL/envs/model/best (1).pt")
#
# EOG_BOX = {"left": 960, "top": 540, "width": 1920, "height": 1080}
#
# sct = mss()
#
# while True:
#
#     start_time = time.time()
#     # get image from screen
#
#     frame = sct.grab(sct.monitors[1])
#     frame_img = np.array(frame)
#     # input image into yolov5 model
#     results = model(frame_img, size=640)
#
#     # get position
#     for result in results.pandas().xyxy[0].to_dict(orient="records"):
#         con = result['confidence']
#         cs = result['class']
#         x1 = int(result['xmin'])
#         y1 = int(result['ymin'])
#         x2 = int(result['xmax'])
#         y2 = int(result['ymax'])
#         name = result['name']
#         # Do whatever you want
#         print(cs, (x2 - x1) / 2 + x1, (y2 - y1) / 2 + y1, name)
#     # frame = np.squeeze(results.imgs)
#     # draw the box
#     frame = np.squeeze(results.render())
#
#     cycle_time = time.time() - start_time
#     cv2.putText(frame, "FPS: {}".format(str(round(1 / cycle_time, 2))), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
#                 (0, 0, 255), 2)
#     cv2.imshow('LOL', frame)
#     # resize the show window
#     cv2.resizeWindow('LOL', 1920, 1080)
#
#     if cv2.waitKey(25) & 0xFF == ord('0'):  # 按q退出，记得输入切成英语再按q
#         cv2.destroyAllWindows()
#         break
