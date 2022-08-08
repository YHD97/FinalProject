import numpy as np
import pyautogui

import cv2
import time

# import DQN
import random
import gym
import gym_LoL


DQN_model_path = "model_gpu"
DQN_log_path = "logs_gpu/"
WIDTH = 1920
HEIGHT = 1080

env = gym.make("gymLoL-v0")
action_size = env.action_space.n
# action[n_choose,j,k,m,r]
# j-attack, k-jump, m-defense, r-dodge, n_choose-do nothing

EPISODES = 300
big_BATCH_SIZE = 16
UPDATE_STEP = 50
# times that evaluate the network
num_step = 0
# used to save log graph
target_step = 0
# used to update target Q network
paused = True
# used to stop training

# agent = DQN(WIDTH, HEIGHT, action_size, DQN_model_path, DQN_log_path)
# DQN init


emergence_break = 0

# emergence_break is used to break down training
# 用于防止出现意外紧急停止训练防止错误训练数据扰乱神经网络
for episode in range(EPISODES):
    state = np.array(env.reset())
    episode_reward = 0
    # count init blood
    target_step = 0
    # used to update target Q network
    done = 0
    total_reward = 0
    stop = 0
    # 用于防止连续帧重复计算reward
    last_time = time.time()
    while True:
        station = np.array(station).reshape(-1, HEIGHT, WIDTH, 1)[0]
        # reshape station for tf input placeholder
        print('loop took {} seconds'.format(time.time() - last_time))
        last_time = time.time()
        target_step += 1
        # get the action by state
        action2 = random.randint(0, 8)
        # action = agent.Choose_Action(station)
        action.take_action(action2, active_player)
        print(action)
        # active_player.update()
        if cv2.waitKey(1) & 0Xff == ord('0'):
            break
    #     # take station then the station change
    #     screen_gray = cv2.cvtColor(grab_screen(window_size), cv2.COLOR_BGR2GRAY)
    #     # collect station gray graph
    #     blood_window_gray = cv2.cvtColor(grab_screen(blood_window), cv2.COLOR_BGR2GRAY)
    #     # collect blood gray graph for count self and boss blood
    #     next_station = cv2.resize(screen_gray, (WIDTH, HEIGHT))
    #     next_station = np.array(next_station).reshape(-1, HEIGHT, WIDTH, 1)[0]
    #
    #     reward, done, stop, emergence_break = action_judge(boss_blood, next_boss_blood,
    #                                                        self_blood, next_self_blood,
    #                                                        stop, emergence_break)
    #     # get action reward
    #     if emergence_break == 100:
    #         # emergence break , save model and paused
    #         # 遇到紧急情况，保存数据，并且暂停
    #         print("emergence_break")
    #         agent.save_model()
    #         paused = True
    #     agent.Store_Data(station, action, reward, next_station, done)
    #     if len(agent.replay_buffer) > big_BATCH_SIZE:
    #         num_step += 1
    #         # save loss graph
    #         # print('train')
    #         agent.Train_Network(big_BATCH_SIZE, num_step)
    #     if target_step % UPDATE_STEP == 0:
    #         agent.Update_Target_Network()
    #         # update target Q network
    #     station = next_station
    #
    #     total_reward += reward
    #
    #     if done == 1:
    #         break
    # if episode % 10 == 0:
    #     agent.save_model()
    #     # save model
    # print('episode: ', episode, 'Evaluation Average Reward:', total_reward / target_step)
    # restart()
