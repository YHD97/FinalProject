import gym
from gym import spaces
import numpy as np
from collections import defaultdict
import cv2

from LoL_utils import create_custom_game, perform_action, get_stats, leave_custom_game, level_up_ability, buy, \
    go_to_Line

from getGameData import player
import time

from version import targetDetacter

from mss import mss

import random


# from .Version.ChampionPosition import ChampionPosition


class LoLEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = spaces.Discrete(14)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(640, 640), dtype=float)
        self.state = {
            'stats':
                {
                    'kills': 0,
                    'deaths': 0,
                    'assists': 0,
                    'minion_kills': 0,
                    'health': 100,
                    'mana': 100,
                    'opponent_health': 100,
                    'Q': False,
                    'W': False,
                    'E': False,
                    'R': False,
                    'D': True,
                    'F': True
                },

            'positions': defaultdict(lambda: None)
        }
        self.champion = 'Vayne'
        self.opponent = 'Ashe'
        self.opponent_template = cv2.imread('D:\FinalProject\Image\Ingame\Ashe.png', 0)
        self.abilitiesQ_template = cv2.imread('D:\FinalProject\Image\Ingame\Q.png', 0)
        self.abilitiesW_template = cv2.imread('D:\FinalProject\Image\Ingame\W.png', 0)
        self.abilitiesE_template = cv2.imread('D:\FinalProject\Image\Ingame\E.png', 0)
        self.abilitiesR_template = cv2.imread('D:\FinalProject\Image\Ingame\R.png', 0)
        self.abilitiesD_template = cv2.imread('D:\FinalProject\Image\Ingame\D.png', 0)
        self.abilitiesF_template = cv2.imread('D:\FinalProject\Image\Ingame\F.png', 0)
        self.sct = mss()
        self.targetDetacter = targetDetacter()

        self.player = player()
        self.start_time = time.time()
        self.away_time = time.time()
        self.playerPosition = None
        self.currentImg = None

    def get_observation(self, sct_img):
        return self.targetDetacter.getTargetpositions(sct_img)

    def get_reward(self, stats, action):
        done = False
        reward = -1
        mk_scaler, health_scaler, opponent_health_scaler = 1, 1, 1.5
        reward += (stats['minion_kills'] - self.state['stats']['minion_kills']) * mk_scaler
        reward += max(self.state['stats']['opponent_health'] - stats['opponent_health'], 0) * opponent_health_scaler
        reward += min(self.state['stats']['opponent_health'] - stats['opponent_health'], 0) * 1.1
        reward -= max(self.state['stats']['health'] - stats['health'], 0) * health_scaler
        if time.time() - self.away_time > 120:
            reward -= 10
        if action == 0:
            reward -= 5
        if action == 7:
            if self.state['stats']['Q'] and self.player.abilitiesQ and self.state['stats']['mana'] > 13:
                reward += 5
            else:
                reward -= 5
        if action == 8:
            reward -= 5

        if action == 9:
            if self.state['stats']['E'] and self.player.abilitiesE and self.state['stats']['mana'] > 35:
                reward += 5
            else:
                reward -= 5

        if action == 10:
            if self.state['stats']['R'] and self.player.abilitiesR and self.state['stats']['mana'] > 22:
                reward += 5
            else:
                reward -= 5

        if action == 11:
            if self.state['stats']['D'] and self.state['stats']['health'] < 50:
                reward += 5
            else:
                reward -= 5

        if action == 12:
            if self.state['stats']['F']:
                reward += 10
            else:
                reward -= 10
        if action == 13:
            if self.state['stats']['health'] < 20:
                reward += 10
            else:
                reward -= 10

        if self.player.kills == 1:
            reward += 100
            done = True

        elif self.player.deaths == 1:
            reward -= 100
            done = True

        return reward, done

    def update_stats(self, stats):
        self.state['stats'] = stats

    def update_positions(self, detections):
        # champion_position = self.state['positions']['myChampion']
        self.state['positions'].clear()
        # self.state['positions']['myChampion'] = champion_position
        for detection in detections:
            self.state['positions'][detection] = detections[detection]
        if (self.state['positions']['enemyMinions'].shape[0] != 0) or (
                self.state['positions']['allyMinions'].shape[0] != 0) or (
                self.state['positions']['enemyChampion'].shape[0] != 0):
            self.away_time = time.time()

    def step(self, action):
        level_up_ability()
        perform_action(action, 'myChampion', 'enemyChampion', self.state['positions'], self.player.get_gold())
        self.player.update()
        sct_img = self.sct.grab(self.sct.monitors[1])
        observation = self.get_observation(sct_img)
        stats = self.state['stats']
        if time.time() - self.start_time > 600:
            reward, done = -10001, True
        else:
            try:
                stats['health'] = int(self.player.health)
                stats['kills'] = int(self.player.kills)
                stats['deaths'] = int(self.player.deaths)
                stats['mana'] = int(self.player.mana)
                stats = get_stats(sct_img, self.state['stats'].copy(), self.opponent_template, self.abilitiesQ_template,
                                  self.abilitiesW_template, self.abilitiesE_template, self.abilitiesR_template,
                                  self.abilitiesD_template, self.abilitiesF_template)
                reward, done = self.get_reward(stats, action)
            except RuntimeError:
                reward, done = -10001, True
        self.update_positions(observation)
        self.update_stats(stats)

        return np.array(sct_img), reward, done, self.state['stats']

    def reset(self, evaluation=False):
        try:
            leave_custom_game()
            time.sleep(20)
        except RuntimeError:
            pass
        create_custom_game()
        open_game = self.player.is_live()
        if open_game == False:
            return self.reset()
        self.start_time = time.time()
        self.away_time = time.time()
        level_up_ability()
        self.player.update()
        self.state = {
            'stats': {
                'kills': self.player.kills,
                'deaths': self.player.deaths,
                'assists': self.player.assists,
                'minion_kills': self.player.creepScore,
                'health': self.player.health,
                'mana': self.player.mana,
                'opponent_health': 100,
                'Q': self.player.abilitiesQ,
                'W': self.player.abilitiesW,
                'E': self.player.abilitiesE,
                'R': self.player.abilitiesR,
                'D': True,
                'F': True
            },

            'positions': defaultdict(lambda: None)
        }
        buy(self.player.get_gold())
        go_to_Line()
        sct_img = self.sct.grab(self.sct.monitors[1])
        observation = self.get_observation(sct_img)
        self.update_positions(observation)
        if evaluation:
            for _ in range(np.random.randint(0, 10)):
                action = np.random.randint(0, self.action_space)
                self.env.step(action)
        return np.array(sct_img)

    def render(self, mode='human'):
        pass


if __name__ == '__main__':

    env = LoLEnv()
    env.reset()
    while True:
        start = time.time()
        img0 = cv2.imread('Image/testImage/Screen14.png')
        env.get_observation(img0)
        # action = random.randint(0, 6)

        print(time.time() - start)
