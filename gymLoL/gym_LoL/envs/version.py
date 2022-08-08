import cv2
import numpy as np
import time
import pyautogui

import ChampionPosition
import TowerPosition
import MinionsPosition


class targetDetacter():
    def __init__(self):
        self.bounding_box = (940, 520, 1940, 1100),
        # self.img = np.array(pyautogui.screenshot(region=self.bounding_box))[:, :, ::-1].copy()
        self.sct_original = None
        self.sct_img = None
        self.ChampionPositionDict = ChampionPosition.ChampionPosition()
        self.allUnitPosition = dict()
        self.minionsDict = MinionsPosition.MinionsPosition()
        self.towerDict = TowerPosition.TowerPosition()

    def getTargetpositions(self, img):
        sct_img_resized = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)
        self.ChampionPositionDict.getTargetpositions(sct_img_resized)
        # self.towerDict.getTargetpositions(img)
        self.minionsDict.getTargetpositions(sct_img_resized)
        self.allUnitPosition.update(self.ChampionPositionDict.myChampionPositionDict)
        self.allUnitPosition.update(self.ChampionPositionDict.enemyChampionPositionDict)
        self.allUnitPosition.update(self.minionsDict.minionsDict)
        # self.allUnitPosition.update(self.towerDict.towerDict)
        # self.allUnitPosition['ChampionPositionDict'] = self.ChampionPositionDict.myChampionPositionDict
        # self.allUnitPosition['enemyChampion'] = self.ChampionPositionDict.enemyChampionPositionDict
        # self.allUnitPosition['Minions'] = self.minionsDict.minionsDict
        # self.allUnitPosition['tower'] = self.towerDict.towerDict
        print("get position")
        return self.allUnitPosition


targetPosition = targetDetacter()


def test(img0=None):
    img0 = cv2.imread(r'D:\FinalProject\Image\testImage\Screen02.png')
    targetPosition.getTargetpositions(img0)
    myChampionPosition = targetPosition.allUnitPosition['myChampion']
    for pos in myChampionPosition:
        cv2.circle(img0, tuple(pos), 10, (255, 255, 0), 5)
        img0[pos[1], pos[0]] = (255, 255, 0)
    enemyChampionPosition = targetPosition.allUnitPosition['enemyChampion']
    # print(type(hero_postion),enemy_heros_postion)

    for pos in enemyChampionPosition:
        # p3 = pos - hero_postion[0]
        # print(p3)
        # p4 = math.hypot(p3[0], p3[1])
        # print(p4)
        cv2.circle(img0, tuple(pos), 10, (255, 255, 255), 5)

    allyMinions = targetPosition.allUnitPosition['allyMinions']
    # img0 = cv2.imread('Image/testImage/Screen02.png')
    for pos in allyMinions:
        cv2.circle(img0, tuple(pos), 15, (255, 0, 255), 5)
    enemyMinionsPostion = targetPosition.allUnitPosition['enemyMinions']
    for pos in enemyMinionsPostion:
        cv2.circle(img0, tuple(pos), 5, (0, 255, 255), 5)
    return img0


if __name__ == '__main__':
    # test()
    while True:
        open_cv_image = np.array(pyautogui.screenshot(region=(940, 520, 1940, 1100)))[:, :, ::-1].copy()
        startTime = time.time()
        open_cv_image = test(open_cv_image)
        cv2.imshow('image', open_cv_image)
        print(1 / (time.time() - startTime))
        if cv2.waitKey(1) & 0Xff == ord('q'):
            break

        # {'myChampion': array([[872, 442]]),
        #  'enemyChampion': array([], dtype=float64),
        #  'allyMinions': array([[1210, 888],
        #                        [1145, 821]]),
        #  'enemyMinions': array([], dtype=float64),
        #  'emenyTurret': array([], dtype=float64),
        #  'allyTurret': array([], dtype=float64)}
