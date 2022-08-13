import cv2
import numpy as np
import colourHelper
import time
import pyautogui
from mss import mss


class MinionsPosition:
    def __init__(self):
        self.path = r'D:\FinalProject\Image\resourceIMage\Minions'
        self.allyMinions = cv2.imread(self.path +r'\test6.png')
        self.enemyMinions = cv2.imread(self.path +r'\test3.png')
        # minions health bar mask
        self.minionsMask = cv2.imread(self.path + '\minionsMask.png')

        self.minionsDict = dict()

    @staticmethod
    def minionsPositionCheck(img, positions, blue=False, green=False, red=False):
        newPositions = []
        for pos in positions:
            x = pos[0]
            y = pos[1]
            colour = img[y + 1:y + 4, x + 1, :]
            # cv2.imwrite(f'pic/{0}.png', img[y + 1:y + 4, x + 1:x+2, :])
            rightBool = colourHelper.colourCheck(colour, blue=blue, red=red)
            if rightBool:
                newPositions.append(pos)

        return newPositions

    def getAllyMinionsPosition(self, img):
        ori_img = img.copy()
        # ori_img = colourHelper.colourClear(ori_img)

        positions = colourHelper.findPositions(ori_img, self.allyMinions, self.minionsMask,
                                               threshold=0.99, colour=[255, 255, 255])
        newPositions = np.array(
            self.minionsPositionCheck(ori_img, positions, blue=True))
        if newPositions.shape[0] != 0:
            newPositions += (34, 36)
        self.minionsDict['allyMinions'] = newPositions

    def getEnemyMinionsPosition(self, img):
        ori_img = img.copy()
        # ori_img = colourHelper.colourClear(ori_img)

        positions = colourHelper.findPositions(ori_img, self.enemyMinions, self.minionsMask,
                                               threshold=0.99, test='', colour=[255, 255, 255])

        newPositions = np.array(
            self.minionsPositionCheck(ori_img, positions, red=True))

        if newPositions.shape[0] != 0:
            newPositions += (34, 36)
        self.minionsDict['enemyMinions'] = newPositions

    def getTargetpositions(self, img):
        self.getAllyMinionsPosition(img)
        self.getEnemyMinionsPosition(img)


targetPosition = MinionsPosition()


def test(img0=None):
    img0 = cv2.imread(r'D:\FinalProject\Image\testImage\Screen02.png')
    targetPosition.getTargetpositions(img0)

    allyMinions = targetPosition.minionsDict['allyMinions']
    # img0 = cv2.imread('Image/testImage/Screen02.png')
    for pos in allyMinions:
        cv2.circle(img0, tuple(pos), 15, (255, 0, 255), 5)
    enemyMinionsPostion = targetPosition.minionsDict['enemyMinions']
    for pos in enemyMinionsPostion:
        cv2.circle(img0, tuple(pos), 5, (0, 255, 255), 5)

    print(allyMinions, enemyMinionsPostion)

    return img0


if __name__ == '__main__':
    # test()

    # while True:
    #
    #     open_cv_image = np.array(pyautogui.screenshot(region=(940, 520, 1940, 1100)))[:, :, ::-1].copy()
    #     startTime = time.time()
    #     open_cv_image = test(open_cv_image)
    #     cv2.imshow('image', open_cv_image)
    #     print(time.time() - startTime)
    #     if cv2.waitKey(1) & 0Xff == ord('q'):
    #         break
    sct = mss()
    EOG_BOX = {"left": 960, "top": 540, "width": 1920, "height": 1080}
    while True:
        sct_img = sct.grab(EOG_BOX)
        sct_img_resized = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)

        # open_cv_image = np.array(pyautogui.screenshot(region=(960, 540, 1920, 1080)))[:, :, ::-1].copy()
        startTime = time.time()
        open_cv_image = test(sct_img_resized)
        cv2.imshow('image', open_cv_image)
        print(time.time() - startTime)
        if cv2.waitKey(1) & 0Xff == ord('q'):
            break
