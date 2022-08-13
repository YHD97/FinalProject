import cv2
import numpy as np
import colourHelper
import time
import pyautogui


class ChampionPosition:
    def __init__(self):
        self.path = r'D:\FinalProject\Image\resourceIMage\Champion'
        #  champion health bar
        self.myChampion = cv2.imread(self.path + r'\test5.png')
        self.enemyChampion = cv2.imread(self.path + r'\test4.png')
        # champion health bar mask
        self.championMask = cv2.imread(self.path + '\championMask.png')

        self.myChampionPositionDict = dict()
        self.enemyChampionPositionDict = dict()

    @staticmethod
    def championPositionCheck(img, positions, blue=False, green=False, red=False):
        newPositions = []
        for pos in positions:
            x = pos[0] + 1
            y = pos[1] + 1
            colour = img[pos[1] + 10:pos[1] + 15, pos[0] + 27, :]
            # cv2.imwrite(f'{x}.png', img[pos[1] + 10:pos[1] + 15, pos[0] + 27:pos[0] + 30, :])
            rightBool = colourHelper.colourCheck(colour, blue=blue, green=green, red=red)
            # colour0 = img[pos[1]:pos[1] + 20, pos[0]:pos[0] + 20, :]
            # # cv2.imwrite(f'{x}.png', img[pos[1]:pos[1] + 20, pos[0]:pos[0] + 20, :])
            # colour = np.sum(colour0, axis=2)
            # rightBool2 = np.max(colour) > 600
            # rightBool = rightBool and rightBool2
            if rightBool:
                # cv2.imwrite('pic/{}f{}-{}.png'.format(rightBool2, pos,np.max(colour)), colour0)
                newPositions.append(pos)
        return newPositions

    def getMyChampionPosition(self, img):
        positions = colourHelper.findPositions(img, self.myChampion, self.championMask, threshold=0.8,
                                               colour=[0, 0, 255])

        newPositions = np.array(self.championPositionCheck(img, positions, green=True))
        if newPositions.shape[0] != 0:
            newPositions += (67, 95)
        self.myChampionPositionDict['myChampion'] = newPositions

    def getEnemyChampionPosition(self, img):
        positions = colourHelper.findPositions(img, self.enemyChampion, self.championMask, threshold=0.8,
                                               colour=[0, 0, 255])
        newPositions = np.array(self.championPositionCheck(img, positions, red=True))
        if newPositions.shape[0] != 0:
            newPositions += (67, 95)
        self.enemyChampionPositionDict['enemyChampion'] = newPositions

    def getTargetpositions(self, img):
        self.getMyChampionPosition(img)
        self.getEnemyChampionPosition(img)


targetPosition = ChampionPosition()


def test(img0=None):
    img0 = cv2.imread(r'D:\FinalProject\Image\testImage\Screen02.png')
    targetPosition.getTargetpositions(img0)

    myChampionPosition = targetPosition.myChampionPositionDict['myChampion']
    for pos in myChampionPosition:
        cv2.circle(img0, tuple(pos), 10, (255, 255, 0), 5)
        img0[pos[1], pos[0]] = (255, 255, 0)
    enemyChampionPosition = targetPosition.enemyChampionPositionDict['enemyChampion']
    # print(type(hero_postion),enemy_heros_postion)

    for pos in enemyChampionPosition:
        # p3 = pos - hero_postion[0]
        # print(p3)
        # p4 = math.hypot(p3[0], p3[1])
        # print(p4)
        cv2.circle(img0, tuple(pos), 10, (255, 255, 255), 5)
    print(targetPosition.myChampionPositionDict)
    print(targetPosition.enemyChampionPositionDict)

    return img0


from mss import mss

if __name__ == '__main__':
    # test()
    sct = mss()
    EOG_BOX = {"left": 960, "top": 540, "width": 1920, "height": 1080}
    while True:
        sct_img = sct.grab(EOG_BOX)
        sct_img_resized = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)

        open_cv_image = np.array(pyautogui.screenshot(region=(960, 540, 1920, 1080)))[:, :, ::-1].copy()
        startTime = time.time()
        open_cv_image = test(sct_img_resized)
        cv2.imshow('image', open_cv_image)
        print(time.time() - startTime)
        if cv2.waitKey(1) & 0Xff == ord('q'):
            break
