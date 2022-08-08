import cv2
import numpy as np
import colourHelper
import time
import pyautogui


class TowerPosition:
    def __init__(self):
        self.path = r'D:\FinalProject\Image\resourceIMage'
        # turret
        self.allyTurret = cv2.imread(self.path + r'\Tower\allyTurret.png')
        self.enemyTurret = cv2.imread(self.path + r'\Tower\enemyTurret.png')
        self.turretMask = cv2.imread(self.path + r'\Tower\turretMask.png')

        # inhibitors
        self.allyInhibitors = cv2.imread(self.path + r'\Tower\allyInhibitors.png')
        self.enemyInhibitors = cv2.imread(self.path + r'\Tower\enemyInhibitors.png')
        self.inhibitorsMask = cv2.imread(self.path + r'\Tower\inhibitorsMask.png')

        self.towerDict = dict()

    @staticmethod
    def towerTurretPositionCheck(img, positions, blue=False, red=False):
        newPositions = []
        for pos in positions:
            x = pos[0]
            y = pos[1]
            colour = img[y + 7, x + 7:x + 10, :]
            rightBool = colourHelper.colourCheck(colour, blue=blue, red=red)
            if rightBool:
                newPositions.append(pos)
        return newPositions

    @staticmethod
    def towerInhibitorsPositionCheck(img, positions, blue=False, red=False):
        newPositions = []
        for pos in positions:
            x = pos[0]
            y = pos[1]
            colour = img[y+7,x+3:x + 5, :]
            rightBool = colourHelper.colourCheck(colour, blue=blue, red=red)
            if rightBool:
                newPositions.append(pos)
        return newPositions

    def getEnemyInhibitorsPosition(self, img):
        ori_img = img.copy()
        # ori_img = colourHelper.colourClear(ori_img)
        positions = colourHelper.findPositions(ori_img, self.enemyInhibitors, self.inhibitorsMask,
                                               threshold=0.95, test='', colour=[255, 255, 255])
        newPositions = np.array(
            self.towerInhibitorsPositionCheck(ori_img, positions, red=True))

        if newPositions.shape[0] != 0:
            newPositions += (50, 150)
        self.towerDict['emenyInhibitors'] = newPositions

    def getAllyInhibitorsPosition(self, img):
        ori_img = img.copy()
        positions = colourHelper.findPositions(ori_img, self.allyInhibitors, self.inhibitorsMask,
                                               threshold=0.95, test=2, colour=[255, 255, 255])
        newPositions = np.array(
            self.towerInhibitorsPositionCheck(ori_img, positions, blue=True))
        if newPositions.shape[0] != 0:
            newPositions += (50, 150)
        self.towerDict['allyInhibitors'] = newPositions

    def getEnemyTurretPosition(self, img):
        ori_img = img.copy()
        positions = colourHelper.findPositions(ori_img, self.enemyTurret, self.turretMask,
                                               threshold=0.95, test=2, colour=[255, 255, 255])
        newPositions = np.array(
            self.towerTurretPositionCheck(ori_img, positions, red=True))

        if newPositions.shape[0] != 0:
            newPositions += (110, 160)
        self.towerDict['emenyTurret'] = newPositions

    def getAllyTurretPosition(self, img):
        ori_img = img.copy()
        positions = colourHelper.findPositions(ori_img, self.allyTurret, self.turretMask,
                                               threshold=0.95, test=2, colour=[255, 255, 255])
        newPositions = np.array(
            self.towerTurretPositionCheck(ori_img, positions, blue=True))
        if newPositions.shape[0] != 0:
            newPositions += (110, 160)

        self.towerDict['allyTurret'] = newPositions

    def getTargetpositions(self, img):
        self.getEnemyInhibitorsPosition(img)
        self.getAllyInhibitorsPosition(img)

        self.getEnemyTurretPosition(img)
        self.getAllyTurretPosition(img)


targetPosition = TowerPosition()


def test(img0=None):
    #img0 = cv2.imread('Image/testImage/Screen03.png')
    img0 = cv2.imread('Image/testImage/Screen14.png')
    targetPosition.getTargetpositions(img0)

    allyTurretpostion = targetPosition.towerDict['emenyInhibitors']
    for pos in allyTurretpostion:
        cv2.circle(img0, tuple(pos), 10, (255, 255, 255), 5)
    allyTurretpostion = targetPosition.towerDict['emenyTurret']
    for pos in allyTurretpostion:
        cv2.circle(img0, tuple(pos), 10, (255, 255, 255), 5)

    allyTurretpostion = targetPosition.towerDict['allyInhibitors']
    for pos in allyTurretpostion:
        cv2.circle(img0, tuple(pos), 10, (255, 255, 255), 5)
    allyTurretpostion = targetPosition.towerDict['allyTurret']
    for pos in allyTurretpostion:
        cv2.circle(img0, tuple(pos), 10, (255, 255, 255), 5)
    return img0



if __name__ == '__main__':
    # test()
    while True:
        open_cv_image = np.array(pyautogui.screenshot(region=(940, 520, 1940, 1100)))[:, :, ::-1].copy()
        startTime = time.time()
        open_cv_image = test()
        cv2.imshow('image', open_cv_image)
        print(time.time() - startTime)
        if cv2.waitKey(1) & 0Xff == ord('q'):
            break
