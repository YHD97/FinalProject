import numpy as np
import cv2


# def colourClear(img):
#     img[np.where(np.sum(img, axis=2) < 120)] = [0, 0, 0]
#     img[np.where((np.max(img, axis=2) - np.min(img, axis=2) < 25) & (np.sum(img, axis=2) < 300))] = [0, 0, 0]
#     return img


def colourCheck(colour, blue=False, green=False, red=False):
    b = colour[:, 0]
    g = colour[:, 1]
    r = colour[:, 2]
    right_bool = False
    if blue:
        right_bool = (((b > r) * (b > g) * (b > 150)) > 0).any()
    if green:
        right_bool = (((g > b) * (g > r) * (g > 100)) > 0).any()
    if red:
        right_bool = (((r > b) * (r > g) * (r > 150)) > 0).any()

    return right_bool


def findPositions(oriImg, targetImg, mask=None, threshold=0.8, test='', colour=[255, 0, 0], maxThreshold=1.1):
    h, w = targetImg.shape[:2]  # rows->h, cols->w
    h2, w2 = oriImg.shape[:2]  # rows->h, cols->w
    img_gray = cv2.cvtColor(oriImg.copy(), cv2.COLOR_BGR2GRAY)
    targetImg_gray = cv2.cvtColor(targetImg.copy(), cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray, targetImg_gray, cv2.TM_CCORR_NORMED)
    img = oriImg.copy()
    # img = oriImg
    positions = []

    max_val = 1
    while (max_val > threshold):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        start_row = np.max([0, max_loc[1] - h // 2])
        end_row = np.min([h2, max_loc[1] + h // 2])
        start_col = np.max([0, max_loc[0] - w // 2])
        end_col = np.min([w2, max_loc[0] + w // 2])
        if max_val > threshold:
            # Prevent start_row, end_row, start_col, end_col be out of range of image
            positions.append(max_loc)
            if test != '':
                img = cv2.rectangle(img, (max_loc[0], max_loc[1]), (max_loc[0] + w + 1, max_loc[1] + h + 1),
                                    colour)
        else:
            break
        res[start_row: end_row, start_col: end_col] = 0
    if test != '':
        cv2.imwrite('result{}.png'.format(test), img)

    return np.array(positions)
