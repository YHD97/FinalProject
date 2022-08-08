import numpy as np
from mss import mss
import pyautogui

import time
import pygetwindow as gw
import re
import CreateLobby
import pytesseract
import pydirectinput
import win32api, win32con
import cv2
from PIL import Image, ImageOps

MAX_WIDTH, MAX_HEIGHT = 1920, 1080

pytesseract.pytesseract.tesseract_cmd = r'D:\anaconda3\envs\yolov5\Library\bin\tesseract.exe'
tessdata_dir_config = '--psm 6 --tessdata-dir "D:/anaconda3/envs/yolov5/Lib/site-packages/pytesseract/tessdata"'

images = {
    "Ashe": "Image/SetGame/Ashe.png",
    "Bottom": "Image/SetGame/Bottom.png",
    "Ezreal": "Image/SetGame/Ezreal.png",
    "LockIn": "Image/SetGame/LockIn.png",
    "StartGame": "Image/SetGame/StartGame.png",
    "Training": "Image/SetGame/Training.png",
    "Vayne": "Image/SetGame/Vayne.png",
    "CheckGame": "Image/Ingame/CheckGame.png",
    "inGame": "Image/Ingame/inGame.png"
}
item_path = [('Doran\'s Blade', 450), ('Doran\'s Blade', 450), ('Doran\'s Blade', 450), ('Doran\'s Blade', 450),
             ('Doran\'s Blade', 450), ('Doran\'s Blade', 450)]
purchased_items = []


def buy(gold):
    current_gold = gold
    pydirectinput.press('p')

    while len(item_path) > 0 and current_gold > item_path[0][1]:
        pydirectinput.keyDown('ctrl')
        pydirectinput.keyDown('l')
        pydirectinput.keyUp('ctrl')
        pydirectinput.keyUp('l')

        pyautogui.write(item_path[0][0])
        current_gold -= item_path[0][1]
        purchased_items.append(item_path.pop(0))
        cast_action('enter', 0.2)
    cast_action('esc', 0.2)


def level_up_ability():
    cast_order = ["r", "q", "e", "w"]
    pydirectinput.keyDown("ctrl")
    for a in cast_order:
        cast_action(a, 0.5)
    pydirectinput.keyUp("ctrl")


def click_image(location, con=0.9):
    spot = pyautogui.locateCenterOnScreen(location, confidence=con)
    if spot:
        pyautogui.click(spot)
    return spot


def cast_action(action, hold):
    pydirectinput.keyDown(action)
    time.sleep(hold)
    pydirectinput.keyUp(action)


# Move the mouse to coordinates
def move_mouse(x, y):
    try:
        win32api.SetCursorPos((x, y))
    except:
        print("Couldn't lock mouse, 10s sleep...")
        time.sleep(10)


# Left click
def left_click(x, y):
    move_mouse(x, y)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
    time.sleep(0.1)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
    time.sleep(0.1)


# Right click
def right_click(x, y):
    move_mouse(x, y)
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0)
    time.sleep(0.1)
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0)
    time.sleep(0.1)


def no_op(champion, opponent, positions, gold):
    pass


def move_up(champion, opponent, positions, gold):
    if positions[champion] is None:
        return
    x, y = positions[champion][0]
    right_click(int(x), max(0, int(y - 100)))


def move_right(champion, opponent, positions, gold):
    if positions[champion].shape[0] == 0:
        return
    x, y = positions[champion][0]
    right_click(int(x + 100), int(y + 50))


def move_down(champion, opponent, positions, gold):
    if positions[champion].shape[0] == 0:
        return
    x, y = positions[champion][0]
    right_click(int(x), int(y + 200))


def move_left(champion, opponent, positions, gold):
    if positions[champion].shape[0] == 0:
        return
    x, y = positions[champion][0]
    right_click(int(x - 100), int(y + 50))


def attack_minion(champion, opponent, positions, gold):
    if positions['enemyMinions'].shape[0] == 0:
        return
    x, y = positions['enemyMinions'][0]

    cast_action('a', 0.5)
    left_click(int(x), int(y))


def attack_champion(champion, opponent, positions, gold):
    if positions[opponent].shape[0] == 0:
        return
    x, y = positions[opponent][0]

    cast_action('a', 0.5)
    left_click(int(x), int(y))


def find_subset_indices(sub, lst):
    indices = []
    ln = len(sub)
    for i in range(len(lst) - ln + 1):
        if lst[i: i + ln] == sub:
            indices.append(i)
    return indices


def wait_for_game():
    retries = 100
    while retries:
        # if not pyautogui.locateOnScreen('Image/Ingame/CheckGame.png', region=(960, 540, 1920, 1080)):
        if not pyautogui.locateOnScreen(images['CheckGame']):
            retries -= 1
            time.sleep(1)
            continue
        time.sleep(60)
        return
    raise TimeoutError('Retry limit exhausted')


def go_to_Line():
    global MAX_WIDTH, MAX_HEIGHT
    MAX_WIDTH = mss().monitors[1]['width']  # 3840
    MAX_HEIGHT = mss().monitors[1]['height']  # 2160
    # mouse_controller.position = (int(0.9223958 * MAX_WIDTH), int(0.8777777 * MAX_HEIGHT))

    right_click(int(0.965 * MAX_WIDTH), int(0.965 * MAX_HEIGHT))
    time.sleep(30)


def create_custom_game():
    try:
        game_window = gw.getWindowsWithTitle('League of Legends')[0]
        assert game_window.title == 'League of Legends'
    except (IndexError, AssertionError):
        raise RuntimeError('League of Legends not running')
    CreateLobby.create()
    time.sleep(5)
    click_image(images.get("StartGame"))
    time.sleep(1)
    click_image(images.get("Bottom"))
    time.sleep(1)
    click_image(images.get("Vayne"))
    time.sleep(1)
    click_image(images.get("LockIn"))
    print('Wait for Start Game')
    wait_for_game()
    cast_action('y', 1)
    time.sleep(0.5)
    pydirectinput.keyDown("shift")
    pydirectinput.press("h")
    pydirectinput.keyUp("shift")


def leave_custom_game():
    try:
        gw.getWindowsWithTitle('League of Legends (TM) Client')[0]
    except IndexError:
        raise RuntimeError('League of Legends client not running')
    time.sleep(10)

    if pyautogui.locateOnScreen(images['inGame'], confidence=0.8):
        pyautogui.keyDown('altleft')
        pyautogui.press('f4')
        pyautogui.press('f4')
        pyautogui.keyUp('altleft')
        time.sleep(10)



def useQ(champion, opponent, positions, gold):
    cast_action('q', 0.5)
    if positions[opponent].shape[0] != 0:
        x, y = positions[opponent][0]
    elif positions['enemyMinions'].shape[0] != 0:
        x, y = positions['enemyMinions'][0]
    else:
        return
    cast_action('a', 0.5)
    left_click(int(x), int(y))


def useW(champion, opponent, positions, gold):
    return


def useE(champion, opponent, positions, gold):
    if positions[opponent].shape[0] != 0:
        x, y = positions[opponent][0]
    elif positions['enemyMinions'].shape[0] != 0:
        x, y = positions['enemyMinions'][0]
    else:
        return
    right_click(int(x), int(y))
    cast_action('e', 0.5)


def useR(champion, opponent, positions, gold):
    cast_action('R', 0.5)


def useD(champion, opponent, positions, gold):
    cast_action('D', 0.5)


def useF(champion, opponent, positions, gold):
    cast_action('F', 0.5)


def goHome(champion, opponent, positions, gold):
    right_click(int(0.88 * MAX_WIDTH), int(0.965 * MAX_HEIGHT))
    time.sleep(8)
    cast_action('b', 0.5)
    time.sleep(8)
    buy(gold)
    go_to_Line()


actions = [
    no_op,
    move_up,
    move_right,
    move_down,
    move_left,
    attack_minion,
    attack_champion,
    useQ,
    useW,
    useE,
    useR,
    useD,
    useF,
    goHome
]


def perform_action(action, champion, opponent, positions, gold):
    global actions
    actions[action](champion, opponent, positions, gold)


def get_stats(sct_img, stats, template, templateQ, templateW, templateE,
              templateR, templateD, templateF):
    orig_img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
    img = ImageOps.invert(orig_img)
    width, height = img.size

    # Network Warning
    region = (int(0.43 * width), int(0.27 * height), int(0.58 * width), int(0.34 * height))
    img_gray = cv2.cvtColor(np.array(orig_img.crop(region))[:, :, ::-1], cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray, cv2.imread(r'D:\FinalProject\Image\Ingame\NetworkWarning.png', 0),
                            cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    if max_val > 0.8:
        (x, y) = (int(0.5 * width), int(0.45 * height))
        left_click(x, y)
        time.sleep(15)
        raise RuntimeError('Inactive for too long')
    # opponent health
    region = (int(0.0703125 * width), int(0.01111111111 * height), int(0.1046875 * width), int(0.07407407 * height))
    img_gray = cv2.cvtColor(np.array(orig_img.crop(region))[:, :, ::-1], cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    if max_val > 0.8:
        region = (int(0.097 * width), int(0.0379629629 * height), int(0.167 * width), int(0.05 * height))
        hsv = cv2.cvtColor(np.array(orig_img.crop(region))[:, :, ::-1], cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
        labels, statistics = cv2.connectedComponentsWithStats(mask, connectivity=8)[1:3]
        try:
            largest_label = 1 + np.argmax(statistics[1:, cv2.CC_STAT_AREA])
            stats['opponent_health'] = int(
                round(np.max(np.argwhere(labels == largest_label)[:, 1]) * 100 / (mask.shape[1] - 1)))
        except ValueError:
            pass
    # minion_kills
    region = (int(0.925 * width), int(0.0009259 * height), int(0.95625 * width), int(0.0259259 * height))
    # cv2.imwrite('result{}.png'.format(1), np.array(orig_img.crop(region))[:, :, ::-1])
    words = pytesseract.image_to_data(img.crop(region), output_type=pytesseract.Output.DICT, config=tessdata_dir_config)
    matches = [i for i, word in enumerate(words['text']) if re.match(r'^[0-9]+$', word) is not None]
    if len(matches) > 0:
        stats['minion_kills'] = int(words['text'][matches[0]].strip())
    # abilitiesQ
    region = (int(0.39 * width), int(0.895 * height), int(0.416 * width), int(0.935 * height))
    img_gray = cv2.cvtColor(np.array(orig_img.crop(region))[:, :, ::-1], cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray, templateQ, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    if max_val > 0.8:
        stats['Q'] = True
    else:
        stats['Q'] = False

    # abilitiesW
    region = (int(0.42 * width), int(0.88 * height), int(0.45 * width), int(0.94 * height))
    img_gray = cv2.cvtColor(np.array(orig_img.crop(region))[:, :, ::-1], cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray, templateW, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    if max_val > 0.8:
        stats['W'] = True
    else:
        stats['W'] = False
    # abilitiesE
    region = (int(0.45 * width), int(0.88 * height), int(0.48 * width), int(0.98 * height))
    img_gray = cv2.cvtColor(np.array(orig_img.crop(region))[:, :, ::-1], cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray, templateE, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)

    if max_val > 0.8:
        stats['E'] = True
    else:
        stats['E'] = False

    # abilitiesR
    region = (int(0.485 * width), int(0.88 * height), int(0.515 * width), int(0.94 * height))
    img_gray = cv2.cvtColor(np.array(orig_img.crop(region))[:, :, ::-1], cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray, templateR, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    if max_val > 0.8:
        stats['R'] = True
    else:
        stats['R'] = False

    # abilitiesD
    region = (int(0.52 * width), int(0.89 * height), int(0.55 * width), int(0.92 * height))
    img_gray = cv2.cvtColor(np.array(orig_img.crop(region))[:, :, ::-1], cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray, templateD, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    if max_val > 0.8:
        stats['D'] = True
    else:
        stats['D'] = False

    # abilitiesF
    region = (int(0.545 * width), int(0.89 * height), int(0.57 * width), int(0.92 * height))
    img_gray = cv2.cvtColor(np.array(orig_img.crop(region))[:, :, ::-1], cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray, templateF, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    if max_val > 0.8:
        stats['F'] = True
    else:
        stats['F'] = False
    print("state updated")
    return stats


if __name__ == '__main__':
    time.sleep(5)
    goHome()
    # img = cv2.imread('Image/testImage/Screen22.png', 0)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # template = cv2.imread('Image/Ingame/Ashe.png', 0)
    # res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    # _, max_val, _, _ = cv2.minMaxLoc(res)
    # if max_val > 0.8:
    #     print(1)

    # state = {
    #
    #     'kills': 0,
    #     'deaths': 0,
    #     'assists': 0,
    #     'minion_kills': 0,
    #     'health': 100,
    #     'mana': 100,
    #     'opponent_health': 100,
    #     'Q': False,
    #     'W': False,
    #     'E': False,
    #     'R': False,
    #     'D': True,
    #     'F': True
    #
    # }
    # sct = mss()
    # EOG_BOX = {"left": 960, "top": 540, "width": 1920, "height": 1080}
    # opponent_template = cv2.imread('D:\FinalProject\Image\Ingame\Ashe.png', 0)
    # abilitiesQ_template = cv2.imread('D:\FinalProject\Image\Ingame\Ashe.png', 0)
    # abilitiesW_template = cv2.imread('D:\FinalProject\Image\Ingame\Ashe.png', 0)
    # abilitiesE_template = cv2.imread('D:\FinalProject\Image\Ingame\Ashe.png', 0)
    # abilitiesR_template = cv2.imread('D:\FinalProject\Image\Ingame\Ashe.png', 0)
    # abilitiesD_template = cv2.imread('D:\FinalProject\Image\Ingame\D.png', 0)
    # abilitiesF_template = cv2.imread('D:\FinalProject\Image\Ingame\D.png', 0)
    # while True:
    #     sct_img = sct.grab(sct.monitors[1])
    #     start = time.time()
    #     # open_cv_image = np.array(pyautogui.screenshot(region=(960, 540, 1920, 1080)))[:, :, ::-1].copy()
    #     stats = get_stats(sct_img, state, opponent_template, abilitiesQ_template,
    #                       abilitiesW_template, abilitiesE_template, abilitiesR_template,
    #                       abilitiesD_template, abilitiesF_template)
    #     print(time.time() - start)
    # cv2.imshow('image', np.array(sct_img))
    # if cv2.waitKey(1) & 0Xff == ord('q'):
    #     break
    # time.sleep(5)
    # PressKeyPynput(DIK_Y)
    # ReleaseKeyPynput(DIK_Y)
    # time.sleep(0.5)
    # mouse_controller.position = (int(0.9223958 * MAX_WIDTH), int(0.8777777 * MAX_HEIGHT))
    # mouse_controller.click(mouse.Button.right)
    # time.sleep(20)
    # create_custom_game()
