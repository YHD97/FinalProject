import pydirectinput
import time
import pyautogui
import math
import CreateLobby
import KeysSimulation


def startGame():
    CreateLobby.create()
    time.sleep(1)
    KeysSimulation.click_image(images.get("StartGame"))
    time.sleep(1)
    KeysSimulation.click_image(images.get("Botton"))
    time.sleep(1)
    KeysSimulation.click_image(images.get("EZ"))
    time.sleep(1)
    KeysSimulation.click_image(images.get("LockIn"))
    screen_dim = pyautogui.size()
    print(screen_dim)
    print('Wait for Start Game')

def cast_action(action, hold):
    pydirectinput.keyDown(action)
    time.sleep(hold)
    pydirectinput.keyUp(action)


def recall(active_player):
    pyautogui.click(1560, 1000)
    time.sleep(8)
    cast_action('b', 1)

    try:
        active_player.update()
    except:
        return
    prev_health = active_player.stats["championStats"]["currentHealth"]
    time.sleep(6.8)
    try:
        active_player.update()
    except:
        return
    curr_health = math.ceil(active_player.stats["championStats"]["currentHealth"])
    is_max = curr_health == math.ceil(prev_health)
    prev_health = math.ceil(prev_health + active_player.get_regen() * 6.8)
    if prev_health not in range(curr_health - 2, curr_health + 2) and not is_max:
        print("Failed recall")
        recall(active_player)
        return
    start_time = time.time()
    while active_player.get_health() < 1.0:
        try:
            active_player.update()
        except:
            return
        if time.time() - start_time > 20:
            print("Timed out on recall")
            recall(active_player)
            return
    print("Full health")
    time.sleep(1)
    # buy_item(active_player)


def take_action(action, active_player):
    if action == 0:  # n_choose
        pass
    elif action == 1:  # q
        cast_action("q", 0.1)
    elif action == 2:  # w
        cast_action("w", 0.1)
    elif action == 3:  # e
        cast_action("e", 0.1)
    elif action == 4:  # r
        cast_action("r", 0.1)
    elif action == 5:  # d
        cast_action("d", 0.1)
    elif action == 6:  # f
        cast_action("f", 0.1)
    elif action == 7:  # go home
        recall(active_player)


# player = getGameData.player()
# time.sleep(5)
# recall(recall)
