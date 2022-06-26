# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


# def print_hi(name):
#     # 在下面的代码行中使用断点来调试脚本。
#     print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。
#
#
# # 按间距中的绿色按钮以运行脚本。
# if __name__ == '__main__':
#     print_hi('PyCharm')


import pyautogui
import numpy as np
import cv2
import time
from PIL import Image
import torch
from mss import mss
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp2/weights/best.pt')

bounding_box = {'top': 0, 'left': 0, 'width': 1980, 'height': 1080}

sct = mss()
images = {
    "play": "play.png"
}

while True:
    if pyautogui.locateOnScreen(images.get("play"), confidence=0.9):
        spot = pyautogui.locateCenterOnScreen(images.get("play"),confidence=0.9)
        if spot:
            # time.sleep(0.05)
            pyautogui.moveTo(spot)
            pyautogui.click(spot)
        print('true')
    start_time = time.time()
    # get image from screen
    frame = sct.grab(bounding_box)
    frame_img = np.array(frame)
    # input image into yolov5 model
    results = model(frame_img,size=640)
    # get
    for result in results.pandas().xyxy[0].to_dict(orient="records"):
        con = result['confidence']
        cs = result['class']
        x1 = int(result['xmin'])
        y1 = int(result['ymin'])
        x2 = int(result['xmax'])
        y2 = int(result['ymax'])
        name = result['name']
        # Do whatever you want
        print(cs,(x2-x1)/2+x1,(y2-y1)/2+y1,name)
    #frame = np.squeeze(results.imgs)
    # draw the box
    frame = np.squeeze(results.render())
    cycle_time = time.time() - start_time
    cv2.putText(frame, "FPS: {}".format(str(round(1 / cycle_time, 2))), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 2)
    cv2.imshow('LOL', frame)
    if cv2.waitKey(25) & 0xFF == ord('0'):  #按q退出，记得输入切成英语再按q
        cv2.destroyAllWindows()
        break
