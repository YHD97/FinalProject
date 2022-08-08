import torch
from mss import mss
import time
import numpy as np
import cv2

# from yolov7.hubconf import custom
# model = custom(path_or_model='yolov7/best.pt')
model = torch.hub.load('yolov5-master', 'custom', path="gymLoL/gym_LoL/envs/model/best (1).pt",source='local')

EOG_BOX = {"left": 960, "top": 540, "width": 1920, "height": 1080}

sct = mss()

while True:

    start_time = time.time()
    # get image from screen

    frame = sct.grab(EOG_BOX)
    frame_img = np.array(frame)
    # input image into yolov5 model
    results = model(frame_img, size=640)

    # get position
    for result in results.pandas().xyxy[0].to_dict(orient="records"):
        if result['confidence'] > 0.8:
            con = result['confidence']
            cs = result['class']
            x1 = int(result['xmin'])
            y1 = int(result['ymin'])
            x2 = int(result['xmax'])
            y2 = int(result['ymax'])
            name = result['name']
            # Do whatever you want
            print(result)
        # print(cs, (x2 - x1) / 2 + x1, (y2 - y1) / 2 + y1, name)
    # frame = np.squeeze(results.imgs)
    # draw the box
    frame = np.squeeze(results.render())

    cycle_time = time.time() - start_time
    cv2.putText(frame, "FPS: {}".format(str(round(1 / cycle_time, 2))), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 2)
    cv2.imshow('LOL', frame)
    # resize the show window
    cv2.resizeWindow('LOL', 1920, 1080)

    if cv2.waitKey(25) & 0xFF == ord('0'):  # 按q退出，记得输入切成英语再按q
        cv2.destroyAllWindows()
        break
