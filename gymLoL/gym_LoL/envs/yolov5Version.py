import cv2
import torch
from PIL import Image
from mss import mss
import time
import numpy as np
from collections import defaultdict


class Detect:
    def __init__(self, conf=0.8, yolo='yolov5s'):
        self.model_path = r'D:\FinalProject\gymLoL\gym_LoL\envs\model\Yolov5model\Yolov5-640p-200time\best.pt'  # path to model
        self.conf = conf  # default is 0.30
        self.yolo = yolo  # default is yolov5s
        self.dict = defaultdict(lambda: None)

    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)  # local model

        return model

    def run(self, img):
        self.dict.clear()
        model = self.load_model()
        model.conf = self.conf
        results = model(img, 1080)
        for result in results.pandas().xyxy[0].to_dict(orient="records"):
            if result['confidence'] > 0.8:
                # con = result['confidence']
                # cs = result['class']
                x1 = int(result['xmin'])
                y1 = int(result['ymin'])
                x2 = int(result['xmax'])
                y2 = int(result['ymax'])
                name = result['name']
                print(result)

                self.dict[name] = [(x2 - x1) / 2 + x1, (y2 - y1) / 2 + y1]

        print(self.dict)
        return self.dict, results


if __name__ == '__main__':

    model = Detect()

    EOG_BOX = {"left": 960, "top": 540, "width": 1920, "height": 1080}

    sct = mss()
    img0 = cv2.imread(r'D:\FinalProject\Image\testImage\Screen02.png')
    while True:

        start_time = time.time()
        # get image from screen

        frame = sct.grab(EOG_BOX)
        frame_img = np.array(frame)
        # input image into yolov5 model
        dict, results = model.run(frame_img)
        print("dict is done")
        # frame = np.squeeze(results.imgs)
        # draw the box
        frame = np.squeeze(results.render())
        frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
        cycle_time = time.time() - start_time
        cv2.putText(frame, "FPS: {}".format(str(round(1 / cycle_time, 2))), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)
        cv2.imshow('LOL', frame)
        # resize the show window
        # cv2.resizeWindow('LOL', 1920,1080)

        if cv2.waitKey(25) & 0xFF == ord('0'):  # 按q退出，记得输入切成英语再按q
            cv2.destroyAllWindows()
            break
