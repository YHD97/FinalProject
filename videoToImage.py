import cv2

vc = cv2.VideoCapture('100zhen.mp4')  # load video
c = 1
d = 0
if vc.isOpened():  #
    rval, frame = vc.read()
else:
    rval = False

timeF = 10  #
# try:

while rval:  #
    rval, frame = vc.read()
    if (c % timeF == 0):
        d = d+1
        cv2.imwrite('ImageTestSpeed/YoloV5ForTrain100frame/' + str(d) + '.jpg', frame)  # 存储为图像
        print(d)
    c = c + 1
    cv2.waitKey(1)
vc.release()