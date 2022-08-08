import cv2

vc = cv2.VideoCapture('1.mp4')  # 读入视频文件
c = 1
d = 0
if vc.isOpened():  # 判断是否正常打开
    rval, frame = vc.read()
else:
    rval = False

timeF = 10  # 视频帧计数间隔频率
# try:

while rval:  # 循环读取视频帧
    rval, frame = vc.read()
    if (c % timeF == 0):  # 每隔timeF帧进行存储操作
        d = d+1
        cv2.imwrite('data/Images2/' + str(d) + '.jpg', frame)  # 存储为图像
        print(d)
    c = c + 1
    cv2.waitKey(1)
vc.release()