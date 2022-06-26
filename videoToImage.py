import cv2

cap = cv2.VideoCapture("1.mp4")
filename = 0
while(True):
    ret, frame = cap.read()

    if filename % 120 == 0:
        cv2.imwrite("./ImageData2/2v1_" + str(filename) + ".jpg", frame)
        cv2.imshow('frame', frame)
    filename+=1
    if cv2.waitKey(1) & 0xFF == ord('0'):
        break

cap.release()
cv2.destroyAllWindows()