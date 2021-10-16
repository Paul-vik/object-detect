import cv2
import numpy as np


video = cv2.VideoCapture('c.mp4')


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]




net = cv2.dnn.readNetFromCaffe('New Text Document.txt','MobileNetSSD_deploy.caffemodel')



while True:
    ret, frame = video.read()
    # frame = cv2.resize(frame, (640,480))
    (h,w) = frame.shape[:2]
    blop = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 0.007843, (300,300), 127.5)
    net.setInput(blop)
    detection = net.forward()
    for i in np.arange(0, detection.shape[2]):
        confidence = detection[0,0,i,2]
        if confidence>0.5:
            id= detection[0,0,i,1]
            box= detection[0,0,i,3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame,(startX-1, startY-40), (endX+1, startY-3), (0,255,0), -1)
            cv2.rectangle(frame,(startX, startY), (endX, endY), (0, 255,0),4)
            cv2.putText(frame, CLASSES[int(id)], (startX+10, startY-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255))
    cv2.imshow("frame", frame)
    k= cv2.waitKey(1)
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()