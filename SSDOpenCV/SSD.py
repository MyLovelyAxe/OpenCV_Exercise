import os
import numpy as np
import cv2

debug = False
webcam = True

PROTOTXT = "MobileNetSSD_deploy.prototxt"
MODEL = "MobileNetSSD_deploy.caffemodel"
INP_VIDEO_PATH = 'cars.gif'

if debug:
    PROTOTXT = os.path.join('SSDOpenCV',PROTOTXT)
    MODEL = os.path.join('SSDOpenCV',MODEL)
    INP_VIDEO_PATH = os.path.join('SSDOpenCV',INP_VIDEO_PATH)

GPU_SUPPORT = 1
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", 
           "bottle", "bus",  "car", "cat", "chair", "cow", 
           "diningtable", "dog", "horse", "motorbike", "person", 
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

if not webcam:
    cap = cv2.VideoCapture(INP_VIDEO_PATH)
else:
    cap = cv2.VideoCapture(0) # choose webcam oject with 0, if only one webcam is available
    cap.set(3,640) # set height, whose id is 3 in setting
    cap.set(4,480) # set width, whose id is 4 in setting
    cap.set(10,20) # set brightness, whose id is 10 in setting


net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
if GPU_SUPPORT:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

while True:
    success, img = cap.read() # read frames one by one, return img and whether it is successful

    if success:
        # cv2.imshow("Image", img) # show the current frame
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        # dimention of detection: [1, 1, 100, 7]
        #    dim-0: number of sample, i.e. one frame to detect
        #    dim-1: number of class, i.e. only "car" class in this case
        #    dim-2: number of anchor boxes
        #    dim-3: [what1, class_idx, confidence, bbox_leftupper_x, bbox_leftupper_y, bbox_rightlower_x, bbox_rightlower_y]
        detections = net.forward()
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = "{}: {:.2f}%".format(CLASSES[idx],confidence*100)
                cv2.rectangle(img, (startX, startY), (endX, endY),    COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(img, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    cv2.imshow('Video', img) # show the current frame
    if cv2.waitKey(1) & 0xFF == ord('q'): # wait for specific keyboard 'q' to close the window
        break
