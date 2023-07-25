import numpy as np
import cv2

cap = cv2.VideoCapture(0) # choose webcam oject with 0, if only one webcam is available
cap.set(3,640) # set height, whose id is 3 in setting
cap.set(4,480) # set width, whose id is 4 in setting
cap.set(10,20) # set brightness, whose id is 10 in setting

whT = 320 # width-height-Target, 320 because the configuration from website is of size 320x320

classesFile = 'coco.names'
classNames = []
with open(classesFile,'rt') as F:
    classesNames = F.read().rstrip('\n').split('\n')
# print(classesNames)
# print(f'We have {len(classNames)} classes')

modelConfiguration = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

### create network
net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

while True:
    success, img = cap.read() # read frames one by one, return img and whether it is successful
    blob = cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    # print(layerNames)
    ### there are 3 output layers in yolo networks, we want all of them
    outputIndex = net.getUnconnectedOutLayers() # [200,227,254] but the index starts from 1, so in order to slice we have to use [199,226,253]
    # print(outputIndex)
    outputNames = [layerNames[i-1] for i in outputIndex] # [200-1,227-1,254-1] = [199,226,253]
    # print(outputNames) # ['yolo_82', 'yolo_94', 'yolo_106']

    outputs = net.forward(outputNames)
    print(f'all outputs: type is {type(outputs)}, length is {len(outputs)}')
    print(f'1st output: type is {type(outputs[0])}, shape is {outputs[0].shape}')
    print(f'2nd output: type is {type(outputs[1])}, shape is {outputs[1].shape}')
    print(f'3rd output: type is {type(outputs[2])}, shape is {outputs[2].shape}')

    cv2.imshow('Video', img) # show the current frame
    if cv2.waitKey(1) & 0xFF == ord('q'): # wait for specific keyboard 'q' to close the window
        break