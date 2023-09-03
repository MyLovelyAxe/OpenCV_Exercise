import numpy as np
import cv2

cap = cv2.VideoCapture(0) # choose webcam oject with 0, if only one webcam is available
cap.set(3,640) # set height, whose id is 3 in setting
cap.set(4,480) # set width, whose id is 4 in setting
cap.set(10,20) # set brightness, whose id is 10 in setting

whT = 320 # width-height-Target, 320 because the configuration from website is of size 320x320
confThreshold = 0.5 # if confidence larger than confThreshold, then we keep the bounding box
nmsThreshold = 0.3 # the lower nmsThreshold is, the less boxes we will get, i.e. more boxes would be suppressed

classesFile = 'coco.names'
classNames = []
with open(classesFile,'rt') as F:
    classNames = F.read().rstrip('\n').split('\n')
# print(classNames)
# print(f'We have {len(classNames)} classes')

## yolo-320
# modelConfiguration = 'yolov3.cfg'
# modelWeights = 'yolov3.weights'
## yolo-tiny
modelConfiguration = 'yolov3-tiny.cfg'
modelWeights = 'yolov3-tiny.weights'

### create network
net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs,img):
    hT,wT,cT = img.shape
    bbox = [] # contain: Cx,Cy,w,h
    classIds = []
    confs = [] # confidence
    # output[0], output[1], output[2] means output of 3 different scales,
    # i.e. output[0]: whT / 32 = 10, shape: [3, 10, 10, 85] -> [3*10*10, 85] -> [300, 85]
    # i.e. output[1]: whT / 16 = 20, shape: [3, 20, 20, 85] -> [3*20*20, 85] -> [1200, 85]
    # i.e. output[2]: whT / 8 = 40, shape: [3, 40, 40, 85] -> [3*40*40, 85] -> [4800, 85]
    # which means [num_anchor_each_scale, scale_size, scale_size, pred_vector]
    # num_anchor_each_scale=3, because yolo algorithm in paper designed 3 anchors for each of 3 scales
    for output in outputs: # 3 outputs: [300,85], [1200,85], [4800,85]
        for det in output: # det: [85,]
            scores = det[5:] # neglect first 5 values
            classId = np.argmax(scores) # find the id of the class with highest confidence
            confidence = scores[classId] # get that confidence
            if confidence > confThreshold:
                # e.g. det[2] is percentage, det[2]*wT is pixel-value
                w,h = int(det[2]*wT), int(det[3]*hT)
                # e.g. det[0] is percentage of center_x, int((det[0]*wT) - w/2) is pixel value of upper-left corner 
                x,y = int((det[0]*wT) - w/2), int((det[1]*hT) - h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    print(f'We have {len(bbox)} bounding boxes')
    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold, nmsThreshold) # Non-Maximum Suppression, output is indices to keep
    print(f'We keep these boxes, whose index are: {indices}')
    for i in indices:
        box = bbox[i]
        x,y,w,h = box
        cv2.rectangle(img,
                      pt1=(x,y),
                      pt2=(x+w,y+h),
                      color=(255,0,255),
                      thickness=2)
        cv2.putText(img,
                    text=f'{classNames[classIds[i]]} {int(confs[i]*100)}%',
                    org=(x,y-10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=(255,0,255),
                    thickness=2)

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
    ### num_box: number of produced bounding-box
    ### box_feature:
    ###         1st/2nd/3rd/4th value: center_x, center_y, width, height (in percentage, not in pixel)
    ###         5th value: confidence that an object appear in this box
    ###         the rest of values: prediction probabilities for all 80 classes
    # print(f'all outputs: type is {type(outputs)}, length is {len(outputs)}')
    # print(f'1st output: type is {type(outputs[0])}, shape is {outputs[0].shape}') # [num_box,box_feature] = [300,85]
    # print(f'2nd output: type is {type(outputs[1])}, shape is {outputs[1].shape}') # [num_box,box_feature] = [1200,85]
    # print(f'3rd output: type is {type(outputs[2])}, shape is {outputs[2].shape}') # [num_box,box_feature] = [4800,85]
    # print(f'0th box in 1st output: type is {type(outputs[0][0])}, shape is {outputs[0][0].shape}') # [num_box,box_feature] = [4800,85]
    # print(f'content inside: {outputs[0][0]}')

    findObjects(outputs,img)

    cv2.imshow('Video', img) # show the current frame
    if cv2.waitKey(1) & 0xFF == ord('q'): # wait for specific keyboard 'q' to close the window
        break