import math
import argparse
import cv2
from ultralytics import YOLO

def get_args():

    parser = argparse.ArgumentParser(description='deploy YOLOV8 model from OpenCV')
    parser.add_argument('--debug',type=bool,default=False,help='debug or not')
    parser.add_argument('--webcam',type=bool,default=False,help='whether to use webcam')
    parser.add_argument('--classes_file',type=str,default='coco.names',help='original file containing names of classes')
    parser.add_argument('--weights',type=str,default='yolov8n.pt',
                        choices=['yolov8n.pt','yolov8m.pt','yolov8l.pt'],
                        help='n: nano verison; m: medium verison; l: large verison;')
    args = parser.parse_args()
    return args

def getCap():

    cap = cv2.VideoCapture(0) # choose webcam oject with 0, if only one webcam is available
    cap.set(3,640) # set height, whose id is 3 in setting
    cap.set(4,480) # set width, whose id is 4 in setting
    cap.set(10,20) # set brightness, whose id is 10 in setting
    return cap

def getClassNames(classesFile):

    classNames = []
    with open(classesFile,'rt') as F:
        classNames = F.read().rstrip('\n').split('\n')
    return classNames

def main():

    args = get_args()
    model = YOLO(args.weights)
    classNames = getClassNames(args.classes_file)
    cap = getCap()
    while True:
        # read frames one by one, return img and whether it is successful
        success, img = cap.read()
        # stream=True will use a generator, making it more efficient
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # you can also use: box.xywd # get x, y, w, d
                x1, y1, x2, y2 = box.xyxy[0] # get xy-coordinates of left-upper and right-lower corners
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                confidence = math.ceil((box.conf[0])*100)/100
                classId = int(box.cls[0])
                print(f'x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2}')
                # mark out the detected object
                cv2.rectangle(img,
                            pt1=(x1,y1),
                            pt2=(x2,y2),
                            color=(255,0,255),
                            thickness=2)
                # note the class name and probility/confidence
                cv2.putText(img,
                            text=f'{classNames[classId]}: {int(confidence*100)}%',
                            org=(x1,y1-10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.6,
                            color=(255,0,255),
                            thickness=2)
            cv2.imshow('Video', img) # show the current frame
        if cv2.waitKey(1) & 0xFF == ord('q'): # wait for specific keyboard 'q' to close the window
            break

if __name__ == "__main__":
    main()
