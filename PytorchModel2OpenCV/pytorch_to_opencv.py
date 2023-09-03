import os
import cv2
import argparse
import numpy as np
import torch
from torchvision import models
from torch.autograd import Variable

"""
This script is a prototype of:
    1. train a model with pytorch (in script, the model is pretrained and offered by pytorch library)
    2. save trained model as .noox file
    3. load .noox into OpenCV
    4. under framework of OpenCV, deploy the trained model for e.g. image classification, object detection .etc
"""

def get_args():

    parser = argparse.ArgumentParser(description='deploy pre-trained model from Pytorch into OpenCV')
    parser.add_argument('--onnx_model_name',type=str,default='resnet50',help='which model from pytorch to import')
    parser.add_argument('--test_img_path',type=str,default='squirrel_cls.jpg',help='image to test model')
    parser.add_argument('--generate_onnx',type=bool,default=False,help='whether export pytorch model as .onnx')
    args = parser.parse_args()

    return args

def getModelPath(modelName):

    onnx_model_name = f"{modelName}.onnx"
    os.makedirs("models", exist_ok=True)
    full_model_path = os.path.join("models", onnx_model_name)

    return full_model_path

def getPretrainedPytorchModel(modelName):

    # get model path
    full_model_path = getModelPath(modelName)

    # get pretrained model from pytorch
    if modelName == 'resnet50':
        original_model = models.resnet50(pretrained=True)

    # generate model input
    generated_input = Variable(torch.randn(1, 3, 224, 224))

    # model export into ONNX format
    torch.onnx.export(
    original_model,
    generated_input,
    full_model_path,
    verbose=True,
    input_names=["input"],
    output_names=["output"],
    opset_version=11
    )

def loadModel(full_model_path, showLayerName=False):
    opencv_net = cv2.dnn.readNetFromONNX(full_model_path)
    if showLayerName:
        print("OpenCV model was successfully read. Layer IDs: \n", opencv_net.getLayerNames())
    return opencv_net

def preprocessImage(img_path):
    # read the image
    input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    input_img = input_img.astype(np.float32)
    input_img = cv2.resize(input_img, (256, 256))
    # define preprocess parameters
    mean = np.array([0.485, 0.456, 0.406]) * 255.0
    scale = 1 / 255.0
    std = [0.229, 0.224, 0.225]
    # prepare input blob to fit the model input:
    # 1. subtract mean
    # 2. scale to set pixel values from 0 to 1
    input_blob = cv2.dnn.blobFromImage(
    image=input_img,
    scalefactor=scale,
    size=(224, 224), # img target size
    mean=mean,
    swapRB=True, # BGR -> RGB
    crop=True # center crop
    )
    # 3. divide by std
    input_blob[0] /= np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    return input_blob

def getImagenetLabels(label_txt):

    imagenet_labels = []
    with open(label_txt, 'r') as file:
        for line in file:
            imagenet_labels.append(line.strip())  # Use strip() to remove leading/trailing whitespace
    print(f'how many classes: {len(imagenet_labels)}')
    return imagenet_labels

def deployNet(full_model_path,test_img_path):
    # get model
    opencv_net = loadModel(full_model_path)
    # get preprocessed image
    preprocessed_img = preprocessImage(test_img_path)
    # get imagenet labels
    imagenet_labels = getImagenetLabels(label_txt='classification_classes_ILSVRC2012.txt')
    # set OpenCV DNN input
    opencv_net.setInput(preprocessed_img)
    # OpenCV DNN inference
    out = opencv_net.forward()
    print("OpenCV DNN prediction: \n")
    print("* shape: ", out.shape)
    # get the predicted class ID
    imagenet_class_id = np.argmax(out)
    # get confidence
    confidence = out[0][imagenet_class_id]
    print("* class ID: {}, label: {}".format(imagenet_class_id, imagenet_labels[imagenet_class_id]))
    print("* confidence: {:.4f}".format(confidence))

if __name__ == "__main__":

    args = get_args()

    if args.generate_onnx:
        getPretrainedPytorchModel(args.onnx_model_name)

    full_model_path = getModelPath(args.onnx_model_name)
    deployNet(full_model_path,args.test_img_path)
