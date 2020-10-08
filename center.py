import os
import random
import darknet
import cv2
import numpy as np

configFile = "yolov3-custom.cfg"
dataFile = "obj.data"
weightsFile = "yolov3-custom_5000.weights"

thresh = 0.5

def draw_points(detections, image, colors):
    boxList = ["People", "Baseball"]
    for label, confidence, bbox in detections:
        left, top, right, bottom = darknet.bbox2points(bbox)
        width_mid = int((left + right) / 2)
        height_mid = int((top + bottom) / 2)

        if label in boxList:
            cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
        else:
            cv2.line(image, (width_mid, height_mid), (width_mid, height_mid), colors[label], 3)
    
    return image

def detection(image, network, class_names, class_colors, thresh):
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height), interpolation = cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)

    image = draw_points(detections, image_resized, class_colors)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections

def main(): 

    random.seed(3)
    network, class_names, class_colors = darknet.load_network(
        configFile, dataFile, weightsFile
    )
    
    while(True):
        imageFile = input("Enter Image Path : ")
        cv2image = cv2.imread(imageFile)
        image, detections = detection(
            cv2image, network, class_names, class_colors, thresh        
        )

        #cv2.imshow('Prediction', image)
        cv2.imwrite('Prediction.jpg', image)

if __name__ == "__main__":
    main()
