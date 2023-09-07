from ultralytics import YOLO
import cv2
import time
import time
from cv2 import *  
import cv2
from ultralytics import YOLO
import os
      
def yolo_process(frame):
    output = frame.copy()
    outs = yolo_model(output, classes=0, conf=0.3)
    boxes = []

    for see in outs:
        for detection in see:  # see == outs[0,1,2,3 ..., n]
            boxes_buf = detection.boxes  #
            box = boxes_buf[0]  # 가장 높은 conf 값을 갖은 것
            x, y, w, h = box.xywh[0].tolist()
            boxes.append([x, y, w, h])

    for i in range(len(boxes)):
        if i < len(boxes):
            x, y, w, h = boxes[i]
            x, y, w, h = int(x), int(y), int(w), int(h)
            x1 = x - w // 2
            x2 = x + w // 2
            y1 = y - h // 2
            y2 = y + h // 2
            # reduce the y2 coordinate by 20% of the original height
            y2_new = int(y1 + (y2 - y1) * 0.3)
            # reduce the x2 coordinate by 50% of the original width
            # x2_new = int(x1 + (x2 - x1) / 4 * 3)
            x2_new = int(x2 - (x2 - x1) / 4)
            # increase the x1 coordinate by 50% of the original width
            x1_new = int(x1 + (x2 - x1) / 4)

            # print(f'x = {x}, {type(x)}, y = {y}, {type(y)}, w = {w}, {type(w)}, h = {h}, {type(h)}')
            region = output[y1:y2_new, x1_new: x2_new]
            height, width = region.shape[:2]
            w = int(width * 0.1)
            h = int(height * 0.1)
            if w <= 0:
                w = 1
            if h <= 0:
                h = 1

            small = cv2.resize(region, (w, h), interpolation=cv2.INTER_AREA)
            mosaic = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
            output[y1:y2_new, x1_new:x2_new] = mosaic
    return output
        
    
        
        
    
        
