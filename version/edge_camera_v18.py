# 230414 Testing Code
import onnx
import numpy as np
import time
from cv2 import *  
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import requests
import threading
import subprocess
from ultralytics import YOLO
'''
def mosaic_area(img, box, ratio=0.05):
    # 모자이크를 적용할 영역을 구합니다.
    for i in range(len(box)):
        if box[i] < 0: 
            box[i] = 0
    start_x, start_y, end_x, end_y = box
    
    region = img[start_y:start_y + end_y, start_x:start_x + end_x]

        # 모자이크를 적용할 영역의 크기를 구합니다.
    height, width = region.shape[:2]
    w = int(width * ratio)
    h = int(height * ratio)
    if w <= 0: w = 1
    if h <= 0: h = 1
    # 영역을 축소합니다.
    small = cv2.resize(region, (w, h), interpolation=cv2.INTER_AREA)
    # 축소한 영역을 다시 확대합니다.
    mosaic = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    
    # 모자이크를 적용한 이미지를 생성합니다.
    img_mosaic = img.copy()
    img_mosaic[start_y:start_y + end_y, start_x:start_x + end_x] = mosaic
    # cv2.imshow("img_mosaic", img_mosaic)
    return img_mosaic


def detectbox(frame):
    img = frame
    # print('detectbox')
    outs = yolo_model(img, classes = 0, conf=0.3)
    # annotated_frame = outs[0].plot()

    # cv2.imwrite(f'camera/original/{timeData}.jpg', annotated_frame)
    
    # -- 탐지한 객체의 클래스 예측
    # class_ids = []
    boxes_buf = []

    # outs == 객체 개수
    # detection == box x1,y1, x2, y2
    for see in outs:
        for detection in see: # see == outs[0,1,2,3 ..., n]
            boxes = detection.boxes # 
            box = boxes[0] # 가장 높은 conf 값을 갖은 것
            x, y, w, h = box.xywh[0].tolist()
            boxes_buf.append([x, y, w, h])
    return boxes_buf
'''
'''
# results = bbox ==? yunet
# boxes = yolo
def visualize(frame, results, boxes, box_color=(255, 0, 0), text_color=(0, 0, 255), fps=None):
    output = frame.copy()
    if fps is not None:
        cv2.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    if boxes == 0:
        for det in (results if results is not None else []):
            bbox = list(map(int, det[:4]))
            
            for i in range(len(bbox)):
                if bbox[i] < 0: 
                    bbox[i] = 0
            start_x, start_y, end_x, end_y = bbox
            
            region = output[start_y:start_y + end_y, start_x:start_x + end_x]
        
                # 모자이크를 적용할 영역의 크기를 구합니다.
            height, width = region.shape[:2]
            w = int(width * 0.05)
            h = int(height * 0.05)
            if w <= 0: w = 1
            if h <= 0: h = 1
            # 영역을 축소합니다.
            small = cv2.resize(region, (w, h), interpolation=cv2.INTER_AREA)
            # 축소한 영역을 다시 확대합니다.
            mosaic = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
            
            # 모자이크를 적용한 이미지를 생성합니다.
            img_mosaic = output.copy()
            img_mosaic[start_y:start_y + end_y, start_x:start_x + end_x] = mosaic
            output = img_mosaic
            #output = mosaic_area(output, bbox)
            # cv2.rectangle(output, bbox , box_color, -1)
'''  
'''
    else:
        # print(f'len(boxes) = {len(boxes)}')
        for i in range(len(boxes)):
            if i < len(boxes):
                # print(f'boxes[{i}] = {boxes[i]}')
                x, y, w, h = boxes[i]
                x, y, w, h = int(x), int(y), int(w), int(h)
                x1 = x - w//2 
                x2 = x + w//2
                y1 = y - h//2
                y2 = y + h//2
                # reduce the y2 coordinate by 20% of the original height
                y2_new = int(y1 + (y2-y1)*0.3)
                # reduce the x2 coordinate by 50% of the original width
                x2_new = int(x1 + (x2-x1)/4 * 3)
                # increase the x1 coordinate by 50% of the original width
                x1_new = int(x1 + (x2-x1)/4)
                detected = False
                # print(x1_new, y1, x2_new, y2_new)
                for det in (results if results is not None else []):
                    bbox = list(map(int, det[:4]))
                    # print(f'99: {bbox}')
                    output = mosaic_area(output, bbox)
                    # print(f'bbox = {bbox}')
                    # cv2.rectangle(output, bbox , box_color, -1)
                    if x1_new < bbox[0] and y1 < bbox[1] and x2_new > bbox[2] and y2_new > bbox[3]:
                        detected = True

                if detected:
                    continue
                else:
                    # print(f'x = {x}, {type(x)}, y = {y}, {type(y)}, w = {w}, {type(w)}, h = {h}, {type(h)}')
                    region = output[y1:y2_new, x1_new: x2_new]
                    height, width = region.shape[:2]
                    w = int(width * 0.1)
                    h = int(height * 0.1)
                    if w <= 0:
                        w = 1
                    if h <= 0:
                        h = 1

                    # 영역을 축소합니다.
                    small = cv2.resize(region, (w, h), interpolation=cv2.INTER_AREA)
                    
                    # 축소한 영역을 다시 확대합니다.
                    mosaic = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
                    
                    # 모자이크를 적용한 이미지를 생성합니다.
                    
                    output[y1:y2_new, x1_new:x2_new] = mosaic
                    # cv2.imshow("img_mosaic", img_mosaic)
'''
    #return output

def send_frame(url, frame, timeData):
    _, img_encoded = cv2.imencode(".jpg", frame)
    requests.post(url, data={'time': timeData, 'cameraID': cameraID}, files={'frame': ('image.jpg', img_encoded, 'image/jpeg')})

def display_original(frame, timeData):
    
    tm2 = cv2.TickMeter()
    tm2.start()
    time.sleep(1/45)  # 30fps
    # ret, OriginFrame = cap.read() 
    OriginFrame = frame.copy()
    tm2.stop()
    fps2 = tm2.getFPS()
    if fps2 is not None:
        cv2.putText(OriginFrame, 'FPS: {:.2f}'.format(fps2), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    # cv2.imshow('Original', OriginFrame)
    # timeData = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # timeData = time.time()
    cv2.imwrite(f'camera/original/{timeData}.jpg', OriginFrame)

def display_process(frame, timeData, face_detector, count, tick, constancy):
            global person
            # If the frame is not 3 channels, convert it to 3 channels
            channels = 1 if len(frame.shape) == 2 else frame.shape[2]
            if channels == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if channels == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            height, width, _ = frame.shape
            face_detector.setInputSize((width, height))



            i = 0

            # Inference
            tm.start()
            _, results = face_detector.detect(frame) # results is a tuple

            for det in (results if results is not None else []):
                i = i + 1

            # print(f'frame_num: {frame_num}, i = {i}, person = {person}, count = {count}, tick = {tick}')
            if person != i:
                # print(f'different!! frame_num = {frame_num}, i = {i}, person = {person}')
                count = 6
                #if tick != 1:
                #    count = 6
                if person > i:
                    tick = 1
                    constancy = person
                    face_detector.setScoreThreshold(0.3)
                    _, results = face_detector.detect(frame)
            person = i

            if count == 6:
                count = count - 1
            elif count > 0:
                count = count - 1
            else:
                constancy = 0

            if i >= constancy and tick == 1:
                tick = 0
                face_detector.setScoreThreshold(0.7)

            # Default fps = tm.getFPS()
            tm.stop()
            
            output = frame.copy()
            fps = tm.getFPS()
            
            if fps is not None:
                cv2.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

            
            for det in (results if results is not None else []):
                bbox = list(map(int, det[:4]))
        
                for i in range(len(bbox)):
                    if bbox[i] < 0: 
                        bbox[i] = 0
                start_x, start_y, end_x, end_y = bbox
            
                region = output[start_y:start_y + end_y, start_x:start_x + end_x]
        
                    # 모자이크를 적용할 영역의 크기를 구합니다.
                height, width = region.shape[:2]
                w = int(width * 0.05)
                h = int(height * 0.05)
                if w <= 0: w = 1
                if h <= 0: h = 1
                # 영역을 축소합니다.
                small = cv2.resize(region, (w, h), interpolation=cv2.INTER_AREA)
                # 축소한 영역을 다시 확대합니다.
                mosaic = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
            
                # 모자이크를 적용한 이미지를 생성합니다.
                img_mosaic = output.copy()
                img_mosaic[start_y:start_y + end_y, start_x:start_x + end_x] = mosaic
                output = img_mosaic
            frame = output
            # cv2.imshow('Mosaic', frame)
            cv2.imwrite(f'camera/process/{timeData}.jpg', frame)

if __name__ == '__main__':

    OriginURL = "http://bangwol08.iptime.org:20002/camera/original"
    ProURL = "http://bangwol08.iptime.org:20002/camera/process"

    face_detector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (320, 320))
    # face_detector = cv2.FaceDetectorYN.create("new1_jodory.onnx", "", (320, 320))
    
    yolo_model = YOLO('yolov8n.onnx')
    
    person = 0

    cap = cv2.VideoCapture(0) #Camera
    # If     Camera Module is not exist, exit this code
    if not cap.isOpened():
        exit()

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # My notebook size: 640.0 x 480.0
    # print("size: {0} x {1}".format(width, height))
    frame_num = 0

    

    while cap.isOpened():
        
        # 카메라ID 설정
        cameraID = 'esqure_01'
        deviceId = 0 
        count = 0
        tick = 0
        constancy = 0

        tm = cv2.TickMeter()
        while cv2.waitKey(1) < 0:
            frame_num += 1
            hasFrame, frame = cap.read()
            if not hasFrame:
                # print('No frames grabbed!')
                break
            
            timeData = time.time()
            # Assuming that the video capture object is named 'cap'
            origin_display_thread = threading.Thread(target=display_original, args=(frame, timeData))
            origin_display_thread.start()
            # if frame is not None:
            #     OriginFrame = frame.copy()
            # timeData = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            display_process_thread = threading.Thread(target=display_process, args=(frame, timeData, face_detector, count, tick, constancy))
            display_process_thread.start()
            display_process_thread.join()

            
            

            
            
            #send_frame(OriginURL, OriginFrame, timeData)
            #send_frame(ProURL, frame, timeData)
            '''
            t1 = threading.Thread(target=send_frame, args=(OriginURL, OriginFrame, timeData))
            t2 = threading.Thread(target=send_frame, args=(ProURL, frame, timeData))
            t1.start()
            t2.start()
            '''
            
            # cv2.imwrite(f'camera/original/{timeData}.jpg', OriginFrame)
        tm.reset()
        cap.release()
        

