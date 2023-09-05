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

def send_frame(url, frame, timeData):
    _, img_encoded = cv2.imencode(".jpg", frame)
    requests.post(url, data={'time': timeData, 'cameraID': cameraID}, files={'frame': ('image.jpg', img_encoded, 'image/jpeg')})

def display_original(frame, timeData):
    # tm2 = cv2.TickMeter()
    # tm2.start()
    OriginFrame = frame.copy()
    # tm2.stop()
    # fps2 = tm2.getFPS()
    # if fps2 is not None:
        # cv2.putText(OriginFrame, 'FPS: {:.2f}'.format(fps2), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    cv2.imwrite(f'camera/original/{timeData}.jpg', OriginFrame)

def display_process(frame, timeData, face_detector, count, tick, constancy):
            global person, process_thread_is_running
            tm = cv2.TickMeter()
            channels = 1 if len(frame.shape) == 2 else frame.shape[2]
            if channels == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if channels == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            height, width, _ = frame.shape
            face_detector.setInputSize((width, height))

            i = 0

            tm.start()
            _, results = face_detector.detect(frame)

            for det in (results if results is not None else []):
                i = i + 1
            if person != i:
                count = 6
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
        
                height, width = region.shape[:2]
                w = int(width * 0.05)
                h = int(height * 0.05)
                if w <= 0: w = 1
                if h <= 0: h = 1
                small = cv2.resize(region, (w, h), interpolation=cv2.INTER_AREA)
                mosaic = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
            
                img_mosaic = output.copy()
                img_mosaic[start_y:start_y + end_y, start_x:start_x + end_x] = mosaic
                output = img_mosaic
            frame = output
        
            cv2.imwrite(f'camera/process/{timeData}.jpg', frame)
            process_thread_is_running = False

if __name__ == '__main__':

    face_detector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (320, 320))
    # face_detector = cv2.FaceDetectorYN.create("new1_jodory.onnx", "", (320, 320))
    
    yolo_model = YOLO('yolov8n.onnx')
    
    person = 0

    cap = cv2.VideoCapture(0) #Camera
    if not cap.isOpened():
        exit()

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    frame_num = 0

    while cap.isOpened():
        
        cameraID = 'esqure_01'
        deviceId = 0 
        count = 0
        tick = 0
        constancy = 0
        
        process_thread_is_running = False

        while cv2.waitKey(1) < 0:
            frame_num += 1
            hasFrame, frame = cap.read()
            if not hasFrame:
                break
            

            tm_main = cv2.TickMeter()
            tm_main.start()
            timeData = time.time()
            origin_display_thread = threading.Thread(target=display_original, args=(frame, timeData))
            display_process_thread = threading.Thread(target=display_process, args=(frame, timeData, face_detector, count, tick, constancy))

            origin_display_thread.start()

            if not process_thread_is_running:
                process_thread_is_running = True
                display_process_thread.start()
                # display_process_thread.join()
            
            tm_main.stop()
            # print(tm_main.getTimeSec())
        tm_main.reset()
        cap.release()
        

