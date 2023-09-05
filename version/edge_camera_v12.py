import cv2
import numpy as np
import multiprocessing as mp
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


def display_original(timeData, shared_array, shape):
    print(f'process1: 시작')
    tm1 = cv2.TickMeter()
    while True:
        tm1.start()
        frame1 = np.frombuffer(shared_array, dtype=np.uint8).reshape(shape)
        tm1.stop()
        fps1 = tm1.getFPS()    
        if fps1 is not None:
            cv2.putText(frame1, 'FPS: {:.2f}'.format(fps1), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        timeData = mp.Array('c', 64)
        cv2.imshow("Original Frame", frame1)
        timeData = mp.Array('c', 64)
        cv2.imwrite('camera/original/' + timeData.value.decode("utf-8") + '.jpg', frame1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

buffer_filled = mp.Value('i', 0)
def display_process(timeData, shared_array, shape):
    global buffer_filled
    if buffer_filled.value:
        return
    buffer_filled.value = 1
    # ... rest of the function logic ...
    buffer_filled.value = 0
    print(f'process2: 시작')
    face_detector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (320, 320))
    while True:
        
        frame = np.frombuffer(shared_array, dtype=np.uint8).reshape(shape)

        frame_num = 0
        person = 0
            
        # 카메라ID 설정
        cameraID = 'esqure_01'
        deviceId = 0 
        count = 0
        tick = 0
        constancy = 0
        boxes = None
        indexes = None

        tm = cv2.TickMeter()
        while True:
            frame_num += 1
                
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
                # print(f'frame_num: {frame_num}')
                # boxes = detectbox(frame)
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
                cv2.putText(output, 'FPS: {:.2f}'.format(fps), (200, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))


        


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


            timeData = mp.Array('c', 64)
            cv2.imshow("Mosaic Frame", frame)            
            timeData = mp.Array('c', 64)
            cv2.imwrite(f'camera/process/{timeData.value.decode("utf-8")}.jpg', frame)
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def main():
    cap = cv2.VideoCapture(0)
    timeData = mp.Array('c', 64)
    ret, frame = cap.read()
    if not ret:
        print("Failed to open camera.")
        return

    shape = frame.shape
    shared_array = mp.Array('B', frame.size, lock=False)
    timeData = mp.Array('c', 64)

    if not cap.isOpened():
        exit()
    

    process1 = mp.Process(target=display_original, args=(timeData, shared_array, shape))
    process2 = mp.Process(target=display_process, args=(timeData, shared_array, shape))
    print(f'process1: 대기중')
    process1.start()
    print(f'process2: 대기중')
    process2.start()

    while True:
        timeData = mp.Array('c', 64)
        ret, frame = cap.read()
        if ret:
            shared_array[:] = frame.flatten()

    cap.release()
    process1.terminate()
    process2.terminate()

if __name__ == '__main__':
    main()
