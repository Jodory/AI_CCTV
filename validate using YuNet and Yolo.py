from ultralytics import YOLO
import cv2
import time
import numpy as np
import time
from cv2 import *  
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from ultralytics import YOLO
import os
import datetime
      
def yunet_process(frame, file_name, face_detector, count, tick, constancy):
            global person, process_thread_is_running
            i = 0  
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
                _, results = face_detector.detect(frame)
            
            output = frame.copy()
            
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
            return output
            # cv2.imwrite(f'camera/process/{file_name}', output)
            # cv2.imwrite(f'camera/esqure_01/process/20230409/0506/{file_name}', output)
            # process_thread_is_running = False

def yolo_process(frame, file_name):
    output = frame.copy()
    outs = yolo_model(output, classes=0, conf=0.3)
    # annotated_frame = outs[0].plot()
    # cv2.imwrite(f'camera/original/{file_name}.jpg', annotated_frame)
    # -- 탐지한 객체의 클래스 예측
    # class_ids = []
    boxes = []

    # outs == 객체 개수
    # detection == box x1,y1, x2, y2
    for see in outs:
        for detection in see:  # see == outs[0,1,2,3 ..., n]
            boxes_buf = detection.boxes  #
            box = boxes_buf[0]  # 가장 높은 conf 값을 갖은 것
            x, y, w, h = box.xywh[0].tolist()
            boxes.append([x, y, w, h])

    # print(f'len(boxes) = {len(boxes)}')
    for i in range(len(boxes)):
        if i < len(boxes):
            # print(f'boxes[{i}] = {boxes[i]}')
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
    # return output
    cv2.imwrite(f'camera/process/{file_name}', output)


if __name__ == '__main__':

    # 모델 load
    yolo_model = YOLO('yolov8n.pt')
    face_detector = cv2.FaceDetectorYN.create(model="face_detection_yunet_2023mar.onnx", config="", input_size=(320, 320), backend_id=3, target_id=0)
    
    # 현재 날짜와 시간 가져오기
    now = datetime.datetime.now()
    
    dir_list = []
    # dir_list = ['20230901/1144', '20230901/1145', '20230901/1145']
    # dir format: yyyymmdd/hhmm 
    dir = (now.strftime('%Y%m%d') + '/' + now.strftime('%H%M'))

    if not dir in dir_list:
        dir_list.append(dir)    
    
    # 이미지가 있는 폴더의 경로
    original_folder = './camera/esqure_01/original/20230409/0506'
    process_folder = './camera/esqure_01/process/20230409/0506'
    yolo_folder = './camera/esqure_01/process/20230409/0506'
    
    # original_folder = './camera//original'
    # process_folder = './camera/process'
    person = 0
    frame_num = 0
     
    start_time = time.time()
    
    # 각 폴더에서 모든 jpg 파일 이름 가져오기
    original_files = set([f for f in os.listdir(original_folder) if f.endswith('.jpg')])
    processed_files = set([f for f in os.listdir(process_folder) if f.endswith('.jpg')])
    
    # original 폴더의 이미지 중 process 폴더에 이미 존재하지 않는 이미지만 처리
    to_process_files = original_files - processed_files
    to_process_files_num = len(to_process_files)
    # print(to_process_files)

    # My notebook size: 640.0 x 480.0
    # print("size: {0} x {1}".format(width, height))
    start_time = time.time()
    
    '''
    end_time = time.time()
    elapsed_time = end_time - start_time  # 걸린 시간 계산
    print(f"처리 시간: {elapsed_time:.4f} 초")
        print(f'fps: {4470 / elapsed_time}')
    '''

    for image_file in to_process_files:
        start_time2 = time.time()
        cameraID = 'esqure_01'
        deviceId = 0 
        count = 0
        tick = 0
        constancy = 0
        
        process_thread_is_running = False
        frame = cv2.imread(os.path.join(original_folder, image_file))

        # frame.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # frame.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        channels = 1 if len(frame.shape) == 2 else frame.shape[2]
        if channels == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if channels == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        height, width, _ = frame.shape
        face_detector.setInputSize((width, height))
        
        # frame = yunet_process(frame, image_file, face_detector, count, tick, constancy)
        yolo_process(frame, image_file)
        
        # frame = yolo_process(frame, image_file)
        # yunet_process(frame, image_file, face_detector, count, tick, constancy)
        
        
        end_time2 = time.time()
        elapsed_time = end_time2 - start_time2  # 걸린 시간 계산
        print(f"처리 시간: {elapsed_time:.4f} 초")
        '''
        yunet_process_thread = threading.Thread(target=yunet_process, args=(frame, image_file, face_detector, count, tick, constancy))
        yolo_process_thread = threading.Thread(target=yolo_process, args=(frame, image_file))
        
        
        yunet_process_thread.start()
        yolo_process_thread.start()
        
        yunet_process_thread.join()
        yolo_process_thread.join()
        
        
        if not process_thread_is_running:
            process_thread_is_running = True
            yunet_process_thread.start()
            yolo_process_thread.start()
            
            yunet_process_thread.join()
            yolo_process_thread.join()
        '''
    end_time = time.time()
    elapsed_time = end_time - start_time  # 걸린 시간 계산
    print(f"처리 시간: {elapsed_time:.4f} 초")
    print(f'fps: {to_process_files_num / elapsed_time}')
