from ultralytics import YOLO
import cv2
import time
import time
from cv2 import *  
import cv2
from ultralytics import YOLO
import os
      
def yolo_process(frame):
    start_time = time.time()
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
    end_time = time.time()
    elapsed_time = end_time - start_time  # 걸린 시간 계산
    print(f"처리 시간: {elapsed_time:.4f} 초")
    return output
    
if __name__ == '__main__':

    # 모델 load
    yolo_model = YOLO('yolov8n.pt')
    
    # 경로 수정해야 함
    process_folder = './camera/esqure_01/process/20230409/0506'
    
    processed_files = set([f for f in os.listdir(process_folder) if f.endswith('.jpg')])
    
    for processed_file in processed_files:
        # mosaic_results 가 모자이크 된 이미지
        mosaic_results =  yolo_process(processed_file)
        # 처리 후 저장을 하든 mp4로 만들든 코딩해야할듯
        # yunet 추가 처리해보는 거는 후에 실험해서 보내줄게.
        
    
        
        
    
        
