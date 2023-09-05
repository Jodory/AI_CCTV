# 230404 Testing Code
# Check Entire Code for Accuracy Improvement

import os
import numpy as np
from cv2 import *  
import cv2
import matplotlib.pyplot as plt

# Use Camera Module for Input Image by frame
capture = cv2.VideoCapture(0) #Camera
# If Camera Module is not exist, exit this code
if not capture.isOpened():
    exit()

# If Camera Modul is exist
# Save width, height about camera module
width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

# My notebook size: 640.0 x 480.0
print("size: {0} x {1}".format(width, height))
 

# Create 'VideoWriter' Instance to save Video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter('test_230404.mp4', fourcc, 24, (int(width), int(height)))  # 24frame is better 

'''If video file is exist in path, process that video by frame'''
# # set the path of the input video file
# video_path = 'image.jpg'
# # create a VideoCapture object to read video from the file
# capture = cv2.VideoCapture(video_path)

# load the model
face_detector = cv2.FaceDetectorYN.create("jodory.onnx", "", (320, 320))

# 1. detect face 2. mosaic face using face bbox
image_buf = 0
faces_buf = []
boxes_prev = []
cnt = 0


while capture.isOpened():
    # Capture frame and load image
    result, image = capture.read()
    if result is False:
        cv2.waitKey(0)
        break
    # image = cv2.imread('image.jpg', cv2.IMREAD_COLOR)
    

    # If the image is not 3 channels, convert it to 3 channels
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # specify the input size
    # My notebook image input size: 640.0 x 480.0
    height, width, _ = image.shape
    face_detector.setInputSize((width, height))

    # detect faces
    ret, faces = face_detector.detect(image)
    
    # face detector
    faces = faces if faces is not None else []
    print(f'cnt = {cnt}, faces = {len(faces)}, faces_buf = {len(faces_buf)}') 

    if cnt == 1:
        boxes_post = []
        for face in faces:
            # bounding box
            boxes_post.append(list(map(int, face[:4])))
        boxes_prev.sort()
        boxes_post.sort()
        if len(boxes_prev) <= len(boxes_post): small, big = boxes_prev, boxes_post
        else: big, small = boxes_prev, boxes_post
        if len(small) == 0 :
            small = big.copy()
        
        boxes_buf = big.copy()
        for i in range(len(small)):
            for j in range(4):
                print(f'buf[before_sum] = {boxes_buf}, {len(boxes_buf)}')
                boxes_buf[i][j] += small[i][j]
                print(f'buf[after_sum] = {boxes_buf}, {len(boxes_buf)}')
                boxes_buf[i][j] = boxes_buf[i][j]//2

        
        #boxes_buf = boxes_prev + boxes_post
        #boxes_buf /= 2
        print(f'buf__before devide 2 = {boxes_buf}, {len(boxes_buf)}')

        # boxes_buf = [(x[0]//2, x[1]//2, x[2]//2, x[3]//2) for x in boxes_buf]

        print(f'big = {big}, {len(big)}')
        print(f'small = {small}, {len(small)}')
        print(f'prev = {boxes_prev}, {len(boxes_prev)}')
        print(f'post = {boxes_post}, {len(boxes_post)}')
        print(f'buf = {boxes_buf}, {len(boxes_buf)}\n')
        #boxes_buf = [x/2 for x in boxes_buf]

        
        for box_buf in boxes_buf:
            # bounding box
            # print(box_buf)
            color = (255, 0, 0)
            thickness = -1
            cv2.rectangle(image_buf, box_buf, color, thickness, cv2.LINE_AA)    # Blue
        writer.write(image_buf)  # Save Frame
        #boxes_prev += box
        #boxes_prev /= 2
        cnt = 0


    # if len(직전 프레임 얼굴 탐지 수) != len(현재 프레임 얼굴 탐지 수): 직전 프레임으로 저장
    '''
    if len(faces) == 0:  
        if(np.all(image_buf == 0)):
            continue
        writer.write(image_buf)  # previous image_save
        continue
    '''
    # 직전 프레임과 현재 프레임의 얼굴 탐지수가 다를 때
    # 경우 1. 첫 프레임, 두번째 프레임 모두 face == 0, pass / write(image)==첫프레임, 두번째 프레임 저장 /
    # 세번째 프레임 faces = 1이면() / 이전 프레임보다 faces 수가 증가되었기 때문에 버퍼 없이 삽입(?)
    if len(faces) != len(faces_buf): # faces == 1, faces_buf == 0, 만약 처음 프레임과 두번째 프레임에 아무것도 잡히지 않았다면 사용 x
        # image_buf가 0 이라면, 즉 직전 프레임이 버퍼에 저장되어 있지 않다면, 현재 프레임 삽입
        
        if(np.all(image_buf == 0)):
            image_buf = image


            boxes_prev = []
            for face in faces:
                # bounding box
                box = list(map(int, face[:4]))
                boxes_prev.append(box)
                # print(box)
                color = (0, 0, 255)
                thickness = -1
                cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)
            writer.write(image)  # Save Frame

            continue
        # writer.write(image_buf)  # previous image_save
        
        if cnt == 0:
            cnt += 1
            #continue
        
        
    
    # print('faces', faces[0])
    # Draw bounding boxes and landmarks for detected faces
    boxes_prev = []
    for face in faces:
        # bounding box
        box = list(map(int, face[:4]))
        boxes_prev.append(box)
        # print(box)
        color = (0, 0, 255)
        thickness = -1
        cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)

    
        #writer.write(image)  # Save Frame
    
    ######################################################## 수정 필요
    if cnt == 0:
        writer.write(image)  # Save Frame
    image_buf = image
    faces_buf = faces
    cv2.imshow("face detection", image)
    key = cv2.waitKey(10)
    if key == ord('q'):
        break
capture.release()
writer.release()  # Save 
cv2.destroyAllWindows()
print('exit')