import cv2
import time

capture = cv2.VideoCapture(0) #Camera

frame_num = 0
while capture.isOpened():
    timeData = time.time()
    # Capture frame and load image
    start = time.time()
    result, image = capture.read()
    end = time.time()
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_num += 1
    print(end - start)
    if result is False:
        cv2.waitKey(0)
        break
    cv2.imwrite(f'camera/original/{timeData}_{int(fps)}_{frame_num}.jpg', image)
    key = cv2.waitKey(32)
    if key == ord('q'):
        break
capture.release()
