import cv2
import numpy as np
import multiprocessing as mp
from datetime import datetime
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
        cv2.imshow("Original Frame", frame1)
        timeData = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cv2.imwrite(f'camera/original/{timeData}.jpg', frame1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def display_process(timeData, shared_array, shape):
    print(f'process2: 시작')
    face_detector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (320, 320))
    while True:
        frame = np.frombuffer(shared_array, dtype=np.uint8).reshape(shape)
        # Ensure frame is 3 channel
        channels = 1 if len(frame.shape) == 2 else frame.shape[2]
        if channels == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if channels == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        # Resize the frame for face detection
        resized_frame = cv2.resize(frame, (320, 320))
        faces = face_detector.detect(resized_frame, 0.5)
        for (x, y, w, h) in faces:
            # Convert the coordinates back to the original size
            x, y, w, h = x*2, y*2, w*2, h*2
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            region = frame[y:y+h, x:x+w]
            region = cv2.resize(region, (w//10, h//10))
            region = cv2.resize(region, (w, h), interpolation=cv2.INTER_NEAREST)
            frame[y:y+h, x:x+w] = region
        
        cv2.imshow("Mosaic Frame", frame)            
        timeData = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cv2.imwrite(f'camera/process/{timeData}.jpg', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def main():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("Failed to open camera.")
        return

    shape = frame.shape
    shared_array = mp.Array('B', frame.size, lock=False)
    timeData = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    if not cap.isOpened():
        exit()
    
    process1 = mp.Process(target=display_original, args=(timeData, shared_array, shape))
    process2 = mp.Process(target=display_process, args=(timeData, shared_array, shape))
    print(f'process1: 대기중')
    process1.start()
    print(f'process2: 대기중')
    process2.start()

    while True:
        ret, frame = cap.read()
        if ret:
            shared_array[:] = frame.flatten()

    cap.release()
    process1.terminate()
    process2.terminate()

if __name__ == '__main__':
    main()


