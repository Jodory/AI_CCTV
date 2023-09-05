import cv2
import os

def create_mp4_from_jpgs(folder_path, output_file):
    # 폴더 내의 모든 jpg 파일을 이름 순으로 검색
    img_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    img_files.sort()

    # 첫 번째 이미지를 사용하여 비디오의 크기를 결정
    frame = cv2.imread(os.path.join(folder_path, img_files[0]))
    h, w, layers = frame.shape
    size = (w, h)

    # .mp4 파일로 저장하기 위한 VideoWriter 객체 생성
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

    # 각 이미지를 비디오 프레임으로 추가
    for i in range(len(img_files)):
        img_path = os.path.join(folder_path, img_files[i])
        img = cv2.imread(img_path)
        out.write(img)

    out.release()

# 사용 예:
create_mp4_from_jpgs('camera/original', 'original.mp4')
create_mp4_from_jpgs('camera/process', 'process.mp4')
