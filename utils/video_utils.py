import cv2
import os
import numpy as np

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)

    index = 0
    frames = []
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)
        cv2.imwrite(f'./outputs/raw_frames/frame_{str(index).zfill(4)}.jpg', frame)
        index+=1

    return frames

def save_video(output_video_path):
    frame_list = os.listdir(f"outputs/annotated_frames")


    for frame_num, frame in enumerate(frame_list):
        annotated = cv2.imread(f"outputs/annotated_frames/{frame}")
        perspective = cv2.imread(f"outputs/perspective_frames/{frame}")
        tracking = cv2.imread(f"outputs/tracking_frames/{frame}")

        perspective = cv2.resize(perspective, (annotated.shape[1], annotated.shape[0] // 2))
        tracking = cv2.resize(tracking, (annotated.shape[1], annotated.shape[0] // 2))
        combined_perspective = np.concatenate((perspective, tracking), axis=0)
        
        frame = np.concatenate((annotated, combined_perspective), axis=1)
        if frame_num == 0:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_video_path, fourcc, 60, (frame.shape[1], frame.shape[0]))
        out.write(frame)
        
    out.release()
