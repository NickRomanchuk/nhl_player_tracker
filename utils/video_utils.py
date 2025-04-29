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
        cv2.imwrite(f'./outputs/raw_frames/frame_{index}.jpg', frame)
        index+=1

    return frames

def save_video(output_video_path):
    frames = []
    frame_list = os.listdir(f"outputs/annotated_frames")

    for frame in frame_list:
        annotated = cv2.imread(f"outputs/annotated_frames/{frame}")
        perspective = cv2.imread(f"outputs/perspective_frames/{frame}")
        perspective = cv2.resize(perspective, (annotated.shape[1], annotated.shape[0]))
        frames.append(np.concatenate((annotated, perspective), axis=1))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 60, (frames[0].shape[1], frames[0].shape[0]))

    for frame in np.array(frames):
        out.write(frame)
        
    out.release()
