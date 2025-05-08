import cv2
import os
import numpy as np

def read_video(video_path):
    # Get video frames
    video = cv2.VideoCapture(video_path)

    # Loop over all the video frames
    index = 0
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        # Save and append video frame 
        cv2.imwrite(f'./outputs/raw_frames/frame_{str(index).zfill(4)}.jpg', frame)
        frames.append(frame)
        index+=1

    return frames

def save_video(output_video_path):
    fps = 60

    for frame_num, frame in enumerate(os.listdir(f"outputs/annotated_frames")):
        # Read the three sets of images to combine
        annotated = cv2.imread(f"outputs/annotated_frames/{frame}")
        perspective = cv2.imread(f"outputs/perspective_frames/{frame}")
        tracking = cv2.imread(f"outputs/tracking_frames/{frame}")

        # Resize the images
        dimensions = (int((annotated.shape[0] // 2) * (200/85)), annotated.shape[0] // 2)
        perspective = cv2.resize(perspective, dimensions)
        tracking = cv2.resize(tracking, dimensions)
        
        # Combine into one frame
        combined_perspective = np.concatenate((perspective, tracking), axis=0)
        frame = np.concatenate((annotated, combined_perspective), axis=1)

        # Write frame to video
        if frame_num == 0:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame.shape[1], frame.shape[0]))
        out.write(frame)
        
    out.release()
