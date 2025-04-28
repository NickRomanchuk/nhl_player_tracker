import cv2

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

def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))

    for frame in output_video_frames:
        out.write(frame)
        
    out.release()