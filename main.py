from utils import read_video, save_video
from tracker import Tracker
# from team_assigner import TeamAssigner
# from camera_movement_estimator import CameraMovementEstimator
# from view_transformer import ViewTransformer
# from speed_and_distance_estimator import SpeedDistanceEstimator
import cv2

def main():
    # Read video, returns array of frames (array of pixels)
    input_video = 'input_video2.mp4'
    video_frames = read_video('input_videos/'+input_video)
    #video_frames = video_frames[:10]
    
    # Initialize Tracker, uses best.pt model
    model = 'best_smalldataset'
    tracker = Tracker(f'./model_training/models/{model}.pt')

    # Track the players in the video
    tracks = tracker.track_players(video_frames)

    # Draw annotations
    annotated_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Save video
    save_video(annotated_video_frames, f'outputs/{model}.avi')

if __name__ == '__main__':
    main()
    print('\n\nDone!\n\n')