from utils import read_video, save_video
from tracker import Tracker
# from team_assigner import TeamAssigner
# from camera_movement_estimator import CameraMovementEstimator
# from view_transformer import ViewTransformer
# from speed_and_distance_estimator import SpeedDistanceEstimator
import cv2

def main():
    # Read video, returns array of frames (array of pixels)
    input_video = 'input_video.mp4'
    video_frames = read_video('input_videos/'+input_video)
    video_frames = video_frames[:10]
    
    # Initialize Tracker, uses best.pt model
    model = 'best_smalldataset'
    tracker = Tracker(f'./model_training/models/{model}.pt')

    # Track the players in the video
    tracks = tracker.track_players(video_frames)

    # Draw annotations
    annotated_video_frames = tracker.draw_annotations(video_frames, tracks)

    # # camera movement estimator
    # camera_movement_esitmator = CameraMovementEstimator(video_frames[0])
    # camera_movement_per_frame = camera_movement_esitmator.get_camera_movement(video_frames,
    #                                                                           read_from_stub=True,
    #                                                                           stub_path='stubs/camera_movement_stub.pk1')
    # camera_movement_esitmator.add_adjust_postions_to_tracks(detections, camera_movement_per_frame)

    # # View Transformer
    # view_transformer = ViewTransformer()
    # view_transformer.add_tranfsormed_position_to_tracks(detections)

    # # interpolate puck positions
    # # detections["puck"] = tracker.interpolate_puck_positions(detections["puck"])
    
    # # Speed and distance estimator
    # #speed_and_distance_estimator = SpeedDistanceEstimator()
    # #speed_and_distance_estimator.add_speed_and_distance_to_tracks(detections)

    # # Draw Camera Movement
    # #output_video_frames = camera_movement_esitmator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    # # Draw Speed and Distance
    # #speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, detections)

    # Save video
    save_video(annotated_video_frames, f'outputs/{model}.avi')

if __name__ == '__main__':
    main()
    print('\n\nDone!\n\n')