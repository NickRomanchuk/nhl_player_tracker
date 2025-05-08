from utils import read_video, save_video
from tracker import Tracker
from homography import PerspectiveTransformer

def main():
    model = 'last'
    input_video = 'input_video2.mp4'

    # Read video, returns array of frames
    video_frames = read_video('input_videos/'+input_video)
    video_frames = video_frames[:3]
    
    # Compute homography from frame to rink model
    transformer = PerspectiveTransformer(f"./homography")
    homographies = transformer.calculate_homographies(video_frames)

    # Track the players in the video via YOLO model
    tracker = Tracker(f'./model_training/models/{model}.pt')
    tracks = tracker.track_players(video_frames, homographies)

    # Draw annotations
    tracker.draw_annotations(video_frames, tracks)
    tracker.draw_player_trajectories(tracks, homographies)

    # Save video
    save_video(f'outputs/{model}.avi')

if __name__ == '__main__':
    main()
    print('\nDone!\n')