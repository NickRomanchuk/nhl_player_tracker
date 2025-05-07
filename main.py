from utils import read_video, save_video
from tracker import Tracker
from homography import PerspectiveTransformer

def main():
    # Read video, returns array of frames (array of pixels)
    input_video = 'input_video2.mp4'
    video_frames = read_video('input_videos/'+input_video)
    #video_frames = video_frames[:100]
    
    # Initialize Tracker, uses best.pt model
    model = 'best'
    tracker = Tracker(f'./model_training/models/{model}.pt')

    # Track the players in the video
    tracks = tracker.track_players(video_frames)

    # Draw annotations
    annotated_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Compute homography
    transformer = PerspectiveTransformer(f"./homography")
    homographies = transformer.calculate_homographies(video_frames)

    # Track player trajectories
    tracker.draw_player_trajectories(video_frames, tracks, homographies)

    # Save video
    save_video(f'outputs/{model}.avi')

if __name__ == '__main__':
    main()
    print('\n\nDone!\n\n')