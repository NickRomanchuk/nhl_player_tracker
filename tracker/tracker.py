from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
import pandas as pd

class Tracker:
    batch_size = 20
    image_size = 640
    confidence = 0.7
    text_param = {'rectangle_width': 40, 'rectangle_height': 20, 'type': cv2.FONT_HERSHEY_SIMPLEX, 'scale': 0.6, 'thickness': 2}
    player_ids = {'home-player':{'count': 0}, 'away-player':{'count': 0}}
    colors = {"text": (0, 0, 0), "away-player": (20, 181, 252), "away-goalie": (20, 181, 252), "home-player": (255, 125, 0), "home-goalie": (255, 125, 0), "referee": (0, 0, 255), "puck": (0,0,0)}

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def track_players(self, frames):
        # Takes array of frames, and returns array of YOLO object detections
        detections = self.detect_players(frames)
        cls_names = detections[0].names

        tracks = {"away-player":[], "away-goalie":[], "home-player":[], "home-goalie":[], "referee":[], "puck":[]}
        for frame_num, frame in enumerate(detections):

            # Convert to supervision detection format and track objects
            detection_supervision = sv.Detections.from_ultralytics(frame)
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            # New dictionary for each frame
            for key in tracks.keys(): tracks[key].append({}) 
            
            # For each detection store information under the tracker id
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                cls_name = cls_names[cls_id]
                track_id = self.update_track_id(cls_name, int(frame_detection[4]))
                tracks[cls_name][frame_num][track_id] = {"bbox":bbox}
        
        # Save the bbox detections
        self.save_detections(frames, tracks)
        
        return tracks
    
    def detect_players(self, frames):
        detections = []
        for i in range(0, len(frames), self.batch_size):
            detections += self.model.predict(frames[i:i+self.batch_size], conf=self.confidence, imgsz=self.image_size)
        return detections
    
    def update_track_id(self, cls_name, track_id):
        if cls_name in self.player_ids:
            cls_dict = self.player_ids[cls_name]
            if track_id not in cls_dict:
                cls_dict['count'] += 1
                cls_dict[track_id] = cls_dict['count']
            track_id = cls_dict[track_id]
                
        return track_id

    def save_detections(self, video_frames, tracks):
        # Loop over each frame
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            # loop over each class
            for key in tracks.keys():
                # loop over each instance of that class in this frame
                for _, object in tracks[key][frame_num].items():
                    # Draw bbox on frame
                    frame = self.draw_box(frame, object["bbox"], self.colors[key])

            # Save the frame
            cv2.imwrite(f'outputs/bbox_frames/frame_{str(frame_num).zfill(4)}.jpg',frame)

    def draw_box(self, frame, bbox, color):
        cv2.rectangle(frame,
                      (int(bbox[0]), int(bbox[1])),
                      (int(bbox[2]), int(bbox[3])),
                      color,
                      2)

        return frame
    
    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []

        #Loop over each frame
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            # loop over each class
            for key in tracks.keys():

                # Annotate each of the detected objects for that class
                for track_id, object in tracks[key][frame_num].items():
                    if key == 'puck':
                        continue
                    else:
                        frame = self.draw_ellipse(frame, object["bbox"], self.colors[key])
                        if 'player' in key:
                            frame = self.draw_track_id(frame, object["bbox"], self.colors[key], track_id)
            
            cv2.imwrite(f'outputs/annotated_frames/frame_{str(frame_num).zfill(4)}.jpg',frame)
            output_video_frames.append(frame)
        
        return output_video_frames
    
    def draw_ellipse(self, frame, bbox, color):
        y2 = int(bbox[3])
        x_center, _ = self.get_center_of_bbox(bbox)
        width = self.get_bbox_width(bbox)

        #Draw Elipse
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(round(width / 2), self.text_param['rectangle_height']//2),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=4,
            lineType=cv2.LINE_4)

        return frame

    def draw_track_id(self, frame, bbox, color, track_id):
        # Draw Rectangle
        y2 = int(bbox[3])
        x_center, _ = self.get_center_of_bbox(bbox)
        x1_rect = round(x_center - self.text_param['rectangle_width'] / 2)
        x2_rect = round(x_center + self.text_param['rectangle_width'] / 2)
        y1_rect = round(y2)
        y2_rect = round(y2 + self.text_param['rectangle_height'])
        cv2.rectangle(frame, (x1_rect, y1_rect), (x2_rect, y2_rect), color, cv2.FILLED)
            
        # Draw tracker id
        text_width, text_height = cv2.getTextSize(f"{track_id}", self.text_param['type'], self.text_param['scale'], self.text_param['thickness'])[0]
        x_center, y_center = self.get_center_of_bbox((x1_rect, y1_rect, x2_rect, y2_rect))
        CenterCoordinates = (x_center - round(text_width / 2), y_center + round(text_height / 2))
        cv2.putText(frame, 
                        f"{track_id}", 
                        CenterCoordinates, 
                        self.text_param['type'],
                        self.text_param['scale'],
                        self.colors['text'],
                        self.text_param['thickness'])

        return frame

    def get_center_of_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        return round((x1+x2)/2), round((y1+y2)/2)
    
    def get_bbox_width(self, bbox):
        x1, _, x2, _ = bbox
        return x2 - x1

    def draw_triangle(self, frame, bbox, color):
        x, y = self.get_center_of_bbox(bbox)
        triangle_points = np.array([[x,y], [x-10, y-20], [x+10, y-20]])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        return frame

    def draw_player_trajectories(self, video_frames, tracks, homographies):
        # Get rink image
        rink = cv2.imread('./homography/geometric_model/rink.png')

        # loop over each frame
        for frame_num, homography in enumerate(homographies):
            # only plot trajectories for skaters
            for cls in ['home-player', 'away-player']:
                players = tracks[cls][frame_num]

                # For each skater on that frame
                for player in players:
                    # Get bottom middle of bounding box
                    bbox = players[player]['bbox']
                    y2 = int(bbox[3])
                    x_center, _ = self.get_center_of_bbox(bbox)
                    bottom_box = [x_center, y2]

                    # transform bottom of bounding box using homography
                    transformed_bottom = cv2.perspectiveTransform(np.array([[bottom_box]], np.float64), homography)[0][0]

                    # plot bottom of box on rink
                    rink = cv2.circle(rink, (round(transformed_bottom[0]), round(transformed_bottom[1])), 2, self.colors[cls], 5)
                    cv2.imwrite(f'./outputs/tracking_frames/frame_{str(frame_num).zfill(4)}.jpg', rink)