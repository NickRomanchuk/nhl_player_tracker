from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2

class Tracker:
    batch_size = 20
    image_size = 640
    confidence = 0.8
    player_counts = {'home-player': {'count':0}, 'away-player':{'count':0}}
    text_param = {'rectangle_width': 40, 
                  'rectangle_height': 20, 
                  'type': cv2.FONT_HERSHEY_SIMPLEX, 
                  'scale': 0.6, 
                  'thickness': 2,
                  "color": (0, 0, 0)}
    colors = {"away-player": (20, 181, 252), 
              "away-goalie": (20, 181, 252), 
              "home-player": (255, 125, 0), 
              "home-goalie": (255, 125, 0), 
              "referee": (0, 0, 255), 
              "puck": (0,0,0)}

    def __init__(self, model_path):
        # initialize YOLO model and ByteTracker
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def track_players(self, frames, homographies):
        # Takes array of frames, and returns array of YOLO object detections
        detections = self.detect_players(frames)
        cls_names = detections[0].names
        
        # Initalize dictionary for player tracks
        tracks = {"away-player":[], "away-goalie":[], "home-player":[], "home-goalie":[], "referee":[], "puck":[]}

        for frame_num, frame in enumerate(detections):
            # New dictionary for each frame
            for key in tracks.keys(): tracks[key].append({}) 

            # Convert to supervision detection format and track objects
            detection_supervision = sv.Detections.from_ultralytics(frame)
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            # For each detection store information under the tracker id
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                cls_name = cls_names[cls_id]
                track_id = self.update_track_id(cls_name, int(frame_detection[4]))
                tracks[cls_name][frame_num][track_id] = {"bbox":bbox}

                # position data
                bottom_box = self.get_bottom_bbox(bbox)
                transformed_bottom = cv2.perspectiveTransform(np.array([[bottom_box]], np.float64), homographies[frame_num])[0][0]
                tracks[cls_name][frame_num][track_id]["position"] = transformed_bottom

                # calculate displacement
                displacement = self.calc_displacement(frame_num, track_id, tracks[cls_name])
                tracks[cls_name][frame_num][track_id]["displacement"] = displacement
                if displacement:
                    tracks[cls_name][frame_num][track_id]["speed"] = displacement * 60

                # Save the bbox detections
                self.save_detection(frame_num, frames[frame_num], tracks)
        
        return tracks
    
    def detect_players(self, frames):
        detections = []
        for i in range(0, len(frames), self.batch_size):
            detections += self.model.predict(frames[i:i+self.batch_size], conf=self.confidence, imgsz=self.image_size)
        return detections
    
    def update_track_id(self, cls_name, track_id):
        # Check if class is one of the players
        if cls_name in self.player_counts:
            # If we havent't seen this track_id yet (means it is new player)
            if track_id not in self.player_counts[cls_name]:
                # Update number of players seen and map from this track_id to new count
                self.player_counts[cls_name]['count'] += 1
                self.player_counts[cls_name][track_id] = self.player_counts[cls_name]['count']
            
            # Update track_id to be the new mapping
            track_id = self.player_counts[cls_name][track_id]
                
        return track_id

    def save_detection(self, frame_num, frame, tracks):
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
    
    def calc_displacement(self, frame_num, track_id, tracks):
        # Get conversion to go from pixels to meters
        meters_conversion = np.load("./homography/geometric_model/meters_conversion.npy")

        if track_id not in  tracks[frame_num-1].keys():
            # If no previous frame displacement is zero
            displacement = 0
        else:
            # Otherwise take difference with previous frame, convert to meters, and calculate hypo
            previous_position =  tracks[frame_num-1][track_id]["position"]
            current_position =  tracks[frame_num][track_id]["position"]
            displacement = np.multiply((current_position - previous_position), meters_conversion)
            displacement = np.hypot(displacement[0], displacement[1])

        return displacement

    def draw_annotations(self, video_frames, tracks):
        #Loop over each frame
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            # loop over each class
            for key in tracks.keys():
                if key == 'puck':
                    continue
                # Annotate each of the detected objects for that class
                for track_id, object in tracks[key][frame_num].items():
                    frame = self.draw_ellipse(frame, object["bbox"], self.colors[key])

                    # If player we will draw speed and track id
                    if 'player' in key:
                        frame = self.draw_track_id(frame, object["bbox"], self.colors[key], track_id)
                        if object.get("speed", None):
                            frame = self.draw_speed(frame, object["bbox"], object["speed"])
            
            cv2.imwrite(f'outputs/annotated_frames/frame_{str(frame_num).zfill(4)}.jpg',frame)
    
    def draw_ellipse(self, frame, bbox, color):
        bottom_box = self.get_bottom_bbox(bbox)
        width = self.get_bbox_width(bbox)

        #Draw Elipse
        cv2.ellipse(
            frame,
            center=bottom_box,
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
        x1_rect, x2_rect = round(x_center - self.text_param['rectangle_width'] / 2), round(x_center + self.text_param['rectangle_width'] / 2)
        y1_rect, y2_rect = round(y2), round(y2 + self.text_param['rectangle_height'])
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
                        self.text_param['color'],
                        self.text_param['thickness'])

        return frame
    
    def draw_speed(self, frame, bbox, speed):
        speed = np.round(speed, 2)
        text = f"{speed} m/s"
        y1 = int(bbox[1])
        x_center, _ = self.get_center_of_bbox(bbox)
            
        # Calculate placement of text
        text_width, text_height = cv2.getTextSize(f"{speed}", self.text_param['type'], self.text_param['scale'], self.text_param['thickness'])[0]
        CenterCoordinates = (x_center - round(text_width / 2), y1 + round(text_height / 2))
        
        # Draw tracker id
        cv2.putText(frame, 
                        text, 
                        CenterCoordinates, 
                        self.text_param['type'],
                        self.text_param['scale'],
                        self.text_param['color'],
                        self.text_param['thickness'])

        return frame

    def draw_player_trajectories(self, tracks, homographies):
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
                    bottom_box = self.get_bottom_bbox(players[player]['bbox'])

                    # transform bottom of bounding box using homography
                    transformed_bottom = cv2.perspectiveTransform(np.array([[bottom_box]], np.float64), homography)[0][0]

                    # plot bottom of box on rink
                    rink = cv2.circle(rink, (round(transformed_bottom[0]), round(transformed_bottom[1])), 1, self.colors[cls], 2)
                    cv2.imwrite(f'./outputs/tracking_frames/frame_{str(frame_num).zfill(4)}.jpg', rink)
    
    def get_center_of_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        return round((x1+x2)/2), round((y1+y2)/2)
    
    def get_bbox_width(self, bbox):
        x1, _, x2, _ = bbox
        return x2 - x1
    
    def get_bottom_bbox(self, bbox):
        y2 = int(bbox[3])
        x_center, _ = self.get_center_of_bbox(bbox)
        return (x_center, y2)