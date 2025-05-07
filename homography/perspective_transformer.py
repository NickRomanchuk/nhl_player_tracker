import os
import cv2
import numpy as np

class PerspectiveTransformer():
    def __init__(self, path):
        self.sift = cv2.SIFT_create()

        self.key_frames = {}
        for file in os.listdir(f"{path}/key_frame_homography/key_frames"):
            name = file.split('.')[0]
            self.key_frames[name] = {'image': cv2.imread(f"{path}/key_frame_homography/key_frames/{name}.jpg")}
            self.key_frames[name]['homography'] = np.load(f"{path}/key_frame_homography/{name}.npy")
        
        self.first_frame = np.load(f"{path}/key_frame_homography/first_frame.npy")
        self.geometric_model = cv2.imread(f"{path}/geometric_model/rink.png")

    def calculate_homographies(self, video_frames):
        prev_homography = self.first_frame
        homographies = []
        name = None

        # For each frame we will calculate homography
        for frame_num, frame in enumerate(video_frames):
            print(frame_num)
            if frame_num == 0:
                homographies.append(self.key_frames['key_frame_1']['homography'] @ self.first_frame)
            else:
                # Calcualte closest key frame and get matching points
                kp_frame, desc_frame = self.sift.detectAndCompute(frame, None)
                kp_frame, kp_key, name = self.calculate_closest_keyframe(kp_frame, desc_frame, name, prev_homography)

                # Compute homography from matching points
                prev_homography, _ = cv2.findHomography(np.array(kp_frame), np.array(kp_key), cv2.RANSAC, 5.0)
                combined_transform = self.key_frames[name]['homography'] @ prev_homography
                homographies.append(combined_transform)

            # Save result (transformed overlayed on rink)
            result = cv2.warpPerspective(frame, homographies[-1], (self.geometric_model.shape[1], self.geometric_model.shape[0]))
            overlayed_frame = cv2.addWeighted(self.geometric_model, 0.4, result, 0.5, 0)
            cv2.imwrite(f'outputs/perspective_frames/frame_{str(frame_num).zfill(4)}.jpg', overlayed_frame)

        return homographies

    def calculate_closest_keyframe(self, kp_frame, desc_frame, key, initial_homography):
        name = None
        most_matches = []
        
        set_key_frames = self.get_set_keys(key)
        for key in set_key_frames:
            kp_key, desc_key = self.sift.detectAndCompute(self.key_frames[key]['image'], None)
            
            # Get inital set of matches
            matches = self.get_knn_matches(desc_frame, desc_key)

            # Further filter matches using previous frame's homography
            frame_pts = np.float32([kp_frame[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
            key_pts = np.float32([kp_key[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
            matches = self.get_dist_matches(frame_pts, key_pts, initial_homography)
            
            print(f"{key}: {len(key_pts)} vs {len(matches)}")
            # Determine closest key frame seen so far
            if len(matches) > len(most_matches):
                most_matches = matches
                name = key

        kp_frame, kp_key = zip(*most_matches)
        return kp_frame, kp_key, name

    def get_set_keys(self, key):
        if key == None:
            return self.key_frames.keys()
        
        if key[-1] == '1':
            keys = [key, key[:-1]+'2']
        elif key[-1] == '3':
            keys = [key, key[:-1]+'2']
        elif key[-1] == '2':
            keys = [key, key[:-1]+'3']

        return keys
    
    def get_knn_matches(self, desc_frame, desc_key):
        index_params = dict(algorithm = 1, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc_frame, desc_key, k=2)

        match = []
        for m,n in matches:
            if m.distance < 0.6*n.distance:
                match.append(m)

        return match

    def get_dist_matches(self, frame_pts, key_pts, initial_homography):
        trans_frame_pts = cv2.perspectiveTransform(frame_pts, initial_homography)

        matches = []
        for i in range(len(key_pts)):
            if np.linalg.norm(key_pts[i]-trans_frame_pts[i]) < 375:
                matches.append((frame_pts[i][0], key_pts[i][0]))

        return matches


