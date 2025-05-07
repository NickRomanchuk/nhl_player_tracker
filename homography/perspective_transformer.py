import os
import cv2
import numpy as np
from scipy.spatial import ConvexHull

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
        prev_homography_queue = [self.first_frame, self.first_frame, self.first_frame]
        homographies = []
        name = None

        # For each frame we will calculate homography
        for frame_num, frame in enumerate(video_frames):
            print(f"Frame: {frame_num}")
            if frame_num == 0:
                homographies.append(self.key_frames['key_frame_1']['homography'] @ self.first_frame)
            else:
                # Calcualte closest key frame and get matching points
                kp_frame, desc_frame = self.sift.detectAndCompute(frame, None)
                average_prev_homography = (prev_homography_queue[0] + prev_homography_queue[1] + prev_homography_queue[2]) / len(prev_homography_queue)
                kp_frame, kp_key, name = self.calculate_closest_keyframe(kp_frame, desc_frame, name, average_prev_homography)

                # Compute homography from matching points
                prev_homography, _ = cv2.findHomography(np.array(kp_frame), np.array(kp_key), cv2.RANSAC, 5.0)
                combined_transform = self.key_frames[name]['homography'] @ prev_homography
                prev_homography_queue.append(prev_homography)
                prev_homography_queue.pop(0)
                homographies.append(combined_transform)

            # Save result (transformed overlayed on rink)
            result = cv2.warpPerspective(frame, homographies[-1], (self.geometric_model.shape[1], self.geometric_model.shape[0]))
            overlayed_frame = cv2.addWeighted(self.geometric_model, 0.4, result, 0.5, 0)
            cv2.imwrite(f'outputs/perspective_frames/frame_{str(frame_num).zfill(4)}.jpg', overlayed_frame)

        return homographies

    def calculate_closest_keyframe(self, kp_frame, desc_frame, key, initial_homography):
        name = None
        maxArea = 0
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
            
            # Determine closest key frame seen so far          
            area = self.area_enclosed(matches)
            print(f"{key}: {len(key_pts)} vs {len(matches)}, area: {area}")
            if (area > maxArea) and (len(matches) > (len(most_matches) / 3)):
                most_matches = matches
                name = key
                maxArea = area
        
        kp_frame, kp_key = zip(*most_matches)
        return kp_frame, kp_key, name
    
    def area_enclosed(self, matches):
        if len(matches) < 4:
            return 0
        
        _, points = zip(*matches)
        points = np.array(points)
        
        pi2 = np.pi/2.

        # get the convex hull for the points
        hull_points = points[ConvexHull(points).vertices]

        # calculate edge angles
        edges = np.zeros((len(hull_points)-1, 2))
        edges = hull_points[1:] - hull_points[:-1]

        angles = np.zeros((len(edges)))
        angles = np.arctan2(edges[:, 1], edges[:, 0])

        angles = np.abs(np.mod(angles, pi2))
        angles = np.unique(angles)

        # find rotation matrices
        rotations = np.vstack([
            np.cos(angles),
            np.cos(angles-pi2),
            np.cos(angles+pi2),
            np.cos(angles)]).T
        rotations = rotations.reshape((-1, 2, 2))

        # apply rotations to the hull
        rot_points = np.dot(rotations, hull_points.T)

        # find the bounding points
        min_x = np.nanmin(rot_points[:, 0], axis=1)
        max_x = np.nanmax(rot_points[:, 0], axis=1)
        min_y = np.nanmin(rot_points[:, 1], axis=1)
        max_y = np.nanmax(rot_points[:, 1], axis=1)

        # find the box with the best area
        areas = (max_x - min_x) * (max_y - min_y)

        return np.min(areas)

    def get_set_keys(self, key):
        if key == None:
            return self.key_frames.keys()
        
        if key[-1] == '1':
            keys = [key, key[:-1]+'2']
        elif key[-1] == '3':
            keys = [key, key[:-1]+'2']
        elif key[-1] == '2':
            keys = self.key_frames.keys()

        return keys
    
    def get_knn_matches(self, desc_frame, desc_key):
        index_params = dict(algorithm = 1, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc_frame, desc_key, k=2)

        match = []
        for m,n in matches:
            if m.distance < 0.8*n.distance:
                match.append(m)

        return match

    def get_dist_matches(self, frame_pts, key_pts, initial_homography):
        trans_frame_pts = cv2.perspectiveTransform(frame_pts, initial_homography)

        matches = []
        for i in range(len(key_pts)):
            if np.linalg.norm(key_pts[i]-trans_frame_pts[i]) < 275:
                matches.append((frame_pts[i][0], key_pts[i][0]))

        return matches


