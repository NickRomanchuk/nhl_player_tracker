import os
import cv2
import numpy as np
import sys
from scipy.spatial import ConvexHull
sys.path.append('../')
from utils import Queue

class PerspectiveTransformer():
    def __init__(self, path):
        self.sift = cv2.SIFT_create()
        key_frame_path = path + '/key_frame_homography'
        self.first_frame_homo = np.load(f"{key_frame_path}/first_frame.npy")    #homography from first frame to key
        self.geometric_model = cv2.imread(f"{path}/geometric_model/rink.png")   #image of rink

        # For each key frame, store the image and its homography to rink model
        self.key_frames = {}
        for file in os.listdir(f"{path}/key_frame_homography/key_frames"):
            name = file[:-4]
            self.key_frames[name] = {'image': cv2.imread(f"{key_frame_path }/key_frames/{name}.jpg")}
            self.key_frames[name]['homography'] = np.load(f"{key_frame_path}/{name}.npy")

    def calculate_homographies(self, video_frames):
        prev_homography_queue = Queue(self.first_frame_homo, 3)
        homographies = []
        name = None

        # For each frame we will calculate homography
        for frame_num, frame in enumerate(video_frames):
            print(f"Frame: {frame_num}")

            if frame_num == 0:
                homographies.append(self.key_frames['key_frame_1']['homography'] @ self.first_frame_homo)
                continue
            
            # Calcualte closest key frame and get matching points
            kp_frame, desc_frame = self.sift.detectAndCompute(frame, None)
            kp_frame, kp_key, name = self.calculate_closest_keyframe(kp_frame, desc_frame, name, prev_homography_queue.average())

            # Compute homography from matching points
            prev_homography, _ = cv2.findHomography(np.array(kp_frame), np.array(kp_key), cv2.RANSAC, 5.0)
            prev_homography_queue.add(prev_homography)

            # Combined transform to go from frame to key frame to rink
            combined_transform = self.key_frames[name]['homography'] @ prev_homography
            homographies.append(combined_transform)

            # Save result (transformed overlayed on rink)
            result = cv2.warpPerspective(frame, combined_transform, (self.geometric_model.shape[1], self.geometric_model.shape[0]))
            overlayed_frame = cv2.addWeighted(self.geometric_model, 0.4, result, 0.5, 0)
            cv2.imwrite(f'outputs/perspective_frames/frame_{str(frame_num).zfill(4)}.jpg', overlayed_frame)

        return homographies

    def calculate_closest_keyframe(self, kp_frame, desc_frame, key, initial_homography):
        # Initialize variables we will compare with neighbouring key frames
        name = None
        maxArea = 0
        most_matches = []
        
        # Get neighbouring key frames
        set_key_frames = self.get_set_keys(key)
        
        for key in set_key_frames:
            # computer swift kp and descriptors
            kp_key, desc_key = self.sift.detectAndCompute(self.key_frames[key]['image'], None)
            
            # Use FLANN to get inital set of matches
            matches = self.get_flann_matches(desc_frame, desc_key)

            # Further filter matches using previous frame's homography
            frame_pts = np.float32([kp_frame[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
            key_pts = np.float32([kp_key[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
            matches = self.get_dist_matches(frame_pts, key_pts, initial_homography)
            
            # Determine closest key frame seen so far, using number of matches and area of matches          
            area = self.area_enclosed(matches)
            print(f"{key}: {len(key_pts)} vs {len(matches)}, area: {area}")
            if (area > maxArea) and (len(matches) > (len(most_matches) / 3)):
                most_matches = matches
                name = key
                maxArea = area
        
        kp_frame, kp_key = zip(*most_matches)
        return kp_frame, kp_key, name
    
    def get_set_keys(self, key):
        # Return neighbouring key frames of current key frame
        match key:
            case None | 'key_frame_2':
                return self.key_frames.keys()
            case 'key_frame_1' | 'key_frame_3':    
                return [key, key[:-1]+'2']
    
    def get_flann_matches(self, desc_frame, desc_key):
        # Get FLANN key point matches
        dist_thres = 0.8
        flann = cv2.FlannBasedMatcher(dict(algorithm = 1, trees = 5), dict(checks=50))
        matches = flann.knnMatch(desc_frame, desc_key, k=2)

        # Eliminate matches where the second closest is within certain distance 
        match = []
        for m,n in matches:
            if m.distance < dist_thres * n.distance:
                match.append(m)

        return match

    def get_dist_matches(self, frame_pts, key_pts, initial_homography):
        # Transform the the potential frame points to be in key frame perspective
        dist_thres = 275
        trans_frame_pts = cv2.perspectiveTransform(frame_pts, initial_homography)

        # Remove matches if the transformed point is not close to the key frame point
        matches = []
        for key_pt, frame_pt, trans_frame_pt in zip(key_pts, frame_pts, trans_frame_pts):
            if np.linalg.norm(key_pt - trans_frame_pt) < dist_thres:
                matches.append((frame_pt[0], key_pt[0]))

        return matches
    
    def area_enclosed(self, matches):
        if len(matches) < 4:
            return 0
        
        _, key_points = zip(*matches)
        points = np.array(key_points)
        
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