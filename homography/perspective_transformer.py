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
        
        self.geometric_model = cv2.imread(f"{path}/geometric_model/rink.png")

    def calculate_homographies(self, video_frames):
        homographies = []
        for frame_num, frame in enumerate(video_frames):
            kp1, des1 = self.sift.detectAndCompute(frame, None)
            good, kp2, name = self.calculate_closest_keyframe(des1)

            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            combined_transform = M @ self.key_frames[name]['homography']
            result = cv2.warpPerspective(frame, combined_transform, (self.geometric_model.shape[1], self.geometric_model.shape[0]))

            overlayed_frame = cv2.addWeighted(self.geometric_model, 0.4, result, 0.5, 0)
            cv2.imwrite(f'outputs/perspective_frames/frame_{frame_num}.jpg', overlayed_frame)
            homographies.append(combined_transform)

        return homographies

    def calculate_closest_keyframe(self, des1):
        dest_good = []
        name = None
        dest_key = None

        for key in self.key_frames:
            kp2, des2 = self.sift.detectAndCompute(self.key_frames[key]['image'], None)

            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)

            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)

            good = []
            for m,n in matches:
                if m.distance < 0.2*n.distance:
                    good.append(m)
            
            if len(good) > len(dest_good):
                dest_good = good
                dest_key = kp2
                name = key
            
        return dest_good, dest_key, name
