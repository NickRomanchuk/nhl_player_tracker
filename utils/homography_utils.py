import cv2
import matplotlib.pyplot as plt
import numpy as np

#class to select keypoints in image
class KeyPointSelector:
    def __init__(self, img, name):
        self.points = []
        self.img = img
        self.name = name

    def select_key_points(self):
        # Show image to
        cv2.imshow(self.name, self.img)
        # Set call back for selecting key points
        cv2.setMouseCallback(self.name, self.select_point)
        # Wait until image is closed
        cv2.waitKey(0)
        # Plot the image with the key points
        self.plot_image()
        # Return selected key points
        return np.array(self.points)

    def plot_image(self):
        # Plot image that has selected keypoints
        plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        plt.show()

    def select_point(self, event, x, y, flags, params):
        # If left mouse clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point to keypoints
            self.points.append((x,y))
            # Add circle to image at selected location
            cv2.circle(self.img, (x,y), 5, (0,0,0), cv2.FILLED)
            cv2.putText(self.img, str(len(self.points)), (x+5,y+5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2)

def get_key_points(path, name):
    # Load image to detect keypoints
    key_frame = cv2.imread(path)

    # Initalize key point selector
    key_frame_points = KeyPointSelector(key_frame, name)
    
    # Select key points in image
    key_points = key_frame_points.select_key_points()
    
    # Return keypoints
    return key_points

def get_homography(src_points, dst_points, name):
    # Compute homography
    M, _ = cv2.findHomography(src_points, dst_points, cv2.LMEDS)    
    # Save the homography
    np.save(f"./{name.split('.')[0]}.npy", M)
    # Return homography matrix
    return M