import numpy as np
import cv2

from maritime_urban_tracking.single_stereo_camera import StereoCamera

class DualStereoCamera:
    def __init__(self, path_to_folder_left, path_to_folder_right, R, t) -> None:
        self.path_to_folder_left = path_to_folder_left
        self.path_to_folder_right = path_to_folder_right
        self.R = R
        self.t = t

        self.cam_left = StereoCamera(path_to_folder_left)
        self.cam_right = StereoCamera(path_to_folder_right)

        K_left, D_left = self.cam_left.K_left, self.cam_left.D_left
        K_right, D_right = self.cam_right.K_left, self.cam_right.D_left # Uses the left image in the right camera

        self.rectifier_wide_from_short = ImageRectifier(R, t, K_left, D_left, K_right, D_right, self.cam_left.image_shape_wh)

        timestamps_left = self.cam_left._timestamps
        timestamps_right = self.cam_right._timestamps
        # (arr_is_it_in_here[:,None] == arr_for_each_row).all(axis=2).any(axis=0)

        is_close_matrix = np.abs(timestamps_left[:,None] - timestamps_right)/(10**6) < 1 # Associate if < 1ms. This will have at max a single 1 for each column or row.
        assert np.count_nonzero(is_close_matrix.sum(axis=1) > 1) == 0, "There should not be multiple associations"
        assert np.count_nonzero(is_close_matrix.sum(axis=0) > 1) == 0, "There should not be multiple associations" 
        is_close_left = is_close_matrix.any(axis=1)
        is_close_right = is_close_matrix.any(axis=0)
        timestamps_left_synchronized = timestamps_left[is_close_left]
        timestamps_right_synchronized = timestamps_right[is_close_right]

        self.cam_left._set_timestamps(timestamps_left_synchronized)
        self.cam_right._set_timestamps(timestamps_right_synchronized)

        assert self.cam_left.fps == self.cam_right.fps
        self.fps = self.cam_left.fps
    
    @property
    def is_playable(self):
        return self.cam_left.is_playable and self.cam_right.is_playable
    
    def get_timestamp(self):
        return self.cam_left.get_timestamp()

    def set_timestamp(self, timestamp):
        is_set_left = self.cam_left.set_timestamp(timestamp)
        is_set_right = self.cam_right.set_timestamp(timestamp)
        return is_set_left and is_set_right
    
    def get_rectified_images(self):
        image_left_unrect = self.cam_left.get_left_image()
        image_right_unrect = self.cam_right.get_left_image()
        image_left, image_right = self.rectifier_wide_from_short.rectify(image_left_unrect, image_right_unrect)
        return image_left, image_right
    
    def move_n_frames(self, n_frames):
        is_moved_left = self.cam_left.move_n_frames(n_frames)
        is_moved_right = self.cam_right.move_n_frames(n_frames)
        return is_moved_left and is_moved_right


class ImageRectifier:
    def __init__(self, R, t, K_left, D_left, K_right, D_right, image_shape_wh) -> None:
        # R1 is R_points_wide_from_short
        self.R1, self.R2, self.left_P, self.right_P, self.Q, roi1, roi2 = cv2.stereoRectify(K_left, D_left, K_right, D_right, image_shape_wh, R, t, alpha=0) # alpha = 0 is zoomed to only valid pixels
        self.map_left_x, self.map_left_y = cv2.initUndistortRectifyMap(K_left, D_left, self.R1, self.left_P, image_shape_wh, cv2.CV_32FC1)
        self.map_right_x, self.map_right_y = cv2.initUndistortRectifyMap(K_right, D_right, self.R2, self.right_P, image_shape_wh, cv2.CV_32FC1)
    
    def rectify(self, image_left, image_right=None):
        image_left_rectified = cv2.remap(image_left, self.map_left_x, self.map_left_y, cv2.INTER_LINEAR)
        if image_right is None:
            return image_left_rectified, None
        
        image_right_rectified = cv2.remap(image_right, self.map_right_x, self.map_right_y, cv2.INTER_LINEAR)
        return image_left_rectified, image_right_rectified
