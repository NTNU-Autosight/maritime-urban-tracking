import glob
import numpy as np
import cv2

from maritime_urban_tracking.timestamp_base import TimestampBase

class StereoCamera(TimestampBase):
    def __init__(self, path_to_folder) -> None:
        """
        These images should already be rectified. This can be seen from the intrinsics matrices. 
        """
        self.path_to_folder = path_to_folder

        self.K_left = np.loadtxt(f"{path_to_folder}/K_left.csv", delimiter=",")
        self.D_left = np.loadtxt(f"{path_to_folder}/D_left.csv", delimiter=",")
        self.K_right = np.loadtxt(f"{path_to_folder}/K_right.csv", delimiter=",")
        self.D_right = np.loadtxt(f"{path_to_folder}/D_right.csv", delimiter=",")
        self.t = np.loadtxt(f"{path_to_folder}/t.csv", delimiter=",")
        self.R = np.loadtxt(f"{path_to_folder}/R.csv", delimiter=",")

        filenames_images_left = glob.glob(f"{path_to_folder}/*_left.jpg")
        timestamps_left = [f.split("/")[-1].split("_")[0] for f in filenames_images_left]
        filenames_images_right = glob.glob(f"{path_to_folder}/*_right.jpg")
        timestamps_right = [f.split("/")[-1].split("_")[0] for f in filenames_images_right]

        timestamps_str = np.intersect1d(timestamps_left, timestamps_right)
        timestamps = np.array([int(s) for s in timestamps_str])
        self._set_timestamps(timestamps)

        self.fps = 30

        image_left_bgr = cv2.imread(f"{self.path_to_folder}/{self._timestamps[0]}_left.jpg")
        height = image_left_bgr.shape[0]
        width = image_left_bgr.shape[1]
        self.image_shape_wh = (width, height)

        # For short baseline disparity calculations
        num_disparities = int(self.image_shape_wh[0]*0.04)
        block_size = 4
        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity = 1,
            numDisparities = num_disparities,
            blockSize = block_size,
            P1 = 8*3*block_size*block_size,
            P2 = 32*3*block_size*block_size,
            disp12MaxDiff = 0,
            uniquenessRatio = 0,
            speckleWindowSize = 200,
            speckleRange = 1,
            mode = cv2.STEREO_SGBM_MODE_SGBM
        )

    def get_left_image(self):
        timestmap = self.get_timestamp()
        image_left_bgr = cv2.imread(f"{self.path_to_folder}/{timestmap}_left.jpg")
        image_left_rgb = cv2.cvtColor(image_left_bgr, cv2.COLOR_BGR2RGB)
        return image_left_rgb

    def get_right_image(self):
        timestmap = self.get_timestamp()
        image_right_bgr = cv2.imread(f"{self.path_to_folder}/{timestmap}_right.jpg")
        image_right_rgb = cv2.cvtColor(image_right_bgr, cv2.COLOR_BGR2RGB)
        return image_right_rgb
    
    def get_disparity(self):
        left_image = self.get_left_image()
        right_image = self.get_right_image()
        disparity = calc_disparity(left_image, right_image, self.stereo_matcher)
        return disparity
    
    def get_pointcloud(self):
        left_image = self.get_left_image()
        disparity = self.get_disparity()

        Q = self.get_Q()
        point_image = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=False)

        valid_indeces = (~np.all(~np.isfinite(point_image), axis=2)) & (point_image[:,:,2] < 150) & (disparity >= 2)
        pc_xyz = point_image[valid_indeces]
        pc_rgb = left_image[valid_indeces] / 255

        return pc_xyz, pc_rgb
    
    def get_Q(self):
        # Create own Q matrix:
        # Could also get it from cv2.stereoRectify if the rectification is not already done
        # OpenCV Q matrix looks like this:
        # See here: https://answers.opencv.org/question/187734/derivation-for-perspective-transformation-matrix-q/
        #  1   0   0       -cx
        #  0   1   0       -cy
        #  0   0   0       f(px)
        #  0   0   -1/Tx   ~0

        K = self.K_left
        T = self.t

        # R should here be identity
        Q = np.array([
            [1, 0, 0, -K[0, 2]],
            [0, 1, 0, -K[1, 2]],
            [0, 0, 0, K[0, 0]],
            [0, 0, - 1 / T[0], 0]
        ])
        return Q
    
    def project_onto_image(self, pc_xyz_camera_frame):
        pts_xy, mask_xyz_to_image = project_onto_image_with_K_D(pc_xyz_camera_frame, self.K_left, self.D_left, self.image_shape_wh)
        return pts_xy, mask_xyz_to_image

def calc_disparity(left_image, right_image, stereo_matcher):
    left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    left = cv2.equalizeHist(left_gray)
    right = cv2.equalizeHist(right_gray)

    disparity_sgbm = stereo_matcher.compute(left, right)
    disparity = disparity_sgbm.astype(np.float32) / 16.0

    disparity[disparity > disparity.max()*0.7] = 0

    return disparity

def project_onto_image_with_K_D(xyz, K, D, image_shape_wh):
        rvec = np.zeros((1,3), dtype=np.float32)
        tvec = np.zeros((1,3), dtype=np.float32)
        image_points, _ = cv2.projectPoints(xyz, rvec, tvec, K, D)
        image_points_squeezed = np.squeeze(image_points, axis=1)
        mask_xyz_to_image = (xyz[:,2] > 0.5) & (image_points_squeezed[:,0] > 0) & (image_points_squeezed[:,1] > 0) & (image_points_squeezed[:,0] < image_shape_wh[0]) & (image_points_squeezed[:,1] < image_shape_wh[1])
        pts_xy = image_points_squeezed[mask_xyz_to_image]

        return pts_xy, mask_xyz_to_image
