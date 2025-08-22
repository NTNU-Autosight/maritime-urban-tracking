import matplotlib.pyplot as plt
import numpy as np

from maritime_urban_tracking.dual_stereo_camera import DualStereoCamera
from maritime_urban_tracking.sequences import SEQUENCE_MULTI_TARGET_STEREO_CALIBRATION_1, SEQUENCE_MULTI_TARGET_STEREO_CALIBRATION_2, SEQUENCE_SINGLE_TARGET_STEREO_CALIBRATION_1, convert_H_to_R_t
from maritime_urban_tracking.single_stereo_camera import StereoCamera


# SEQUENCE = SEQUENCE_SINGLE_TARGET_STEREO_CALIBRATION_1
# SEQUENCE = SEQUENCE_MULTI_TARGET_STEREO_CALIBRATION_1
SEQUENCE = SEQUENCE_MULTI_TARGET_STEREO_CALIBRATION_2

def read_single_stereo_camera():
    cam = StereoCamera(SEQUENCE.path_stereo_camera_left)

    cam.set_timestamp(1721207739738390000)
    while cam.is_playable:
        timestamp = cam.get_timestamp()
        print(f"Timestamp: {timestamp}")

        image_left = cam.get_left_image()
        image_right = cam.get_right_image()

        plt.ion()
        plt.figure("Images")
        plt.clf()
        plt.subplot(1,2,1)
        plt.imshow(image_left)
        plt.axis("off")
        plt.subplot(1,2,2)
        plt.imshow(image_right)
        plt.axis("off")

        plt.pause(0.1)
        plt.show()

        cam.move_n_frames(int(cam.fps*0.1))

def read_dual_stereo_camera():
    R, t = convert_H_to_R_t(SEQUENCE.H_points_right_cam_from_left_cam)
    cam = DualStereoCamera(
        path_to_folder_left=SEQUENCE.path_stereo_camera_left, 
        path_to_folder_right=SEQUENCE.path_stereo_camera_right, 
        R=R, 
        t=t
    )

    while cam.is_playable:
        timestamp = cam.get_timestamp()
        print(f"Timestamp: {timestamp}")

        image_left, image_right = cam.get_rectified_images()

        ys = np.array([range(0, cam.cam_left.image_shape_wh[1], 70)])
        xs = np.array([0, cam.cam_left.image_shape_wh[0]-1])
        Xs, Ys = np.meshgrid(xs, ys)

        plt.ion()
        plt.figure("Images")
        plt.clf()
        plt.subplot(1,2,1)
        plt.imshow(image_left)
        plt.plot(Xs.T, Ys.T, color="green", linewidth=1)
        plt.plot
        plt.axis("off")
        plt.subplot(1,2,2)
        plt.imshow(image_right)
        plt.plot(Xs.T, Ys.T, color="green", linewidth=1)
        plt.axis("off")

        plt.pause(0.01)
        plt.show()

        # while True:
        #     plt.pause(1)

        cam.move_n_frames(int(cam.fps*1))

if __name__ == "__main__":
    # read_single_stereo_camera()
    read_dual_stereo_camera()
