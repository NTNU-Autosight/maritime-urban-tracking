import matplotlib.pyplot as plt
import numpy as np

from maritime_urban_tracking.gnss import GNSS
from maritime_urban_tracking.ins import INS
from maritime_urban_tracking.lidar import Lidar
from maritime_urban_tracking.polarized_rig import PolarizedRig
from maritime_urban_tracking.o3d_pc_visualizer import O3DPointCloudVisualizer
from maritime_urban_tracking.sequences import SEQUENCE_MULTI_TARGET_CROSS, \
    SEQUENCE_MULTI_TARGET_DAY_CRUISER_UNDOCK, \
    SEQUENCE_MULTI_TARGET_KAYAK_UNDOCK_1, \
    SEQUENCE_MULTI_TARGET_KAYAK_UNDOCK_2, \
    SEQUENCE_MULTI_TARGET_LIDAR_CAMERA_CALIBRATION_1, \
    SEQUENCE_MULTI_TARGET_LIDAR_CAMERA_CALIBRATION_2, \
    SEQUENCE_MULTI_TARGET_MANEUVER_2, \
    SEQUENCE_MULTI_TARGET_PASS_1, \
    SEQUENCE_MULTI_TARGET_PASS_2, \
    SEQUENCE_MULTI_TARGET_PASS_3, \
    SEQUENCE_MULTI_TARGET_PASS_4, \
    SEQUENCE_MULTI_TARGET_PASS_5, \
    SEQUENCE_MULTI_TARGET_STEREO_CALIBRATION_1, \
    SEQUENCE_MULTI_TARGET_STEREO_CALIBRATION_2
from maritime_urban_tracking.sequences import SEQUENCE_SINGLE_TARGET_180_1, \
    SEQUENCE_SINGLE_TARGET_180_2, \
    SEQUENCE_SINGLE_TARGET_CAMERA_LIDAR_CALIBRATION_1, \
    SEQUENCE_SINGLE_TARGET_CAMERA_LIDAR_CALIBRATION_2, \
    SEQUENCE_SINGLE_TARGET_DOCKING, \
    SEQUENCE_SINGLE_TARGET_MANEUVER_1, \
    SEQUENCE_SINGLE_TARGET_MANEUVER_2, \
    SEQUENCE_SINGLE_TARGET_MAPPING, \
    SEQUENCE_SINGLE_TARGET_PASS_NORTH, \
    SEQUENCE_SINGLE_TARGET_PASS_SOUTH, \
    SEQUENCE_SINGLE_TARGET_STEREO_CALIBRATION_1, \
    SEQUENCE_SINGLE_TARGET_STEREO_CALIBRATION_2, \
    SEQUENCE_SINGLE_TARGET_UNDOCK_SOUTH, \
    SEQUENCE_SINGLE_TARGET_UNDOCK_STILL_2, \
    SEQUENCE_SINGLE_TARGET_WEST_1, \
    SEQUENCE_SINGLE_TARGET_WEST_2
from maritime_urban_tracking.sequences import H_points_vessel_from_lidar, homogeneous_multiplication, invert_transformation, convert_H_to_pose
from maritime_urban_tracking.single_stereo_camera import StereoCamera, project_onto_image_with_K_D


O3D_CONF_PATH = "./examples/o3d_conf.json"

# SEQUENCE = SEQUENCE_SINGLE_TARGET_STEREO_CALIBRATION_1
# SEQUENCE = SEQUENCE_SINGLE_TARGET_CAMERA_LIDAR_CALIBRATION_1
# SEQUENCE = SEQUENCE_SINGLE_TARGET_WEST_1
SEQUENCE = SEQUENCE_SINGLE_TARGET_MANEUVER_1
# SEQUENCE = SEQUENCE_SINGLE_TARGET_MANEUVER_2
# SEQUENCE = SEQUENCE_SINGLE_TARGET_WEST_2
# SEQUENCE = SEQUENCE_SINGLE_TARGET_PASS_NORTH
# SEQUENCE = SEQUENCE_SINGLE_TARGET_PASS_SOUTH
# SEQUENCE = SEQUENCE_SINGLE_TARGET_UNDOCK_SOUTH
# SEQUENCE = SEQUENCE_SINGLE_TARGET_UNDOCK_STILL_2
# SEQUENCE = SEQUENCE_SINGLE_TARGET_180_1
# SEQUENCE = SEQUENCE_SINGLE_TARGET_180_2
# SEQUENCE = SEQUENCE_SINGLE_TARGET_DOCKING
# SEQUENCE = SEQUENCE_SINGLE_TARGET_MAPPING
# SEQUENCE = SEQUENCE_SINGLE_TARGET_CAMERA_LIDAR_CALIBRATION_2
# SEQUENCE = SEQUENCE_SINGLE_TARGET_STEREO_CALIBRATION_2

# SEQUENCE = SEQUENCE_MULTI_TARGET_STEREO_CALIBRATION_1
# SEQUENCE = SEQUENCE_MULTI_TARGET_LIDAR_CAMERA_CALIBRATION_1
# SEQUENCE = SEQUENCE_MULTI_TARGET_PASS_1
# SEQUENCE = SEQUENCE_MULTI_TARGET_PASS_2
# SEQUENCE = SEQUENCE_MULTI_TARGET_KAYAK_UNDOCK_1
# SEQUENCE = SEQUENCE_MULTI_TARGET_KAYAK_UNDOCK_2
# SEQUENCE = SEQUENCE_MULTI_TARGET_DAY_CRUISER_UNDOCK
# SEQUENCE = SEQUENCE_MULTI_TARGET_PASS_3
# SEQUENCE = SEQUENCE_MULTI_TARGET_PASS_4
# SEQUENCE = SEQUENCE_MULTI_TARGET_PASS_5
# SEQUENCE = SEQUENCE_MULTI_TARGET_MANEUVER_2
# SEQUENCE = SEQUENCE_MULTI_TARGET_CROSS
# SEQUENCE = SEQUENCE_MULTI_TARGET_LIDAR_CAMERA_CALIBRATION_2
# SEQUENCE = SEQUENCE_MULTI_TARGET_STEREO_CALIBRATION_2


def play_all_together():
    lidar = Lidar(SEQUENCE.path_lidar_and_ins)
    ins = INS(SEQUENCE.path_lidar_and_ins)
    stereo_camera_left = StereoCamera(SEQUENCE.path_stereo_camera_left)
    stereo_camera_right = StereoCamera(SEQUENCE.path_stereo_camera_right)
    if SEQUENCE.path_polarized_rig == "":
        polarized_rig = None
    else:
        polarized_rig = PolarizedRig(SEQUENCE.path_polarized_rig)
    H_points_lidar_from_right_cam = invert_transformation(SEQUENCE.H_points_right_cam_from_lidar)
    H_points_right_cam_from_lidar = SEQUENCE.H_points_right_cam_from_lidar
    H_points_right_cam_from_left_cam = SEQUENCE.H_points_right_cam_from_left_cam
    H_points_polarized_left_from_stereo_rl = invert_transformation(SEQUENCE.H_points_stereo_rl_from_polarized_left)

    gnss_s: list[GNSS] = []
    for gnss_name, path_gnss in SEQUENCE.paths_gnss.items():
        gnss = GNSS(path_gnss)
        gnss_s.append(gnss)

    o3d_vis = O3DPointCloudVisualizer(
        visualization_parameter_path=O3D_CONF_PATH,
        should_start_with_conf=True
    )
    o3d_pc_lidar = o3d_vis.create_point_cloud()
    o3d_pc_stereo_left = o3d_vis.create_point_cloud()
    o3d_pc_stereo_right = o3d_vis.create_point_cloud()

    o3d_gnss_s = [o3d_vis.create_point(color=(0, 1.0, 0)) for _ in SEQUENCE.paths_gnss.keys()]

    o3d_coord_frame_vessel = o3d_vis.create_coord_frame()
    o3d_coord_frame_lidar = o3d_vis.create_coord_frame()
    o3d_coord_frame_stereo_left = o3d_vis.create_coord_frame()
    o3d_coord_frame_stereo_right = o3d_vis.create_coord_frame()

    while lidar.is_playable:
        timestamp = lidar.get_timestamp()
        median_timestamp = lidar.get_median_valid_point_timestamp()
        ins.set_timestamp(median_timestamp)
        # stereo_camera_right.set_timestamp(timestamp)
        stereo_camera_left.set_timestamp(median_timestamp)
        if polarized_rig is not None:
            polarized_rig.set_timestamp(timestamp)

        for gnss in gnss_s:
            gnss.set_timestamp(timestamp)

        pc_txyz_lidar = lidar.get_pointcloud()
        pc_xyz_lidar = pc_txyz_lidar[:,1:]
        pc_rgb = np.tile(np.array([255, 0, 0], dtype=np.uint8)/255, (pc_xyz_lidar.shape[0], 1))

        for i in range(len(gnss_s)):
            gnss = gnss_s[i]
            o3d_gnss = o3d_gnss_s[i]
            if gnss.is_playable:
                pos_gnss = gnss.get_pos_ned()
                pos_gnss[2] = 0
                o3d_vis.update_point(o3d_gnss, pos_gnss)

        if ins.is_playable:
            pose_ins = ins.get_pose()
            H_points_piren_from_vessel = ins.get_H_points_piren_from_vessel()
            H_points_piren_from_lidar = H_points_piren_from_vessel @ H_points_vessel_from_lidar
            pose_lidar = convert_H_to_pose(H_points_piren_from_lidar)

            pc_xyz_piren = homogeneous_multiplication(H_points_piren_from_lidar, pc_xyz_lidar)

            o3d_vis.update_coord_frame(o3d_coord_frame_vessel, pose_ins[:3], pose_ins[3:])
            o3d_vis.update_coord_frame(o3d_coord_frame_lidar, pose_lidar[:3], pose_lidar[3:])
            o3d_vis.update_point_cloud(o3d_pc_lidar, pc_xyz_piren, pc_rgb)

            # if stereo_camera_right.is_playable:
            #     pc_stereo_right_xyz_right_cam, pc_stereo_right_rgb = stereo_camera_right.get_pointcloud()
            #     H_points_piren_from_right_cam = H_points_piren_from_lidar @ H_points_lidar_from_right_cam
            #     pc_stereo_right_xyz_piren = homogeneous_multiplication(H_points_piren_from_right_cam, pc_stereo_right_xyz_right_cam)

            #     pose_stereo_right = convert_H_to_pose(H_points_piren_from_right_cam)

            #     o3d_vis.update_point_cloud(o3d_pc_stereo_right, pc_stereo_right_xyz_piren, pc_stereo_right_rgb)
            #     o3d_vis.update_coord_frame(o3d_coord_frame_stereo_right, pose_stereo_right[:3], pose_stereo_right[3:])
            
            # if stereo_camera_left.is_playable:
            #     pc_stereo_left_xyz_left_cam, pc_stereo_left_rgb = stereo_camera_left.get_pointcloud()
            #     H_points_piren_from_left_cam = H_points_piren_from_lidar @ H_points_lidar_from_right_cam @ H_points_right_cam_from_left_cam
            #     pc_stereo_left_xyz_piren = homogeneous_multiplication(H_points_piren_from_left_cam, pc_stereo_left_xyz_left_cam)

            #     pose_stereo_left = convert_H_to_pose(H_points_piren_from_left_cam)

            #     o3d_vis.update_point_cloud(o3d_pc_stereo_left, pc_stereo_left_xyz_piren, pc_stereo_left_rgb)
            #     o3d_vis.update_coord_frame(o3d_coord_frame_stereo_left, pose_stereo_left[:3], pose_stereo_left[3:])
        
        if stereo_camera_left.is_playable:
            H_points_left_cam_from_lidar = invert_transformation(H_points_lidar_from_right_cam @ H_points_right_cam_from_left_cam)
            pc_xyz_left_cam = homogeneous_multiplication(H_points_left_cam_from_lidar, pc_xyz_lidar)

            pts_xy_left_cam_float, mask_xy_from_xyz = stereo_camera_left.project_onto_image(pc_xyz_left_cam)
            pts_xy = np.array(pts_xy_left_cam_float, dtype=np.uint)

            plt.ion()
            plt.figure("Lidar projected onto image")
            plt.clf()
            # brighten image
            image_left_uint = stereo_camera_left.get_left_image()
            image_bright_uint = (image_left_uint.astype(np.float32)*127/255+128).astype(np.uint8)
            plt.imshow(image_bright_uint)
            plt.scatter(pts_xy[:,0], pts_xy[:,1], s=3, marker="o")
            plt.axis("off")
        
        if polarized_rig is not None and polarized_rig.is_playable:
            H_points_polarized_left_cam_from_lidar = H_points_polarized_left_from_stereo_rl @ H_points_right_cam_from_lidar
            pc_xyz_polarized_left_cam = homogeneous_multiplication(H_points_polarized_left_cam_from_lidar, pc_xyz_lidar)

            pts_xy_polarized_left_cam_float, mask_xy_from_xyz = project_onto_image_with_K_D(pc_xyz_polarized_left_cam, polarized_rig.K_left, polarized_rig.D_left, polarized_rig.image_shape_wh)
            pts_xy = np.array(pts_xy_polarized_left_cam_float, dtype=np.uint)

            plt.ion()
            plt.figure("Lidar projected onto polarized image")
            plt.clf()
            # brighten image
            image_polarized_left_uint, _ = polarized_rig.get_images(should_only_visual=True, should_use_learned=False)
            image_bright_uint = (image_polarized_left_uint.astype(np.float32)*127/255+128).astype(np.uint8)
            plt.imshow(image_bright_uint)
            plt.scatter(pts_xy[:,0], pts_xy[:,1], s=3, marker="o")
            plt.axis("off")



        o3d_vis.render()

        plt.pause(0.1)
        lidar.move_n_frames(int(lidar.fps*1))

if __name__ == "__main__":
    play_all_together()
