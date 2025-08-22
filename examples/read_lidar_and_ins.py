import matplotlib.pyplot as plt
import numpy as np

from maritime_urban_tracking.ins import INS
from maritime_urban_tracking.lidar import Lidar
from maritime_urban_tracking.o3d_pc_visualizer import O3DPointCloudVisualizer
from maritime_urban_tracking.sequences import SEQUENCE_SINGLE_TARGET_STEREO_CALIBRATION_1, H_points_vessel_from_lidar, homogeneous_multiplication, convert_H_to_pose


SEQUENCE = SEQUENCE_SINGLE_TARGET_STEREO_CALIBRATION_1
O3D_CONF_PATH = "./examples/o3d_conf.json"


def read_lidar():
    lidar = Lidar(SEQUENCE.path_lidar_and_ins)

    o3d_vis = O3DPointCloudVisualizer()
    pcd = o3d_vis.create_point_cloud()

    lidar.set_timestamp(1721207739738390000)
    while lidar.is_playable:
        timestamp = lidar.get_timestamp()
        print(f"Timestamp: {timestamp}")

        pc_xyz = lidar.get_pointcloud()
        pc_rgb = np.tile(np.array([255, 0, 0], dtype=np.uint8)/255, (pc_xyz.shape[0], 1))

        o3d_vis.update_point_cloud(pcd, pc_xyz, pc_rgb)
        o3d_vis.render()

        plt.pause(0.1)
        lidar.move_n_frames(int(lidar.fps*0.1))

def read_ins():
    ins = INS(SEQUENCE.path_lidar_and_ins)

    ins.set_timestamp(1721207739738390000)
    while ins.is_playable:
        timestamp = ins.get_timestamp()

        pose = ins.get_pose()
        print(f"Timestamp: {timestamp}, n: {pose[0]: 0.0f}, e: {pose[1]: 0.0f}")

        plt.pause(0.01)
        ins.move_n_frames(int(ins.fps*0.1))

def read_lidar_and_ins():
    lidar = Lidar(SEQUENCE.path_lidar_and_ins)
    ins = INS(SEQUENCE.path_lidar_and_ins)

    o3d_vis = O3DPointCloudVisualizer(
        visualization_parameter_path=O3D_CONF_PATH,
        should_start_with_conf=True
    )
    o3d_pc_lidar = o3d_vis.create_point_cloud()
    o3d_coord_frame_vessel = o3d_vis.create_coord_frame()
    o3d_coord_frame_lidar = o3d_vis.create_coord_frame()

    TIMESTAMP = 1721207989732741120
    lidar.set_timestamp(TIMESTAMP)
    ins.set_timestamp(TIMESTAMP)

    while lidar.is_playable and ins.is_playable:
        timestamp_lidar = lidar.get_timestamp()
        timestamp_ins = ins.get_timestamp()
        print(f"Timestamp: {timestamp_lidar}. Diff to INS: {(timestamp_lidar-timestamp_ins)/(10**6): 0.3f}ms")

        pc_txyz_lidar = lidar.get_pointcloud()
        pc_xyz_lidar = pc_txyz_lidar[:,1:]
        pc_rgb = np.tile(np.array([255, 0, 0], dtype=np.uint8)/255, (pc_xyz_lidar.shape[0], 1))

        pose_ins = ins.get_pose()
        H_points_piren_from_vessel = ins.get_H_points_piren_from_vessel()
        H_points_piren_from_lidar = H_points_piren_from_vessel @ H_points_vessel_from_lidar
        pose_lidar = convert_H_to_pose(H_points_piren_from_lidar)

        pc_xyz_piren = homogeneous_multiplication(H_points_piren_from_lidar, pc_xyz_lidar)

        o3d_vis.update_coord_frame(o3d_coord_frame_vessel, pose_ins[:3], pose_ins[3:])
        o3d_vis.update_coord_frame(o3d_coord_frame_lidar, pose_lidar[:3], pose_lidar[3:])
        o3d_vis.update_point_cloud(o3d_pc_lidar, pc_xyz_piren, pc_rgb)
        o3d_vis.render()

        plt.pause(0.1)
        lidar.move_n_frames(int(lidar.fps*1))
        timestamp_lidar = lidar.get_timestamp()
        ins.set_timestamp(timestamp_lidar)

if __name__ == "__main__":
    # read_lidar()
    # read_ins()
    read_lidar_and_ins()
