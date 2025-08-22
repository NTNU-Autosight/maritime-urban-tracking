from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation as R


DATASET_FOLDER = "/media/nicholas/T7 Shield/datasets/Maritime Urban Tracking"


def invert_transformation(H):
    R = H[:3,:3]
    T = H[:3,3]
    H_transformed = np.block([
        [R.T, -R.T.dot(T)[:,np.newaxis]],
        [np.zeros((1,3)), np.ones((1,1))]
    ])
    return H_transformed

def create_transform(R=np.eye(3), T=np.zeros((3,1))):
    H = np.block([
        [R, T],
        [np.zeros((1,3)), np.ones((1,1))]
    ])
    return H

def homogeneous_multiplication(H_np1xnp1, pts_mxn):
        m, n = pts_mxn.shape
        np1 = H_np1xnp1.shape[0]
        assert H_np1xnp1.shape[0] == H_np1xnp1.shape[1]
        assert n+1 == np1

        pts_t = H_np1xnp1.dot(np.r_[pts_mxn.T, np.ones((1, m))])[0:n, :].T
        return pts_t

def convert_pose_to_H(pose):
    """
    Pose: n, e, d, x, y, z, w
    H: 4x4 matrix
    convert_pose_to_H(pose_vessel_in_piren) creates H_points_piren_from_vessel
    """
    t = pose[:3]
    rot_quat = pose[3:]
    rot_mat = R.from_quat(rot_quat).as_matrix()
    H = create_transform(rot_mat, t[:,np.newaxis])
    return H

def convert_H_to_pose(H):
    """
    convert_H_to_pose(H_points_piren_from_lidar) creates the pose of the lidar in the PIREN frame
    """
    rot_mat = H[0:3, 0:3]
    rot = R.from_matrix(rot_mat)
    rot_quat = rot.as_quat()
    t = H[0:3, 3]
    pose = np.array([*t, *rot_quat])
    return pose

def convert_R_t_to_H(R=np.eye(3), T=np.zeros((3,1))):
    return create_transform(R, T)

def convert_H_to_R_t(H):
    R = H[:3,:3]
    T = H[:3,3]
    return R, T

@dataclass
class Sequence:
    path_stereo_camera_left: str
    path_stereo_camera_right: str
    H_points_right_cam_from_left_cam: np.ndarray
    path_lidar_and_ins: str
    H_points_right_cam_from_lidar: np.ndarray
    paths_gnss: dict
    path_polarized_rig: str
    H_points_stereo_rl_from_polarized_left: np.ndarray

# Defines the PIREN frame where the INS has its measurements from. 
# Note that the RTK used seems to have a bias that we compensate for with these offsets. 
PIREN_LAT = 63.4389029083
PIREN_LON = 10.39908278
PIREN_ALT = 39.923
OFFSET_LAT = 0
OFFSET_LON = 0

OFFSET_LAT = 1.20917e-05
OFFSET_LON = -2.278e-05

PIREN_LAT = PIREN_LAT + OFFSET_LAT
PIREN_LON = PIREN_LON + OFFSET_LON

# INS to lidar calibration
r = np.array([
    [-0.831867720239427, 0.554971722221873, 0.00157593075568076],
    [0.554878227534351, 0.831668129225735, 0.0209350289826501],
    [0.0103076977058998, 0.0182896244973725, -0.99977959621296]
])
# r_roll_tweak = R.from_euler("x", [-1.5], degrees=True).as_matrix()[0]
t = np.array([
    [-4.10094915569212],
    [-1.22739005452833],
    [-1.34745524250482]
])
# H_tweak = np.array([[ 0.95082714, -0.03797034, -0.30738575, -0.58890096],
#        [-0.05036569,  0.96029111, -0.27441625, -0.52134405],
#        [ 0.30559948,  0.27640411,  0.91115845, -0.16577111],
#        [ 0.        ,  0.        ,  0.        ,  1.        ]])
# R_tweak,_ = convert_H_to_R_t(H_tweak)
# H_tweak = convert_R_t_to_H(R=R_tweak)
H_points_vessel_from_lidar = create_transform(r, t)

# # From MA2
# ROT_FLOOR_TO_LIDAR = np.array([[-8.27535228e-01,  5.61392452e-01,  4.89505779e-03],
#        [ 5.61413072e-01,  8.27516685e-01,  5.61236993e-03],
#        [-8.99999879e-04,  7.39258326e-03, -9.99972269e-01]])
# TRANS_FLOOR_TO_LIDAR = np.array([-4.1091, -1.1602, -1.015 ])
# H_POINTS_FLOOR_FROM_LIDAR = np.block([
#     [ROT_FLOOR_TO_LIDAR, TRANS_FLOOR_TO_LIDAR[:,np.newaxis]], 
#     [np.zeros((1,3)), np.ones((1,1))]
# ])
# ROT_VESSEL_TO_FLOOR = np.array([[1., 0., 0.],
#        [0., 1., 0.],
#        [0., 0., 1.]])
# TRANS_VESSEL_TO_FLOOR = np.array([ 0. ,  0. , -0.3])
# H_POINTS_VESSEL_FROM_FLOOR = np.block([
#     [ROT_VESSEL_TO_FLOOR, TRANS_VESSEL_TO_FLOOR[:,np.newaxis]], 
#     [np.zeros((1,3)), np.ones((1,1))]
# ])
# H_points_vessel_from_lidar = H_POINTS_VESSEL_FROM_FLOOR @ H_POINTS_FLOOR_FROM_LIDAR


####
# Other calibration is specific to each day
####

####
# Single Target day
####
H_points_right_cam_from_lidar = np.array([
    [-4.94306887e-01, -8.69287318e-01,  5.10348622e-04, 4.30159488e-01],
    [ 2.53792341e-02, -1.50183450e-02, -9.99565077e-01, 1.02473391e-01],
    [ 8.68916910e-01, -4.94078950e-01,  2.94855177e-02, 6.81931136e-02],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]
])

r = np.array([
    [ 0.98406923,  0.00575494,  0.17769252],
    [ 0.00217145,  0.99901233, -0.04438065],
    [-0.17777243,  0.04405949,  0.9830848 ]
])
t = np.array([
    [-1.63168186],
    [ 0.00384789],
    [ 0.13875284]
])
H_points_right_cam_from_left_cam = create_transform(r, t)

r = np.array([
    [ 9.97358961e-01, -6.49543208e-04,  7.26270024e-02],
    [-1.90787968e-03,  9.99380642e-01,  3.51381954e-02],
    [-7.26048441e-02, -3.51839576e-02,  9.96739999e-01]
])
t = np.array([
    [-1.24682263],
    [ 0.02522048],
    [ 0.07710963]
])
H_points_stereo_rl_from_polarized_left = create_transform(r, t)

SEQUENCE_SINGLE_TARGET_STEREO_CALIBRATION_1 = Sequence(
    path_stereo_camera_left=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-11-15-20_SN46650164",
    path_stereo_camera_right=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-11-15-22_SN47641998",
    H_points_right_cam_from_left_cam=H_points_right_cam_from_left_cam,
    path_lidar_and_ins=f"{DATASET_FOLDER}/2024-07-17 Single Target/lidar_and_ins/2024-07-17-11-15-16",
    H_points_right_cam_from_lidar=H_points_right_cam_from_lidar,
    paths_gnss={},
    path_polarized_rig=f"{DATASET_FOLDER}/2024-07-17 Single Target/polarized/09-15-21_images",
    H_points_stereo_rl_from_polarized_left=H_points_stereo_rl_from_polarized_left
)

SEQUENCE_SINGLE_TARGET_CAMERA_LIDAR_CALIBRATION_1 = Sequence(
    path_stereo_camera_left=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-11-28-19_SN46650164",
    path_stereo_camera_right=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-11-28-21_SN47641998",
    H_points_right_cam_from_left_cam=H_points_right_cam_from_left_cam,
    path_lidar_and_ins=f"{DATASET_FOLDER}/2024-07-17 Single Target/lidar_and_ins/2024-07-17-11-28-15",
    H_points_right_cam_from_lidar=H_points_right_cam_from_lidar,
    paths_gnss={},
    path_polarized_rig=f"{DATASET_FOLDER}/2024-07-17 Single Target/polarized/09-28-22_images",
    H_points_stereo_rl_from_polarized_left=H_points_stereo_rl_from_polarized_left
)

SEQUENCE_SINGLE_TARGET_WEST_1 = Sequence(
    path_stereo_camera_left=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-12-16-15_SN46650164",
    path_stereo_camera_right=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-12-16-13_SN47641998",
    H_points_right_cam_from_left_cam=H_points_right_cam_from_left_cam,
    path_lidar_and_ins=f"{DATASET_FOLDER}/2024-07-17 Single Target/lidar_and_ins/2024-07-17-12-16-10",
    H_points_right_cam_from_lidar=H_points_right_cam_from_lidar,
    paths_gnss={
         "day cruiser front": f"{DATASET_FOLDER}/2024-07-17 Single Target/gnss/day cruiser front/2024-07-17-12-18-02.pos",
         "day cruiser back": f"{DATASET_FOLDER}/2024-07-17 Single Target/gnss/day cruiser back/2024-07-17-12-17-46.pos"
    },
    path_polarized_rig=f"{DATASET_FOLDER}/2024-07-17 Single Target/polarized/10-16-16_images",
    H_points_stereo_rl_from_polarized_left=H_points_stereo_rl_from_polarized_left
)

SEQUENCE_SINGLE_TARGET_MANEUVER_1 = Sequence(
    path_stereo_camera_left=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-12-20-46_SN46650164",
    path_stereo_camera_right=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-12-20-47_SN47641998",
    H_points_right_cam_from_left_cam=H_points_right_cam_from_left_cam,
    path_lidar_and_ins=f"{DATASET_FOLDER}/2024-07-17 Single Target/lidar_and_ins/2024-07-17-12-20-43",
    H_points_right_cam_from_lidar=H_points_right_cam_from_lidar,
    paths_gnss={
         "day cruiser front": f"{DATASET_FOLDER}/2024-07-17 Single Target/gnss/day cruiser front/2024-07-17-12-21-09.pos",
         "day cruiser back": f"{DATASET_FOLDER}/2024-07-17 Single Target/gnss/day cruiser back/2024-07-17-12-21-19.pos"
    },
    path_polarized_rig=f"{DATASET_FOLDER}/2024-07-17 Single Target/polarized/10-20-49_images",
    H_points_stereo_rl_from_polarized_left=H_points_stereo_rl_from_polarized_left
)

SEQUENCE_SINGLE_TARGET_MANEUVER_2 = Sequence(
    path_stereo_camera_left=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-12-24-56_SN46650164",
    path_stereo_camera_right=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-12-24-57_SN47641998",
    H_points_right_cam_from_left_cam=H_points_right_cam_from_left_cam,
    path_lidar_and_ins=f"{DATASET_FOLDER}/2024-07-17 Single Target/lidar_and_ins/2024-07-17-12-24-52",
    H_points_right_cam_from_lidar=H_points_right_cam_from_lidar,
    paths_gnss={
         "day cruiser front": f"{DATASET_FOLDER}/2024-07-17 Single Target/gnss/day cruiser front/2024-07-17-12-25-40.pos",
         "day cruiser back": f"{DATASET_FOLDER}/2024-07-17 Single Target/gnss/day cruiser back/2024-07-17-12-25-40.pos"
    },
    path_polarized_rig=f"{DATASET_FOLDER}/2024-07-17 Single Target/polarized/10-24-57_images",
    H_points_stereo_rl_from_polarized_left=H_points_stereo_rl_from_polarized_left
)

SEQUENCE_SINGLE_TARGET_WEST_2 = Sequence(
    path_stereo_camera_left=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-12-28-25_SN46650164",
    path_stereo_camera_right=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-12-28-26_SN47641998",
    H_points_right_cam_from_left_cam=H_points_right_cam_from_left_cam,
    path_lidar_and_ins=f"{DATASET_FOLDER}/2024-07-17 Single Target/lidar_and_ins/2024-07-17-12-28-35",
    H_points_right_cam_from_lidar=H_points_right_cam_from_lidar,
    paths_gnss={
         "day cruiser front": f"{DATASET_FOLDER}/2024-07-17 Single Target/gnss/day cruiser front/2024-07-17-12-29-04.pos",
         "day cruiser back": f"{DATASET_FOLDER}/2024-07-17 Single Target/gnss/day cruiser back/2024-07-17-12-29-03.pos"
    },
    path_polarized_rig=f"{DATASET_FOLDER}/2024-07-17 Single Target/polarized/10-28-33_images",
    H_points_stereo_rl_from_polarized_left=H_points_stereo_rl_from_polarized_left
)

SEQUENCE_SINGLE_TARGET_PASS_NORTH = Sequence(
    path_stereo_camera_left=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-12-33-59_SN46650164",
    path_stereo_camera_right=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-12-34-00_SN47641998",
    H_points_right_cam_from_left_cam=H_points_right_cam_from_left_cam,
    path_lidar_and_ins=f"{DATASET_FOLDER}/2024-07-17 Single Target/lidar_and_ins/2024-07-17-12-33-55",
    H_points_right_cam_from_lidar=H_points_right_cam_from_lidar,
    paths_gnss={
         "day cruiser front": f"{DATASET_FOLDER}/2024-07-17 Single Target/gnss/day cruiser front/2024-07-17-12-34-20.pos",
         "day cruiser back": f"{DATASET_FOLDER}/2024-07-17 Single Target/gnss/day cruiser back/2024-07-17-12-34-20.pos"
    },
    path_polarized_rig=f"{DATASET_FOLDER}/2024-07-17 Single Target/polarized/10-34-03_images",
    H_points_stereo_rl_from_polarized_left=H_points_stereo_rl_from_polarized_left
)

SEQUENCE_SINGLE_TARGET_PASS_SOUTH = Sequence(
    path_stereo_camera_left=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-12-38-13_SN46650164",
    path_stereo_camera_right=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-12-38-14_SN47641998",
    H_points_right_cam_from_left_cam=H_points_right_cam_from_left_cam,
    path_lidar_and_ins=f"{DATASET_FOLDER}/2024-07-17 Single Target/lidar_and_ins/2024-07-17-12-38-09",
    H_points_right_cam_from_lidar=H_points_right_cam_from_lidar,
    paths_gnss={
         "day cruiser front": f"{DATASET_FOLDER}/2024-07-17 Single Target/gnss/day cruiser front/2024-07-17-12-38-39.pos",
         "day cruiser back": f"{DATASET_FOLDER}/2024-07-17 Single Target/gnss/day cruiser back/2024-07-17-12-38-40.pos"
    },
    path_polarized_rig=f"{DATASET_FOLDER}/2024-07-17 Single Target/polarized/10-38-16_images",
    H_points_stereo_rl_from_polarized_left=H_points_stereo_rl_from_polarized_left
)

SEQUENCE_SINGLE_TARGET_UNDOCK_SOUTH = Sequence(
    path_stereo_camera_left=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-12-47-27_SN46650164",
    path_stereo_camera_right=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-12-47-29_SN47641998",
    H_points_right_cam_from_left_cam=H_points_right_cam_from_left_cam,
    path_lidar_and_ins=f"{DATASET_FOLDER}/2024-07-17 Single Target/lidar_and_ins/2024-07-17-12-47-21",
    H_points_right_cam_from_lidar=H_points_right_cam_from_lidar,
    paths_gnss={
         "day cruiser front": f"{DATASET_FOLDER}/2024-07-17 Single Target/gnss/day cruiser front/2024-07-17-12-48-19.pos",
         "day cruiser back": f"{DATASET_FOLDER}/2024-07-17 Single Target/gnss/day cruiser back/2024-07-17-12-48-21.pos"
    },
    path_polarized_rig=f"{DATASET_FOLDER}/2024-07-17 Single Target/polarized/10-47-30_images",
    H_points_stereo_rl_from_polarized_left=H_points_stereo_rl_from_polarized_left
)

SEQUENCE_SINGLE_TARGET_UNDOCK_STILL_2 = Sequence(
    path_stereo_camera_left=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-13-42-01_SN46650164",
    path_stereo_camera_right=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-13-42-01_SN47641998",
    H_points_right_cam_from_left_cam=H_points_right_cam_from_left_cam,
    path_lidar_and_ins=f"{DATASET_FOLDER}/2024-07-17 Single Target/lidar_and_ins/2024-07-17-13-41-57",
    H_points_right_cam_from_lidar=H_points_right_cam_from_lidar,
    paths_gnss={
         "day cruiser front": f"{DATASET_FOLDER}/2024-07-17 Single Target/gnss/day cruiser front/2024-07-17-13-42-11.pos",
         "day cruiser back": f"{DATASET_FOLDER}/2024-07-17 Single Target/gnss/day cruiser back/2024-07-17-13-42-11.pos"
    },
    path_polarized_rig=f"{DATASET_FOLDER}/2024-07-17 Single Target/polarized/11-42-02_images",
    H_points_stereo_rl_from_polarized_left=H_points_stereo_rl_from_polarized_left
)

SEQUENCE_SINGLE_TARGET_CROSS = Sequence(
    path_stereo_camera_left=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-13-49-30_SN46650164",
    path_stereo_camera_right=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-13-49-31_SN47641998",
    H_points_right_cam_from_left_cam=H_points_right_cam_from_left_cam,
    path_lidar_and_ins=f"{DATASET_FOLDER}/2024-07-17 Single Target/lidar_and_ins/2024-07-17-13-49-27",
    H_points_right_cam_from_lidar=H_points_right_cam_from_lidar,
    paths_gnss={
         "day cruiser front": f"{DATASET_FOLDER}/2024-07-17 Single Target/gnss/day cruiser front/2024-07-17-13-49-40.pos",
         "day cruiser back": f"{DATASET_FOLDER}/2024-07-17 Single Target/gnss/day cruiser back/2024-07-17-13-49-40.pos"
    },
    path_polarized_rig=f"{DATASET_FOLDER}/2024-07-17 Single Target/polarized/11-49-33_images",
    H_points_stereo_rl_from_polarized_left=H_points_stereo_rl_from_polarized_left
)

SEQUENCE_SINGLE_TARGET_180_1 = Sequence(
    path_stereo_camera_left=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-13-57-11_SN46650164",
    path_stereo_camera_right=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-13-57-12_SN47641998",
    H_points_right_cam_from_left_cam=H_points_right_cam_from_left_cam,
    path_lidar_and_ins=f"{DATASET_FOLDER}/2024-07-17 Single Target/lidar_and_ins/2024-07-17-13-56-06",
    H_points_right_cam_from_lidar=H_points_right_cam_from_lidar,
    paths_gnss={},
    path_polarized_rig=f"{DATASET_FOLDER}/2024-07-17 Single Target/polarized/11-56-12_images",
    H_points_stereo_rl_from_polarized_left=H_points_stereo_rl_from_polarized_left
)

SEQUENCE_SINGLE_TARGET_180_2 = Sequence( # No polarized data recorded
    path_stereo_camera_left=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-14-40-42_SN46650164",
    path_stereo_camera_right=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-14-40-43_SN47641998",
    H_points_right_cam_from_left_cam=H_points_right_cam_from_left_cam,
    path_lidar_and_ins=f"{DATASET_FOLDER}/2024-07-17 Single Target/lidar_and_ins/2024-07-17-14-40-41",
    H_points_right_cam_from_lidar=H_points_right_cam_from_lidar,
    paths_gnss={},
    path_polarized_rig=f"",
    H_points_stereo_rl_from_polarized_left=H_points_stereo_rl_from_polarized_left
)

SEQUENCE_SINGLE_TARGET_DOCKING = Sequence(
    path_stereo_camera_left=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-14-55-18_SN46650164",
    path_stereo_camera_right=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-14-55-20_SN47641998",
    H_points_right_cam_from_left_cam=H_points_right_cam_from_left_cam,
    path_lidar_and_ins=f"{DATASET_FOLDER}/2024-07-17 Single Target/lidar_and_ins/2024-07-17-14-55-26",
    H_points_right_cam_from_lidar=H_points_right_cam_from_lidar,
    paths_gnss={},
    path_polarized_rig=f"{DATASET_FOLDER}/2024-07-17 Single Target/polarized/12-54-47_images",
    H_points_stereo_rl_from_polarized_left=H_points_stereo_rl_from_polarized_left
)

SEQUENCE_SINGLE_TARGET_MAPPING = Sequence( # The polairzed rig projects lidar too far to the right and stops very early. 
    path_stereo_camera_left=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-15-05-32_SN46650164",
    path_stereo_camera_right=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-15-05-31_SN47641998",
    H_points_right_cam_from_left_cam=H_points_right_cam_from_left_cam,
    path_lidar_and_ins=f"{DATASET_FOLDER}/2024-07-17 Single Target/lidar_and_ins/2024-07-17-15-05-31",
    H_points_right_cam_from_lidar=H_points_right_cam_from_lidar,
    paths_gnss={},
    path_polarized_rig=f"{DATASET_FOLDER}/2024-07-17 Single Target/polarized/13-05-56_images",
    H_points_stereo_rl_from_polarized_left=H_points_stereo_rl_from_polarized_left
)

SEQUENCE_SINGLE_TARGET_CAMERA_LIDAR_CALIBRATION_2 = Sequence( # No polarized data recorded
    path_stereo_camera_left=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-15-27-30_SN46650164",
    path_stereo_camera_right=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-15-27-28_SN47641998",
    H_points_right_cam_from_left_cam=H_points_right_cam_from_left_cam,
    path_lidar_and_ins=f"{DATASET_FOLDER}/2024-07-17 Single Target/lidar_and_ins/2024-07-17-15-27-27",
    H_points_right_cam_from_lidar=H_points_right_cam_from_lidar,
    paths_gnss={},
    path_polarized_rig=f"",
    H_points_stereo_rl_from_polarized_left=H_points_stereo_rl_from_polarized_left
)

SEQUENCE_SINGLE_TARGET_STEREO_CALIBRATION_2 = Sequence( # No polarized data recorded
    path_stereo_camera_left=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-15-34-46_SN46650164",
    path_stereo_camera_right=f"{DATASET_FOLDER}/2024-07-17 Single Target/images/2024-07-17-15-34-45_SN47641998",
    H_points_right_cam_from_left_cam=H_points_right_cam_from_left_cam,
    path_lidar_and_ins=f"{DATASET_FOLDER}/2024-07-17 Single Target/lidar_and_ins/2024-07-17-15-34-45",
    H_points_right_cam_from_lidar=H_points_right_cam_from_lidar,
    paths_gnss={},
    path_polarized_rig=f"",
    H_points_stereo_rl_from_polarized_left=H_points_stereo_rl_from_polarized_left
)


####
# Multi target day
####
H_points_right_cam_from_lidar = np.array([
    [-0.5675599 , -0.82332899, -0.00226229,  0.44340955],
    [ 0.02076975, -0.01157061, -0.99971733,  0.10048742],
    [ 0.82307009, -0.56744646,  0.02366736,  0.10781797],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
])

# Both sequences
r = np.array([[ 0.9999918 , -0.00166961,  0.00369083],
    [ 0.00173772,  0.99982679, -0.01853033],
    [-0.00365925,  0.01853659,  0.99982149]])
t = np.array([[-1.53902361e+00],
    [-5.57686850e-04],
    [-1.29203031e-02]])
H_points_right_cam_from_left_cam = create_transform(r, t)

r = np.array([
    [ 9.99960151e-01,  7.59820691e-06, -8.92733199e-03],
    [ 2.90602827e-04,  9.99441976e-01,  3.34013806e-02],
    [ 8.92260411e-03, -3.34026439e-02,  9.99402147e-01]
])
t = np.array([
    [-1.20023506],
    [ 0.02330838],
    [-0.07169089]
])
H_points_stereo_rl_from_polarized_left = create_transform(r, t)

SEQUENCE_MULTI_TARGET_STEREO_CALIBRATION_1 = Sequence(
    path_stereo_camera_left=f"{DATASET_FOLDER}/2024-07-19 Multi target/images/2024-07-19-09-10-54_SN46650164",
    path_stereo_camera_right=f"{DATASET_FOLDER}/2024-07-19 Multi target/images/2024-07-19-09-10-54_SN47641998",
    H_points_right_cam_from_left_cam=H_points_right_cam_from_left_cam,
    path_lidar_and_ins=f"{DATASET_FOLDER}/2024-07-19 Multi target/lidar_and_ins/2024-07-19-09-10-53",
    H_points_right_cam_from_lidar=H_points_right_cam_from_lidar,
    paths_gnss={},
    path_polarized_rig=f"{DATASET_FOLDER}/2024-07-19 Multi target/polarized/07-11-01_images",
    H_points_stereo_rl_from_polarized_left=H_points_stereo_rl_from_polarized_left
)

SEQUENCE_MULTI_TARGET_LIDAR_CAMERA_CALIBRATION_1 = Sequence(
    path_stereo_camera_left=f"{DATASET_FOLDER}/2024-07-19 Multi target/images/2024-07-19-09-20-13_SN46650164",
    path_stereo_camera_right=f"{DATASET_FOLDER}/2024-07-19 Multi target/images/2024-07-19-09-20-15_SN47641998",
    H_points_right_cam_from_left_cam=H_points_right_cam_from_left_cam,
    path_lidar_and_ins=f"{DATASET_FOLDER}/2024-07-19 Multi target/lidar_and_ins/2024-07-19-09-20-09",
    H_points_right_cam_from_lidar=H_points_right_cam_from_lidar,
    paths_gnss={},
    path_polarized_rig=f"{DATASET_FOLDER}/2024-07-19 Multi target/polarized/07-20-16_images",
    H_points_stereo_rl_from_polarized_left=H_points_stereo_rl_from_polarized_left
)

SEQUENCE_MULTI_TARGET_PASS_1 = Sequence(
    path_stereo_camera_left=f"{DATASET_FOLDER}/2024-07-19 Multi target/images/2024-07-19-09-37-01_SN46650164",
    path_stereo_camera_right=f"{DATASET_FOLDER}/2024-07-19 Multi target/images/2024-07-19-09-37-02_SN47641998",
    H_points_right_cam_from_left_cam=H_points_right_cam_from_left_cam,
    path_lidar_and_ins=f"{DATASET_FOLDER}/2024-07-19 Multi target/lidar_and_ins/2024-07-19-09-36-59",
    H_points_right_cam_from_lidar=H_points_right_cam_from_lidar,
    paths_gnss={
         "day cruiser": f"{DATASET_FOLDER}/2024-07-19 Multi target/gnss/day cruiser/2024-07-19-09-37-34.pos",
         "kayak": f"{DATASET_FOLDER}/2024-07-19 Multi target/gnss/kayak/2024-07-19-09-37-32.pos"
    },
    path_polarized_rig=f"{DATASET_FOLDER}/2024-07-19 Multi target/polarized/07-36-59_images",
    H_points_stereo_rl_from_polarized_left=H_points_stereo_rl_from_polarized_left
)

SEQUENCE_MULTI_TARGET_PASS_2 = Sequence(
    path_stereo_camera_left=f"{DATASET_FOLDER}/2024-07-19 Multi target/images/2024-07-19-09-40-05_SN46650164",
    path_stereo_camera_right=f"{DATASET_FOLDER}/2024-07-19 Multi target/images/2024-07-19-09-40-07_SN47641998",
    H_points_right_cam_from_left_cam=H_points_right_cam_from_left_cam,
    path_lidar_and_ins=f"{DATASET_FOLDER}/2024-07-19 Multi target/lidar_and_ins/2024-07-19-09-39-59",
    H_points_right_cam_from_lidar=H_points_right_cam_from_lidar,
    paths_gnss={
         "day cruiser": f"{DATASET_FOLDER}/2024-07-19 Multi target/gnss/day cruiser/2024-07-19-09-40-40.pos",
         "kayak": f"{DATASET_FOLDER}/2024-07-19 Multi target/gnss/kayak/2024-07-19-09-40-32.pos"
    },
    path_polarized_rig=f"{DATASET_FOLDER}/2024-07-19 Multi target/polarized/07-40-07_images",
    H_points_stereo_rl_from_polarized_left=H_points_stereo_rl_from_polarized_left
)

SEQUENCE_MULTI_TARGET_KAYAK_UNDOCK_1 = Sequence(
    path_stereo_camera_left=f"{DATASET_FOLDER}/2024-07-19 Multi target/images/2024-07-19-09-44-31_SN46650164",
    path_stereo_camera_right=f"{DATASET_FOLDER}/2024-07-19 Multi target/images/2024-07-19-09-44-32_SN47641998",
    H_points_right_cam_from_left_cam=H_points_right_cam_from_left_cam,
    path_lidar_and_ins=f"{DATASET_FOLDER}/2024-07-19 Multi target/lidar_and_ins/2024-07-19-09-44-25",
    H_points_right_cam_from_lidar=H_points_right_cam_from_lidar,
    paths_gnss={
         "day cruiser": f"{DATASET_FOLDER}/2024-07-19 Multi target/gnss/day cruiser/2024-07-19-09-45-10.pos",
         "kayak": f"{DATASET_FOLDER}/2024-07-19 Multi target/gnss/kayak/2024-07-19-09-45-04.pos"
    },
    path_polarized_rig=f"{DATASET_FOLDER}/2024-07-19 Multi target/polarized/07-44-32_images",
    H_points_stereo_rl_from_polarized_left=H_points_stereo_rl_from_polarized_left
)

SEQUENCE_MULTI_TARGET_KAYAK_UNDOCK_2 = Sequence(
    path_stereo_camera_left=f"{DATASET_FOLDER}/2024-07-19 Multi target/images/2024-07-19-09-47-27_SN46650164",
    path_stereo_camera_right=f"{DATASET_FOLDER}/2024-07-19 Multi target/images/2024-07-19-09-47-28_SN47641998",
    H_points_right_cam_from_left_cam=H_points_right_cam_from_left_cam,
    path_lidar_and_ins=f"{DATASET_FOLDER}/2024-07-19 Multi target/lidar_and_ins/2024-07-19-09-47-21",
    H_points_right_cam_from_lidar=H_points_right_cam_from_lidar,
    paths_gnss={
         "day cruiser": f"{DATASET_FOLDER}/2024-07-19 Multi target/gnss/day cruiser/2024-07-19-09-48-02.pos",
         "kayak": f"{DATASET_FOLDER}/2024-07-19 Multi target/gnss/kayak/2024-07-19-09-48-02.pos"
    },
    path_polarized_rig=f"{DATASET_FOLDER}/2024-07-19 Multi target/polarized/07-47-28_images",
    H_points_stereo_rl_from_polarized_left=H_points_stereo_rl_from_polarized_left
)

SEQUENCE_MULTI_TARGET_DAY_CRUISER_UNDOCK = Sequence(
    path_stereo_camera_left=f"{DATASET_FOLDER}/2024-07-19 Multi target/images/2024-07-19-09-54-31_SN46650164",
    path_stereo_camera_right=f"{DATASET_FOLDER}/2024-07-19 Multi target/images/2024-07-19-09-54-32_SN47641998",
    H_points_right_cam_from_left_cam=H_points_right_cam_from_left_cam,
    path_lidar_and_ins=f"{DATASET_FOLDER}/2024-07-19 Multi target/lidar_and_ins/2024-07-19-09-54-27",
    H_points_right_cam_from_lidar=H_points_right_cam_from_lidar,
    paths_gnss={
         "day cruiser": f"{DATASET_FOLDER}/2024-07-19 Multi target/gnss/day cruiser/2024-07-19-09-55-04.pos",
         "kayak": f"{DATASET_FOLDER}/2024-07-19 Multi target/gnss/kayak/2024-07-19-09-55-04.pos"
    },
    path_polarized_rig=f"{DATASET_FOLDER}/2024-07-19 Multi target/polarized/07-54-29_images",
    H_points_stereo_rl_from_polarized_left=H_points_stereo_rl_from_polarized_left
)

SEQUENCE_MULTI_TARGET_PASS_3 = Sequence(
    path_stereo_camera_left=f"{DATASET_FOLDER}/2024-07-19 Multi target/images/2024-07-19-10-09-25_SN46650164",
    path_stereo_camera_right=f"{DATASET_FOLDER}/2024-07-19 Multi target/images/2024-07-19-10-09-25_SN47641998",
    H_points_right_cam_from_left_cam=H_points_right_cam_from_left_cam,
    path_lidar_and_ins=f"{DATASET_FOLDER}/2024-07-19 Multi target/lidar_and_ins/2024-07-19-10-09-18",
    H_points_right_cam_from_lidar=H_points_right_cam_from_lidar,
    paths_gnss={
         "day cruiser": f"{DATASET_FOLDER}/2024-07-19 Multi target/gnss/day cruiser/2024-07-19-10-09-49.pos",
         "kayak": f"{DATASET_FOLDER}/2024-07-19 Multi target/gnss/kayak/2024-07-19-10-09-51.pos"
    },
    path_polarized_rig=f"{DATASET_FOLDER}/2024-07-19 Multi target/polarized/08-09-22_images",
    H_points_stereo_rl_from_polarized_left=H_points_stereo_rl_from_polarized_left
)

SEQUENCE_MULTI_TARGET_PASS_4 = Sequence(
    path_stereo_camera_left=f"{DATASET_FOLDER}/2024-07-19 Multi target/images/2024-07-19-10-11-31_SN46650164",
    path_stereo_camera_right=f"{DATASET_FOLDER}/2024-07-19 Multi target/images/2024-07-19-10-11-32_SN47641998",
    H_points_right_cam_from_left_cam=H_points_right_cam_from_left_cam,
    path_lidar_and_ins=f"{DATASET_FOLDER}/2024-07-19 Multi target/lidar_and_ins/2024-07-19-10-11-32",
    H_points_right_cam_from_lidar=H_points_right_cam_from_lidar,
    paths_gnss={
         "day cruiser": f"{DATASET_FOLDER}/2024-07-19 Multi target/gnss/day cruiser/2024-07-19-10-12-10.pos",
         "kayak": f"{DATASET_FOLDER}/2024-07-19 Multi target/gnss/kayak/2024-07-19-10-12-10.pos"
    },
    path_polarized_rig=f"{DATASET_FOLDER}/2024-07-19 Multi target/polarized/08-11-33_images",
    H_points_stereo_rl_from_polarized_left=H_points_stereo_rl_from_polarized_left
)

SEQUENCE_MULTI_TARGET_PASS_5 = Sequence( # Misses kayak GT. 
    path_stereo_camera_left=f"{DATASET_FOLDER}/2024-07-19 Multi target/images/2024-07-19-10-13-33_SN46650164",
    path_stereo_camera_right=f"{DATASET_FOLDER}/2024-07-19 Multi target/images/2024-07-19-10-13-34_SN47641998",
    H_points_right_cam_from_left_cam=H_points_right_cam_from_left_cam,
    path_lidar_and_ins=f"{DATASET_FOLDER}/2024-07-19 Multi target/lidar_and_ins/2024-07-19-10-13-33",
    H_points_right_cam_from_lidar=H_points_right_cam_from_lidar,
    paths_gnss={
         "day cruiser": f"{DATASET_FOLDER}/2024-07-19 Multi target/gnss/day cruiser/2024-07-19-10-14-10.pos"
    },
    path_polarized_rig=f"{DATASET_FOLDER}/2024-07-19 Multi target/polarized/08-13-35_images",
    H_points_stereo_rl_from_polarized_left=H_points_stereo_rl_from_polarized_left
)

SEQUENCE_MULTI_TARGET_MANEUVER_2 = Sequence(
    path_stereo_camera_left=f"{DATASET_FOLDER}/2024-07-19 Multi target/images/2024-07-19-10-23-39_SN46650164",
    path_stereo_camera_right=f"{DATASET_FOLDER}/2024-07-19 Multi target/images/2024-07-19-10-23-40_SN47641998",
    H_points_right_cam_from_left_cam=H_points_right_cam_from_left_cam,
    path_lidar_and_ins=f"{DATASET_FOLDER}/2024-07-19 Multi target/lidar_and_ins/2024-07-19-10-23-36",
    H_points_right_cam_from_lidar=H_points_right_cam_from_lidar,
    paths_gnss={
         "day cruiser": f"{DATASET_FOLDER}/2024-07-19 Multi target/gnss/day cruiser/2024-07-19-10-24-15.pos",
         "kayak": f"{DATASET_FOLDER}/2024-07-19 Multi target/gnss/kayak/2024-07-19-10-24-32.pos"
    },
    path_polarized_rig=f"{DATASET_FOLDER}/2024-07-19 Multi target/polarized/08-23-40_images",
    H_points_stereo_rl_from_polarized_left=H_points_stereo_rl_from_polarized_left
)

SEQUENCE_MULTI_TARGET_CROSS = Sequence(
    path_stereo_camera_left=f"{DATASET_FOLDER}/2024-07-19 Multi target/images/2024-07-19-10-27-44_SN46650164",
    path_stereo_camera_right=f"{DATASET_FOLDER}/2024-07-19 Multi target/images/2024-07-19-10-27-44_SN47641998",
    H_points_right_cam_from_left_cam=H_points_right_cam_from_left_cam,
    path_lidar_and_ins=f"{DATASET_FOLDER}/2024-07-19 Multi target/lidar_and_ins/2024-07-19-10-27-35",
    H_points_right_cam_from_lidar=H_points_right_cam_from_lidar,
    paths_gnss={
         "day cruiser": f"{DATASET_FOLDER}/2024-07-19 Multi target/gnss/day cruiser/2024-07-19-10-28-14.pos",
         "kayak": f"{DATASET_FOLDER}/2024-07-19 Multi target/gnss/kayak/2024-07-19-10-28-10.pos"
    },
    path_polarized_rig=f"{DATASET_FOLDER}/2024-07-19 Multi target/polarized/08-27-42_images",
    H_points_stereo_rl_from_polarized_left=H_points_stereo_rl_from_polarized_left
)

SEQUENCE_MULTI_TARGET_LIDAR_CAMERA_CALIBRATION_2 = Sequence( # No polarized data recorded
    path_stereo_camera_left=f"{DATASET_FOLDER}/2024-07-19 Multi target/images/2024-07-19-10-40-27_SN46650164",
    path_stereo_camera_right=f"{DATASET_FOLDER}/2024-07-19 Multi target/images/2024-07-19-10-40-28_SN47641998",
    H_points_right_cam_from_left_cam=H_points_right_cam_from_left_cam,
    path_lidar_and_ins=f"{DATASET_FOLDER}/2024-07-19 Multi target/lidar_and_ins/2024-07-19-10-40-14",
    H_points_right_cam_from_lidar=H_points_right_cam_from_lidar,
    paths_gnss={},
    path_polarized_rig=f"",
    H_points_stereo_rl_from_polarized_left=H_points_stereo_rl_from_polarized_left
)

SEQUENCE_MULTI_TARGET_STEREO_CALIBRATION_2 = Sequence( # No polarized data recorded
    path_stereo_camera_left=f"{DATASET_FOLDER}/2024-07-19 Multi target/images/2024-07-19-10-43-46_SN46650164",
    path_stereo_camera_right=f"{DATASET_FOLDER}/2024-07-19 Multi target/images/2024-07-19-10-43-47_SN47641998",
    H_points_right_cam_from_left_cam=H_points_right_cam_from_left_cam,
    path_lidar_and_ins=f"{DATASET_FOLDER}/2024-07-19 Multi target/lidar_and_ins/2024-07-19-10-43-41",
    H_points_right_cam_from_lidar=H_points_right_cam_from_lidar,
    paths_gnss={},
    path_polarized_rig=f"",
    H_points_stereo_rl_from_polarized_left=H_points_stereo_rl_from_polarized_left
)

