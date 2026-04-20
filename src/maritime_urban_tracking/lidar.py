import glob
import numpy as np
import open3d as o3d

from maritime_urban_tracking.timestamp_base import TimestampBase

class Lidar(TimestampBase):
    def __init__(self, path_to_folder) -> None:
        """
         
        """
        self.path_to_folder = path_to_folder

        filepaths = glob.glob(f"{path_to_folder}/*.bytes")
        timestamps_str = [f.split("/")[-1].split(".")[0] for f in filepaths]
        timestamps = np.array(timestamps_str, dtype=int)
        self._set_timestamps(timestamps)

        self.fps = 10
        self._point_step = 48

    def get_pointcloud(self):
        message_bytes = self._get_message_bytes()

        """
        The point cloud message looks like this:

        data = array([0, 0, 0, ..., 0, 0, 0], dtype=uint8)
        All fields have count=1, INT8=1, UINT8=2, INT16=3, UINT16=4, INT32=5, UINT32=6, FLOAT32=7, FLOAT64=8, __msgtype__='sensor_msgs/msg/PointField'
        fields =[
            0 = sensor_msgs__msg__PointField(name='x', offset=0, datatype=7)
            1 = sensor_msgs__msg__PointField(name='y', offset=4, datatype=7)
            2 = sensor_msgs__msg__PointField(name='z', offset=8, datatype=7)
            3 = sensor_msgs__msg__PointField(name='intensity', offset=16, datatype=7)
            4 = sensor_msgs__msg__PointField(name='t', offset=20, datatype=6)
            5 = sensor_msgs__msg__PointField(name='reflectivity', offset=24, datatype=4)
            6 = sensor_msgs__msg__PointField(name='ring', offset=26, datatype=4)
            7 = sensor_msgs__msg__PointField(name='ambient', offset=28, datatype=4)
            8 = sensor_msgs__msg__PointField(name='range', offset=32, datatype=6)
            len() = 9
        ]
        header = std_msgs__msg__Header(stamp=builtin_interfaces__msg__Time(sec=1721207717, nanosec=241221888, __msgtype__='builtin_interfaces/msg/Time'), frame_id='lidar_aft/os_sensor', __msgtype__='std_msgs/msg/Header')
        special variables
        frame_id = 'lidar_aft/os_sensor'
        stamp = builtin_interfaces__msg__Time(sec=1721207717, nanosec=241221888, __msgtype__='builtin_interfaces/msg/Time')
        height = 32
        is_bigendian = False
        is_dense = True
        point_step = 48
        row_step = 49152
        width = 1024

        It is important that we include the time in the data, so that we can later compensate for slow Lidar. 
        """

        xyz = message_bytes[:,:12].view(dtype=np.float32)
        t = message_bytes[:,20:24].view(dtype=np.uint32)

        txyz = np.concatenate((t, xyz), axis=1)
        return txyz

    def get_median_valid_point_timestamp(self):
        message_bytes = self._get_message_bytes()
        t = message_bytes[:,20:24].view(dtype=np.uint32).reshape(-1)
        r = message_bytes[:,32:36].view(dtype=np.uint32).reshape(-1)
        valid = r > 0
        if not np.any(valid):
            return None
        t_median = int(np.median(t[valid]))
        return int(self.get_timestamp()) + t_median
    
    def _get_message_bytes(self):
        timestamp = self.get_timestamp()
        filepath = f"{self.path_to_folder}/{timestamp}.bytes"
        message_bytes = np.fromfile(filepath, dtype=np.uint8).reshape(-1, self._point_step)
        return message_bytes
