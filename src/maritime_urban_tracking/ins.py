import numpy as np
import pandas as pd

from maritime_urban_tracking.sequences import convert_pose_to_H
from maritime_urban_tracking.timestamp_base import TimestampBase

class INS(TimestampBase):
    def __init__(self, path_to_folder) -> None:
        """
        INS: Inertial Navigation System. Includes GNSS and IMU. 
        """
        self.path_to_folder = path_to_folder

        filepath = f"{path_to_folder}/ins.csv"
        df = pd.read_csv(filepath, sep="\t")
        data_timestamps_ned_xyzw = df.to_numpy()
        timestamps = np.array(data_timestamps_ned_xyzw[:,0], dtype=int)
        self._set_timestamps(timestamps)
        self._data_ned_xyzw = data_timestamps_ned_xyzw[:,1:]

        self.fps = 100

    def get_pose(self):
        return self._data_ned_xyzw[self._index_timestamps_current]
    
    def get_H_points_piren_from_vessel(self):
        pose = self.get_pose()
        H_points_piren_from_vessel = convert_pose_to_H(pose)
        return H_points_piren_from_vessel
