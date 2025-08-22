# ps_geo =[]
#     for ts, long, lat, height in np.array(df_vals).T:
#         # The KB2 box requires handling of leap seconds. 
#         # See: https://stackoverflow.com/questions/33415475/how-to-get-current-date-and-time-from-gps-unsegment-time-in-python
#         ts_with_leap = ts + (-37+19) * 10**9
#         ps_geo.append([ts_with_leap, long, lat, height])
#     pts = np.array(ps_geo)
#     ps_enu = lon_lat_to_xy_enu(pts)
#     return ps_enu



import pandas as pd
import numpy as np
import pymap3d

from maritime_urban_tracking.sequences import PIREN_ALT, PIREN_LAT, PIREN_LON
from maritime_urban_tracking.timestamp_base import TimestampBase


class GNSS(TimestampBase):
    def __init__(self, path_to_gnss_file) -> None:
        """
        
        """
        df = pd.read_table(path_to_gnss_file, sep='\s+', header=24, parse_dates={'Timestamp': [0, 1]})
        df_vals = df["Timestamp"].astype(int).to_numpy(), df["longitude(deg)"].to_numpy(), df["latitude(deg)"].to_numpy(), df["height(m)"].to_numpy()
        pts_t_lon_lat_h = np.array(df_vals).T

        timestamps = pts_t_lon_lat_h[:,0]
        # The KB2 box requires handling of leap seconds. 
        # See: https://stackoverflow.com/questions/33415475/how-to-get-current-date-and-time-from-gps-unsegment-time-in-python
        timestamps_w_leap = timestamps + (-37+19) * 10**9

        self._set_timestamps(timestamps_w_leap)
        pts_lon_lat_h = pts_t_lon_lat_h[:,1:]
        self._pts_piren_ned = convert_lon_lat_h_to_piren_n_e_d(pts_lon_lat_h)

        self.fps = 5

    def get_pos_ned(self):
        return self._pts_piren_ned[self._index_timestamps_current]

def convert_lon_lat_h_to_piren_n_e_d(pts_lon_lat_h):
    """
    From Longitude, latitude and height 
    To North, east and down in meters, with respect to PIREN frame
    """
    ps = []
    for pt_llh in pts_lon_lat_h:
        lon, lat, h = pt_llh
        n, e, d = pymap3d.geodetic2ned(lat, lon, h, PIREN_LAT, PIREN_LON, PIREN_ALT)
        ps.append([n, e, d])
    return np.array(ps)
