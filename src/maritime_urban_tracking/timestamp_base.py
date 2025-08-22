import numpy as np

class TimestampBase:
    def __init__(self) -> None:
        self._timestamps = np.array([])

        self._index_timestamps_current = 0
        self.is_playable = False
    
    def _set_timestamps(self, timestamps):
        timestamps = np.array(sorted(timestamps))
        self._timestamps = timestamps
        if len(timestamps) > 0:
            self.is_playable = True
            self._index_timestamps_current = 0
            return True
        else:
            self.is_playable = False
            self._index_timestamps_current = 0
            return False
    
    def get_timestamp(self):
        return self._timestamps[self._index_timestamps_current]
    
    @property
    def length(self):
        return len(self._timestamps)
    
    def set_timestamp(self, timestamp):
        i = np.searchsorted(self._timestamps, timestamp, side="left")

        if i == 0 and timestamp < self._timestamps[0]:
            self.is_playable = False
            return False
        elif i == 0:
            self._index_timestamps_current = i
            self.is_playable = True
            return True
        elif i == len(self._timestamps):
            self.is_playable = False
            return False
        else:
            self.is_playable = True
            if np.abs(timestamp - self._timestamps[i]) < np.abs(timestamp - self._timestamps[i-1]):
                self._index_timestamps_current = i
            else:
                self._index_timestamps_current = i-1
            return True

    def move_n_frames(self, n_frames: int):
        frame_number = self._index_timestamps_current + n_frames
        return self._set_timestamp_frame_number(frame_number)
    
    def _set_timestamp_frame_number(self, frame_number):
        if self.length > 0 and  0 <= frame_number < self.length:
            self._index_timestamps_current = frame_number
            self.is_playable = True
            return True
        else:
            self.is_playable = False
            return False
