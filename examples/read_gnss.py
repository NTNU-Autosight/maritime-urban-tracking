import matplotlib.pyplot as plt

from maritime_urban_tracking.gnss import GNSS
from maritime_urban_tracking.sequences import SEQUENCE_SINGLE_TARGET_CROSS


SEQUENCE = SEQUENCE_SINGLE_TARGET_CROSS
O3D_CONF_PATH = "./examples/o3d_conf.json"


def read_single_gnss():
    gnss = GNSS(SEQUENCE.paths_gnss["day cruiser front"])

    ns = []
    es = []
    while gnss.is_playable:
        timestamp = gnss.get_timestamp()
        pos = gnss.get_pos_ned()

        print(f"Timestamp: {timestamp}, pos: {pos}")

        ns.append(pos[0])
        es.append(pos[1])

        plt.ion()
        plt.figure("Position NE")
        plt.clf()
        plt.plot(es, ns)

        plt.pause(0.2)
        plt.show()

        gnss.move_n_frames(int(gnss.fps*0.2))

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    read_single_gnss()