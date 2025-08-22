import matplotlib.pyplot as plt
import numpy as np

from maritime_urban_tracking.sequences import SEQUENCE_SINGLE_TARGET_STEREO_CALIBRATION_1
from maritime_urban_tracking.polarized_rig import PolarizedRig, create_colormap_figure


SEQUENCE = SEQUENCE_SINGLE_TARGET_STEREO_CALIBRATION_1


def read_polarized_rig():
    cam = PolarizedRig(SEQUENCE.path_polarized_rig)
    cam.set_timestamp(1721208033499757568)

    plt.figure("Colormap")
    plt.clf()
    image_colormap_rgb = create_colormap_figure()
    fig = plt.gcf()
    fig.set_figwidth(21/6)
    fig.set_figheight(21/6)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    plt.imshow(image_colormap_rgb)
    plt.axis("off")

    while cam.is_playable:
        timestamp = cam.get_timestamp()
        print(f"Timestamp: {timestamp}")

        image_left, image_right, image_p_left, image_p_right = cam.get_images(should_only_visual=False, should_use_learned=False)

        plt.ion()

        plt.figure("Images")
        plt.clf()
        fig = plt.gcf()
        fig.set_figwidth(86/6)
        fig.set_figheight(70/6)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        plt.subplot(2,2,1)
        plt.imshow(image_left)
        plt.axis("off")
        plt.subplot(2,2,2)
        plt.imshow(image_right)
        plt.axis("off")
        plt.subplot(2,2,3)
        plt.imshow(image_p_left)
        plt.axis("off")
        plt.subplot(2,2,4)
        plt.imshow(image_p_right)
        plt.axis("off")

        plt.figure("Calibration check")
        plt.clf()
        images_combined = np.hstack((image_left, image_right))
        fig = plt.gcf()
        fig.set_figwidth(86/6)
        fig.set_figheight(35/6)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        plt.imshow(images_combined)
        plt.hlines([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], xmin = 0, xmax = 1224*2)
        plt.axis("off")

        plt.pause(0.01)
        plt.show()

        cam.move_n_frames(14)

if __name__ == "__main__":
    read_polarized_rig()
