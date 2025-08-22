import glob
import numpy as np
import cv2
import json
from skimage.color import lch2lab, lab2rgb
import torch
import os

from maritime_urban_tracking.timestamp_base import TimestampBase

class PolarizedRig(TimestampBase):
    def __init__(self, path_to_folder: str) -> None:
        """
        The rig is from the paper: A Lightweight, Polarization-Camera Equipped Sensor Rig for the Development of Autonomous Surface Vehicles
        The polarization is explained there. 

        Note: We need to rectify the images. 
        """
        super().__init__()
        self.path_to_folder = path_to_folder

        folderpath_cam_parameters = f'{"/".join(path_to_folder.split("/")[0:-1])}/parameters'
        K_left, D_left, K_right, D_right, t, R = get_cam_parameters(folderpath_cam_parameters)
        self.image_shape_wh = (1224, 1024)

        # R1 and R2 are the rotations that are made for each of the images. R_points_after_from_before
        # The projection matrices are for projection after the rectification. 
        R1, R2, self.left_P, self.right_P, self.Q, roi1, roi2 = cv2.stereoRectify(K_left, D_left, K_right, D_right, self.image_shape_wh, R, t, alpha=0) # alpha = 0 is zoomed to only valid pixels
        Tx = self.right_P[0,3]/self.right_P[0,0] # See the documentation at https://amroamroamro.github.io/mexopencv/matlab/cv.stereoRectify.html
        # Getting the parameters after rectification
        self.t = np.array([Tx, 0, 0])
        self.K_left = self.left_P[:3,:3]
        self.K_right = self.right_P[:3,:3]
        self.D_left = np.array([0,0,0,0,0], dtype=np.float32)[None,:] # This is the format that opencv wants. Shape (1,5)
        self.D_right = np.array([0,0,0,0,0], dtype=np.float32)[None,:]
        self.R = np.eye(3)
        self.map_left_x, self.map_left_y = cv2.initUndistortRectifyMap(K_left, D_left, R1, self.left_P, self.image_shape_wh, cv2.CV_32FC1)
        self.map_right_x, self.map_right_y = cv2.initUndistortRectifyMap(K_right, D_right, R2, self.right_P, self.image_shape_wh, cv2.CV_32FC1)

        filepaths = glob.glob(f"{path_to_folder}/*.json")

        timestamps = []
        self.map_timestamp_to_filename = {}
        for filepath in filepaths:
            with open(filepath) as file:
                data = json.load(file)
                assert data[0]["cam_serial"] == "220300007" # Ensure consistent order
                timestamp_0 = data[0]["timestamp_ns"]
                timestamp_1 = data[1]["timestamp_ns"]
                timestamp = int((timestamp_0+timestamp_1)/2)

                filename = filepath.split("/")[-1].split(".")[0]
                
                timestamps.append(timestamp)
                self.map_timestamp_to_filename[timestamp] = filename
        
        timestamps = np.array(sorted(timestamps))

        self._set_timestamps(timestamps)

        self.fps = 14
    
    def get_images(self, 
                   should_only_visual=True, 
                   should_use_learned=False
    ):
        """
            Get the images. Potentially 2 visual images and 2 polarized images.

            If Only visible: returns left and right image
            Else: returns left and right visual images, then left and right polarized images in rgb format. This is slow as finding the rgb values for the polarized images is slow in this implementation. 

            Learned means that the debayering / demosaicing is using learned weights from the paper: A Lightweight, Polarization-Camera Equipped Sensor Rig for the Development of Autonomous Surface Vehicles
            Otherwise, OpenCV is used for demosaicing also the polarized images. 

            Full res will potentially halucinate as it interpolates values. Only valid option when the learned weights are used. 
        """
        S0, S1, S2 = self._get_stokes(
            should_use_learned=should_use_learned
        )

        image_left_rgb_unrectified, image_right_rgb_unrectified = calc_visual_rgb_from_stokes(S0)
        image_left_rgb, image_right_rgb = self._rectify_images(image_left_rgb_unrectified, image_right_rgb_unrectified)

        if should_only_visual: return image_left_rgb, image_right_rgb

        image_polarized_left_rgb_unrectified, image_polarized_right_rgb_unrectified = calc_polarized_rgb_from_stokes(S0, S1, S2)
        image_polarized_left_rgb, image_polarized_right_rgb = self._rectify_images(image_polarized_left_rgb_unrectified, image_polarized_right_rgb_unrectified)

        return  image_left_rgb, image_right_rgb, image_polarized_left_rgb, image_polarized_right_rgb
        
    
    def _get_stokes(self, should_use_learned):
        images_bayer = self._get_images_bayer()

        if not should_use_learned:
            I0, I45, I90, I135 = calc_polarized_intensities_from_images_bayer_classic(images_bayer)
            S0, S1, S2 = calc_stokes_from_polarized_intensities_classic(I0, I45, I90, I135)
        else:
            S0, S1, S2 = calc_stokes_from_images_bayer_learned(images_bayer)
        
        return S0, S1, S2
    
    def _rectify_images(self, l_unrectified, r_unrectified):
        l = cv2.remap(l_unrectified, self.map_left_x, self.map_left_y, cv2.INTER_LINEAR)
        r = cv2.remap(r_unrectified, self.map_right_x, self.map_right_y, cv2.INTER_LINEAR)

        return l, r
    
    def _get_images_bayer(self):
        images_12bit_uint8 = self._get_12bit_uint8_images()

        width = 2448
        height = 2*2048

        # Ensure that the images has 3 uint8 values for each 12bit actual value
        assert len(images_12bit_uint8) == width * height * 3//2

        # The image is encoded to use 12bits for each cell, however it is stored to save data into
        # uint8, bytes. It is stored in the manner of 3 bytes corresponding to 2 cells, like so:
        # | 8bit | 4bit 4bit | 8bit |
        # | lsb  | msb  lsb  | msb  |
        # We have to transform this into
        # | 16bit      | 16bit      |
        # Though it takes more space, this can be used further in standard data handling and
        # visualization. 
        # Something similar is done here: https://stackoverflow.com/questions/70515648/reading-and-saving-12bit-raw-bayer-image-using-opencv-python
        # I am not sure why the msb of the middle byte is used for the third byte instead of the first byte. 
        # It kind of seems like the two words (4-bit groups) are switched in the middle (2nd) byte. 

        images_12bit = images_12bit_uint8.astype(np.uint16) # Needs to be 16 bit so that we can bitshift. 
        images_16bit = np.zeros(width * height, np.uint16)
        images_16bit[0::2] = images_12bit[0::3] + ((images_12bit[1::3]&(0b00001111)) << 8)
        images_16bit[1::2] = (images_12bit[1::3] >> 4) + (images_12bit[2::3] << 4)

        images_bayer = np.reshape(images_16bit, (height, width))
        return images_bayer
    
    def _get_12bit_uint8_images(self):
        timestamp = self.get_timestamp()
        filename = self.map_timestamp_to_filename[timestamp]

        with open(f"{self.path_to_folder}/{filename}.raw", "rb") as file:
            images_12bit_uint8 = np.fromfile(file, np.uint8, -1)
        
        return images_12bit_uint8

def calc_polarized_intensities_from_images_bayer_classic(images_bayer: np.ndarray):
    """
        Output is in rgb format, meaning it is demosaiced (interpolated) and has depth 3, one for each color channel. 
    """
    # Extract polarization channels (assuming Lucid's 2×2 block pattern)
    I90     = images_bayer[0::2, 0::2]  # 90°  (Vertical)
    I45     = images_bayer[0::2, 1::2]  # 45°  (Diagonal)
    I135    = images_bayer[1::2, 0::2]  # 135°  (Diagonal)
    I0      = images_bayer[1::2, 1::2]  # 0°   (Horizontal)

    I0_rgb = cv2.demosaicing(I0, cv2.COLOR_BayerRGGB2RGB) # cv2.COLOR_BayerBG2BGR == cv2.COLOR_BayerRGGB2BGR
    I45_rgb = cv2.demosaicing(I45, cv2.COLOR_BayerRGGB2RGB)
    I90_rgb = cv2.demosaicing(I90, cv2.COLOR_BayerRGGB2RGB)
    I135_rgb = cv2.demosaicing(I135, cv2.COLOR_BayerRGGB2RGB)

    return I0_rgb, I45_rgb, I90_rgb, I135_rgb

def calc_stokes_from_polarized_intensities_classic(I0_rgb: np.ndarray, I45_rgb: np.ndarray, I90_rgb: np.ndarray, I135_rgb: np.ndarray):
    I0, I45, I90, I135 = I0_rgb.astype(np.int64), I45_rgb.astype(np.int64), I90_rgb.astype(np.int64), I135_rgb.astype(np.int64) # Avoid under- and overflow

    # Compute Stokes parameters
    S0 = I0 + I90
    S1 = I0 - I90
    S2 = I45 - I135

    return S0, S1, S2

def calc_visual_rgb_from_stokes(S0_rgb: np.ndarray):
    """
    The input is S0 for each r,g,b
    It is in int64, but represents the summation of two uint12, meaning it can have value up to 2**13=8190
    """
    # Compute rgb images
    # images_rgb = np.round(S0_rgb.astype(float) * (255/8190)).astype(np.uint8) # Scaled to 0-255 using the most significant bits
    images_rgb = np.round(np.clip(S0_rgb.astype(float) * (255/8190)*2,0,255)).astype(np.uint8) # Scaled to 0-255 using less significant bits

    image_right_rgb = images_rgb[1024:,:,:]
    image_left_rgb = images_rgb[:1024,:,:]

    return image_left_rgb, image_right_rgb

def calc_polarized_rgb_from_stokes(S0_rgb, S1_rgb, S2_rgb):
    S1_squared = np.power(S1_rgb,2)
    S2_squared = np.power(S2_rgb,2)

    dolp = np.divide(np.sqrt(S1_squared+S2_squared),S0_rgb)
    dolp = np.mean(dolp, axis=2)
    dolp[dolp >1] = 1 # Put ceiling on maximum degree of polarization. Might be due to noise. Most should be between 0 and 1. 

    aolp = np.arctan2(S2_rgb,S1_rgb)/2
    aolp = np.mean(aolp, axis=2)

    images_polarized_rgb = convert_adolp_to_rgb(aolp, dolp) # This takes about 0.4 seconds!

    image_polarized_right_rgb = images_polarized_rgb[1024:,:,:]
    image_polarized_left_rgb = images_polarized_rgb[:1024,:,:]
    return image_polarized_left_rgb, image_polarized_right_rgb

def calc_stokes_from_images_bayer_learned(images_bayer):
    dirname = os.path.dirname(__file__)
    weights = torch.load(f"{dirname}/weights/weights_half_stokes.pt", weights_only=True, map_location="cpu")

    images_convolvable = transform_images_bayer_to_convolvable_torch(images_bayer)
    images_convolved = torch.nn.functional.conv2d(images_convolvable, weights, padding="same")
    images_stokes = transform_images_convolvable_to_bayer(images_convolved)

    N, M = images_stokes.shape[:2]
    S0, S1, S2 = images_stokes.reshape((N,M,3,3)).transpose((3,0,1,2))
    S0_rgb = (np.clip(S0, 0, 1)*8190).astype(np.int64) # uint12, but in an int64
    S1_rgb = (np.clip(S1, -0.5, 0.5)*8190).astype(np.int64)
    S2_rgb = (np.clip(S2, -0.5, 0.5)*8190).astype(np.int64)
    return S0_rgb, S1_rgb, S2_rgb

def transform_images_bayer_to_convolvable_torch(images_bayer: np.ndarray)->torch.Tensor:
    N, M = images_bayer.shape
    images_reshaped = images_bayer.reshape((N//4,2,2,M//4,2,2)).transpose((0,3,1,4,2,5)).reshape(N//4,M//4,16)
    images_convolvable = torch.from_numpy(images_reshaped.astype(np.float32)/(2**12)).unflatten(0,(2,-1)).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)
    return images_convolvable

def transform_images_convolvable_to_bayer(images_convolvable: torch.Tensor)->np.ndarray:
    N = images_convolvable.shape[2]*8
    M = images_convolvable.shape[3]*4
    assert images_convolvable.is_contiguous(memory_format=torch.channels_last)

    images_to_reshape = images_convolvable.permute(0, 2, 3, 1).to(memory_format=torch.contiguous_format).flatten(0,1).numpy()
    # images_to_reshape_scaled = np.clip(images_to_reshape*255, 0, 255).astype(np.uint16)
    C = images_to_reshape.shape[2]//4
    images_bayer = images_to_reshape.reshape((N//4,M//4,2,2,C)).transpose((0,2,1,3,4)).reshape((N//2,M//2,C))
    
    # C = images_to_reshape_scaled.shape[2]//16
    # images_bayer = images_to_reshape_scaled.reshape((N//4,M//4,2,2,2,2,C)).transpose((0,2,4,1,3,5,6)).reshape((N,M,C))
    return images_bayer

def convert_adolp_to_rgb(aolp: np.ndarray, dolp: np.ndarray):
    """
    AoLP: Angle of Linear Polarization. Bewteen -pi and pi
    DoLP: Degree of Linear Polarization. Between 0 and 1. 

    returns RGB based on LCH. See the resulting color map figure in the function create_colormap_figure(). 
    """
    luminescence = dolp*70 # 0-100
    chroma = dolp*40 # 0-100
    hue = np.mod(aolp, np.pi)*2 # 0-2pi

    lch = np.stack((luminescence, chroma, hue), axis=2)
    lab = lch2lab(lch)
    rgb = lab2rgb(lab)

    image_polarized_rgb = (rgb*255).astype(np.uint8)
    return image_polarized_rgb

def create_colormap_figure():
    size = 500

    ys, xs = np.mgrid[:2*size,:2*size] # mgrid[:HEIGHT, :WIDTH] returns 2 images: ys, xs. Can give list of pts (x,y) by np.mgrid[:HEIGHT,:WIDTH].T.reshape(-1,2)
    xs_rel = xs - size
    ys_rel = -(ys-2*size) - size
    rs = np.sqrt(np.power(ys_rel,2) + np.power(xs_rel,2))
    image_dolp = rs / size
    image_aolp = np.arctan2(ys_rel, xs_rel)

    image_dolp[rs>size] = 0

    image_rgb = convert_adolp_to_rgb(image_aolp, image_dolp)
    return image_rgb

def get_cam_parameters(folder_path):
    K_left = np.loadtxt(f"{folder_path}/K_l.csv", delimiter=",", skiprows=1)
    D_left = np.loadtxt(f"{folder_path}/dist_l.csv", delimiter=",", skiprows=1)
    K_right = np.loadtxt(f"{folder_path}/K_r.csv", delimiter=",", skiprows=1)
    D_right = np.loadtxt(f"{folder_path}/dist_r.csv", delimiter=",", skiprows=1)
    t = np.loadtxt(f"{folder_path}/T.csv", delimiter=",", skiprows=1)
    R = np.loadtxt(f"{folder_path}/R.csv", delimiter=",", skiprows=1)

    return K_left, D_left, K_right, D_right, t, R
