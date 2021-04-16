import random

import cv2
import numpy as np
import scipy
import scipy.signal
from skimage.filters import gaussian, median
from skimage.util import random_noise
from skimage.morphology import ball
from random import randint

from BoneEnhance.bone_enhance.utilities import print_orthogonal

from solt.core import (
    BaseTransform,
    ImageTransform,
    InterpolationPropertyHolder,
    MatrixTransform,
    PaddingPropertyHolder,
)
from solt.transforms import Rotate
from solt.constants import (
    ALLOWED_BLURS,
    ALLOWED_COLOR_CONVERSIONS,
    ALLOWED_CROPS,
    ALLOWED_INTERPOLATIONS,
    ALLOWED_GRIDMASK_MODES,
)
from solt.core import Stream
from solt.core import DataContainer, Keypoints
from solt.utils import (
    ensure_valid_image,
    validate_numeric_range_parameter,
    validate_parameter,
)


class Crop(BaseTransform):
    """Center / Random crop transform.

    Object performs center or random cropping depending on the parameters.

    Parameters
    ----------
    crop_to : tuple or int or None
        Size of the crop ``(height_new, width_new, ...)``. If ``int``, then a square crop will be made.
    crop_mode : str
        Crop mode. Can be either ``'c'`` - center or ``'r'`` - random.

    See also
    --------
    solt.constants.ALLOWED_CROPS

    """

    serializable_name = "crop"
    """How the class should be stored in the registry"""

    def __init__(self, magnification, crop_to=None, crop_mode="c"):
        super(Crop, self).__init__(p=1, data_indices=None)

        self.mag = magnification
        self.crop_to = crop_to
        self.crop_mode = crop_mode
        self.offsets_small_s = None
        self.offsets_small_e = None
        self.offsets_large_s = None
        self.offsets_large_e = None
        self.frame_small = None
        self.frame_large = None

    def sample_transform(self, data: DataContainer):
        if self.crop_to is not None:
            rand = random.random()
            self.frame_small = data.data[0].shape[:-1]
            self.frame_large = data.data[1].shape[:-1]

            ndim = len(self.frame_small)

            # Crop begin
            if self.crop_mode == "r":
                self.offsets_small_s = [int(rand * (self.frame_small[i] - self.crop_to[0][i])) for i in range(ndim)]
                self.offsets_large_s = [int(rand * (self.frame_large[i] - self.crop_to[1][i])) for i in range(ndim)]
            else:
                self.offsets_small_s = [(self.frame_small[i] - self.crop_to[0][i]) // 2 for i in range(ndim)]
                self.offsets_large_s = [(self.frame_large[i] - self.crop_to[1][i]) // 2 for i in range(ndim)]

            # Crop end
            self.offsets_small_e = [self.offsets_small_s[i] + self.crop_to[0][i] for i in range(ndim)]
            self.offsets_large_e = [self.offsets_large_s[i] + self.crop_to[1][i] for i in range(ndim)]

    def __crop_img_or_mask(self, img_mask):
        if self.crop_to is not None and img_mask.shape[:-1] == self.frame_small:
            ndim = len(self.offsets_small_s)
            sel = [slice(self.offsets_small_s[i], self.offsets_small_e[i]) for i in range(ndim)]
            sel = tuple(sel + [...,])
            return img_mask[sel]
        elif self.crop_to is not None and img_mask.shape[:-1] == self.frame_large:
            ndim = len(self.offsets_large_s)
            sel = [slice(self.offsets_large_s[i], self.offsets_large_e[i]) for i in range(ndim)]
            sel = tuple(sel + [..., ])
            return img_mask[sel]
        else:
            raise Exception('Image size mismatch')

    def _apply_img(self, img: np.ndarray, settings: dict):
        return self.__crop_img_or_mask(img)

    def _apply_mask(self, mask: np.ndarray, settings: dict):
        return self.__crop_img_or_mask(mask)

    def _apply_labels(self, labels, settings: dict):
        return labels

    def _apply_pts(self, pts: Keypoints, settings: dict):
        if self.crop_to is None:
            return pts
        pts_in = pts.data.copy()
        pts_out = np.empty_like(pts_in)

        for i in range(len(self.offsets_s)):
            pts_out[:, i] = pts_in[:, i] - self.offsets_s[i]

        return Keypoints(pts_out, frame=self.crop_to)


class Pad(BaseTransform, PaddingPropertyHolder):
    """Pads the input to a given size.

    Parameters
    ----------
    pad_to : tuple or int or None
        Target size ``(new_height, new_width, ...)``. Trailing channel dimension
        is kept unchanged and the corresponding padding must be excluded.
        The padding is computed using the following equations:

        ``pre_pad[k] = (pad_to[k] - shape_in[k]) // 2``
        ``post_pad[k] = pad_to[k] - shape_in[k] - pre_pad[k]``

    padding : str
        Padding type.

    See also
    --------
    solt.constants.allowed_paddings

    """

    serializable_name = "pad"
    """How the class should be stored in the registry"""

    def __init__(self, pad_to=None, padding=None):
        BaseTransform.__init__(self, p=1)
        PaddingPropertyHolder.__init__(self, padding)

        self.pad_to = pad_to
        self.offsets_sml_s = None
        self.offsets_sml_e = None
        self.offsets_lrg_s = None
        self.offsets_lrg_e = None
        self.frame_sml = None
        self.frame_lrg = None

    def sample_transform(self, data: DataContainer):
        if self.pad_to is not None:
            self.frame_sml = data.data[0].shape[:-1]
            self.frame_lrg = data.data[1].shape[:-1]

            ndim = len(self.frame_sml)

            self.offsets_sml_s = [(self.pad_to[0][i] - self.frame_sml[i]) // 2 for i in range(ndim)]
            self.offsets_lrg_s = [(self.pad_to[1][i] - self.frame_lrg[i]) // 2 for i in range(ndim)]

            self.offsets_sml_e = [self.pad_to[0][i] - self.frame_sml[i] - self.offsets_sml_s[i] for i in range(ndim)]
            self.offsets_lrg_e = [self.pad_to[1][i] - self.frame_lrg[i] - self.offsets_lrg_s[i] for i in range(ndim)]

            # If padding is negative, do not pad and do not raise the error
            for i in range(ndim):
                if self.offsets_sml_s[i] < 0:
                    self.offsets_sml_s[i] = 0
                if self.offsets_lrg_s[i] < 0:
                    self.offsets_lrg_s[i] = 0
                if self.offsets_sml_e[i] < 0:
                    self.offsets_sml_e[i] = 0
                if self.offsets_lrg_e[i] < 0:
                    self.offsets_lrg_e[i] = 0

    def _apply_img_or_mask(self, img_mask: np.ndarray, settings: dict):
        if img_mask.shape[:-1] == self.frame_sml:
            offsets_s = self.offsets_sml_s
            offsets_e = self.offsets_sml_e
        elif img_mask.shape[:-1] == self.frame_lrg:
            offsets_s = self.offsets_lrg_s
            offsets_e = self.offsets_lrg_e
        else:
            raise Exception('Image size mismatch!')

        if self.pad_to is not None:
            pad_width = [(s, e) for s, e in zip(offsets_s, offsets_e)]
            if img_mask.ndim > len(pad_width):
                pad_width = pad_width + [
                    (0, 0),
                ]

            if settings["padding"][1] == "strict":
                padding = settings["padding"][0]
            else:
                padding = self.padding[0]
            mode = {"z": "constant", "r": "reflect"}[padding]

            return np.pad(img_mask, pad_width=pad_width, mode=mode)
        else:
            return img_mask

    def _apply_img(self, img: np.ndarray, settings: dict):
        return self._apply_img_or_mask(img, settings)

    def _apply_mask(self, mask: np.ndarray, settings: dict):
        return self._apply_img_or_mask(mask, settings)

    def _apply_labels(self, labels, settings: dict):
        return labels

    def _apply_pts(self, pts: Keypoints, settings: dict):
        if self.pad_to is None:
            return pts
        if self.padding[0] != "z":
            raise ValueError
        pts_in = pts.data.copy()
        pts_out = np.empty_like(pts_in)
        ndim = len(self.offsets_s)

        for i in range(ndim):
            pts_out[:, i] = pts_in[:, i] + self.offsets_s[i]

        frame = [self.offsets_s[i] + pts.frame[i] + self.offsets_e[i] for i in range(ndim)]

        return Keypoints(pts_out, frame=frame)


class Brightness(ImageTransform):
    """Performs a random brightness augmentation

    Parameters
    ----------
    p : float
        Probability of applying this transform,
    brightness_range: tuple or None
        brightness_range shift range. If None, then ``brightness_range=(0, 0)``.
    data_indices : tuple or None
        Indices of the images within the data container to which this transform needs to be applied.
        Every element within the tuple must be integer numbers.
        If None, then the transform will be applied to all the images withing the DataContainer.

    """

    _default_range = (0, 0)

    serializable_name = "brightness"
    """How the class should be stored in the registry"""

    def __init__(self, brightness_range=None, data_indices=None, p=0.5):
        super(Brightness, self).__init__(p=p, data_indices=data_indices)
        self.brightness_range = validate_numeric_range_parameter(brightness_range, self._default_range)

    def sample_transform(self, data):
        brightness_fact = random.uniform(self.brightness_range[0], self.brightness_range[1])
        lut = np.arange(0, 256) + brightness_fact
        lut = np.clip(lut, 0, 255).astype("uint8")
        self.state_dict = {"brightness_fact": brightness_fact, "LUT": lut}

    @ensure_valid_image(num_dims_spatial=(3,))
    def _apply_img(self, img: np.ndarray, settings: dict):
        return cv2.LUT(img, self.state_dict["LUT"])


class Contrast(ImageTransform):
    """Transform randomly changes the contrast

    Parameters
    ----------
    p : float
        Probability of applying this transform,
    contrast_range : tuple or float or None
        Gain of the noise. Indicates percentage of indices, which will be changed.
        If float, then ``gain_range = (1-contrast_range, 1+contrast_range)``.
    data_indices : tuple or None
        Indices of the images within the data container to which this transform needs to be applied.
        Every element within the tuple must be integer numbers.
        If None, then the transform will be applied to all the images withing the DataContainer.

    """

    _default_range = (1, 1)
    serializable_name = "contrast"
    """How the class should be stored in the registry"""

    def __init__(self, p=0.5, contrast_range=0.1, data_indices=None):
        super(Contrast, self).__init__(p=p, data_indices=data_indices)

        if isinstance(contrast_range, float):
            contrast_range = (1 - contrast_range, 1 + contrast_range)

        self.contrast_range = validate_numeric_range_parameter(contrast_range, self._default_range, 0)

    def sample_transform(self, data):
        contrast_mul = random.uniform(self.contrast_range[0], self.contrast_range[1])
        lut = np.arange(0, 256) * contrast_mul
        lut = np.clip(lut, 0, 255).astype("uint8")
        self.state_dict = {"contrast_mul": contrast_mul, "LUT": lut}

    @ensure_valid_image(num_dims_spatial=(3,))
    def _apply_img(self, img: np.ndarray, settings: dict):
        return cv2.LUT(img, self.state_dict["LUT"])


class Blur(ImageTransform):
    """Transform blurs an image

    Parameters
    ----------
    p : float
        Probability of applying this transform,
    blur_type : str
        Blur type. See allowed blurs in `solt.constants`
    k_size: int or tuple
        Kernel sizes of the blur. if int, then sampled from ``(k_size, k_size)``. If tuple,
        then sampled from the whole tuple. All the values here must be odd.
    gaussian_sigma: int or float or tuple
        Gaussian sigma value. Used for both X and Y axes. If None, then gaussian_sigma=1.
    data_indices : tuple or None
        Indices of the images within the data container to which this transform needs to be applied.
        Every element within the tuple must be integer numbers.
        If None, then the transform will be applied to all the images withing the DataContainer.

    See also
    --------
    solt.constants.ALLOWED_BLURS

    """

    _default_range = (1, 1)

    serializable_name = "blur"
    """How the class should be stored in the registry"""

    def __init__(self, p=0.5, blur_type="g", k_size=3, gaussian_sigma=None, data_indices=None):
        super(Blur, self).__init__(p=p, data_indices=data_indices)
        if not isinstance(k_size, (int, tuple, list)):
            raise TypeError("Incorrect kernel size")

        if isinstance(k_size, list):
            k_size = tuple(k_size)

        if isinstance(k_size, int):
            k_size = (k_size, k_size)

        for k in k_size:
            if k % 2 == 0 or k < 1 or not isinstance(k, int):
                raise ValueError

        if isinstance(gaussian_sigma, (int, float)):
            gaussian_sigma = (gaussian_sigma, gaussian_sigma)

        self.blur = validate_parameter(blur_type, ALLOWED_BLURS, "g", basic_type=str, heritable=False)
        self.k_size = k_size
        self.gaussian_sigma = validate_numeric_range_parameter(gaussian_sigma, self._default_range, 0)

    def sample_transform(self, data):
        k = random.choice(self.k_size)
        s = random.uniform(self.gaussian_sigma[0], self.gaussian_sigma[1])
        self.state_dict = {"k_size": k, "sigma": s}

        if self.blur == "mo":
            if self.k_size[0] <= 2:
                raise ValueError("Lower bound for blur kernel size cannot be less than 2 for motion blur")

            kernel = np.zeros((k, k), dtype=np.uint8)
            xs, xe = random.randint(0, k - 1), random.randint(0, k - 1)

            if xs == xe:
                ys, ye = random.sample(range(k), 2)
            else:
                ys, ye = random.randint(0, k - 1), random.randint(0, k - 1)
            cv2.line(kernel, (xs, ys), (xe, ye), 1, thickness=1)
            kernel = kernel / np.sum(kernel)
            self.state_dict.update({"motion_kernel": kernel})

    @ensure_valid_image(num_dims_spatial=(3,))
    def _apply_img(self, img: np.ndarray, settings: dict):
        if self.blur == "g":
            return gaussian(img, sigma=self.state_dict["sigma"])
        if self.blur == "m":
            return median(img, selem=np.expand_dims(ball(self.state_dict["k_size"]), axis=3))

        if self.blur == "mo":
            return cv2.filter2D(img, -1, self.state_dict["motion_kernel"])


class Flip(BaseTransform):
    """Random Flipping transform.

    Parameters
    ----------
    p : float
        Probability of flip
    axis : int
        Flipping axis. 0 - vertical, 1 - horizontal, etc. -1 - all axes.
    """

    serializable_name = "flip"
    """How the class should be stored in the registry"""

    def __init__(self, p=0.5, axis=1, data_indices=None):
        super(Flip, self).__init__(p=p, data_indices=data_indices)
        if axis not in [-1, 0, 1]:
            raise ValueError("Incorrect Value of axis!")

        self.axis = axis
        self.orientation = None

    def sample_transform(self, data: DataContainer):
        self.orientation = randint(0, 6)

    @ensure_valid_image(num_dims_spatial=(3,))
    def _apply_img(self, img: np.ndarray, settings: dict):
        return self._flip(img)

    @ensure_valid_image(num_dims_total=(3,))
    def _apply_mask(self, mask: np.ndarray, settings: dict):
        return self._flip(mask)

    def _flip(self, img):
        if self.axis == 0:
            return np.ascontiguousarray(img[::-1, ...])
        elif self.axis == 1:
            return np.ascontiguousarray(img[:, ::-1, ...])
        elif self.axis == 2:
            return np.ascontiguousarray(img[:, :, ::-1, ...])
        else:
            orientation = self.orientation
            if orientation == 0:
                return np.ascontiguousarray(img[::-1, ...])
            elif orientation == 1:
                return np.ascontiguousarray(img[:, ::-1, ...])
            elif orientation == 2:
                return np.ascontiguousarray(img[:, :, ::-1, ...])
            elif orientation == 3:
                return np.ascontiguousarray(img[::-1, ::-1, ...])
            elif orientation == 4:
                return np.ascontiguousarray(img[:, ::-1, ::-1, ...])
            elif orientation == 5:
                return np.ascontiguousarray(img[::-1, :, ::-1, ...])
            else:
                return np.ascontiguousarray(img[::-1, ::-1, ::-1, ...])

    def _apply_labels(self, labels, settings: dict):
        return labels

    def _apply_pts(self, pts: Keypoints, settings: dict):
        # We should guarantee that we do not change the original data.
        pts_data = pts.data.copy()
        if self.axis == 0:
            pts_data[:, 1] = pts.frame[0] - 1 - pts_data[:, 1]
        elif self.axis == 1:
            pts_data[:, 0] = pts.frame[1] - 1 - pts_data[:, 0]
        elif self.axis == -1:
            pts_data[:, 1] = pts.frame[0] - 1 - pts_data[:, 1]
            pts_data[:, 0] = pts.frame[1] - 1 - pts_data[:, 0]

        return Keypoints(pts=pts_data, frame=pts.frame)


class Rotate90(Rotate):
    """Random rotation around the center by 90 degrees.

    Parameters
    ----------
    k : int
        How many times to rotate the data. If positive, indicates the clockwise direction.
        Zero by default.
    p : float
        Probability of using this transform

    """

    serializable_name = "rotate_90"
    """How the class should be stored in the registry"""

    def __init__(self, k=0, p=0.5, ignore_fast_mode=False):
        if not isinstance(k, int):
            raise TypeError("Argument `k` must be an integer!")
        super(Rotate90, self).__init__(p=p, angle_range=(k * 90, k * 90), ignore_fast_mode=ignore_fast_mode)
        self.k = k
        self.axis = None

    def sample_transform(self, data: DataContainer):
        self.axis = randint(0, 2)

    @ensure_valid_image(num_dims_spatial=(3,))
    def _apply_img(self, img: np.ndarray, settings: dict):
        # Rotate to random axis along 3D
        axis = self.axis
        if axis != 2:
            return np.ascontiguousarray(np.rot90(img, -self.k, axes=(axis, axis + 1)))
        else:
            return np.ascontiguousarray(np.rot90(img, -self.k, axes=(0, axis)))

    @ensure_valid_image(num_dims_total=(3,))
    def _apply_mask(self, mask: np.ndarray, settings: dict):
        axis = self.axis
        if axis != 2:
            return np.ascontiguousarray(np.rot90(mask -self.k, axes=(axis, axis + 1)))
        else:
            return np.ascontiguousarray(np.rot90(mask, -self.k, axes=(0, axis)))


class Noise(ImageTransform):
    """Adds noise to an image. Other types of data than the image are ignored.

    Parameters
    ----------
    p : float
        Probability of applying this transform,
    gain_range : tuple or float or None
        Gain of the noise. Final image is created as ``(1-gain)*img + gain*noise``.
        If float, then ``gain_range = (0, gain_range)``. If None, then ``gain_range=(0, 0)``.
    data_indices : tuple or None
        Indices of the images within the data container to which this transform needs to be applied.
        Every element within the tuple must be integer numbers.
        If None, then the transform will be applied to all the images withing the DataContainer.

    """

    _default_range = (0, 0)

    serializable_name = "noise"
    """How the class should be stored in the registry"""

    def __init__(self, p=0.5, gain_range=0.1, data_indices=None, mode='gaussian'):
        super(Noise, self).__init__(p=p, data_indices=data_indices)
        if isinstance(gain_range, float):
            gain_range = (0, gain_range)

        self.type = mode
        self.gain_range = validate_numeric_range_parameter(gain_range, self._default_range, min_val=0, max_val=1)

    def sample_transform(self, data: DataContainer):
        super(Noise, self).sample_transform(data)
        gain = random.uniform(self.gain_range[0], self.gain_range[1])
        self.state_dict = {"gain": gain}

    @ensure_valid_image(num_dims_spatial=(3,))
    def _apply_img(self, img: np.ndarray, settings: dict):
        if self.type == 'gaussian' or self.type == 'speckle':
            return random_noise(img, mode=self.type, var=self.state_dict['gain']) * 255
        elif self.type == 's&p':
            return random_noise(img, mode=self.type, amount=self.state_dict['gain']) * 255
        else:
            return random_noise(img, mode=self.type) * 255


