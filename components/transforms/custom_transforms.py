import random

import cv2
import numpy as np
import scipy
import scipy.signal

from solt.core import (
    BaseTransform,
    ImageTransform,
    InterpolationPropertyHolder,
    MatrixTransform,
    PaddingPropertyHolder,
)
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