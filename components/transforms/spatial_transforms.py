import copy
import random
from abc import ABCMeta, abstractmethod
import cv2
import numpy as np

from scipy.ndimage import rotate, shift

from solt.utils import Serializable
from solt.constants import ALLOWED_INTERPOLATIONS, ALLOWED_PADDINGS
from solt.core._data import DataContainer, Keypoints
from solt.core._base_transforms import BaseTransform, InterpolationPropertyHolder, PaddingPropertyHolder
from solt.utils import ensure_valid_image, validate_parameter, validate_numeric_range_parameter
from solt.core import ImageTransform


class MatrixTransform(BaseTransform, InterpolationPropertyHolder, PaddingPropertyHolder):
    """Matrix Transform abstract class. (Affine and Homography).
    Does all the transforms around the image /  center.

    Parameters
    ----------
    interpolation : str
        Interpolation mode.
    padding : str or None
        Padding Mode.
    p : float
        Probability of transform's execution.
    ignore_state : bool
        Whether to ignore the pre-calculated transformation or not. If False,
        then it will lead to an incorrect behavior when the objects are of different sizes.
        Should be used only when it is assumed that the image, mask and keypoints are of
        the same size.

    """

    def __init__(
        self, interpolation="bilinear", padding="z", p=0.5, ignore_state=True, affine=True, ignore_fast_mode=False,
    ):
        BaseTransform.__init__(self, p=p, data_indices=None)
        InterpolationPropertyHolder.__init__(self, interpolation=interpolation)
        PaddingPropertyHolder.__init__(self, padding=padding)

        self.ignore_fast_mode = ignore_fast_mode
        self.fast_mode = False
        self.affine = affine
        self.ignore_state = ignore_state
        self.reset_state()

    def reset_state(self):
        BaseTransform.reset_state(self)
        self.state_dict["transform_matrix"] = np.eye(3)

    def fuse_with(self, trf):
        """
        Takes a transform an performs a matrix fusion. This is useful to optimize the computations

        Parameters
        ----------
        trf : MatrixTransform

        """

        if trf.padding is not None:
            self.padding = trf.padding
        self.interpolation = trf.interpolation

        self.state_dict["transform_matrix"] = trf.state_dict["transform_matrix"] @ self.state_dict["transform_matrix"]

    def sample_transform(self, data):
        """Samples the transform and corrects for frame change.

        Returns
        -------
        None

        """
        super(MatrixTransform, self).sample_transform(data)
        self.sample_transform_matrix(data)  # Only this method needs to be implemented!

        # If we are in fast mode, we do not have to recompute the the new coordinate frame!
        if "P" not in data.data_format and not self.ignore_fast_mode:
            width = self.state_dict["frame"][1]
            height = self.state_dict["frame"][0]
            origin = [(width - 1) // 2, (height - 1) // 2]
            # First, let's make sure that our transformation matrix is applied at the origin
            transform_matrix_corr = MatrixTransform.move_transform_to_origin(
                self.state_dict["transform_matrix"], origin
            )
            self.state_dict["frame_new"] = list(copy.copy(self.state_dict["frame"]))

            self.state_dict["transform_matrix_corrected"] = transform_matrix_corr
        else:
            # If we have the keypoints or the transform is a homographic one, we can't use the fast mode at all.
            self.correct_transform()

    @staticmethod
    def move_transform_to_origin(transform_matrix, origin):
        # First we correct the transformation so that it is performed around the origin
        transform_matrix = transform_matrix.copy()
        t_origin = np.array([1, 0, -origin[0], 0, 1, -origin[1], 0, 0, 1]).reshape((3, 3))

        t_origin_back = np.array([1, 0, origin[0], 0, 1, origin[1], 0, 0, 1]).reshape((3, 3))
        transform_matrix = np.dot(t_origin_back, np.dot(transform_matrix, t_origin))

        return transform_matrix

    @staticmethod
    def recompute_coordinate_frame(transform_matrix, width, height):
        coord_frame = np.array([[0, 0, 1], [0, height, 1], [width, height, 1], [width, 0, 1]])
        new_frame = np.dot(transform_matrix, coord_frame.T).T
        new_frame[:, 0] /= new_frame[:, -1]
        new_frame[:, 1] /= new_frame[:, -1]
        new_frame = new_frame[:, :-1]
        # Computing the new coordinates

        # If during the transform, we obtained negative coordinates, we have to move to the origin
        if np.any(new_frame[:, 0] < 0):
            new_frame[:, 0] += abs(new_frame[:, 0].min())
        if np.any(new_frame[:, 1] < 0):
            new_frame[:, 1] += abs(new_frame[:, 1].min())

        new_frame[:, 0] -= new_frame[:, 0].min()
        new_frame[:, 1] -= new_frame[:, 1].min()
        w_new = int(np.round(new_frame[:, 0].max()))
        h_new = int(np.round(new_frame[:, 1].max()))

        return h_new, w_new

    @staticmethod
    def correct_for_frame_change(transform_matrix: np.ndarray, width: int, height: int):
        """Method takes a matrix transform, and modifies its origin.

        Parameters
        ----------
        transform_matrix : numpy.ndarray
            Transform (3x3) matrix
        width : int
            Width of the coordinate frame
        height : int
            Height of the coordinate frame
        Returns
        -------
        out : numpy.ndarray
            Modified Transform matrix

        """
        origin = [(width - 1) // 2, (height - 1) // 2]
        # First, let's make sure that our transformation matrix is applied at the origin
        transform_matrix = MatrixTransform.move_transform_to_origin(transform_matrix, origin)
        # Now, if we think of scaling, rotation and translation, the image size gets increased
        # when we apply any geometric transform. Default behaviour in OpenCV is designed to crop the
        # image edges, however it is not desired when we want to deal with Keypoints (don't want them
        # to exceed teh image size).

        # If we imagine that the image edges are a rectangle, we can rotate it around the origin
        # to obtain the new coordinate frame
        h_new, w_new = MatrixTransform.recompute_coordinate_frame(transform_matrix, width, height)
        transform_matrix[0, -1] += w_new // 2 - origin[0]
        transform_matrix[1, -1] += h_new // 2 - origin[1]

        return transform_matrix, w_new, h_new

    @abstractmethod
    def sample_transform_matrix(self, data):
        """Method that is called to sample the transform matrix

        """

    def correct_transform(self):
        h, w = self.state_dict["frame"][:2]
        tm = self.state_dict["transform_matrix"]
        tm_corr, w_new, h_new = MatrixTransform.correct_for_frame_change(tm, w, h)
        self.state_dict["frame_new"] = [h_new, w_new]
        self.state_dict["transform_matrix_corrected"] = tm_corr

    def parse_settings(self, settings):
        interp = ALLOWED_INTERPOLATIONS[self.interpolation[0]]
        if settings["interpolation"][1] == "strict":
            interp = ALLOWED_INTERPOLATIONS[settings["interpolation"][0]]

        padding = ALLOWED_PADDINGS[self.padding[0]]
        if settings["padding"][1] == "strict":
            padding = ALLOWED_PADDINGS[settings["padding"][0]]

        return interp, padding

    def _apply_img_or_mask(self, img: np.ndarray, settings: dict):
        """Applies a transform to an image or mask without controlling the shapes.

        Parameters
        ----------
        img : numpy.ndarray
            Image or mask
        settings : dict
            Item-wise settings

        Returns
        -------
        out : numpy.ndarray
            Warped image

        """

        if self.affine:
            return self._apply_img_or_mask_affine(img, settings)
        else:
            return self._apply_img_or_mask_perspective(img, settings)

    def _apply_img_or_mask_perspective(self, img: np.ndarray, settings: dict):
        h_new = self.state_dict["frame_new"][0]
        w_new = self.state_dict["frame_new"][1]
        interp, padding = self.parse_settings(settings)
        transf_m = self.state_dict["transform_matrix_corrected"]
        return cv2.warpPerspective(img, transf_m, (w_new, h_new), flags=interp, borderMode=padding)

    def _apply_img_or_mask_affine(self, img: np.ndarray, settings: dict):
        h_new = self.state_dict["frame_new"][0]
        w_new = self.state_dict["frame_new"][1]
        interp, padding = self.parse_settings(settings)
        transf_m = self.state_dict["transform_matrix_corrected"]
        return cv2.warpAffine(img, transf_m[:2, :], (w_new, h_new), flags=interp, borderMode=padding)

    @ensure_valid_image(num_dims_spatial=(3,))
    def _apply_img(self, img: np.ndarray, settings: dict):
        """Applies a matrix transform to an image.
        If padding is None, the default behavior (zero padding) is expected.

        Parameters
        ----------
        img : numpy.ndarray
            Input Image
        settings : dict
            Item-wise settings

        Returns
        -------
        out : numpy.ndarray
            Output Image

        """
        return self._apply_img_or_mask(img, settings)

    def _apply_mask(self, mask: np.ndarray, settings: dict):
        """Abstract method, which defines the transform's behaviour when it is applied to masks HxW.

        If padding is None, the default behavior (zero padding) is expected.

        Parameters
        ----------
        mask : numpy.ndarray
            Mask to be augmented
        settings : dict
            Item-wise settings

        Returns
        -------
        out : numpy.ndarray
            Result

        """
        return self._apply_img_or_mask(mask, settings)

    def _apply_labels(self, labels, settings: dict):
        """Transform's application to labels. Simply returns them back without modifications.

        Parameters
        ----------
        labels : numpy.ndarray
            Array of labels.
        settings : dict
            Item-wise settings

        Returns
        -------
        out : numpy.ndarray
            Result

        """
        return labels

    def _apply_pts(self, pts: Keypoints, settings: dict):
        """Abstract method, which defines the transform's behaviour when it is applied to keypoints.

        Parameters
        ----------
        pts : Keypoints
            Keypoints object
        settings : dict
            Item-wise settings

        Returns
        -------
        out : Keypoints
            Result

        """
        if self.padding[0] == "r":
            raise ValueError("Cannot apply transform to keypoints with reflective padding!")

        pts_data = pts.data.copy()

        tm_corr = self.state_dict["transform_matrix_corrected"]

        pts_data = np.hstack((pts_data, np.ones((pts_data.shape[0], 1))))
        pts_data = np.dot(tm_corr, pts_data.T).T

        pts_data[:, 0] /= pts_data[:, 2]
        pts_data[:, 1] /= pts_data[:, 2]

        return Keypoints(pts_data[:, :-1], frame=self.state_dict["frame_new"])


class Rotate(ImageTransform):
    """Random rotation around the center clockwise

    Parameters
    ----------
    angle_range : tuple or float or None
        Range of rotation.
        If float, then (-angle_range, angle_range) will be used for transformation sampling.
        if None, then angle_range=(0,0).
    interpolation : str or tuple or None
        Interpolation type. Check the allowed interpolation types.
    padding : str or tuple or None
        Padding mode. Check the allowed padding modes.
    p : float
        Probability of using this transform
    ignore_state : bool
        Whether to ignore the state. See details in the docs for `MatrixTransform`.

    """

    _default_range = (0, 0)

    serializable_name = "rotate"
    """How the class should be stored in the registry"""

    def __init__(
        self, angle_range=None, p=0.5, data_indices=None, vol=False
    ):
        super(Rotate, self).__init__(p=p, data_indices=data_indices)

        if isinstance(angle_range, (int, float)):
            angle_range = (-angle_range, angle_range)

        self.angle_range = validate_numeric_range_parameter(angle_range, self._default_range)
        self.vol = vol

    def sample_transform(self, data: DataContainer):
        super(Rotate, self).sample_transform(data)
        self.state_dict['rot'] = random.uniform(self.angle_range[0], self.angle_range[1])
        self.state_dict['axes'] = random.sample([0, 1, 2], k=2)

    def _apply_img(self, img: np.ndarray, settings: dict):
        if self.vol:
            return rotate(img, angle=self.state_dict['rot'], axes=self.state_dict['axes'], reshape=False)
        else:
            return rotate(img, angle=self.state_dict['rot'], reshape=False)


class Translate(ImageTransform):
    """Random Translate transform

    Parameters
    ----------
    range_x: tuple or int or None
        Translation range along the horizontal axis. If int, then range_x=(-range_x, range_x).
        If None, then range_x=(0,0).
    range_y: tuple or int or None
        Translation range along the vertical axis. If int, then range_y=(-range_y, range_y).
        If None, then range_y=(0,0).
    p: float
        probability of applying this transform.
    """

    _default_range = (0, 0)

    serializable_name = "translate"
    """How the class should be stored in the registry"""

    def __init__(
                 self,
                 range_x=None,
                 range_y=None,
                 range_z=None,
                 p=0.5,
                 data_indices=None,
                 magnification=4,
    ):
        super(Translate, self).__init__(p=p, data_indices=data_indices)
        if isinstance(range_x, (int, float)):
            range_x = (min(range_x, -range_x), max(range_x, -range_x))

        if isinstance(range_y, (int, float)):
            range_y = (min(range_y, -range_y), max(range_y, -range_y))

        if isinstance(range_z, (int, float)):
            range_z = (min(range_z, -range_z), max(range_z, -range_z))

        self.range_x = validate_numeric_range_parameter(range_x, self._default_range)
        self.range_y = validate_numeric_range_parameter(range_y, self._default_range)

        # Range z only in 3D
        if range_z is not None:
            self.range_z = validate_numeric_range_parameter(range_z, self._default_range)
        else:
            self.range_z = None

        # Other attributes
        self.magnification = magnification
        self.frame_sml = None
        self.frame_lrg = None

    def sample_transform(self, data: DataContainer):
        super(Translate, self).sample_transform(data)

        self.frame_sml = data.data[0].shape[:-1]
        self.frame_lrg = data.data[1].shape[:-1]

        if self.range_z is not None:
            self.state_dict['translate'] = [random.uniform(self.range_x[0], self.range_x[1]),
                                            random.uniform(self.range_y[0], self.range_y[1]),
                                            random.uniform(self.range_z[0], self.range_z[1]),
                                            0]

            self.state_dict['translate_target'] = [self.state_dict['translate'][0] * self.magnification,
                                                   self.state_dict['translate'][1] * self.magnification,
                                                   self.state_dict['translate'][2] * self.magnification,
                                                   0]
        else:
            self.state_dict['translate'] = [random.uniform(self.range_x[0], self.range_x[1]),
                                            random.uniform(self.range_y[0], self.range_y[1]),
                                            0]

            self.state_dict['translate_target'] = [self.state_dict['translate'][0] * self.magnification,
                                                   self.state_dict['translate'][1] * self.magnification,
                                                   0]

    def _apply_img(self, img: np.ndarray, settings: dict):
        if img.shape[:-1] == self.frame_sml:
            return shift(img, shift=self.state_dict['translate'])
        elif img.shape[:-1] == self.frame_lrg:
            return shift(img, shift=self.state_dict['translate_target'])
        else:
            raise Exception('Image size mismatch!')
