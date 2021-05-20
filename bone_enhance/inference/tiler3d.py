import numpy as np
import torch
import cv2
from typing import List
import math
from scipy.ndimage import gaussian_filter


class Tiler3D:
    def __init__(self, image_shape, tile, step, out, mag=4, weight='mean'):

        tile_out = tuple([s * mag for s in tile])
        dim = len(tile)

        self.input = image_shape
        self.mag = mag
        self.out = out
        self.dim = dim
        self.tile = tile
        self.tile = np.min((tile, self.input[:-1]), axis=0)  # Remove channel dimension
        self.step = tuple([s // step for s in tile])
        self.tile_out = np.min((tile_out, out), axis=0)
        self.step_out = tuple([s // step for s in tile_out])

        overlap = [(self.tile[x] - self.step[x]) for x in range(dim)]

        if weight == 'mean':
            self.weight = self._mean(self.tile_out)
        elif weight == 'gaussian':
            self.weight = self._gaussian(self.tile_out, step)
        else:
            raise Exception('Weight not implemented!')

        # Margins
        n_tiles = [max(1, math.ceil((self.input[x] - overlap[x]) / self.step[x])) for x in range(dim)]
        residuals = [self.step[x] * n_tiles[x] - (self.input[x] - overlap[x]) for x in range(dim)]

        self.margin_begin = [residuals[x] // 2 for x in range(dim)]
        self.margin_end = [residuals[x] - self.margin_begin[x] for x in range(dim)]
        self.margin_begin_out = [(residuals[x] * mag) // 2 for x in range(dim)]
        self.margin_end_out = [residuals[x] * mag - self.margin_begin_out[x] for x in range(dim)]

        crops = []
        crops_out = []
        bbox_crops = []

        # 3D tiling
        if self.dim == 3:
            for x in range(
                    0, self.input[0] + self.margin_begin[0] + self.margin_end[0] - self.tile[0] + 1, self.step[0]
            ):
                for y in range(
                        0, self.input[1] + self.margin_begin[1] + self.margin_end[1] - self.tile[1] + 1, self.step[1]
                ):
                    for z in range(
                            0, self.input[2] + self.margin_begin[2] + self.margin_end[2] - self.tile[2] + 1, self.step[2]
                    ):
                        crops.append((x, y, z, self.tile[0], self.tile[1], self.tile[2]))
                        bbox_crops.append((x - self.margin_begin[0], y - self.margin_begin[1], z - self.margin_begin[2],
                                           self.tile[0], self.tile[1], self.tile[2]))

                        crops_out.append((x * mag, y * mag, z * mag, self.tile_out[0], self.tile_out[1], self.tile_out[2]))
        else:
            for x in range(
                    0, self.input[0] + self.margin_begin[0] + self.margin_end[0] - self.tile[0] + 1, self.step[0]
            ):
                for y in range(
                        0, self.input[1] + self.margin_begin[1] + self.margin_end[1] - self.tile[1] + 1, self.step[1]
                ):
                    crops.append((x, y, self.tile[0], self.tile[1]))
                    bbox_crops.append((x - self.margin_begin[0], y - self.margin_begin[1],
                                       self.tile[0], self.tile[1]))

                    crops_out.append(
                        (x * mag, y * mag, self.tile_out[0], self.tile_out[1]))


        self.crops = np.array(crops)
        self.crops_out = np.array(crops_out)
        self.bbox_crops = np.array(bbox_crops)

    def split(self, image):

        # Pad image to correct size
        size = tuple([image.shape[x] + self.margin_begin[x] + self.margin_end[x] for x in range(self.dim)])
        self.image_pad = size
        image_pad = np.zeros(size + (image.shape[-1],))
        #image_pad[self.margin_begin:image.shape[:-1] + self.margin_begin] = image
        if self.dim == 3:
            image_pad[self.margin_begin[0]:image.shape[0] + self.margin_begin[0],
                      self.margin_begin[1]:image.shape[1] + self.margin_begin[1],
                      self.margin_begin[2]:image.shape[2] + self.margin_begin[2], :] = image
        else:
            image_pad[self.margin_begin[0]:image.shape[0] + self.margin_begin[0],
                      self.margin_begin[1]:image.shape[1] + self.margin_begin[1], :] = image
        image = image_pad

        tiles = []
        if self.dim == 3:
            for x, y, z, tile_x, tile_y, tile_z in self.crops:
                tile = image[x: x + tile_x, y: y + tile_y, z: z + tile_z].copy()
                if tile.shape[0] != self.tile[0]:
                    print(tile.shape[0])
                assert tile.shape[1] == self.tile[1]
                assert tile.shape[2] == self.tile[2]

                tiles.append(tile)
        else:
            for x, y, tile_x, tile_y in self.crops:
                tile = image[x: x + tile_x, y: y + tile_y].copy()
                if tile.shape[0] != self.tile[0]:
                    print(tile.shape[0])
                assert tile.shape[1] == self.tile[1]

                tiles.append(tile)
        return tiles

    def crop_to_orignal_size(self, image):
        """
        Crop the predicted image based on the padded margins.
        """
        cr = self.margin_begin_out
        if self.dim == 3:
            crop = image[
                   cr[0]:self.out[0] + cr[0],
                   cr[1]:self.out[1] + cr[1],
                   cr[2]:self.out[2] + cr[2],
                   ]
        else:
            crop = image[
                   cr[0]:self.out[0] + cr[0],
                   cr[1]:self.out[1] + cr[1],
                   ]
        return crop

    def merge(self, tiles: List[np.ndarray], channels, dtype=np.float16):
        if len(tiles) != len(self.crops):
            raise ValueError

        target_shape = self.target_shape

        image = np.zeros((channels, target_shape[0], target_shape[1], target_shape[2]), dtype=dtype)
        norm_mask = np.zeros((1, target_shape[0], target_shape[1], target_shape[2]), dtype=dtype)

        #w = np.dstack([self.weight] * channels)
        w = np.expand_dims(self.weight, axis=0)

        for tile, (x, y, z, tile_x, tile_y, tile_z) in zip(tiles, self.crops_out):
            image[:, x: x + tile_x, y: y + tile_y, z: z + tile_z] += tile * w
            norm_mask[:, x: x + tile_x, y: y + tile_y, z: z + tile_z] += w

        # Ensure no division by 0
        norm_mask = np.clip(norm_mask, a_min=np.finfo(norm_mask.dtype).eps, a_max=None)
        # Account for overlapping regions
        image = np.divide(image, norm_mask).astype(dtype)
        image = np.moveaxis(image, 0, -1).astype(dtype)
        image = self.crop_to_orignal_size(image)
        return image

    @property
    def target_shape(self):
        target_shape = tuple([self.image_pad[x] * self.mag + self.margin_begin[x] + self.margin_end[x] for x in range(self.dim)])
        return target_shape

    def _mean(self, tile_size):
        return np.ones(tile_size, dtype=np.float32)

    def _gaussian(self, tile_size, step):
        m = np.zeros(tile_size, dtype=np.float32)
        dim = np.array(m.shape) // 2
        if len(dim) == 3:
            m[dim[0], dim[1], dim[2]] = 1
        else:
            m[dim[0], dim[1]] = 1
        return gaussian_filter(m, sigma=dim[0] // step)


class TileMerger3D:
    """
    Helper class to merge final image on GPU. This generally faster than moving individual tiles to CPU.
    """

    def __init__(self, image_shape, channels, weight, cuda=True):
        """

        :param image_shape: Shape of the source image
        :param image_margin:
        :param weight: Weighting matrix
        """

        self.weight = torch.from_numpy(np.expand_dims(weight, axis=0)).float()
        if len(image_shape) == 3:
            self.image = torch.zeros((channels, image_shape[0], image_shape[1], image_shape[2]), dtype=torch.float32)
            self.norm_mask = torch.zeros((1, image_shape[0], image_shape[1], image_shape[2]), dtype=torch.float32)
        else:
            self.image = torch.zeros((channels, image_shape[0], image_shape[1]), dtype=torch.float32)
            self.norm_mask = torch.zeros((1, image_shape[0], image_shape[1]), dtype=torch.float32)
        if cuda:
            self.weight = self.weight.cuda()
            self.image = self.image.cuda()
            self.norm_mask = self.norm_mask.cuda()

    def integrate_batch(self, batch: torch.Tensor, crop_coords):
        """
        Accumulates batch of tile predictions
        :param batch: Predicted tiles
        :param crop_coords: Corresponding tile crops w.r.t to original image
        """
        if len(batch) != len(crop_coords):
            raise ValueError("Number of images in batch does not correspond to number of coordinates")

        # 3D merge
        if len(batch.size()) == 5:
            for tile, (x, y, z, tile_x, tile_y, tile_z) in zip(batch, crop_coords):
                self.image[:, x: x + tile_x, y: y + tile_y, z: z + tile_z] += tile * self.weight
                self.norm_mask[:, x: x + tile_x, y: y + tile_y, z: z + tile_z] += self.weight
        # 2D merge
        else:
            for tile, (x, y, tile_x, tile_y) in zip(batch, crop_coords):
                self.image[:, x: x + tile_x, y: y + tile_y] += tile * self.weight
                self.norm_mask[:, x: x + tile_x, y: y + tile_y] += self.weight

    def merge(self) -> torch.Tensor:
        return self.image / self.norm_mask


class ImageSlicer:
    """
    Helper class to slice image into tiles and merge them back. Modified from Pytorch-toolbelt to include a
    gaussian window.
    """

    def __init__(self, image_shape, tile_size, tile_step=0, image_margin=0, weight="mean"):
        """

        :param image_shape: Shape of the source image (H, W)
        :param tile_size: Tile size (Scalar or tuple (H, W)
        :param tile_step: Step in pixels between tiles (Scalar or tuple (H, W))
        :param image_margin:
        :param weight: Fusion algorithm. 'mean' - avergaing
        """
        self.image_height = image_shape[0]
        self.image_width = image_shape[1]

        if isinstance(tile_size, (tuple, list)):
            assert len(tile_size) == 2
            self.tile_size = int(tile_size[0]), int(tile_size[1])
        else:
            self.tile_size = int(tile_size), int(tile_size)

        if isinstance(tile_step, (tuple, list)):
            assert len(tile_step) == 2
            self.tile_step = int(tile_step[0]), int(tile_step[1])
        else:
            self.tile_step = int(tile_step), int(tile_step)

        weights = {"mean": self._mean, "pyramid": self._pyramid, "gaussian": self._gaussian}

        if weight == 'gaussian':
            self.weight = weights[weight](self.tile_size, tile_size[0] // tile_step[0])
        else:
            self.weight = weight if isinstance(weight, np.ndarray) else weights[weight](self.tile_size)

        if self.tile_step[0] < 1 or self.tile_step[0] > self.tile_size[0]:
            raise ValueError()
        if self.tile_step[1] < 1 or self.tile_step[1] > self.tile_size[1]:
            raise ValueError()

        overlap = [self.tile_size[0] - self.tile_step[0], self.tile_size[1] - self.tile_step[1]]

        self.margin_left = 0
        self.margin_right = 0
        self.margin_top = 0
        self.margin_bottom = 0

        if image_margin == 0:
            # In case margin is not set, we compute it manually

            nw = max(1, math.ceil((self.image_width - overlap[1]) / self.tile_step[1]))
            nh = max(1, math.ceil((self.image_height - overlap[0]) / self.tile_step[0]))

            extra_w = self.tile_step[1] * nw - (self.image_width - overlap[1])
            extra_h = self.tile_step[0] * nh - (self.image_height - overlap[0])

            self.margin_left = extra_w // 2
            self.margin_right = extra_w - self.margin_left
            self.margin_top = extra_h // 2
            self.margin_bottom = extra_h - self.margin_top

        else:
            if (self.image_width - overlap[1] + 2 * image_margin) % self.tile_step[1] != 0:
                raise ValueError()

            if (self.image_height - overlap[0] + 2 * image_margin) % self.tile_step[0] != 0:
                raise ValueError()

            self.margin_left = image_margin
            self.margin_right = image_margin
            self.margin_top = image_margin
            self.margin_bottom = image_margin

        crops = []
        bbox_crops = []

        for y in range(
            0, self.image_height + self.margin_top + self.margin_bottom - self.tile_size[0] + 1, self.tile_step[0]
        ):
            for x in range(
                0, self.image_width + self.margin_left + self.margin_right - self.tile_size[1] + 1, self.tile_step[1]
            ):
                crops.append((x, y, self.tile_size[1], self.tile_size[0]))
                bbox_crops.append((x - self.margin_left, y - self.margin_top, self.tile_size[1], self.tile_size[0]))

        self.crops = np.array(crops)
        self.bbox_crops = np.array(bbox_crops)

    def split(self, image, border_type=cv2.BORDER_CONSTANT, value=0):
        assert image.shape[0] == self.image_height
        assert image.shape[1] == self.image_width

        orig_shape_len = len(image.shape)
        image = cv2.copyMakeBorder(
            image,
            self.margin_top,
            self.margin_bottom,
            self.margin_left,
            self.margin_right,
            borderType=border_type,
            value=value,
        )

        # This check recovers possible lack of last dummy dimension for single-channel images
        if len(image.shape) != orig_shape_len:
            image = np.expand_dims(image, axis=-1)

        tiles = []
        for x, y, tile_width, tile_height in self.crops:
            tile = image[y : y + tile_height, x : x + tile_width].copy()
            assert tile.shape[0] == self.tile_size[0]
            assert tile.shape[1] == self.tile_size[1]

            tiles.append(tile)

        return tiles

    def cut_patch(self, image: np.ndarray, slice_index, border_type=cv2.BORDER_CONSTANT, value=0):
        assert image.shape[0] == self.image_height
        assert image.shape[1] == self.image_width

        orig_shape_len = len(image.shape)
        image = cv2.copyMakeBorder(
            image,
            self.margin_top,
            self.margin_bottom,
            self.margin_left,
            self.margin_right,
            borderType=border_type,
            value=value,
        )

        # This check recovers possible lack of last dummy dimension for single-channel images
        if len(image.shape) != orig_shape_len:
            image = np.expand_dims(image, axis=-1)

        x, y, tile_width, tile_height = self.crops[slice_index]

        tile = image[y : y + tile_height, x : x + tile_width].copy()
        assert tile.shape[0] == self.tile_size[0]
        assert tile.shape[1] == self.tile_size[1]
        return tile

    @property
    def target_shape(self):
        target_shape = (
            self.image_height + self.margin_bottom + self.margin_top,
            self.image_width + self.margin_right + self.margin_left,
        )
        return target_shape

    def merge(self, tiles: List[np.ndarray], dtype=np.float32):
        if len(tiles) != len(self.crops):
            raise ValueError

        channels = 1 if len(tiles[0].shape) == 2 else tiles[0].shape[2]
        target_shape = (
            self.image_height + self.margin_bottom + self.margin_top,
            self.image_width + self.margin_right + self.margin_left,
            channels,
        )

        image = np.zeros(target_shape, dtype=np.float64)
        norm_mask = np.zeros(target_shape, dtype=np.float64)

        w = np.dstack([self.weight] * channels)

        for tile, (x, y, tile_width, tile_height) in zip(tiles, self.crops):
            # print(x, y, tile_width, tile_height, image.shape)
            image[y : y + tile_height, x : x + tile_width] += tile * w
            norm_mask[y : y + tile_height, x : x + tile_width] += w

        # print(norm_mask.min(), norm_mask.max())
        norm_mask = np.clip(norm_mask, a_min=np.finfo(norm_mask.dtype).eps, a_max=None)
        normalized = np.divide(image, norm_mask).astype(dtype)
        crop = self.crop_to_orignal_size(normalized)
        return crop

    def crop_to_orignal_size(self, image):
        assert image.shape[0] == self.target_shape[0]
        assert image.shape[1] == self.target_shape[1]
        crop = image[
            self.margin_top : self.image_height + self.margin_top,
            self.margin_left : self.image_width + self.margin_left,
        ]
        assert crop.shape[0] == self.image_height
        assert crop.shape[1] == self.image_width
        return crop

    def _mean(self, tile_size):
        return np.ones((tile_size[0], tile_size[1]), dtype=np.float32)

    def _pyramid(self, tile_size):
        w, _, _ = compute_pyramid_patch_weight_loss(tile_size[0], tile_size[1])
        return w

    def _gaussian(self, tile_size, step):
        m = np.zeros(tile_size, dtype=np.float32)
        dim = np.array(m.shape) // 2
        m[dim[0], dim[1]] = 1
        m = gaussian_filter(m, sigma=dim[0] // step)
        return m / np.max(m)


def compute_pyramid_patch_weight_loss(width: int, height: int) -> np.ndarray:
    """Compute a weight matrix that assigns bigger weight on pixels in center and
    less weight to pixels on image boundary.
    This weight matrix then used for merging individual tile predictions and helps dealing
    with prediction artifacts on tile boundaries.

    :param width: Tile width
    :param height: Tile height
    :return: Since-channel image [Width x Height]
    """
    xc = width * 0.5
    yc = height * 0.5
    xl = 0
    xr = width
    yb = 0
    yt = height
    Dc = np.zeros((width, height))
    De = np.zeros((width, height))

    Dcx = np.square(np.arange(width) - xc + 0.5)
    Dcy = np.square(np.arange(height) - yc + 0.5)
    Dc = np.sqrt(Dcx[np.newaxis].transpose() + Dcy)

    De_l = np.square(np.arange(width) - xl + 0.5) + np.square(0.5)
    De_r = np.square(np.arange(width) - xr + 0.5) + np.square(0.5)
    De_b = np.square(0.5) + np.square(np.arange(height) - yb + 0.5)
    De_t = np.square(0.5) + np.square(np.arange(height) - yt + 0.5)

    De_x = np.sqrt(np.minimum(De_l, De_r))
    De_y = np.sqrt(np.minimum(De_b, De_t))
    De = np.minimum(De_x[np.newaxis].transpose(), De_y)

    alpha = (width * height) / np.sum(np.divide(De, np.add(Dc, De)))
    W = alpha * np.divide(De, np.add(Dc, De))
    return W, Dc, De
