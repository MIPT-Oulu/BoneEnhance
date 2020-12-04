import numpy as np
import torch
import cv2
from typing import List
import math


class Tiler3D:
    def __init__(self, image_shape, tile, step, out, mag=4):

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

        self.weight = self._mean(self.tile_out)

        # Margins
        n_tiles = [max(1, math.ceil((self.input[x] - overlap[x]) / self.step[x])) for x in range(dim)]
        residuals = [self.step[x] * n_tiles[x] - (self.input[x] - overlap[x]) for x in range(dim)]

        self.margin_begin = [residuals[x] // 2 for x in range(dim)]
        self.margin_end = [residuals[x] - self.margin_begin[x] for x in range(dim)]

        crops = []
        crops_out = []
        bbox_crops = []
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

        self.crops = np.array(crops)
        self.crops_out = np.array(crops_out)
        self.bbox_crops = np.array(bbox_crops)

    def split(self, image):

        # Pad image to correct size
        size = tuple([image.shape[x] + self.margin_begin[x] + self.margin_end[x] for x in range(self.dim)])
        self.image_pad = size
        image_pad = np.zeros(size + (image.shape[-1],))
        #image_pad[self.margin_begin:image.shape[:-1] + self.margin_begin] = image
        image_pad[self.margin_begin[0]:image.shape[0] + self.margin_begin[0],
                  self.margin_begin[1]:image.shape[1] + self.margin_begin[1],
                  self.margin_begin[2]:image.shape[2] + self.margin_begin[2], :] = image
        image = image_pad

        tiles = []
        for x, y, z, tile_x, tile_y, tile_z in self.crops:
            tile = image[x: x + tile_x, y: y + tile_y, z: z + tile_z].copy()
            if tile.shape[0] != self.tile[0]:
                print(tile.shape[0])
            assert tile.shape[1] == self.tile[1]
            assert tile.shape[2] == self.tile[2]

            tiles.append(tile)
        return tiles

    def crop_to_orignal_size(self, image):
        crop = image[
            self.margin_begin[0]: self.out[0] + self.margin_end[0],
            self.margin_begin[1]: self.out[1] + self.margin_end[1],
            self.margin_begin[2]: self.out[2] + self.margin_end[2],
        ]
        return crop

    def merge(self, tiles: List[np.ndarray], channels, dtype=np.float32):
        if len(tiles) != len(self.crops):
            raise ValueError

        target_shape = self.target_shape

        image = np.zeros((channels, target_shape[0], target_shape[1], target_shape[2]), dtype=np.float32)
        norm_mask = np.zeros((1, target_shape[0], target_shape[1], target_shape[2]), dtype=np.float32)

        #w = np.dstack([self.weight] * channels)
        w = np.expand_dims(self.weight, axis=0)

        for tile, (x, y, z, tile_x, tile_y, tile_z) in zip(tiles, self.crops_out):
            image[:, x: x + tile_x, y: y + tile_y, z: z + tile_z] += tile * w
            norm_mask[:, x: x + tile_x, y: y + tile_y, z: z + tile_z] += w

        #norm_mask = np.clip(norm_mask, a_min=np.finfo(norm_mask.dtype).eps, a_max=None)
        image = np.divide(image, norm_mask).astype(dtype)
        image = np.moveaxis(image, 0, -1).astype('float32')
        image = self.crop_to_orignal_size(image)
        return image

    @property
    def target_shape(self):
        target_shape = tuple([self.image_pad[x] * self.mag + self.margin_begin[x] + self.margin_end[x] for x in range(self.dim)])
        return target_shape

    def _mean(self, tile_size):
        return np.ones(tile_size, dtype=np.float32)


class CudaTileMerger3D:
    """
    Helper class to merge final image on GPU. This generally faster than moving individual tiles to CPU.
    """

    def __init__(self, image_shape, channels, weight):
        """

        :param image_shape: Shape of the source image
        :param image_margin:
        :param weight: Weighting matrix
        """

        self.weight = torch.from_numpy(np.expand_dims(weight, axis=0)).float().cuda()
        self.image = torch.zeros((channels, image_shape[0], image_shape[1], image_shape[2])).cuda()
        self.norm_mask = torch.zeros((1, image_shape[0], image_shape[1], image_shape[2])).cuda()

    def integrate_batch(self, batch: torch.Tensor, crop_coords):
        """
        Accumulates batch of tile predictions
        :param batch: Predicted tiles
        :param crop_coords: Corresponding tile crops w.r.t to original image
        """
        if len(batch) != len(crop_coords):
            raise ValueError("Number of images in batch does not correspond to number of coordinates")

        for tile, (x, y, z, tile_x, tile_y, tile_z) in zip(batch, crop_coords):
            try:
                self.image[:, x: x + tile_x, y: y + tile_y, z: z + tile_z] += tile * self.weight
            except RuntimeError:
                print()
            self.norm_mask[:, x: x + tile_x, y: y + tile_y, z: z + tile_z] += self.weight

    def merge(self) -> torch.Tensor:
        return self.image / self.norm_mask