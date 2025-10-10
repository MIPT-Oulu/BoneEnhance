from random import choice, uniform

from skimage.filters import gaussian

import cv2
import h5py
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize
from pathlib import Path

from bone_enhance.utilities import print_images, print_orthogonal, downscale_image, load_neighbor_slices

_DEFAULT_ANTIALIASING_KERNEL = 5
_DEFAULT_ANTIALIASING_SIGMA = 0.5
_DEFAULT_SEGMENTATION_THRESHOLD = 90

def parse_grayscale(root, entry, transform, data_key, target_key, debug=False, config=None):

    if config.training.rgb:
        target = cv2.imread(str(entry.target_fname), -1)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target[:, :, 1] = target[:, :, 0]
        target[:, :, 2] = target[:, :, 0]
    else:
        target = cv2.imread(str(entry.target_fname), -1)
        # Make sure the target is in grayscale
        if target.ndim == 3:
            target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    data_max = np.iinfo(target.dtype).max
    # Magnification
    mag = config.training.magnification
    # Antialiasing kernel size
    if config.training.antialiasing is None:
        config.training.antialiasing = _DEFAULT_ANTIALIASING_KERNEL
    if config.training.sigma is None:
        config.training.sigma = _DEFAULT_ANTIALIASING_SIGMA

    # Resize target to 4x magnification respect to input
    if config is not None and not config.training.crossmodality:
        # Add noise if given, always do antialiasing with Gaussian blur
        input_img = downscale_image(target, im_size=(target.shape[1] // mag, target.shape[0] // mag),
                                    add_noise=config.training.noise,
                                    blur=True, sigma=config.training.sigma)
    # Co-registered images
    elif config is not None:

        # Read image and target
        if config.training.rgb:
            input_img = cv2.imread(str(entry.fname), -1)
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            input_img[:, :, 1] = input_img[:, :, 0]
            input_img[:, :, 2] = input_img[:, :, 0]
        else:
            input_img = cv2.imread(str(entry.fname), -1)
            # Make sure the input is in grayscale
            if target.ndim == 3:
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

        # If the image sizes do not match, rescale the target image to match the input image
        if target.shape != tuple([mag * x for x in input_img.shape]):
            target = downscale_image(target, im_size=(target.shape[1] * mag, target.shape[0] * mag),
                                     add_noise=config.training.noise)
    else:
        raise NotImplementedError

    # Make sure that grayscale images also possess channel dimension
    if len(input_img.shape) != 3:
        input_img = np.expand_dims(input_img, -1)
    if len(target.shape) != 3:
        target = np.expand_dims(target, -1)

    # Apply random transforms. Images are returned in format 3xHxW
    input_img, target = transform((input_img, target))

    # Target is scaled to -1 to +1 range (for tanh activation)
    target = (target / float(data_max) - 0.5) * 2

    # Plot a small random portion of image-target pairs during debug
    if debug and uniform(0, 1) >= 0.995:
        fig = plt.figure(dpi=300)
        ax1 = fig.add_subplot(221)
        im = ax1.imshow(np.asarray(input_img[0, :, :]), cmap='gray')
        plt.colorbar(im, orientation='horizontal')
        plt.title('Input')

        ax2 = fig.add_subplot(222)
        im2 = ax2.imshow(np.asarray(target[0, :, :]), cmap='gray')
        plt.colorbar(im2, orientation='horizontal')
        plt.title('Target')

        ax3 = fig.add_subplot(223)
        ax3.hist(np.asarray(input_img).ravel(), bins=2 ** 10)
        ax4 = fig.add_subplot(224)
        ax4.hist(np.asarray(target).ravel(), bins=2 ** 10)
        plt.show()

    return {data_key: input_img, target_key: target}


def parse_adjacent_prediction(root, entry, transform, data_key, target_key, debug=False, config=None):
    """
    Loads every mag (e.g. 4th) image as adjacent input images (idx -4, 0, 4) and
    the next mag adjacent slices as target (idx 0, 1, 2, 3).
    """
    # Load the center slice
    center_img = cv2.imread(str(entry.target_fname), -1)
    data_max = np.iinfo(center_img.dtype).max

    # Antialiasing kernel size
    if config.training.sigma is None:
        config.training.sigma = _DEFAULT_ANTIALIASING_SIGMA
    mag = config.training.magnification

    # Build target: slices at indices 0, 1, 2, 3
    target = np.zeros((center_img.shape[0], center_img.shape[1], mag), dtype=center_img.dtype)
    target[:, :, 0] = center_img

    # Load next 3 slices for target (idx 1, 2, 3)
    for i in range(1, mag):
        next_slice = Path(entry.target_fname.parent,
                          entry.target_fname.stem[:-8] +
                          str(int(entry.target_fname.stem[-8:]) + i).zfill(8) +
                          entry.target_fname.suffix)
        if next_slice.exists():
            target[:, :, i] = cv2.imread(str(next_slice), -1)
        else:
            target[:, :, i] = center_img  # Fallback to center if not exists

    # Build input: slices at indices -4, 0, 4
    input_img = np.zeros((center_img.shape[0], center_img.shape[1], 3), dtype=center_img.dtype)
    input_img[:, :, 1] = center_img  # Center at channel 1

    # Load slice at idx -4
    prev_slice = Path(entry.fname.parent,
                      entry.fname.stem[:-8] +
                      str(int(entry.fname.stem[-8:]) - 4).zfill(8) +
                      entry.fname.suffix)
    if prev_slice.exists():
        input_img[:, :, 0] = cv2.imread(str(prev_slice), -1)
    else:
        input_img[:, :, 0] = center_img

    # Load slice at idx +4
    next_slice = Path(entry.fname.parent,
                      entry.fname.stem[:-8] +
                      str(int(entry.fname.stem[-8:]) + 4).zfill(8) +
                      entry.fname.suffix)
    if next_slice.exists():
        input_img[:, :, 2] = cv2.imread(str(next_slice), -1)
    else:
        input_img[:, :, 2] = center_img

    input_img = downscale_image(input_img, im_size=(input_img.shape[0] // mag, input_img.shape[1] // mag),
                                add_noise=config.training.noise, sigma=config.training.sigma)

    # Apply random transforms. Images are returned in format CxHxW
    input_img, target = transform((input_img, target))

    if config.training.no_mean_std:
        # Scale input to from 0 to +1 range
        input_img = input_img / float(data_max)

    # Target is scaled to -1 to +1 range (tanh activation)
    target = (target / float(data_max) - 0.5) * 2

    # Plot a small random portion of image-target pairs during debug
    if debug and uniform(0, 1) >= 0.995:
        fig, ax = plt.subplots(2, 3, figsize=(12, 8))

        # Input channels (idx -4, 0, 4)
        for i in range(3):
            im = ax[0, i].imshow(np.asarray(input_img[i, :, :]), cmap='gray')
            fig.colorbar(im, ax=ax[0, i], orientation='horizontal')
            ax[0, i].set_title(f'Input ch{i} (idx {-4 + i * 4})')

        # Target channels (idx 0, 1, 2)
        for i in range(3):
            im = ax[1, i].imshow(np.asarray(target[i, :, :]), cmap='gray')
            fig.colorbar(im, ax=ax[1, i], orientation='horizontal')
            ax[1, i].set_title(f'Target ch{i} (idx {i})')

        plt.tight_layout()
        plt.show()

    return {data_key: input_img, target_key: target}


def parse_3ch(root, entry, transform, data_key, target_key, debug=False, config=None):
    """
    Loads neighboring slices as a 3-channel image. If rgb is set to true, target includes neighboring slices.
    If false, target includes only the center slice and adjacent slices are used as supporting information for input.
    """
    # Try to load neighbouring slices
    target = load_neighbor_slices(entry.target_fname)
    data_max = np.iinfo(target.dtype).max

    # Antialiasing kernel size
    if config.training.sigma is None:
        config.training.sigma = _DEFAULT_ANTIALIASING_SIGMA
    mag = config.training.magnification

    # Resize target to 4x magnification respect to input
    if config is not None and not config.training.crossmodality:
        input_img = downscale_image(target, im_size=(target.shape[1] // mag, target.shape[0] // mag),
                                    add_noise=config.training.noise, sigma=config.training.sigma)

    # Co-registered images
    elif config is not None:
        # Try to load neighbouring slices for input
        input_img = load_neighbor_slices(entry.fname)

        # If the image sizes (not counting channel dim) do not match, rescale the target image to match the input image
        if target.shape[:-1] != tuple([mag * x for x in input_img.shape[:-1]]):
            target = downscale_image(target, im_size=(input_img.shape[1] * mag, input_img.shape[0] * mag), add_noise=False)
    else:
        raise NotImplementedError

    # Make sure that grayscale images also possess channel dimension
    #if len(input_img.shape) != 3:
    #    input_img = np.expand_dims(input_img, -1)
    #if len(target.shape) != 3:
    #    target = np.expand_dims(target, -1)

    # Apply random transforms. Images are returned in format 3xHxW
    input_img, target = transform((input_img, target))

    # Target is scaled to -1 to +1 range (tanh activation)
    target = (target / float(data_max) - 0.5) * 2

    # Keep only the center slice of target if rgb is not True
    if not config.training.rgb:
        target = target[[1], :, :]

    # Plot a small random portion of image-target pairs during debug
    if debug and uniform(0, 1) >= 0.995:
        fig, ax = plt.subplots(2, 2)
        im = ax[0, 0].imshow(np.asarray(input_img[0, :, :]), cmap='gray')
        fig.colorbar(im, ax=ax[0, 0], orientation='horizontal')
        ax[0, 0].set_title('Input')

        im = ax[0, 1].imshow(np.asarray(target[0, :, :]), cmap='gray')
        fig.colorbar(im, ax=ax[0, 1], orientation='horizontal')
        ax[0, 1].set_title('Target')

        # Histogram
        ax[1, 0].hist(np.asarray(input_img).ravel(), bins=2 ** 10)
        ax[1, 1].hist(np.asarray(target).ravel(), bins=2 ** 10)
        plt.show()

    return {data_key: input_img, target_key: target}


def parse_segmentation(root, entry, transform, data_key, target_key, debug=False, config=None):

    # Read image and target
    if config.training.rgb:
        img = cv2.imread(str(entry.fname), -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img[:, :, 1] = img[:, :, 0]
        img[:, :, 2] = img[:, :, 0]
    else:
        img = cv2.imread(str(entry.fname), cv2.IMREAD_GRAYSCALE)

    # Segmentation mask
    target = cv2.imread(str(entry.target_fname), cv2.IMREAD_GRAYSCALE)

    # Magnification
    mag = config.training.magnification
    # Antialiasing kernel size
    if config.training.antialiasing is None:
        config.training.antialiasing = _DEFAULT_ANTIALIASING_KERNEL
    if config.training.sigma is None:
        config.training.sigma = _DEFAULT_ANTIALIASING_SIGMA

    # Binarize µCT image, then downscale
    if not config.training.crossmodality:
        # Get the downscaled input image
        new_size = (img.shape[1] // mag, img.shape[0] // mag)
        # Antialiasing and downscaling
        img = cv2.resize(cv2.GaussianBlur(target, ksize=(config.training.antialiasing, config.training.antialiasing),
                                          sigmaX=config.training.sigma), new_size)

        img = np.expand_dims(img, -1)
        if config.training.rgb:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # No modifications needed when using CBCT img

    # Segmentation target
    if type(config.training.threshold) is not int:
        config.training.threshold = _DEFAULT_SEGMENTATION_THRESHOLD

    target = (target > config.training.threshold).astype('uint8')

    # Set target size to 4x input
    #new_size = (img.shape[1] * mag, img.shape[0] * mag)
    #target = cv2.resize(target, new_size, interpolation=cv2.INTER_NEAREST)

    # Set input size to match target
    new_size = (target.shape[1], target.shape[0])
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)

    # Make sure that grayscale images also possess channel dimension
    #if len(img.shape) != 3:

    #if len(target.shape) != 3:
    #    target = np.expand_dims(target, -1)

    # Apply random transforms. Images are returned in format 3xHxW
    img, target = transform((img, target))

    # Target is scaled to 0-1 range
    #target = target / 255.

    # Plot a small random portion of image-target pairs during debug
    if debug and uniform(0, 1) >= 0.99:
        fig = plt.figure(dpi=300)
        ax1 = fig.add_subplot(121)
        im = ax1.imshow(np.asarray(img[0, :, :] / 255.), cmap='gray')
        plt.colorbar(im, orientation='horizontal')
        plt.title('Input')

        ax2 = fig.add_subplot(122)
        im2 = ax2.imshow(np.asarray(target[0, :, :]), cmap='gray')
        plt.colorbar(im2, orientation='horizontal')
        plt.title('Target')
        plt.show()

    return {data_key: img, target_key: target}


def parse_3d(root, entry, transform, data_key, target_key, debug=False, config=None):
    # Load target with hdf5
    with h5py.File(entry.target_fname, 'r') as f:
        target = f['data'][:]

    # Data type maximum value
    data_max = float(np.iinfo(target.dtype).max)

    # Magnification
    mag = config.training.magnification

    # Downscaling should be done outside training
    if config is not None:

        # Load input with hdf5
        with h5py.File(entry.fname, 'r') as f:
            img = f['data'][:]

            # Resize the target to match input in case of a mismatch
            new_size = (int(img.shape[0] * mag), int(img.shape[1] * mag), int(img.shape[2] * mag))
            if target.shape != new_size:
                if data_max == 65535:
                    target = resize(target.astype('float64'), new_size, order=0, anti_aliasing=True,
                                    preserve_range=True).astype('uint16')
                elif data_max == 255:
                    target = resize(target.astype('float64'), new_size, order=0, anti_aliasing=True,
                                    preserve_range=True).astype('uint8')
    else:
        raise NotImplementedError

    # Channel dimension
    if config.training.rgb:
        target = np.stack((target,) * 3, axis=-1)
        img = np.stack((img,) * 3, axis=-1)
    else:
        target = np.stack((target,), axis=-1)  # One-channel
        img = np.stack((img,), axis=-1)

    # Apply random transforms
    img, target = transform((img, target))

    # Images are in the format 3xHxWxD
    # and scaled to 0-1 range
    if config.training.no_mean_std:
        img /= data_max
    # Target is scaled to -1 to +1 range
    target = (target / data_max - 0.5) * 2

    # Plot a small random portion of image-target pairs during debug
    if debug and uniform(0, 1) >= 0.98:
        print_images([img[0, 7, :, :].numpy() / data_max, img[0, :, 7, :].numpy() / data_max,
                      target[0, 7 * mag, :, :].numpy(), target[0, :, 7 * mag, :].numpy()])

    return {data_key: img, target_key: target}


def parse_3d_debug(root, entry, transform, data_key, target_key, debug=False, config=None):
    """
    Note! Works only in downsampling.
    For cross-modality, transfer to 2D should be done simultaneously for img and target.
    """
    # Load target with hdf5
    with h5py.File(entry.target_fname, 'r') as f:
        target = f['data'][:]

    # Magnification, kernel size
    mag = config.training.magnification
    k = choice([5])


    # Resize target to 4x magnification respect to input
    if config is not None and not config.training.crossmodality:
        # Factor for OpenCV
        new_size = (target.shape[1] // mag, target.shape[0] // mag)
        # Factor for skimage
        new_size = (target.shape[0] // mag, target.shape[1] // mag)
        # Factor for 3D
        new_size = (target.shape[0] // mag, target.shape[1] // mag, target.shape[2] // mag)

        # Downscale and antialias
        #img = cv2.resize(blur_2d(target, k, 0.5), new_size)
        #img = resize(blur_3d(target, k, 0.5), new_size, order=1, preserve_range=True).astype(np.uint8)
        img = resize(target, new_size, order=1, preserve_range=True).astype(np.uint8)

    elif config is not None:

        # Load input with hdf5
        with h5py.File(entry.fname, 'r') as f:
            img = f['data'][:]

        # Resize the target to match input in case of a mismatch
        new_size = (int(img.shape[0] * mag), int(img.shape[1] * mag), int(img.shape[2] * mag))
        if target.shape != new_size:
            target = resize(target.astype('float64'), new_size, order=0, anti_aliasing=True,
                            preserve_range=True).astype('uint8')

    else:
        raise NotImplementedError

    # Create 2D images
    #img, target = transfer_3d_to_random_2d([img, target])

    # Channel dimension
    if config.training.rgb:
        target = np.stack((target,) * 3, axis=-1)
        img = np.stack((img,) * 3, axis=-1)
    else:
        target = np.stack((target,), axis=-1)  # One-channel
        img = np.stack((img,), axis=-1)

    # Apply random transforms
    img, target = transform((img, target))

    # Images are in the format 3xHxW
    # and scaled to 0-1 range
    # Target is scaled to -1 to +1 range
    target = (target / 255. - 0.5) * 2

    # Plot a small random portion of image-target pairs during debug
    if debug and uniform(0, 1) >= 0.95 and len(img.shape) != 4:
        fig = plt.figure(dpi=300)
        ax1 = fig.add_subplot(121)
        im = ax1.imshow(np.asarray(img.permute(1, 2, 0) / 255.), cmap='gray')
        plt.colorbar(im, orientation='horizontal')
        plt.title('Input')

        ax2 = fig.add_subplot(122)
        im2 = ax2.imshow(np.asarray(target.permute(1, 2, 0)), cmap='gray')
        plt.colorbar(im2, orientation='horizontal')
        plt.title('Target')
        plt.show()

    return {data_key: img, target_key: target}


def parse_autoencoder_2d(root, entry, transform, data_key, target_key, debug=False, config=None):

    if config.training.rgb:
        target = cv2.imread(str(entry.target_fname), -1)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target[:, :, 1] = target[:, :, 0]
        target[:, :, 2] = target[:, :, 0]
    else:
        target = cv2.imread(str(entry.target_fname), cv2.IMREAD_GRAYSCALE)

    # Magnification
    mag = config.training.magnification
    k = choice([5])

    # Resize target to 4x magnification respect to input
    if config is not None and not config.training.crossmodality:

        # Resize target to a relevant size (from the 3.2µm resolution to 51.2µm
        new_size = (target.shape[1] // 16, target.shape[0] // 16)

        # Antialiasing
        target = cv2.GaussianBlur(target, ksize=(k, k), sigmaX=0)

        target = cv2.resize(target.copy(), new_size)  # .transpose(1, 0, 2)
        #target = resize(target.astype('float64'), new_size, order=0, anti_aliasing=True, preserve_range=True).astype('uint8')

        new_size = (target.shape[1] // mag, target.shape[0] // mag)

        # No antialias
        #img = cv2.resize(target, new_size, interpolation=cv2.INTER_LANCZOS4)
        # Antialias
        img = cv2.resize(cv2.GaussianBlur(target, ksize=(k, k), sigmaX=0), new_size)
        #img = resize(target.astype('float64'), new_size, order=0, anti_aliasing=True, preserve_range=True, anti_aliasing_sigma=k).astype('uint8')
    elif config is not None:

        # Read image and target
        if config.training.rgb:
            img = cv2.imread(str(entry.fname), -1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img[:, :, 1] = img[:, :, 0]
            img[:, :, 2] = img[:, :, 0]
        else:
            img = cv2.imread(str(entry.fname), cv2.IMREAD_GRAYSCALE)


        new_size = (img.shape[1] * mag, img.shape[0] * mag)
        target = cv2.GaussianBlur(target, ksize=(k, k), sigmaX=0)
        target = cv2.resize(target, new_size)
        #target = resize(target.astype('float64'), new_size, order=0, anti_aliasing=True, preserve_range=True, anti_aliasing_sigma=k).astype('uint8')
    else:
        raise NotImplementedError

    # Make sure that grayscale images also possess channel dimension
    if len(img.shape) != 3:
        img = np.expand_dims(img, -1)
    if len(target.shape) != 3:
        target = np.expand_dims(target, -1)

    # Apply random transforms. Images are returned in format 3xHxW
    img, target = transform((img, target))

    # Target is scaled to -1 to +1 range
    target = (target / 255. - 0.5) * 2

    # Plot a small random portion of image-target pairs during debug
    if debug and uniform(0, 1) >= 0.99:
        fig = plt.figure(dpi=300)
        ax1 = fig.add_subplot(121)
        im = ax1.imshow(np.asarray(img[0, :, :] / 255.), cmap='gray')
        plt.colorbar(im, orientation='horizontal')
        plt.title('Input')

        ax2 = fig.add_subplot(122)
        im2 = ax2.imshow(np.asarray(target[0, :, :]), cmap='gray')
        plt.colorbar(im2, orientation='horizontal')
        plt.title('Target')
        plt.show()

    return {data_key: target, target_key: target}


def parse_autoencoder_3d(root, entry, transform, data_key, target_key, debug=False, config=None):
    # Load target with hdf5
    with h5py.File(entry.target_fname, 'r') as f:
        target = f['data'][:]

    # Magnification
    mag = config.training.magnification

    #cm = choice([True, False])
    cm = config.training.crossmodality

    # Resize target to 4x magnification respect to input
    #if config is not None and not config.training.crossmodality:
    if not cm:

        # Resize target with the given magnification to provide the input image
        new_size = (target.shape[0] // mag, target.shape[1] // mag, target.shape[2] // mag)

        sigma = choice([0.5])
        img = resize(target.astype('float64'), new_size, order=0, anti_aliasing=True, preserve_range=True, anti_aliasing_sigma=sigma).astype('uint8')

    elif config is not None:

        # Load input with hdf5
        with h5py.File(entry.fname, 'r') as f:
            img = f['data'][:]

        # Resize the target to match input in case of a mismatch
        new_size = (int(img.shape[0] * mag), int(img.shape[1] * mag), int(img.shape[2] * mag))
        if target.shape != new_size:
            target = resize(target.astype('float64'), new_size, order=0, anti_aliasing=True, preserve_range=True).astype('uint8')
    else:
        raise NotImplementedError

    # Channel dimension
    if config.training.rgb:
        target = np.stack((target,) * 3, axis=-1)
        img = np.stack((img,) * 3, axis=-1)
    else:
        target = np.stack((target,), axis=-1)  # One-channel
        img = np.stack((img,), axis=-1)

    # Apply random transforms
    img, target = transform((img, target))

    # Target is scaled to -1 to +1 range
    target = (target / 255. - 0.5) * 2

    # Plot a small random portion of image-target pairs during debug
    if debug and uniform(0, 1) >= 0.95 and len(img.shape) != 4:
        res = 0.2  # In mm
        print_orthogonal(img[0, :, :, :].numpy() / 255, title='Input', res=res)

        print_orthogonal(target[0, :, :, :].numpy(), title='Target', res=res / mag)

    return {data_key: target, target_key: target}