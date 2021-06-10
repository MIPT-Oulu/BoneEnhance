from random import choice, uniform

import cv2
import h5py
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize

from bone_enhance.utilities import print_images, print_orthogonal


def parse_grayscale(root, entry, transform, data_key, target_key, debug=False, config=None):

    if config.training.rgb:
        target = cv2.imread(str(entry.target_fname), -1)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target[:, :, 1] = target[:, :, 0]
        target[:, :, 2] = target[:, :, 0]
    else:
        target = cv2.imread(str(entry.target_fname), cv2.IMREAD_GRAYSCALE)

    # Magnification
    mag = config.training.magnification
    # Antialiasing kernel size
    if config.training.antialiasing is not None:
        k = config.training.antialiasing
    else:
        k = 5
    if config.training.sigma is not None:
        s = config.training.sigma
    else:
        s = 0

    # Resize target to 4x magnification respect to input
    if config is not None and not config.training.crossmodality:

        # Resize target to a relevant size (from the 3.2µm resolution to 51.2µm
        #new_size = (target.shape[1] // 16, target.shape[0] // 16)

        # Antialiasing
        #target = cv2.GaussianBlur(target, ksize=(k, k), sigmaX=0)

        #target = cv2.resize(target.copy(), new_size)  # .transpose(1, 0, 2)
        #target = resize(target.astype('float64'), new_size, order=0, anti_aliasing=True, preserve_range=True).astype('uint8')

        new_size = (target.shape[1] // mag, target.shape[0] // mag)

        # No antialias
        #img = cv2.resize(target, new_size, interpolation=cv2.INTER_LANCZOS4)
        # Antialias
        img = cv2.resize(cv2.GaussianBlur(target, ksize=(k, k), sigmaX=s, sigmaY=s), new_size)
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
        target = cv2.GaussianBlur(target, ksize=(k, k), sigmaX=s, sigmaY=s)
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
    if debug and uniform(0, 1) >= 0.999:
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
    k = choice([5])

    # Binarize µCT image, then downscale
    if not config.training.crossmodality:
        # Get the downscaled input image
        new_size = (img.shape[1], img.shape[0])
        # Antialiasing and downscaling
        img = cv2.resize(cv2.GaussianBlur(target, ksize=(k, k), sigmaX=0), new_size)

        img = np.expand_dims(img, -1)
        if config.training.rgb:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # No modifications needed when using CBCT img

    # Segmentation target
    threshold = 90
    target = (target > threshold).astype('uint8')

    # Set target size to 4x input
    new_size = (img.shape[1] * mag, img.shape[0] * mag)
    target = cv2.resize(target, new_size, interpolation=cv2.INTER_NEAREST)

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

    # Magnification
    mag = config.training.magnification

    #cm = choice([True, False])
    cm = config.training.crossmodality

    # Downscaling should be done outside training
    if config is not None:

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

    # Images are in the format 3xHxWxD
    # and scaled to 0-1 range
    #img /= 255.
    # Target is scaled to -1 to +1 range
    target = (target / 255. - 0.5) * 2

    # Plot a small random portion of image-target pairs during debug
    if debug and uniform(0, 1) >= 0.98:
        #res = 0.2  # In mm
        #print_orthogonal(img[0, :, :, :].numpy() / 255, title='Input', res=res)

        #print_orthogonal(target[0, :, :, :].numpy(), title='Target', res=res / mag)
        dims = img.size()
        #print_images([img[0, dims[1] // 2, :, :].numpy() / 255., img[0, :, dims[2] // 2, :].numpy() / 255.,
        #              target[0, dims[1] // 2 * mag, :, :].numpy(), target[0, :, dims[2] // 2 * mag, :].numpy()])
        print_images([img[0, 7, :, :].numpy() / 255., img[0, :, 7, :].numpy() / 255.,
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