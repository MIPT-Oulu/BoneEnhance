import numpy as np
from scipy.signal import fftconvolve, convolve

def calculate_bvtv(mask, voi, percentage=True):
    """
    Calculates the bone volume fraction
    :param mask: Array with the trabecular mask. 
    :param voi: Array with the full volume, including areas between trabeculae.
    :param percetage:  Whether to output a decimal or percentage number as result.
    :return: Bone volume fraction
    """
    # Make sure that the images are in right format
    mask = (mask / np.max(mask)).astype(np.uint8)
    voi = (voi / np.max(voi)).astype(np.uint8)

    bone = np.sum(mask)
    volume = np.sum(voi)

    # No VOI to be evaluated
    if volume == 0:
        print('Empty VOI')
        return 0

    fraction = bone / volume

    if percentage:
        return fraction * 100
    else:
        return fraction


def make_2d_gauss(ks, sigma):
    """Gaussian kernel"""
    # Mean indices
    c = ks // 2

    # Exponents
    x = (np.linspace(0, ks - 1, ks) - c) ** 2
    y = (np.linspace(0, ks - 1, ks) - c) ** 2

    # Denominator
    denom = np.sqrt(2 * np.pi * sigma ** 2)

    # Evaluate gaussians
    ex = np.exp(-0.5 * x / sigma ** 2) / denom
    ey = np.exp(-0.5 * y / sigma ** 2) / denom

    # Iterate over kernel size
    kernel = np.zeros((ks, ks))
    for k in range(ks):
        kernel[k, :] = ey[k] * ex

    # Normalize so kernel sums to 1
    kernel /= kernel.sum()

    return kernel


def make_3d_gauss(ks, sigma, normalize_max_one=True):
    """Gaussian kernel"""
    # Mean indices
    c = ks // 2

    # Exponents symmetric for xyz
    loc = (np.linspace(0, ks - 1, ks) - c) ** 2

    # Denominator
    denom = np.sqrt(2 * np.pi * sigma ** 2)

    # Evaluate gaussians
    gauss = np.exp(-0.5 * loc / sigma ** 2) / denom

    # Iterate over kernel size
    kernel = np.zeros((ks, ks, ks))
    for x in range(ks):
        for y in range(ks):
            for z in range(ks):
                kernel[x, y, z] = gauss[x] * gauss[y] * gauss[z]

    if normalize_max_one:
        kernel /= kernel[c, c, c]
    else:
        # Normalize so kernel sums to 1
        kernel /= kernel.sum()

    return kernel


def blur_3d(img, ks, sigma):
    kernel = make_3d_gauss(ks, sigma, normalize_max_one=True)
    return convolve(img, kernel, mode='same')