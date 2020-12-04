import numpy as np


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