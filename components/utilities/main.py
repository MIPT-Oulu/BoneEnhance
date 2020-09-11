import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import cv2
from tqdm import tqdm
from joblib import Parallel, delayed
#from skimage import measure


def load_images(path, n_jobs=12, rgb=False, uCT=False):
    """
    Loads multiple images from directory and stacks them into 3D numpy array

    Parameters
    ----------
    path : str
        Path to image stack.
    axis : tuple
        Order of loaded sample axes.
    n_jobs : int
        Number of parallel workers. Check N of CPU cores.
    Returns
    -------
    Loaded stack of images as 3D numpy array.
    """

    # List all files in alphabetic order
    files = os.listdir(path)
    files.sort()

    # Exclude extra files
    newlist = []
    if uCT:
        for file in files:
            if file.endswith('.png') or file.endswith('.bmp') or file.endswith('.tif'):
                try:
                    int(file[-7:-4])
                    newlist.append(file)
                except ValueError:
                    continue
    else:
        for file in files:
            if file.endswith('.png') or file.endswith('.bmp') or file.endswith('.tif'):
                newlist.append(file)
    files = newlist[:]  # replace list
    files.sort()

    # Load images
    if rgb:
        data = Parallel(n_jobs=n_jobs)(delayed(read_image_rgb)(path, file) for file in tqdm(files, 'Loading'))
        return files, np.array(data)
    else:
        data = Parallel(n_jobs=n_jobs)(delayed(read_image_gray)(path, file) for file in tqdm(files, 'Loading'))
        return files, np.array(data)


def load(path, axis=(0, 1, 2), n_jobs=12, rgb=False):
    """
    Loads an image stack as numpy array.

    Parameters
    ----------
    path : str
        Path to image stack.
    axis : tuple
        Order of loaded sample axes.
    n_jobs : int
        Number of parallel workers. Check N of CPU cores.
    Returns
    -------
    Loaded stack as 3D numpy array.
    """
    files = os.listdir(path)
    files.sort()
    # Exclude extra files
    newlist = []
    for file in files:
        if file.endswith('.png') or file.endswith('.bmp') or file.endswith('.tif'):
            try:
                int(file[-7:-4])
                newlist.append(file)
            except ValueError:
                continue
    files = newlist[:]  # replace list
    # Load images
    if rgb:
        data = Parallel(n_jobs=n_jobs)(delayed(read_image_rgb)(path, file) for file in files)
    else:
        data = Parallel(n_jobs=n_jobs)(delayed(read_image_gray)(path, file) for file in files)
    # Transpose array
    if axis != (0, 1, 2) and rgb:
        return np.transpose(np.array(data), axis + (3,)), files
    elif axis != (0, 1, 2):
        return np.transpose(np.array(data), axis), files
    return np.array(data), files


def read_image_gray(path, file):
    """Reads image from given path."""
    # Image
    f = os.path.join(path, file)
    image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    #image = cv2.imread(f, -1)
    return image


def read_image_rgb(path, file):
    """Reads image from given path."""
    # Image
    f = os.path.join(path, file)
    image = cv2.imread(f, cv2.IMREAD_COLOR)

    return image


def save(path, file_name, data, n_jobs=12, dtype='.png'):
    """
    Save a volumetric 3D dataset in given directory.

    Parameters
    ----------
    path : str
        Directory for dataset.
    file_name : str
        Prefix for the image filenames.
    data : 3D numpy array
        Volumetric data to be saved.
    n_jobs : int
        Number of parallel workers. Check N of CPU cores.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    nfiles = np.shape(data)[2]

    if data[0, 0, 0].dtype is bool:
        data = data * 255

    # Parallel saving (nonparallel if n_jobs = 1)
    if type(file_name) is list:
        Parallel(n_jobs=n_jobs)(delayed(cv2.imwrite)
                                (path + '/' + file_name[k][:-4] + dtype,
                                 data[:, :, k].astype('uint8'))
                                for k in tqdm(range(nfiles), 'Saving dataset'))

    else:
        Parallel(n_jobs=n_jobs)(delayed(cv2.imwrite)
                                (path + '/' + file_name + '_' + str(k).zfill(8) + dtype,
                                 data[:, :, k].astype('uint8'))
                                for k in tqdm(range(nfiles), 'Saving dataset'))


def save_images(path, file_names, data, n_jobs=12, crop=False):
    """
    Save a set of RGB images in given directory.

    Parameters
    ----------
    path : str
        Directory for dataset.
    file_name : str
        Prefix for the image filenames.
    data : 3D numpy array
        Volumetric data to be saved.
    n_jobs : int
        Number of parallel workers. Check N of CPU cores.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    nfiles = len(data)

    # Parallel saving (nonparallel if n_jobs = 1)
    if crop:
        Parallel(n_jobs=n_jobs)(delayed(cv2.imwrite)
                                (path + '/' + file_names[k][:-4] + '.png', data[k][:512, :])
                                for k in tqdm(range(nfiles), 'Saving images'))
    else:
        Parallel(n_jobs=n_jobs)(delayed(cv2.imwrite)
                                (path + '/' + file_names[k][:-4] + '.png', data[k][:, :])
                                for k in tqdm(range(nfiles), 'Saving images'))


def bounding_box(image, largest=True):
    """
    Return bounding box and contours of a mask.

    Parameters
    ----------
    image : 2D numpy array
        Input mask
    largest : bool
        Option to return only the largest contour. All contours returned otherwise.
    Returns
    -------
    Bounding box coordinates (tuple) and list of contours (or largest contour).
    """
    # Threshold and create Mat
    _, mask = cv2.threshold(image, thresh=0.5, maxval=1, type=0)
    # All contours
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # Largest contour
    c = max(contours, key=cv2.contourArea)

    # Return bounding rectangle for largest contour
    if largest:
        return cv2.boundingRect(c), c
    else:
        return cv2.boundingRect(c), contours


def print_orthogonal(data, mask=None, invert=True, res=3.2, title=None, cbar=True, cmap='gray', savepath=None, scale_factor=1000):
    """Print three orthogonal planes from given 3D-numpy array.

    Set pixel resolution in µm to set axes correctly.

    Parameters
    ----------
    data : 3D numpy array
        Three-dimensional input data array.
    savepath : str
        Full file name for the saved image. If not given, Image is only shown.
        Example: C:/path/data.png
    invert : bool
        Choose whether to invert y-axis of the data
    res : float
        Imaging resolution. Sets tick frequency for plots.
    title : str
        Title for the image.
    cbar : bool
        Choose whether to use colorbar below the images.
    cmap : str
        Colormap for the images
    """
    alpha = 0.5
    cmap_mask = 'autumn'
    dims = np.array(np.shape(data)) // 2
    dims2 = np.array(np.shape(data))
    x = np.linspace(0, dims2[0], dims2[0])
    y = np.linspace(0, dims2[1], dims2[1])
    z = np.linspace(0, dims2[2], dims2[2])
    scale = 1 / res
    if dims2[0] < scale_factor * scale:
        xticks = np.arange(0, dims2[0], scale_factor * scale / 4)
    else:
        xticks = np.arange(0, dims2[0], scale_factor * scale / 2)
    if dims2[1] < scale_factor * scale:
        yticks = np.arange(0, dims2[1], scale_factor * scale / 4)
    else:
        yticks = np.arange(0, dims2[1], scale_factor * scale / 2)
    if dims2[2] < scale_factor * scale:
        zticks = np.arange(0, dims2[2], scale_factor * scale / 4)
    else:
        zticks = np.arange(0, dims2[2], scale_factor * scale / 2)

    # Plot figure
    fig = plt.figure(dpi=300)
    ax1 = fig.add_subplot(131)
    cax1 = ax1.imshow(data[:, :, dims[2]].T, cmap=cmap)
    if cbar and not isinstance(data[0, 0, dims[2]], np.bool_):
        cbar1 = fig.colorbar(cax1, ticks=[np.min(data[:, :, dims[2]]), np.max(data[:, :, dims[2]])],
                             orientation='horizontal')
        cbar1.solids.set_edgecolor("face")
    if mask is not None:
        m = mask[:, :, dims[2]].T
        ax1.imshow(np.ma.masked_array(m, m == 0), cmap=cmap_mask, alpha=alpha)
    plt.title('Transaxial (xy)')
    ax2 = fig.add_subplot(132)
    cax2 = ax2.imshow(data[:, dims[1], :].T, cmap=cmap)
    if cbar and not isinstance(data[0, dims[1], 0], np.bool_):
        cbar2 = fig.colorbar(cax2, ticks=[np.min(data[:, dims[1], :]), np.max(data[:, dims[1], :])],
                             orientation='horizontal')
        cbar2.solids.set_edgecolor("face")
    if mask is not None:
        m = mask[:, dims[1], :].T
        ax2.imshow(np.ma.masked_array(m, m == 0), cmap=cmap_mask, alpha=alpha)
    plt.title('Coronal (xz)')
    ax3 = fig.add_subplot(133)
    cax3 = ax3.imshow(data[dims[0], :, :].T, cmap=cmap)
    if cbar and not isinstance(data[dims[0], 0, 0], np.bool_):
        cbar3 = fig.colorbar(cax3, ticks=[np.min(data[dims[0], :, :]), np.max(data[dims[0], :, :])],
                             orientation='horizontal')
        cbar3.solids.set_edgecolor("face")
    if mask is not None:
        m = mask[dims[0], :, :].T
        ax3.imshow(np.ma.masked_array(m, m == 0), cmap=cmap_mask, alpha=alpha)
    plt.title('Sagittal (yz)')

    # Give plot a title
    if title is not None:
        plt.suptitle(title)

    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / scale))
    ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y / scale))
    ticks_z = ticker.FuncFormatter(lambda z, pos: '{0:g}'.format(z / scale))
    ax1.xaxis.set_major_formatter(ticks_x)
    ax1.yaxis.set_major_formatter(ticks_y)
    ax2.xaxis.set_major_formatter(ticks_x)
    ax2.yaxis.set_major_formatter(ticks_z)
    ax3.xaxis.set_major_formatter(ticks_y)
    ax3.yaxis.set_major_formatter(ticks_z)
    ax1.set_xticks(xticks)
    ax1.set_yticks(yticks)
    ax2.set_xticks(xticks)
    ax2.set_yticks(zticks)
    ax3.set_xticks(yticks)
    ax3.set_yticks(zticks)

    # Invert y-axis
    if invert:
        ax1.invert_yaxis()
        ax2.invert_yaxis()
        ax3.invert_yaxis()
    plt.tight_layout()

    # Save the image
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", transparent=True)
    plt.show()


def print_images(images, masks=None, title=None, subtitles=None, save_path=None, sample=None, transparent=False):
    """Print three images from list of three 2D images.

    Parameters
    ----------
    images : list
        List containing three 2D numpy arrays
    save_path : str
        Full file name for the saved image. If not given, Image is only shown.
        Example: C:/path/images.png
    subtitles : list
        List of titles to be shown above each plot.
    sample : str
        Name for the image.
    title : str
        Title for the image.
    transparent : bool
        Choose whether to have transparent image background.
    """
    alpha = 0.3
    cmap = plt.cm.tab10  # define the colormap
    cmap2 = 'Dark2_r'
    """
    cmap2 = plt.cm.tab10  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    cmaplist[0] = (.5, .5, .5, 1.0)

    # create the new map
    cmap2 = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)
    """

    # Configure plot
    fig = plt.figure(dpi=300)
    if title is not None:
        fig.suptitle(title, fontsize=16)

    ax1 = fig.add_subplot(131)
    cax1 = ax1.imshow(images[0], cmap=cmap2)
    if not isinstance(images[0][0, 0], np.bool_):  # Check for boolean image
        cbar1 = fig.colorbar(cax1, ticks=[np.min(images[0]), np.max(images[0])], orientation='horizontal')
        cbar1.solids.set_edgecolor("face")
    if subtitles is not None:
        plt.title(subtitles[0])
    if masks is not None:
        m = masks[0]
        ax1.imshow(np.ma.masked_array(m, m == 0), cmap=cmap, alpha=alpha)

    ax2 = fig.add_subplot(132)
    cax2 = ax2.imshow(images[1], cmap=cmap2)
    if not isinstance(images[1][0, 0], np.bool_):
        cbar2 = fig.colorbar(cax2, ticks=[np.min(images[1]), np.max(images[1])], orientation='horizontal')
        cbar2.solids.set_edgecolor("face")
    if subtitles is not None:
        plt.title(subtitles[1])
    if masks is not None:
        m = masks[1]
        ax2.imshow(np.ma.masked_array(m, m == 0), cmap=cmap, alpha=alpha)

    ax3 = fig.add_subplot(133)
    cax3 = ax3.imshow(images[2], cmap=cmap2)
    if not isinstance(images[2][0, 0], np.bool_):
        cbar3 = fig.colorbar(cax3, ticks=[np.min(images[2]), np.max(images[2])], orientation='horizontal')
        cbar3.solids.set_edgecolor("face")
    if subtitles is not None:
        plt.title(subtitles[2])
    if masks is not None:
        m = masks[2]
        ax3.imshow(np.ma.masked_array(m, m == 0), cmap=cmap, alpha=alpha)

    # Save or show
    if save_path is not None and sample is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        plt.tight_layout()  # Make sure that axes are not overlapping
        fig.savefig(save_path + sample, transparent=transparent)
        plt.close(fig)
    else:
        plt.show()
