import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import torch
import os
import cv2
from copy import deepcopy
from functools import partial
from pydicom import dcmread, dcmwrite, Dataset
from pydicom.pixel_data_handlers.util import apply_modality_lut
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID
from datetime import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
from glob import glob
#from skimage import measure


def threshold(data, method='otsu', block=11):
    """Thresholds 3D or 2D array using the Otsu method. Returns mask and threshold value."""

    # Select thresholding method
    available_methods = {
        'otsu': cv2.THRESH_OTSU,
        'gaussian': cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        'mean': cv2.ADAPTIVE_THRESH_MEAN_C
    }
    th = available_methods[method]

    if len(data.shape) == 2:
        val, mask = cv2.threshold(data.astype('uint8'), 0, 255, th)
        return mask, val

    mask1 = np.zeros(data.shape)
    mask2 = np.zeros(data.shape)
    values1 = np.zeros(data.shape[0])
    values2 = np.zeros(data.shape[1])
    if method == 'otsu':
        for i in range(data.shape[0]):
            values1[i], mask1[i, :, :] = cv2.threshold(data[i, :, :].astype('uint8'), 0, 255, th)
        for i in range(data.shape[1]):
            values2[i], mask2[:, i, :] = cv2.threshold(data[:, i, :].astype('uint8'), 0, 255, th)
        value = (np.mean(values1) + np.mean(values2)) // 2
        return data > value, value
    else:
        for i in range(data.shape[0]):
            mask1[i, :, :] = cv2.adaptiveThreshold(data[i, :, :].astype('uint8'), 255,
                                                               adaptiveMethod=th, thresholdType=cv2.THRESH_BINARY,
                                                               blockSize=block, C=0)
        return mask1.astype('bool'), 0


def load_logfile(path: str, first=True) -> dict:
    """
    Read and return logfile with extension .log
    :param path: Path to the log file
    :param first: Return only first log file obtained
    :return: Log file as dict
    """

    # Return a list of log files
    log = glob(path + '/**.log')

    # Read the first log file in list
    if first:
        log = log[0]
        try:
            with open(log) as f:
                # Read and split along newline (\n)
                log_file = f.read().splitlines()
        except UnicodeDecodeError: # ANSI encoding
            with open(log, encoding='gbk') as f:
                # Read and split along newline (\n)
                log_file = f.read().splitlines()
    # Concatenate all log files into one list
    else:
        log_file = []
        for l in log:
            with open(l) as f:
                log_file.append(f.read().splitlines())

    # Remove titles (lines without equality sign)
    extras = []
    # Find extra lines
    for line in range(len(log_file)):
        if '=' not in log_file[line]:
            extras.append(log_file[line])
    # Remove extra lines
    for e in extras:
        log_file.remove(e)

    # Split from the first "=" sign
    parameter, value = [], []
    for line in log_file:
        parts = line.split('=', 1)
        parameter.append(parts[0])
        value.append(parts[1:][0])

    # Compile dictionary
    log_file = dict(zip(parameter, value))

    return log_file


def calculate_mean_std(array, rgb=False):
    """
    Calculates mean and standard deviation from array.
    :param array:
    :param rgb:
    :return:
    """
    mean = torch.Tensor([np.mean(array) / 255])
    std = torch.Tensor([np.std(array) / 255])

    if rgb:
        mean = np.repeat(mean, 3, axis=-1)
        std = np.repeat(std, 3, axis=-1)
    return mean, std


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


def load(path, axis=(0, 1, 2), n_jobs=12, rgb=False, dicom=False):
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
        if file.endswith('.png') or file.endswith('.bmp') or file.endswith('.tif') \
                or file.endswith('.dcm') or file.endswith('.ima') or dicom:
            try:
                if file.endswith('.dcm') or file.endswith('.ima') or dicom:
                    newlist.append(file)
                    dicom = True
                    continue

                int(file[-7:-4])

                # Do not load files with different prefix into the stack
                if len(newlist) != 0 and file.rsplit('_', 1)[0] != newlist[-1].rsplit('_', 1)[0]:
                    break

                newlist.append(file)
            except ValueError:
                continue
    files = newlist[:]  # replace list
    # Load images
    if dicom:
        data = Parallel(n_jobs=n_jobs)(delayed(read_image_dicom)(path, file) for file in files)
    elif rgb:
        data = Parallel(n_jobs=n_jobs)(delayed(read_image_rgb)(path, file) for file in files)
    else:
        data = Parallel(n_jobs=n_jobs)(delayed(read_image_gray)(path, file) for file in files)

    data = np.array(data)

    if len(data.shape) == 1:
        Warning('Image dimensions are not consistent! Returning a list of images.')
        return data, files

    # Zero the data (remove negative values from HU units)
    if np.min(data) < 0:
        data -= np.min(data)

    # Transpose array
    if axis != (0, 1, 2) and rgb and data.ndim == 4:
        return np.transpose(data, axis + (3,)), files
    elif axis != (0, 1, 2):
        data = np.transpose(data, axis)
        return data, files
    return data, files


def read_image_gray(path, file):
    """Reads image from given path."""
    # Image
    f = os.path.join(path, file)
    image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    #image = cv2.imread(f, -1)  # Might read a 3-channel image
    return image


def read_image_dicom(path, file):
    """Reads image from given path."""
    # Image
    f = os.path.join(path, file)
    image = dcmread(f)

    return apply_modality_lut(image.pixel_array, image)
    #return image.SOPInstanceUID


def read_image_rgb(path, file):
    """Reads image from given path."""
    # Image
    f = os.path.join(path, file)
    image = cv2.imread(f, cv2.IMREAD_COLOR)

    return image


def save(path, file_name, data, n_jobs=12, dtype='.png', verbose=True, template='../../BoneEnhance/template.dcm'):
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
    dtype : str
        File name extension.
    verbose : bool
        Whether to show progress bar during saving.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    nfiles = np.shape(data)[2]

    # Save bool as uint8, 255
    if data[0, 0, 0].dtype is bool:
        data = data * 255

    # 8- or 16-bit
    if dtype == '.tif' or dtype == '.tiff':
        n_bits = 'uint16'
    else:
        n_bits = 'uint8'

    # Save as dicom or image
    if dtype == '.dcm' or dtype == '':
        #writer = write_image_dicom
        writer = write_dicom_template
        # Read template image
        dcm_template = dcmread(template)
        # Save as default
        writer = partial(writer, template=dcm_template)
    else:
        writer = cv2.imwrite

    # Parallel saving (nonparallel if n_jobs = 1)
    if type(file_name) is list:
        Parallel(n_jobs=n_jobs)(delayed(writer)
                                (path + '/' + file_name[k][:-4] + dtype,
                                 data[:, :, k].astype(n_bits))
                                for k in tqdm(range(nfiles), 'Saving dataset', disable=not verbose))

    else:
        Parallel(n_jobs=n_jobs)(delayed(writer)
                                (path + '/' + file_name + '_' + str(k).zfill(8) + dtype,
                                 data[:, :, k].astype(n_bits))
                                for k in tqdm(range(nfiles), 'Saving dataset', disable=not verbose))


def write_image_dicom(path, image, res=0.05):
    # Create a DICOM dataset with filler values (except for the image file)
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = UID('1.2.840.10008.5.1.4.1.1.2')
    file_meta.MediaStorageSOPInstanceUID = UID("1.2.3.123")
    file_meta.ImplementationClassUID = UID("1.2.3.4")
    file_meta.TransferSyntaxUID = UID('1.2.840.10008.1.2')  # Little endian implicit

    ds = FileDataset(path, Dataset(), file_meta=file_meta, preamble=b"\0" * 128)

    # Dataset metadata
    ds.PatientName = "Bone Enhance"
    ds.PatientID = "240322-1111"
    ds.PatientBirthDate = datetime.now().strftime('%Y%m%d')

    # Set creation date/time
    dt = datetime.now()
    ds.ContentDate = dt.strftime('%Y%m%d')
    timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
    ds.ContentTime = timeStr

    # Set the transfer syntax
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    ds.Modality = 'CT'
    ds.PixelSpacing = [res, res]
    ds.SliceThickness = res

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.HighBit = 7
    ds.BitsStored = 8
    ds.BitsAllocated = 8
    ds.Columns = image.shape[1]
    ds.Rows = image.shape[0]
    ds.WindowCenter = 1060

    # Set the slice position
    slice_idx = str.rsplit(path, '_', 1)[1]  # Slice idx separated by _
    slice_idx = int(os.path.splitext(slice_idx)[0])  # Remove extension

    ds.file_meta.MediaStorageSOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID[:-3] + str(slice_idx + 1)
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.ImagePositionPatient = [0, 0, np.round(res * slice_idx, 3)]

    # Add the image data to dicom file
    ds.PixelData = image.astype('uint8').tobytes()
    ds[0x7fe00010].maxBytesToDisplay = 8
    ds[0x7fe00010].VR = 'OW'
    ds.SmallestImagePixelValue = np.min(image).astype('uint8')
    ds[0x00280106].VR = 'US'
    ds.LargestImagePixelValue = np.max(image).astype('uint8')
    ds[0x00280107].VR = 'US'

    # Write dicom file to path
    dcmwrite(path, ds)


def write_dicom_template(path, image, template, res=0.05):
    """
    Write a dicom image based on an existing image template for metadata.
    :param path:
    :param image:
    :param template:
    :param res: Resolution of the reconstructed image (assumed isotropic)
    :return:
    """
    # Make sure the template is not altered
    ds = deepcopy(template)

    # Set creation date/time
    dt = datetime.now()
    ds.ContentDate = dt.strftime('%Y%m%d')
    timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
    ds.ContentTime = timeStr

    # Image metadata

    ds.PixelSpacing = [res, res]
    ds.SliceThickness = res
    #ds.SpacingBetweenSlices = res
    ds.HighBit = 7
    ds.BitsStored = 8  # Standard requires 12-16
    ds.BitsAllocated = 8  # Standard requires 16 for CT
    ds.Columns = image.shape[1]
    ds.Rows = image.shape[0]

    # Add the image data to dicom file

    ds.PixelData = image.astype('uint8').tobytes()
    ds[0x7fe00010].maxBytesToDisplay = 8
    ds[0x7fe00010].VR = 'OW'
    ds.SmallestImagePixelValue = np.min(image).astype('uint8')
    ds[0x00280106].VR = 'US'
    ds.LargestImagePixelValue = np.max(image).astype('uint8')
    ds[0x00280107].VR = 'US'
    ds.WindowCenter = 500  # Display settings (-1000 to 2000)
    ds.WindowWidth = 3000
    ds.RescaleIntercept = -1000  # How the output is rescaled (-1000 to 2000)
    ds.RescaleSlope = 13.5

    # Set the slice position
    diff = 1  # Difference in numbering?
    slice_idx = str.rsplit(path, '_', 1)[1]  # Slice idx separated by _
    slice_idx = int(os.path.splitext(slice_idx)[0])  # Remove extension
    hundreds = (slice_idx - diff) // 100  # UID changes every hundred slices
    tens = (slice_idx - diff) // 10  # UID changes every ten slices

    # Patient name
    ds.PatientName = str.rsplit(path, '/')[-2]  # Folder name is also sample/patient name
    ds.PatientID = dt.strftime('%d%m') + dt.strftime('%Y')[2:] + '-' + str(ds.PatientName)  # Savedata-samplename

    # UID
    uid = ds.SOPInstanceUID.rsplit('.', 2)
    uid[1] = str.rsplit(path, '/')[-2]  # Folder name is also sample/patient name
    sample_id = [str(ord(x)) for x in uid[1]]  # Convert text to numbers
    uid[1] = ''.join(sample_id)
    ms_uid = ds.file_meta.MediaStorageSOPInstanceUID.rsplit('.', 2)
    #ds.SOPInstanceUID = f'{uid[0]}.{int(uid[1]) + hundreds}.{str(slice_idx + 1).zfill(5)}'
    ds.SOPInstanceUID = f'{uid[0]}.{int(uid[1])}.{str(slice_idx + 1).zfill(5)}'
    ds.FrameOfReferenceUID = f'{uid[0]}.{int(uid[1])}1'
    ds.StudyInstanceUID = f'{uid[0]}.{int(uid[1])}'
    ds.SeriesInstanceUID = f'{uid[0]}.{int(uid[1])}2'
    ds.InstanceNumber = slice_idx + 1
    #ds.file_meta.MediaStorageSOPInstanceUID = f'{ms_uid[0]}.{int(ms_uid[1]) + hundreds}.{str(slice_idx + 1).zfill(5)}'
    ds.file_meta.MediaStorageSOPInstanceUID = f'{ms_uid[0]}.{int(ms_uid[1])}.{str(slice_idx + 1).zfill(5)}'

    #del ds.SOPInstanceUID
    #del ds.file_meta.MediaStorageSOPInstanceUID

    # Position
    ds.ImagePositionPatient[2] = np.round(res * (slice_idx + diff), 3)
    ds.SliceLocation = np.round(res * (slice_idx + diff), 3)

    # Write dicom file to path
    dcmwrite(path, ds)


def save_images(path, file_names, data, n_jobs=12, dicom=False):
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
    if dicom:
        Parallel(n_jobs=n_jobs)(delayed(dcmwrite)
                                (path + '/' + file_names[k][:-4] + '.dcm', data[k][:, :])
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


def print_orthogonal(data, mask=None, invert=True, res=200, title=None, cbar=True, cmap='gray', savepath=None, scale_factor=2):
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
    #cmap2 = 'Dark2_r'
    cmap2 = 'gray'
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

    ax1 = fig.add_subplot(221)
    cax1 = ax1.imshow(images[0], cmap=cmap2)
    if not isinstance(images[0][0, 0], np.bool_):  # Check for boolean image
        cbar1 = fig.colorbar(cax1, ticks=[np.min(images[0]), np.max(images[0])], orientation='horizontal')
        cbar1.solids.set_edgecolor("face")
    if subtitles is not None:
        plt.title(subtitles[0])
    if masks is not None:
        m = masks[0]
        ax1.imshow(np.ma.masked_array(m, m == 0), cmap=cmap, alpha=alpha)

    ax2 = fig.add_subplot(222)
    cax2 = ax2.imshow(images[1], cmap=cmap2)
    if not isinstance(images[1][0, 0], np.bool_):
        cbar2 = fig.colorbar(cax2, ticks=[np.min(images[1]), np.max(images[1])], orientation='horizontal')
        cbar2.solids.set_edgecolor("face")
    if subtitles is not None:
        plt.title(subtitles[1])
    if masks is not None:
        m = masks[1]
        ax2.imshow(np.ma.masked_array(m, m == 0), cmap=cmap, alpha=alpha)

    ax3 = fig.add_subplot(223)
    cax3 = ax3.imshow(images[2], cmap=cmap2)
    if not isinstance(images[2][0, 0], np.bool_):
        cbar3 = fig.colorbar(cax3, ticks=[np.min(images[2]), np.max(images[2])], orientation='horizontal')
        cbar3.solids.set_edgecolor("face")
    if subtitles is not None:
        plt.title(subtitles[2])
    if masks is not None:
        m = masks[2]
        ax3.imshow(np.ma.masked_array(m, m == 0), cmap=cmap, alpha=alpha)

    ax4 = fig.add_subplot(224)
    cax4 = ax4.imshow(images[3], cmap=cmap2)
    if not isinstance(images[3][0, 0], np.bool_):
        cbar4 = fig.colorbar(cax4, ticks=[np.min(images[3]), np.max(images[3])], orientation='horizontal')
        cbar4.solids.set_edgecolor("face")
    if subtitles is not None:
        plt.title(subtitles[3])
    if masks is not None:
        m = masks[3]
        ax4.imshow(np.ma.masked_array(m, m == 0), cmap=cmap, alpha=alpha)

    # Save or show
    if save_path is not None and sample is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        plt.tight_layout()  # Make sure that axes are not overlapping
        fig.savefig(save_path + sample, transparent=transparent)
        plt.close(fig)
    else:
        plt.show()


def convert_3d_to_random_2d(stack, mag=None):
    """If two stacks, input smaller first!"""
    axis = random.choice([0, 1, 2])

    if isinstance(stack, (list, tuple)):
        dim = stack[0].shape[axis]
        slice = random.randint(0, dim - 1)
        if mag is None:
            mag = stack[1].shape[axis] // dim

        if axis == 0:
            return [stack[0][slice, :, :], stack[1][slice * mag, :, :]]
        elif axis == 1:
            return [stack[0][:, slice, :], stack[1][:, slice * mag, :]]
        else:
            return [stack[0][:, :, slice], stack[1][:, :, slice * mag]]
    else:
        dim = stack.shape[axis]
        slice = random.randint(0, dim - 1)

        if axis == 0:
            return stack[slice, :, :]
        elif axis == 1:
            return stack[:, slice, :]
        else:
            return stack[:, :, slice]


def convert_3d_tensor_to_random_2d(stack, mag=None):
    """If two stacks, input smaller first!"""
    axis = random.choice([0, 1, 2])

    if isinstance(stack, (list, tuple)):
        dim = stack[0].size(axis)
        slice = random.randint(0, dim - 1)
        if mag is None:
            mag = stack[1].size(axis) // dim

        if axis == 0:
            return [stack[0][:, :, slice, :, :], stack[1][:, :, slice * mag, :, :]]
        elif axis == 1:
            return [stack[0][:, :, :, slice, :], stack[1][:, :, :, slice * mag, :]]
        else:
            return [stack[0][:, :, :, :, slice], stack[1][:, :, :, :, slice * mag]]
    else:
        dim = stack.size(axis)
        slice = random.randint(0, dim - 1)

        if axis == 0:
            return stack[:, :, slice, :, :]
        elif axis == 1:
            return stack[:, :, :, slice, :]
        else:
            return stack[:, :, :, :, slice]
