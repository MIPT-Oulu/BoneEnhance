import os
import numpy as np
from pathlib import Path
import cv2
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from bone_enhance.training.session import init_experiment
from bone_enhance.utilities.main import load, save, print_orthogonal, read_image_gray
from scipy.ndimage import zoom, median_filter
from skimage.transform import resize
from imageio import imread
from astra.data2d import create, get
from astra import astra_dict, algorithm
from astra.creators import create_proj_geom, create_backprojection3d_gpu, create_vol_geom, create_sino3d_gpu, create_backprojection, create_projector, create_sino



if __name__ == "__main__":
    images_rec = Path('/media/santeri/Transcend/train_data/reconstruction/KP03-L6-4MD2')
    images_sino = Path('/media/santeri/Transcend/train_data/sinograms_old/KP03-L6-4MD2')
    #images_sino = Path('/media/santeri/Transcend/DL reconstruction data/KP03-L6-4MD2/')

    images_save = Path('/media/santeri/Transcend/train_data/results')
    images_save.mkdir(exist_ok=True)

    # Imaging parameters
    rows = 1344
    cols = 2016

    cols = 1364

    pixel_size = 3.2 / 1000  # µm
    obj_source = 48.31606  # mm
    det_source = 271.75368

    pixel_size = 1.5
    obj_source = 2000  # mm
    det_source = 2600

    num_of_projections = 2000
    #num_of_projections = 180
    #angles = np.linspace(0, 2 * np.pi, num=num_of_projections, endpoint=False)
    angles = np.linspace(0, np.pi, num=num_of_projections, endpoint=False)

    create_sinogram = True
    anim = False


    #data, files = load(str(images_rec), rgb=False, axis=(1, 2, 0))
    data = imread(str(images_sino / 'KP03-L6-4MD2_00000426.tif')).astype(float)
    data_ref = read_image_gray(images_rec, 'KP03-L6-4MD2__rec00000426.bmp')[:1364, :1364]
    # Scale the 16-bit projection
    data /= 255
    #vol = create_vol_geom(rows, rows)

    if create_sinogram:
        # 2D reconstruction geometry
        geometry_vol = create_vol_geom(cols, cols)
        id_vol = create('-vol', geometry_vol, data=data_ref)

        # 2D projection geometry
        geometry = create_proj_geom('fanflat', pixel_size, cols, angles, det_source, det_source - obj_source)
        #geometry = create_proj_geom('parallel', pixel_size, cols, angles, det_source, det_source - obj_source)

        geom_id = create_projector('strip_fanflat', geometry, geometry_vol)
        #geom_id = create_projector('line', geometry, geometry_vol)
        id, sinogram = create_sino(data_ref, geom_id)
    else:

        # 2D projection geometry
        #geometry = create_proj_geom('fanflat', pixel_size, cols, angles, det_source, det_source - obj_source)
        geometry = create_proj_geom('fanflat', pixel_size, cols, angles, det_source, 0)
        id = create('-sino', geometry, data.transpose())

        # 2D reconstruction geometry
        geometry_vol = create_vol_geom(cols, cols)
        id_vol = create('-vol', geometry_vol, data=0)

        # Algorithm
        alg = astra_dict('FBP_CUDA')
        alg['ProjectionDataId'] = id
        alg['ReconstructionDataId'] = id_vol
        alg_id = algorithm.create(alg)

        # Run recon
        algorithm.run(alg_id)
        rec = get(id_vol)

        # scale
        rec[rec < 0] = 0
        rec /= np.max(rec)
        rec = np.round(rec * 255).astype(np.uint8)

        #geometry = create_proj_geom('cone', pixel_size, pixel_size, rows, cols, angles, det_source, det_source - obj_source)
        #id = create_projector('linear', geometry, vol)
        #rec = create_backprojection(data, id)

    #plt.imshow(data, cmap='gray')
    #plt.show()

    if anim:
        fig, ax = plt.subplots()

        line, = ax.plot(sinogram[0, :], label='0$^\circ$')
        line.set_color('black')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 24
        plt.xlabel('Detector')
        plt.ylabel('Intensity')
        plt.ylim([0, 1.1 * np.max(sinogram)])
        plt.xticks([])
        plt.yticks([])
        L = plt.legend(loc='upper left')

        def animate(i):
            label = '%d$^\circ$' % (i)
            line.set_ydata(sinogram[i, :])  # update the data.
            L.get_texts()[0].set_text(label)
            return line,


        ani = animation.FuncAnimation(
            fig, animate, interval=1, blit=True, save_count=num_of_projections)

        # To save the animation, use e.g.
        #
        ani.save('/home/santeri/Astra_figure/sinogram5.mp4', fps=15, writer='imagemagick')
        #
        # or
        #
        #writer = animation.FFMpegWriter(
        #     fps=6, metadata=dict(artist='Santeri Rytky'), bitrate=1800)
        #ani.save("/home/santeri/Astra_figure/sinogram.mp4", writer=writer)

    if create_sinogram:
        fig = plt.figure(figsize=(4, 5))
        #plt.tight_layout()
        plt.imshow(sinogram, cmap='gray')  # , aspect='auto')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 24
        np = num_of_projections
        #plt.yticks([0, np // 4 - 1, np // 2 - 1, np * 3 // 4 - 1, np - 1], [0, 90, 180, 270, 360])
        plt.yticks([0, np // 4 - 1, np // 2 - 1, np * 3 // 4 - 1, np - 1], ['0$^\circ$', '45$^\circ$', '90$^\circ$', '135$^\circ$', '180$^\circ$'])
        plt.xlabel('Detector')
        plt.ylabel('Angle')
        plt.xticks([])
        plt.savefig('/home/santeri/Astra_figure/sino_fan.tif', dpi=300)
        plt.show()

        # Projection plots

        plt.plot(sinogram[0, :], '--', label='0 degrees')
        plt.xticks([])
        plt.yticks([])
        plt.plot(sinogram[num_of_projections // 4, :], '-', label='90 degrees')
        plt.plot(sinogram[num_of_projections // 8, :], ':', label='45 degrees')
        plt.plot(sinogram[num_of_projections // 8 * 3, :], ':', label='135 degrees')
        plt.legend()
        plt.show()

        # Summed image
        plt.plot(np.sum(data_ref, 0), '--', label='0 degrees')
        plt.plot(np.sum(data_ref, 1), '-', label='90 degrees')
        plt.title('Sum image')
        plt.legend()
        plt.show()
    else:
        plt.imshow(rec, cmap='gray')
        plt.show()
    plt.imshow(data_ref, cmap='gray')
    plt.show()




