FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3.7\
    libsm6 \
    libxext6 \
    libxrender-dev \
    zlib1g-dev \
    libjpeg-dev \
    vim \
    git \
    bash
    #python3.7.9 \
    #python3-pip

# Python package management and basic dependencies
#RUN apt-get install -y curl python3.7 python3.7-dev python3.7-distutils

# Register the version in alternatives
#RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1

# Set python 3 as the default python
#RUN update-alternatives --set python /usr/bin/python3.7

# Upgrade pip to latest version
#RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
#    python get-pip.py --force-reinstall && \
#    rm get-pip.py

WORKDIR /home/BoneEnhance

RUN git clone https://github.com/MIPT-Oulu/BoneEnhance.git@development

# Install dependencies
RUN pip3 install --no-cache-dir --trusted-host pypi.python.org h5py==2.8.0 \
    deep-pipeline==0.2.5 \
    git+https://github.com/MIPT-Oulu/Collagen.git@collagen-super-resolution \
    git+https://github.com/imedslab/solt.git \
    h5py==2.10.0 \
    omegaconf==2.0.0 \
    opencv-python==4.3.0.36 \
    opencv-python-headless==4.3.0.36 \
    pillow==6.1.0 \
    pretrainedmodels==0.7.4 \
    pydicom \
    pytorch-toolbelt==0.3.2 \
    segmentation-models-pytorch==0.1.0 \
    solt==0.1.9 \
    tensorboard==2.3.0 \
    tensorboardx==2.1 \
    termcolor==1.1.0 \
    torchcontrib==0.0.2 \
    torchfile==0.1.0 \
    torchnet==0.0.4 \
    torchvision==0.7.0 \
    vtk==9.0.1
