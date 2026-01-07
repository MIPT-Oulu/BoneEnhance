# Nvidia CUDA image with Ubuntu 20.04 LTS
FROM nvidia/cudagl:11.2.1-devel-ubuntu20.04

# Conda environment

# Install package prerequisite software with auto-confirmation (-y)
# ppa:deadsnakes allows installation of old Python versions that would not be available on current Ubuntu
RUN apt-get update  \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get install -y libglib2.0-0 \
    python3-pip \
    libglib2.0-0 \
    zlib1g-dev \
    libjpeg-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    wget \
    curl \
    gnupg \
    ca-certificates \
    software-properties-common \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3.7-venv \
    && rm -rf /var/lib/apt/lists/*

# Move to home directory
WORKDIR /home

# Clone BoneEnhance (development branch)
# TODO freeze commit
RUN git clone --branch development https://github.com/MIPT-Oulu/BoneEnhance.git
#RUN git clone --branch collagen-super-resolution https://github.com/MIPT-Oulu/Collagen.git


WORKDIR /home/BoneEnhance

# Conda environment with Python 3.7
RUN apt-get install -y wget \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh
# Add Conda to path variable
ENV PATH=/opt/conda/bin:$PATH

# Local files TODO
COPY requirements_full.txt /home/BoneEnhance/requirements_full.txt
#COPY environment.yml /home/BoneEnhance/environment.yml

# Accept Anaconda terms of service
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda env create -n boneenhance -f environment.yml
#RUN conda create -n boneenhance python=3.7 cudatoolkit=10.1 -y
RUN conda run -n boneenhance pip install --no-cache-dir "protobuf==3.15.2"

# Downgrade packages for compatibility
#RUN conda run -n boneenhance pip install "protobuf<3.20" --force-reinstall
WORKDIR /home
RUN git clone --branch scalable-augmentations https://github.com/sarytky/solt.git \
    && cd solt \
    && pip install -e .

# Add Python to virtual environment
#RUN python3.7 -m venv /venv
#ENV PATH=/venv/bin:$PATH

# Install Python dependencies
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install pypi packages
#RUN pip install -r requirements_full.txt \
RUN conda run -n boneenhance pip install --trusted-host pypi.python.org \
    jupyterlab \
    notebook \
    ipykernel \
    ipywidgets==7.7.1 \
    numpy==1.21.6 \
    jupyterlab_widgets

# Downgrade protobuf
RUN conda run -n boneenhance pip install --force-reinstall "protobuf==3.9.2"

# Register boneenhance to Jupyter kernels
RUN conda run -n boneenhance python -m ipykernel install --user --name boneenhance --display-name "Python (boneenhance)"

# Copy in username file
WORKDIR /home
COPY users.txt /home/
COPY create_users.sh /home/
RUN ./create_users.sh && \
    rm users.txt create_users.sh

WORKDIR /home/BoneEnhance
RUN git reset --hard HEAD && git pull

# Ensure the user has required permissions
ARG GROUPNAME
ARG USERNAME
RUN mkdir -p /home/predictions \
    && chown -R $USERNAME:$GROUPNAME /home/predictions  \
    && chmod -R 770 /home/predictions \
    && chown -R $USERNAME:$GROUPNAME /home/BoneEnhance \
    && chmod -R 770 /home/BoneEnhance \
    && mkdir -p /home/santeri \
    && chown -R $USERNAME:$GROUPNAME /home/santeri \
    && chmod -R 770 /home/santeri \
    && mkdir -p /home/.local/share/jupyter \
    && chown -R $USERNAME:$GROUPNAME /home/.local/share/jupyter \
    && chmod -R 770 /home/.local/share/jupyter

ENV PYTHONPATH=/home/BoneEnhance
WORKDIR /home

# Mount position for data and snapshots
RUN mkdir Data
VOLUME ["/home/Data"]
RUN mkdir Workdir
VOLUME ["/home/Workdir"]

# Expose Jupyter port
EXPOSE 8888

# Activate environment by default
SHELL ["conda", "run", "-n", "boneenhance", "/bin/bash", "-c"]

#RUN conda run -n boneenhance python -c "import collagen; print(collagen.__version__)"

# Default command (can be overridden)
#CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
CMD ["conda", "run", "--no-capture-output", "-n", "boneenhance", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]


