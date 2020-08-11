FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

RUN apt-get -y update
RUN apt-get install -y --no-install-recommends \
      build-essential \
      git \
      libgoogle-glog-dev \
      libgtest-dev \
      libiomp-dev \
      libleveldb-dev \
      liblmdb-dev \
      libopencv-dev \
      libopenmpi-dev \
      libsnappy-dev \
      libprotobuf-dev \
      openmpi-bin \
      openmpi-doc \
      protobuf-compiler \
      python-dev \
      python-pip   
RUN pip install --upgrade pip
RUN pip install setuptools                       
RUN pip install --user \
      future \
      numpy \
      protobuf \
      typing \
      hypothesis
RUN apt-get install -y --no-install-recommends \
      libgflags-dev \
      cmake

RUN git clone --branch master --recursive https://github.com/pytorch/pytorch.git
RUN pip install typing pyyaml
WORKDIR /pytorch
RUN git submodule update --init --recursive
RUN python setup.py install

RUN git clone https://github.com/facebookresearch/detectron /detectron

# Install Python dependencies
RUN pip install -U pip
RUN pip install -r /detectron/requirements.txt

# Install the COCO API
RUN git clone https://github.com/cocodataset/cocoapi.git /cocoapi
WORKDIR /cocoapi/PythonAPI

ENV PYTHONPATH /usr/local
ENV Caffe2_DIR=/usr/local/lib/python2.7/dist-packages/torch/share/cmake/Caffe2/
ENV PYTHONPATH=${PYTHONPATH}:/pytorch/build
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

ENV LD_LIBRARY_PATH=/usr/local/lib/python2.7/dist-packages/torch/lib/:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=/usr/local/lib/python2.7/dist-packages/torch/lib/:${LIBRARY_PATH}

ENV C_INCLUDE_PATH=/usr/local/lib/python2.7/dist-packages/torch/lib/include/:${C_INCLUDE_PATH}
ENV CPLUS_INCLUDE_PATH=/usr/local/lib/python2.7/dist-packages/torch/lib/include/:${CPLUS_INCLUDE_PATH}

ENV C_INCLUDE_PATH=/pytorch/:${C_INCLUDE_PATH}
ENV CPLUS_INCLUDE_PATH=/pytorch/:${CPLUS_INCLUDE_PATH}

ENV C_INCLUDE_PATH=/pytorch/build/:${C_INCLUDE_PATH}
ENV CPLUS_INCLUDE_PATH=/pytorch/build/:${CPLUS_INCLUDE_PATH}

ENV C_INCLUDE_PATH=/pytorch/torch/lib/include/:${C_INCLUDE_PATH}
ENV CPLUS_INCLUDE_PATH=/pytorch/torch/lib/include/:${CPLUS_INCLUDE_PATH}

RUN make install

WORKDIR /detectron

RUN make
#RUN make ops

RUN apt-get -y update \
    && apt-get -y install \
        wget \
        software-properties-common

# VideoPose3d
# get ffmpeg
RUN add-apt-repository ppa:mc3man/trusty-media  
RUN apt-get update  
RUN apt-get install ffmpeg  
RUN apt-get install frei0r-plugins  

RUN git clone https://github.com/facebookresearch/VideoPose3D.git /VideoPose3D
RUN mkdir /VideoPose3D/checkpoint && cd /VideoPose3D/checkpoint
RUN wget https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin

RUN cp /VideoPose3D/inference/infer_video.py /detectron/tools/