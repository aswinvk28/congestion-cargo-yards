FROM openvino/ubuntu18_data_dev:latest

USER root
RUN apt-get update
RUN apt-get install curl -y
RUN apt-get install build-essential g++ gcc make cmake -y
RUN apt-get install libgtest-dev -y
RUN mkdir /home/project
RUN apt-get install python3 python3-pip -y
RUN pip3 install tensorflow scikit-build python-dotenv opencv-python
RUN pip3 install pycocotools
RUN apt update && apt install -y libsm6 libxext6 libfontconfig1 libxrender1 -y
RUN pip3 install h5py==2.9.0
RUN apt-get install git -y
RUN pip3 install paho-mqtt
RUN pip3 install tensorflow_constrained_optimization
RUN git config --global user.email "aswinkvj@gmail.com"
RUN git config --global user.name "Aswin Vijayakumar"

RUN pip3 install rtsp

WORKDIR /home/project

ENTRYPOINT "/bin/bash"