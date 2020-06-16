FROM ubuntu:18.04
# set noninteractive installation
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y install tzdata


MAINTAINER https://github.com/ooza/Textar
LABEL version="1.0"

## Pyton installation ##
RUN apt-get update && apt-get install -y python3.7 python3-pip git

#RUN mkdir -p /input/

COPY requirements.txt /input/requirements.txt

#RUN apt-get install libpcap-dev libpq-dev


RUN pip3 install -r /input/requirements.txt

#WORKDIR /input/

COPY . /input/

RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev

#WORKDIR /input/artext_detection

RUN cd /input/artextDetection/utils/bbox && chmod +x make.sh && ./make.sh

WORKDIR /input/
