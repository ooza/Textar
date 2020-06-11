FROM ubuntu:18.04
# set noninteractive installation
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y install tzdata


MAINTAINER https://github.com/ooza/Textar
LABEL version="1.0"

## Pyton installation ##
RUN apt-get update && apt-get install -y python3.7 python3-pip git

RUN mkdir -p /home/textar/

COPY requirements.txt .

#RUN apt-get install libpcap-dev libpq-dev


RUN pip3 install -r requirements.txt

WORKDIR /home/textar/

COPY . .

RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev

WORKDIR /home/textar/artext_detection

RUN cd utils/bbox && chmod +x make.sh && ./make.sh

WORKDIR /home/textar/
