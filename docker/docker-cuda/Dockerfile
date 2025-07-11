ARG BASE_IMAGE=ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.1.0-gpu-cuda12.9-cudnn9.9
FROM ${BASE_IMAGE}

ARG HTTP_PROXY=""
ARG PIP_INDEX=https://pypi.org/simple

ENV http_proxy="${HTTP_PROXY}"
ENV https_proxy="${HTTP_PROXY}"

RUN pip config set global.index-url "${PIP_INDEX}"

WORKDIR /root/
RUN git clone https://github.com/PaddlePaddle/ERNIE.git

WORKDIR /root/ERNIE
RUN pip install -r requirements/gpu/requirements.txt 
RUN pip install -e .

ENV http_proxy ""
ENV https_proxy ""

RUN pip config unset global.index-url
