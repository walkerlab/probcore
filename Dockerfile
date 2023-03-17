#FROM python:3.12.0a6-bullseye 
FROM walkerlab/pytorch-jupyter:cuda-11.7.1-pytorch-1.13.1-torchvision-0.13.0-torchaudio-0.11.0-ubuntu-20.04

COPY . /src/gensn

RUN python3 -m pip install --upgrade pip &&\
    python3 -m pip install -e /src/gensn
