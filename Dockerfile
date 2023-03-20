FROM walkerlab/pytorch-jupyter:cuda-11.8.0-pytorch-1.13.0-torchvision-0.14.0-torchaudio-0.13.0-ubuntu-22.04
COPY . /src/gensn

# RUN python3 -m pip install --upgrade pip &&\
#     python3 -m pip install -e /src/gensn
# TODO: turn this into dev requirement
RUN pip3 install pytest

RUN pip3 install -e /src/gensn