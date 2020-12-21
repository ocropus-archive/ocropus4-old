FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime
ENV LC_ALL=C
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -qqy curl git strace
RUN ln -s /opt/conda/bin/python3 /usr/bin/python3
RUN ln -s /opt/conda/bin/conda /usr/bin/conda
RUN conda install pip
RUN ln -s /opt/conda/bin/pip /usr/bin/pip3
RUN pip3 install numpy scipy
RUN pip3 install git+git://github.com/tmbdev/webdataset

RUN apt-get install -qqy gcc build-essential vim-nox
RUN pip3 install editdistance

COPY ocropus4.tar /tmp
RUN mkdir /tmp/ocropus4 && cd /tmp/ocropus4 && \
    tar xvf ../ocropus4.tar && \
    pip3 install -r requirements.txt && \
    python3 setup.py install
