FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
ENV LC_ALL=C
ENV DEBIAN_FRONTEND=noninteractive

# Basic Packages
RUN apt-get -qqy update
RUN apt-get install -qqy curl daemon wamerican-huge
RUN apt-get install -qqy python3 python3-scipy python3-matplotlib python3-pip graphviz
RUN apt-get install -qqy wamerican-huge

# Python3 PIP Packages
RUN pip3 install --no-cache-dir --upgrade virtualenv
RUN pip3 install --no-cache-dir --upgrade six future msgpack pyzmq simplejson braceexpand natsort

# PyTorch
#RUN pip3 install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install torch torchvision

# Jupyter
RUN pip3 install --no-cache-dir --upgrade ipython jupyter bash_kernel pydot rise
RUN jupyter-nbextension install rise --py --sys-prefix
RUN python3 -m bash_kernel.install

# additional packages
RUN apt-get install -qqy git
RUN pip3 install --no-cache-dir --upgrade git+git://github.com/tmbdev/hocr-tools
RUN pip3 install --no-cache-dir --upgrade git+git://github.com/tmbdev/webdataset
RUN pip3 install --no-cache-dir --upgrade git+git://github.com/tmbdev/torchmore
RUN pip3 install --no-cache-dir --upgrade git+git://github.com/NVlabs/tensorcom
RUN pip3 install --no-cache-dir --upgrade git+git://github.com/NVlabs/torchtrainers
RUN pip3 install --no-cache-dir --upgrade editdistance
