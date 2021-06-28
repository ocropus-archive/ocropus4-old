FROM ubuntu:20.04
RUN date
RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository -y ppa:alex-p/tesseract-ocr
RUN apt-get update && apt-get install -y tesseract-ocr-all 

RUN apt-get -qqy update && apt-get install -qqy \
        curl \
        gcc \
        apt-transport-https \
        lsb-release \
        openssh-client \
        git \
        gnupg

RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz
RUN mkdir -p /usr/local/gcloud && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz
RUN /usr/local/gcloud/google-cloud-sdk/install.sh
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

RUN apt-get install -qqy iputils-ping
RUN apt-get install -qqy dnsutils
RUN apt-get install -qqy parallel
RUN apt-get install -qqy graphicsmagick imagemagick

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ="America/Los Angeles"
RUN echo $TZ > /etc/timezone
RUN rm -f /etc/localtime && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime || true
RUN apt-get install -qqy tzdata

RUN apt-get install -qqy python3-dev python3-setuptools python3-pip
RUN pip3 install lxml
RUN pip3 install reportlab
RUN pip3 install Pillow
RUN pip3 install beautifulsoup4
RUN pip3 install numpy
RUN pip3 install scipy
RUN pip3 install matplotlib
RUN pip3 install hocr-tools
RUN pip3 install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

ADD ocropus4.tar.gz /tmp/
RUN cd /tmp/ocropus4 && pip3 install -r requirements.txt
RUN cd /tmp/ocropus4 && pip3 install -r requirements.dev.txt
RUN cd /tmp/ocropus4 && python3 setup.py install

COPY *.sh /usr/local/bin/
RUN chmod 755 /usr/local/bin/*.sh

RUN mkdir /work
WORKDIR /work
