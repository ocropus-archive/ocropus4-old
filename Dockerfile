FROM ocropus4-base
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get install -qqy rename

WORKDIR /work
RUN pip3 install git+git://github.com/tmbdev/tarproc.git#egg=tarproc
RUN . venv/bin/activate && \
    pip3 install git+git://github.com/tmbdev/webdataset.git#egg=webdataset && \
    pip3 install git+git://github.com/tmbdev/tarproc.git#egg=tarproc
COPY ocropus4/ ocropus4
RUN . venv/bin/activate && \
    pip3 install ./ocropus4
