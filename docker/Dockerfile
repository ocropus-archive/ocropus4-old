FROM ocropus4-base

RUN pip3 install -e git+git://github.com/tmbdev/webdataset.git@v2#egg=webdataset
RUN pip3 install -e git+git://github.com/NVlabs/ocrodeg.git#egg=ocrodeg
RUN pip3 install -e git+git://github.com/NVlabs/tensorcom.git#egg=tensorcom
RUN pip3 install -e git+git://github.com/tmbdev/tarproc.git#egg=tarproc
RUN pip3 install -e git+git://github.com/tmbdev/torchmore.git#egg=torchmore

COPY ocropus4.tar /tmp
RUN mkdir /tmp/ocropus4 && cd /tmp/ocropus4 && \
    tar xvf ../ocropus4.tar && \
    pip3 install -r requirements.txt && \
    python3 setup.py install
RUN pip3 install -e git+git://github.com/tmbdev/webdataset.git@v2#egg=webdataset
