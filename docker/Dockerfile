FROM ocropus4-base

COPY ocropus4.tar /tmp
RUN mkdir /tmp/ocropus4 && cd /tmp/ocropus4 && \
    tar xvf ../ocropus4.tar && \
    python3 setup.py install && \
    pip3 install -r -U requirements.txt && \
    true
