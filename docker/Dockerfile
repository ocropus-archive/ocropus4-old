FROM ocropus4-base

COPY ocropus4.tar /tmp
RUN mkdir /tmp/ocropus4 && cd /tmp/ocropus4 && \
    tar xvf ../ocropus4.tar && \
    pip3 install -r requirements.txt && \
    python3 setup.py install
