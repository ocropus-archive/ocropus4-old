#
# Copyright (c) 2017-2019 TBD. All rights reserved.
# This file is part of TBD (see TBD).
# See the LICENSE file for licensing terms (TBD).
#

import os
import pytest
import webdataset as wds
from ocropus import ocrorec
from ocropus import ocroseg
from ocropus import slog
from ocropus import nlbin
from ocropus import ocrobin
from ocropus import ocrorot
from ocropus import ocroscale
from ocropus import ocroskew

bucket = "pipe:curl -sL https://storage.googleapis.com/ocropus4-test"
mbucket = "pipe:curl -sL https://storage.googleapis.com/ocropus4-models"

def test_data():
    ds = wds.WebDataset(f"{bucket}/gsub-words-test.tar")
    next(iter(ds))


def test_ocrorec_pretrained(tmpdir):
    mname = "wordmodel.pth"
    assert 0 == os.system(f"curl -sL {mbucket}/{mname} > {tmpdir}/{mname}")
    ocrorec.recognize(f"{bucket}/gsub-words-test.tar",
                      model=f"{tmpdir}/{mname}", display=False, limit=3)


def test_ocroseg_pretrained(tmpdir):
    mname = "wsegmodel.pth"
    assert 0 == os.system(f"curl -sL {mbucket}/{mname} > {tmpdir}/{mname}")
    ocroseg.segment(f"{bucket}/gsub-test.tar",
                    model=f"{tmpdir}/{mname}", display=False, limit=3)


def test_ocrorec(tmpdir):
    ocrorec.train(f"{bucket}/gsub-words-test.tar",
                  log_to=f"{tmpdir}/ocrorec-train.sqlite3", ntrain=100)
    slog.getbest(f"{tmpdir}/ocrorec-train.sqlite3", f"{tmpdir}/ocrorec.pth")
    ocrorec.recognize(f"{bucket}/gsub-words-test.tar",
                      model=f"{tmpdir}/ocrorec.pth", display=False, limit=3)


def test_ocroseg(tmpdir):
    ocroseg.train(f"{bucket}/gsub-wseg-test.tar",
                  log_to=f"{tmpdir}/ocroseg-train.sqlite3", ntrain=100)
    slog.getbest(f"{tmpdir}/ocroseg-train.sqlite3", f"{tmpdir}/ocroseg.pth")
    ocroseg.predict(f"{bucket}/gsub-wseg-test.tar",
                    model=f"{tmpdir}/ocroseg.pth")
    ocroseg.predict(f"{bucket}/gsub-test.tar", model=f"{tmpdir}/ocroseg.pth")


def test_nlbin(tmpdir):
    nlbin.binarize(f"{bucket}/gsub-test.tar",
                   output=f"{tmpdir}/binarized.tar", maxrec=3)
    nlbin.binarize(f"{bucket}/gsub-test.tar",
                   output=f"{tmpdir}/binarized.tar", deskew=True, maxrec=3)


def test_ocrobin(tmpdir):
    ocrobin.train([f"{bucket}/gsub-bin-test.tar"], nsamples=10,
                  log_to=f"{tmpdir}/ocrobin-train.sqlite3")
    slog.getbest(f"{tmpdir}/ocrobin-train.sqlite3", f"{tmpdir}/ocrobin.pth")
    ocrobin.binarize(f"{bucket}/gsub-test.tar", iext="jpeg",
                     output=f"{tmpdir}/binarized.tar", model=f"{tmpdir}/ocrobin.pth", limit=10)


def test_ocrorot(tmpdir):
    ocrorot.train([f"{bucket}/gsub-bin-test.tar"], nsamples=10,
                  log_to=f"{tmpdir}/ocrorot-train.sqlite3")
    slog.getbest(f"{tmpdir}/ocrorot-train.sqlite3", f"{tmpdir}/ocrorot.pth")
    ocrorot.correct([f"{bucket}/gsub-bin-test.tar"], nsamples=10,
                    output=f"{tmpdir}/rotated.tar", model=f"{tmpdir}/ocrorot.pth")


def test_ocroscale(tmpdir):
    ocroscale.train([f"{bucket}/gsub-bin-test.tar"], nsamples=10,
                    log_to=f"{tmpdir}/ocroscale-train.sqlite3")
    slog.getbest(f"{tmpdir}/ocroscale-train.sqlite3",
                 f"{tmpdir}/ocroscale.pth")
    ocroscale.correct([f"{bucket}/gsub-bin-test.tar"], nsamples=10,
                      output=f"{tmpdir}/scaleated.tar", model=f"{tmpdir}/ocroscale.pth")


def test_ocroskew(tmpdir):
    ocroskew.train([f"{bucket}/gsub-bin-test.tar"], nsamples=10,
                   log_to=f"{tmpdir}/ocroskew-train.sqlite3")
    slog.getbest(f"{tmpdir}/ocroskew-train.sqlite3", f"{tmpdir}/ocroskew.pth")
    ocroskew.correct([f"{bucket}/gsub-bin-test.tar"], nsamples=10,
                     output=f"{tmpdir}/skewated.tar", model=f"{tmpdir}/ocroskew.pth")


def test_pagerec(tmpdir):
    pass