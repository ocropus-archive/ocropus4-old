#
# Copyright (c) 2017-2019 TBD. All rights reserved.
# This file is part of TBD (see TBD).
# See the LICENSE file for licensing terms (TBD).
#

from ocropus import extract_seg
import webdataset as wds


def test_hocr2seg(tmp_path):
    extract_seg.hocr2seg(
        "testdata/pages.tar",
        output=f"{tmp_path}/_seg.tar",
        maxcount=50,
        ignore_errors=False,
        check="none",
    )
    count = 0
    for sample in wds.WebDataset(f"{tmp_path}/_seg.tar"):
        count += 1
    assert count == 50


def test_hocr2seg_pipe(tmp_path):
    extract_seg.hocr2seg(
        "pipe:cat testdata/pages.tar",
        output=f"pipe:dd of={tmp_path}/_seg2.tar",
        maxcount=50,
        ignore_errors=False,
        check="none",
    )
    count = 0
    for sample in wds.WebDataset(f"{tmp_path}/_seg2.tar"):
        count += 1
    assert count == 50
