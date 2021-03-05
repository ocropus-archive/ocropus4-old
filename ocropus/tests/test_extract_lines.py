#
# Copyright (c) 2017-2019 TBD. All rights reserved.
# This file is part of TBD (see TBD).
# See the LICENSE file for licensing terms (TBD).
#

from ocropus import extract_lines
import webdataset as wds


#def test_hocr2rec(tmp_path):
#    extract_seg.hocr2seg(
#        "testdata/pages.tar", output=f"{tmp_path}/_rec.tar", maxcount=50, ignore_errors=False
#    )
#    count = 0
#    for sample in wds.Dataset(f"{tmp_path}/_rec.tar"):
#        count += 1
#    assert count == 50
