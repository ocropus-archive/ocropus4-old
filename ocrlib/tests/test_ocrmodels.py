#
# Copyright (c) 2017-2019 TBD. All rights reserved.
# This file is part of TBD (see TBD).
# See the LICENSE file for licensing terms (TBD).
#
from __future__ import unicode_literals

from ocrlib import ocrmodels


def test_ocr_models():
    ocrmodels.make_conv_only()
    ocrmodels.make_conv_resnet()
    ocrmodels.make_lstm_ctc()
    ocrmodels.make_lstm_normalized()
    ocrmodels.make_lstm_transpose()
    ocrmodels.make_lstm_keep()
    ocrmodels.make_lstm_resnet()
    ocrmodels.make_lstm_unet()
    ocrmodels.make_lstm2_ctc()


def test_seg_models():
    ocrmodels.make_seg_conv()
    ocrmodels.make_seg_lstm()
    ocrmodels.make_seg_unet()
