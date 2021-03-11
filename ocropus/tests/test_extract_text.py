#
# Copyright (c) 2017-2019 TBD. All rights reserved.
# This file is part of TBD (see TBD).
# See the LICENSE file for licensing terms (TBD).
#

from ocropus import extract_text


def test_acceptable_words():
    f = extract_text.acceptable_words()
    assert f("123")
    assert f("$123.00")
    assert f("-$123.00")
    assert f("-123.")
    assert f("hello")
    assert not f("fwlkfj")
    assert f("klsjdf-")
