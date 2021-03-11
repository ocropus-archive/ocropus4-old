#
# Copyright (c) 2017-2019 TBD. All rights reserved.
# This file is part of TBD (see TBD).
# See the LICENSE file for licensing terms (TBD).
#

from ocropus import extract_text


def test_acceptable_words():
    f = extract_text.acceptable_words()
    assert f("7")
    assert f("123")
    assert f("$9")
    assert f("9%")
    assert f("%123")
    assert f("123%")
    assert f("0.123%")
    assert f("$123.00")
    assert f("-$123.00")
    assert f("-123.")
    assert f("hello")
    assert f("ABC")
    assert f("AB")
    assert f("A.B.")
    assert not f("fwlkfj")
    assert f("klsjdf-")
