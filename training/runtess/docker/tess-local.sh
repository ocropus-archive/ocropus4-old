#!/bin/bash

die() {
    echo "ERROR: $*" 1>&2
    exit 1
}

test -f $1 || die "$1: not found"
test -f $2 && die "$2: output already exists"

tarp proc -c '
input=$(ls sample.* | egrep -i "(jpg|jpeg|png)$")
echo $(cat sample.__key__) $input
tesseract $input sample -l eng hocr
' "$1" -o "$2"
