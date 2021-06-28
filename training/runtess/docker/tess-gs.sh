#!/bin/bash

set -e
set -x

if test -f /auth.json; then
    echo "activating service account from /auth.json"
    gcloud auth activate-service-account --key-file=/auth.json
fi

if gsutil ls "$2"; then
    echo "output already exists"
    exit
fi

gsutil cat "$1" |
tarp proc -c '
input=$(ls sample.* | egrep -i "(jpg|jpeg|png)$")
tesseract $input sample -l eng hocr
echo $(cat sample.__key__) : $input : $(ls)
' - -o - |
gsutil cp - "$2"
