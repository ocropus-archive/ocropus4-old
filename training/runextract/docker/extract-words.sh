#!/bin/bash

set -e
set -x

if test -f /auth.json; then
    echo "activating service account from /auth.json"
    gcloud auth activate-service-account --key-file=/auth.json
fi

input="$1"; shift
output="$1"; shift

if gsutil ls "$output"; then
    echo "output already exists"
    exit
fi

ocropus=${ocropus:-ocropus4}
page=${page:-"page.jpg;jpg;jpeg"}
hocr=${hocr:-"page.hocr;hocr.html;hocr"}
extensions=${extensions:-"$page $hocr"}

gsutil cat "$input" |
$ocropus extract-rec hocr2rec --extensions="$extensions" - --output - "$@" |
gsutil cp - "$output"
