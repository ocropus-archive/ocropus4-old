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

gsutil cat "$input" |
ocropus4 extract-seg hocr2seg --element ocr_line - --output - "$@" |
gsutil cp - "$output"
