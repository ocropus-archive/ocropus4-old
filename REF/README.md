ICDAR 2019 Worksheets for Tutorial
==================================

These are worksheets for the
[ICDAR 2019 Tutorial on Deep Learning for Document Analysis](https://github.com/tmbdev/icdar2019-tutorial).
They implement data generation and full training for text recognition (OCR) and document segmentation
using a variety of common approaches, including convolution, U-net, LSTM, 2D LSTM, transpose convolutions,
and upscaling.

Please see the individual worksheets for details. The models themselves are found in `ocrlib/ocrmodels.py`.

Note that the worksheets checked into the repository are only partially trained (using `run-all`).
That is, they are only trained enough to make sure that the code works upon checkin.
For good models, train for at least 10 epochs or more.

Running the Code
================

You can run the Jupyter server with `./run-jupyter` and then connect to the server at `http://localhost:9888`.
This will build a Docker container and then execute it as you in the current directory.

Data
====

You can download the training data used with these notebooks by running `run-download.

You can generate OCR data using the `word-image-generation.ipynb` notebook.
You need to have a basic set of TrueType fonts installed for that.
