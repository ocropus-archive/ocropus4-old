# OCRopus4

This is a **very preliminary** version of OCRopus 4. Components are:

- ocrobin -- binarization (hardcoded binarizer is also available)
- ocrorot -- orientation
- ocroskew -- small page rotations ("skew")
- ocroscale -- scale determination
- ocroseg -- word, line, and block segmentation of pages
- ocrorec -- text recognition from words and lines

Status is:

- DL training and recognition for text recognition and page layout analysis are working
- training fully works on UW3 and G1000
- we have large datasets extracted from the Tobacco corpus and Google 1000 Books

Warnings:

- **The models are very preliminary and have only been trained on UW3.**
- **The code needs to be refactored and cleaned up.**

# Getting Started

Look at `g1000test/quicktest.ipynb` to see all the components in action.

# Other Comments

- OCRopus4 provides both command line interfaces and Python APIs.
    - command line APIs are provided by `Typer`
    - command invocation generally looks like this:
        - `python3 -m ocropus.ocroseg train ...`
        - that is, commands are simply part of the library files
        - the toplevel `ocropus4` script provides a nicer interface
- All the computations are carried out in PyTorch.
- OCRopus4 uses some extra external and internal libraries:
    - `typer` for command line interfaces
    - `webdatset` for large scale I/O
    - `torchmore` for easy model building (shape inference and extra layers)
    - `tarp` for fast processing of large datasets
    - `ocropus/slog.py` for logging (into sqlite3 databases)

# Directory Structure

- `REF` - various old notebooks and code snippets kept for reference (they don't necessarily run)
- `TESTS` - various manual tests
- `ocropus` - most Python code (will be renamed to `ocropus`)
- `testdata` - datasets used by PyTest tests
- `models` - trained models (must download with `./run download`)
- `prep` - notebooks and scripts for data preparation (mostly UW3 examples)
- `*.ipynb` - various scripts
- `ocropus4.yaml` - configuration file for ocropus4 command

# Commands:

- `./ocropus4` -- the main command for OCRopus 4
- `./run` -- various utility commands for building and working with the repo
    - clean -- remove temporary files
    - cleanlogs -- clean up log files
    - download -- download ./models from Google Cloud
    - upload -- upload ./models to Google Cloud
    - info -- get info about models from all logfiles in the current directory
    - getbest -- get the best model out of a logfile
    - val2model -- fix for validation data into model
    - venv -- set up a virtualenv
    - testline -- test line recognition
    - testseg -- test segmentation
    - uw3dewline -- train on pre-dewarped lines (uw3-dew)
    - uw3rawline -- train on raw lines (uw3-rawlines)
    - uw3words -- train on extracted words (uw3-lwords)
    - uw3wseg -- train a word segmenter (uw3-wordseg-markers-masked-patches)
    - uw3wwseg -- train a word segmenter (uw3-wordseg-markers-masked-patches)
    - uw3lseg -- train a line segmenter (uw3-lineseg-markers-patches)
    - uw3wlseg -- train a line segmenter (uw3-lineseg-markers-patches)
    - uw3zones -- train a block segmenter (uw3-zoneseg)
    - uw3wzones -- train a block segmenter (uw3-zoneseg)
    - lab -- run jupyter lab in the environment
    - build -- 
    - test -- 
    - help -- help message
- `./train` -- miscellaneous training scripts
