import os
from glob import glob

import pytest
import numpy as np
from PIL import Image

from ocropus.pdf_extraction import run_pdf_extraction
from ocropus.table_analysis import find_slice_boundaries, generate_table_html
from ocropus.tt_utils import create_probmap_from_coords

test_dir = "../../tt_testdata"

def get_map():
    prob_map = f'{test_dir}/test-pic.png'
    cell_coords = [[60, 70, 300, 90], [355, 70, 605, 90], [50, 130, 190, 170], [206, 140, 310, 175],
                   [360, 140, 485, 175], [505, 140, 600, 175], [50, 260, 180, 290], [208, 220, 310, 240],
                   [370, 220, 485, 240], [510, 220, 600, 240], [212, 310, 310, 330], [370, 300, 485, 330],
                   [510, 300, 600, 330]]

    prob_map_size = np.array(Image.open(prob_map)).shape
    map_from_coords = create_probmap_from_coords(cell_coords, prob_map_size)

    return map_from_coords, cell_coords

@pytest.mark.parametrize(
    "test_input, result",
    [
        ({"prob_map": get_map()[0], "axis": 0, "fil_var": 5.0}, [197, 336, 496]),
        ({"prob_map": get_map()[0], "axis": 1, "fil_var": 5.0}, [114, 198, 272])
    ]
)
def test_find_slice_boundaries(test_input, result):
    proj = test_input["prob_map"][:, :, 0].sum(test_input["axis"])
    res = find_slice_boundaries(proj=proj, filter_variance=test_input["fil_var"])
    print(res[1])
    assert res[1]==result

@pytest.mark.parametrize(
    "test_input, result",
    [
        ({"map": get_map(), "row_idxs": [114, 198, 272, 400], "col_idxs": [197, 336, 496, 640]},
         '<html><head><style> table, th, td {border: 1px solid black;} </style></head><body><table><tr><td class="ocr_table_cell" title="bbox 60 70 300 90" colspan="2"> TEXT </td><td class="ocr_table_cell" title="bbox 355 70 605 90" colspan="2"> TEXT </td></tr><tr><td class="ocr_table_cell" title="bbox 50 130 190 170"> TEXT </td><td class="ocr_table_cell" title="bbox 206 140 310 175"> TEXT </td><td class="ocr_table_cell" title="bbox 360 140 485 175"> TEXT </td><td class="ocr_table_cell" title="bbox 505 140 600 175"> TEXT </td></tr><tr><td class="ocr_table_cell" title="bbox 50 260 180 290" rowspan="2"> TEXT </td><td class="ocr_table_cell" title="bbox 208 220 310 240"> TEXT </td><td class="ocr_table_cell" title="bbox 370 220 485 240"> TEXT </td><td class="ocr_table_cell" title="bbox 510 220 600 240"> TEXT </td></tr><tr><td class="ocr_table_cell" title="bbox 212 310 310 330"> TEXT </td><td class="ocr_table_cell" title="bbox 370 300 485 330"> TEXT </td><td class="ocr_table_cell" title="bbox 510 300 600 330"> TEXT </td></tr></table></body></html>'
         )
    ]
)
def test_generate_table_html(test_input, result):

    res = generate_table_html(test_input["map"][0], test_input["map"][1])
    assert res==result

@pytest.mark.parametrize(
    "test_input, result",
    [
        ({"pdf_data": [(f'{test_dir}/pdf_extract/out-17.jpg', f'{test_dir}/pdf_extract/out-17.tables.json')],
          "out-dir":f'{test_dir}/pdf_extract', "pdf_file_path": f'{test_dir}/pdf_extract/12818436.pdf'},
         '<html><head><style> table, th, td {border: 1px solid black;} </style></head><body><table><tr><td class="ocr_table_cell" title="bbox 5 0 166 14">Other operating expenses</td><td class="ocr_table_cell" title="bbox 529 1 568 13">-427,0</td><td class="ocr_table_cell" title="bbox 639 1 666 13">96,5</td></tr><tr><td class="ocr_table_cell" title="bbox 6 23 143 36">Net operating income</td><td class="ocr_table_cell" title="bbox 534 23 568 36">258,9</td><td class="ocr_table_cell" title="bbox 639 23 666 36">96,5</td></tr><tr><td class="ocr_table_cell" title="bbox 6 45 143 58">Net operating income</td><td class="ocr_table_cell" title="bbox 534 45 569 58">816,7</td><td class="ocr_table_cell" title="bbox 647 45 666 58">0,0</td></tr><tr><td class="ocr_table_cell" title="bbox 5 65 102 77">Overhead costs</td><td class="ocr_table_cell" title="bbox 529 66 568 78">-385,6</td><td class="ocr_table_cell" title="bbox 643 66 666 78">-0,8</td></tr></table></body></html>'
         )
    ]
)
def test_run_pdf_extraction(test_input, result):
    for i in glob(f"{test_dir}/pdf_extract/out-17/*.html"):
        os.remove(i)

    run_pdf_extraction(test_input["pdf_data"], test_input["out-dir"], test_input["pdf_file_path"])
    with open(f'{test_dir}/pdf_extract/out-17/tab0.html') as fp:
        res = fp.readline().strip()

    assert res==result