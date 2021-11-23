from typing import List, Tuple, Union, Iterable, Dict

import click
import numpy as np
import scipy
import json

from scipy.ndimage import gaussian_filter
from ocropus.tt_utils import create_probmap_from_coords, get_right_cell_format

@click.group()
def cli():
    pass

def find_local_minima(
        proj: np.array,
        filter_variance: float
) -> List[int]:
    """
    Finds all local minima in a projection

    Parameters
    ----------
    proj
        An array of projection across an axis of 2D data
    filter_variance
        Variance of gaussian filter
    """
    threshold = np.mean(proj)
    smoothed = gaussian_filter(proj, filter_variance) # blurs the image
    local_minima = (np.roll(smoothed, -1) > smoothed) * (np.roll(smoothed, 1) > smoothed)
    boundaries = local_minima * (proj < threshold)

    return boundaries


def slice_proj_image(
        slices: Union[List, Iterable]
) -> List:
    """
    Finds indexes for row/column boundaries

    Parameters
    ----------
    slices
        A list of slices
    """
    idxs = []
    prev = 0
    for sli in slices:
        bound_start = sli[0].start
        bound_stop = sli[0].stop
        if prev == 0:
            prev = bound_stop
        else:
            bound_mark = int(0.5 * (bound_start + prev))
            idxs.append(bound_mark)
            prev = bound_stop

    return idxs


def find_slice_boundaries(
        proj: np.array,
        filter_variance: float
) -> Tuple[List, List]:
    """
    Finds start and end of boundary in a projection

    Parameters
    ----------
    proj
        An array of projection across an axis of 2D data
    filter_variance
        Variance of gaussian filter
    """
    smoothed = gaussian_filter(proj, filter_variance)
    threshold = 0.5 * np.percentile(smoothed, 90)
    column_markers = (smoothed > threshold)
    labeled, n = scipy.ndimage.label(column_markers)
    slices = list(scipy.ndimage.find_objects(labeled))
    idxs = slice_proj_image(slices)
    return slices, idxs


def get_proj(
        binary_map: np.array,
        ax: int,
        filter_variance: float,
        minima_method: bool = False
) -> Tuple[List, List]:
    """
    Find indexes for row/column boundaries

    Parameters
    ---------
    binary_map
        Probability map where cell areas equals to 255 and remaining area is 0
    ax
        axis to compute projection over rows or columns
    filter_variance
        Variance of gaussian filter
    """

    new_arr = binary_map[:, :, 0].sum(ax)

    if minima_method:
        bounds = find_local_minima(new_arr, filter_variance)
        idxs = [idx for idx, val in enumerate(bounds) if val == True]
    else:
        bounds, idxs = find_slice_boundaries(new_arr, filter_variance)
        if ax == 0:
            idxs.append(binary_map.shape[1])
        elif ax == 1:
            idxs.append(binary_map.shape[0])
    print("IDXS where boundaries are made:", idxs)
    return bounds, idxs


def get_span(
        tab_cell_bounds: Tuple,
        boundaries: List,
        prop_threshold=0.10
) -> Dict:

    """
    Finds cells that span over rows or columns

    Parameters
    ---------
    tab_cell_bounds
        Tuple with coordinates of single axes for a table cell
    boundaries
        row/column boundaries
    prop_threshold
        Threshold value to check a row or column is actually spanning or not
    """
    cell_start, cell_end = tab_cell_bounds
    res = {}

    N = len(boundaries)

    for idx_start in range(N):

        if cell_start < boundaries[idx_start]:
            res["start"] = idx_start

            for idx_end in range(idx_start, N):
                if cell_end < boundaries[idx_end]:

                    if idx_end > idx_start:
                        extend_prop = (cell_end - boundaries[idx_end - 1]) / (cell_end - cell_start)

                        if extend_prop > prop_threshold:
                            res["end"] = idx_end
                        else:
                            res["end"] = idx_end - 1
                    else:
                        res["end"] = idx_end
                    return res


def get_row_col(
        tab_cell_coord: List,
        row_idxs: List,
        col_idxs: List
) -> Dict:

    """
    Find the row/column start and end for text boxes (table cell coordinates)

    Parameters
    ---------
    tab_cell_coord
        A list of bbox coordinates of table cell
    row_idxs
        List with indexes where a row starts
    col_idxs
        List with indexes where a column starts
   """


    x1, y1, x2, y2 = tab_cell_coord
    res = {}

    rows = get_span((y1, y2), row_idxs)
    cols = get_span((x1, x2), col_idxs)

    res.update({f"row_{k}": v for k, v in rows.items()})
    res.update({f"col_{k}": v for k, v in cols.items()})

    res["bbox"] = [x1, y1, x2, y2]
    return res

def get_merged_cells(
        old_cell: Dict,
        new_cell:Dict
) -> Dict:
    """
    Checks if the new cell is before or after the old cell and returns the merged cell

    Parameters
    ---------
    old_cell
        Dict containing start/end of row and column for a cell
    new_cell
        Dict containing start/end of row and column for a cell

    """
    if new_cell["bbox"][0] < old_cell["bbox"][0]:
        old_cell["bbox"][0] = new_cell["bbox"][0]
        old_cell["bbox"][1] = new_cell["bbox"][1]

    if new_cell["bbox"][2] > old_cell["bbox"][2]:
        old_cell["bbox"][2] = new_cell["bbox"][2]
        old_cell["bbox"][3] = new_cell["bbox"][3]

    return old_cell

def get_table_structure(
        tab_cell_coords: List[List],
        row_idxs: List,
        col_idxs: List
) -> List[List]:
    """
    Find the table structure

    Parameters
    ---------
    tab_cell_coords
        A list of bbox coordinates of table cells from table segmentation
    row_idxs
        List with indexes where a row starts
    col_idxs
        List with indexes where a column starts
    """
    tab_struct = [[None] * len(col_idxs) for i in range(len(row_idxs))]

    for tab_cell_coord in tab_cell_coords:
        cell_struct_data = get_row_col(tab_cell_coord, row_idxs, col_idxs)
        row_start, col_start = cell_struct_data["row_start"], cell_struct_data["col_start"]
        old_cell = tab_struct[row_start][col_start]

        if old_cell:
            tab_struct[row_start][col_start] = get_merged_cells(old_cell, cell_struct_data)
        else:
            tab_struct[row_start][col_start] = cell_struct_data

    return tab_struct


def get_html_data(
        table_struct: List
) -> str:

    """
    Creates html data for the table structure

    Parameters
    ---------
    table_struct
        A list of lists which contains dictionary with row_start/end for each table cell
    """

    tab_html = [[] for _ in range(len(table_struct))]

    for idx, row in enumerate(table_struct):
        for cell in row:
            if cell:
                row_span = cell["row_end"] - cell["row_start"] + 1
                col_span = cell["col_end"] - cell["col_start"] + 1
                bbox = [str(i) for i in cell["bbox"]]

                if row_span > 1 and col_span > 1:
                    td_html = f'<td class="ocr_table_cell" title="bbox {" ".join(bbox)}" rowspan="{row_span}" colspan="{col_span}"> TEXT </td>'
                elif row_span > 1:
                    td_html = f'<td class="ocr_table_cell" title="bbox {" ".join(bbox)}" rowspan="{row_span}"> TEXT </td>'
                elif col_span > 1:
                    td_html = f'<td class="ocr_table_cell" title="bbox {" ".join(bbox)}" colspan="{col_span}"> TEXT </td>'
                else:
                    td_html = f'<td class="ocr_table_cell" title="bbox {" ".join(bbox)}"> TEXT </td>'

                tab_html[idx].append(td_html)

    tab_html = '<table>' + ''.join(['<tr>' + ''.join(row) + '</tr>' for row in tab_html]) + '</table>'

    html_table_start = "<html><head><style> table, th, td {border: 1px solid black;} </style></head><body>"
    html_table_end = "</body></html>"

    return  html_table_start + tab_html + html_table_end


def generate_table_html(
        binary_map: np.array,
        tab_cell_coords: List[List]
) -> str:
    """
    Converts binary table and cell coordinates to html table.

    Parameters
    ----------
    binary_map
        Probability map where cell areas equals to 255 and remaining area is 0
    tab_cell_coords
        A list of bbox coordinates of table cells from table segmentation
    """

    col_bounds, col_idxs = get_proj(binary_map, 0, filter_variance=5.0, minima_method=False)
    row_bounds, row_idxs = get_proj(binary_map, 1, filter_variance=5.0, minima_method=False)

    table_structure = get_table_structure(tab_cell_coords, row_idxs, col_idxs)
    html_table = get_html_data(table_structure)

    return html_table

def run_table_analysis(
        tab_cell_coords_path: str,
        image_size: (int, int),

) -> None:
    """
    Run table analysis pipeline and generates html table

    Parameters
    ---------
    tab_cell_coords_path
        Path of file containing table cell coordinates given by table segmentation
    image_size
        A tuple of table image size. (Width, Height)
    """

    with open(tab_cell_coords_path) as fp:
        tab_cell_coords = get_right_cell_format(json.load(fp))

    binary_map = create_probmap_from_coords(tab_cell_coords, image_size)

    html_table = generate_table_html(binary_map, tab_cell_coords)

    with open('tab.html', 'w') as fp:
        fp.write(html_table)


@cli.command()
@click.option('--filename', required=True, help="Filepath of cell coordinates for a table")
@click.option('--image_size', nargs=2, type=int, required=True, help="Size (height, width) of table")
def table_analysis(filename, image_size):
    run_table_analysis(filename, image_size)

if __name__=='__main__':
    cli()