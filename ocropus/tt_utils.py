from typing import List, Tuple

import tarfile
import os
import numpy as np


def open_tar_file(
        tar_file: str,
        out_dir: str
) -> None:
    """
    Extracts the contents of tar file and store them in a new folder

    Parameters
    ---------
    tar_file
        Path of tar file which contains output of page or table segmentation
    out_dir
        Folder name to store contents of tar file
    """

    if not os.path.exists(out_dir):
        my_tar = tarfile.open(tar_file)
        my_tar.extractall('./' + out_dir)
        my_tar.close()

def get_folder_names(
        out_dir: str
) -> List:
    """
    Returns names of directories for which Ocropus detects tables

    Parameters
    ---------
    out_dir
        Output folder (untar) name given by table/page segmentation
    """
    os.chdir(out_dir)
    dirnames_with_tab = []
    for name in os.listdir("."):
        if os.path.isdir(name):
            dirnames_with_tab.append(name)
    os.chdir("../")

    return dirnames_with_tab

def get_data(
        dirnames_with_tab: List,
        untar_dir_name: str
) -> List:
    """
    Returns paths of two files (table coordinates, pdf image) for each pdf page which contains table
    
    Parameters
    ---------
    dirnames_with_tab
        Directory names for which Ocropus detects tables
    untar_dir_name
        Output folder (untar) name given by table/page segmentation
    """
    pdf_data = []
    for dirname_with_tab in dirnames_with_tab:
        pdf_image = os.path.join(untar_dir_name, dirname_with_tab + ".jpg")
        tab_cords = os.path.join(untar_dir_name, dirname_with_tab + ".tables.json")
        pdf_data.append((pdf_image, tab_cords))

    return pdf_data

def make_text(
        words: List
) -> str:
    """
    Return text string output of get_text("words"). Word items are sorted for reading sequence left to right, top to bottom.

    Parameters
    ---------
    words
        list of words on a pdf page
    """
    line_dict = {}  # key: vertical coordinate, value: list of words
    words.sort(key=lambda w: w[0])  # sort by horizontal coordinate
    for w in words:  # fill the line dictionary
        y1 = round(w[3], 1)  # bottom of a word: don't be too picky!
        word = w[4]  # the text of the word
        line = line_dict.get(y1, [])  # read current line content
        line.append(word)  # append new word
        line_dict[y1] = line  # write back to dict
    lines = list(line_dict.items())
    lines.sort()  # sort vertically
    return "\n".join([" ".join(line[1]) for line in lines])

def create_probmap_from_coords(
        tab_cell_coords: List[List],
        image_size: Tuple
) -> np.array:
    """
    Creates Probability map where cell areas equals to 255 and remaining area is 0

    Parameters
    ---------
    tab_cell_coords
        A list of bbox co-ordinates of table cells from table segmentation
    image_size
        A tuple of table image size. (Width, Height)
    """
    binary_map = np.zeros([image_size[0], image_size[1], 3])
    for tab_cell_coord in tab_cell_coords:
        binary_map[tab_cell_coord[1]:tab_cell_coord[3], tab_cell_coord[0]:tab_cell_coord[2]] = 255
    #Image.fromarray(image_array[:, :, 0]).convert('RGB').save('16.png')
    return binary_map

def get_right_cell_format(
        tab_cell_coords: List
) -> List:
    """
    Creates cell coordinates in the format accepted by table_analysis

    Parameters
    ----------
    tab_cell_coords
        A list of bbox co-ordinates of table cells from table segmentation
    """
    new_tab_cell_coords = []
    for line in tab_cell_coords:
        new_tab_cell_coords.append([line[1][0], line[0][0], line[1][1], line[0][1]]) #x1, y1, x2, y2

    return new_tab_cell_coords