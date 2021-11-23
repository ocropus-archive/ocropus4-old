from typing import List

import json

import click
import fitz
import numpy as np

from PIL import Image
from ocropus.tt_utils import make_text, get_right_cell_format, open_tar_file, get_folder_names, get_data
from ocropus.table_analysis import generate_table_html
from bs4 import BeautifulSoup

@click.group()
def cli():
    pass

def image2grid(
        image_array: np.array,
        tab_cell_coords: List
) -> np.array:
    """
    Creates Probability map where cell areas equals to 255 and remaining area is 0

    Parameters
    -----------
    image_array
        An array with shape equals to table image
    tab_cell_coords
        A list of bbox co-ordinates of table cells from table segmentation
    """
    for tab_cell_coord in tab_cell_coords:
        x11 = int(tab_cell_coord[1][0])
        y11 = int(tab_cell_coord[0][0])
        x21 = int(tab_cell_coord[1][1])
        y21 = int(tab_cell_coord[0][1])

        image_array[y11:y21, x11:x21] = 255  #image_array[y-cords, x-cords]

    return image_array


def insert_pdf_text(
        tab_struct: str,
        x1: int,
        y1: int,
        tab2page_scale_x: float,
        tab2page_scale_y: float,
        scale_x: float,
        scale_y: float,
        words: List
) -> str:
    """
    Replaces TEXT in tab_struct with the text present in pdf at given bbox coordinates

    Parameters
    -----------
    tab_struct
        Table structure in html format
    x1, y1
        x, y coordinates of table image given by page segmentation
    tab2page_scale_x, tab2page_scale_y
        scales of table image w/r/t pdf page image
    scale_x, scale_y
        scales of pdf_page_image w/r/t original pdf
    words
        list of words on a pdf page
    """

    soup = BeautifulSoup(tab_struct, 'lxml')
    for row1 in soup.find_all('tr'):
        for row2 in row1.find_all('td'):
            bounds = row2.attrs['title'][5:].split(" ")

            x11 = (x1 + (int(bounds[0]) * tab2page_scale_x))  # tab_cell[1][0] = x1
            y11 = (y1 + (int(bounds[1]) * tab2page_scale_y))  # y1
            x21 = (x1 + (int(bounds[2]) * tab2page_scale_x))  # x2
            y21 = (y1 + (int(bounds[3]) * tab2page_scale_y))  # y2

            rect = fitz.fitz.Rect(x11 * scale_x, y11 * scale_y, x21 * scale_x, y21 * scale_y)
            mywords = [w for w in words if fitz.Rect(w[:4]).intersects(rect)]
            cell_text = make_text(mywords)
            row2.string.replace_with(cell_text)
    return  str(soup)


def extract_tabtext_from_cellcords_proj(
        res_dir: str,
        pdf_file_path: str,
        pdf_image_path: str,
        tab_coords: List,
        pdf_img_idx: str
) -> None:
    """
    Writes table contents in the html format

    Parameters
    -----------
    res_dir
        Directory to store output of pdf_extraction pipeline
    pdf_file_path
        Path of the whole pdf
    pdf_image_path
        Path of the single pdf page image which contains table
    tab_coords
        Table coordinates for table detected by page segmentation
    pdf_img_idx
        Index of page in the original pdf which contains table
    """

    pdf_doc = fitz.Document(pdf_file_path)
    pdf_page_image = Image.open(pdf_image_path)
    pdf_doc_page = pdf_doc[int(pdf_img_idx)]#page in pdf where table is present

    # scales of pdf_page_image w/r/t original pdf
    scale_x = pdf_doc_page.rect[2] / pdf_page_image.size[0]
    scale_y = pdf_doc_page.rect[3] / pdf_page_image.size[1]

    for idx, tab_coord in enumerate(tab_coords):

        tab_cells = json.load(open(f"{res_dir}/out-{int(pdf_img_idx)}/tab{idx}.cells.json"))
        tab_image = Image.open(f"{res_dir}/out-{int(pdf_img_idx)}/tab{idx}.jpg")
        image_array = np.zeros([tab_image.size[1], tab_image.size[0], 3])

        print(f"Running for {pdf_file_path}, and image idx of {pdf_img_idx}, table: tab{idx}.jpg")

        tab_x, tab_y = tab_image.size

        x1 = tab_coord[1]["start"]
        y1 = tab_coord[0]["start"]
        x2 = tab_coord[1]["stop"]
        y2 = tab_coord[0]["stop"]

        # Scale of tab image w/r/t pdf_page_image
        tab2page_scale_x = (x2 - x1) / tab_x
        tab2page_scale_y = (y2 - y1) / tab_y

        words = pdf_doc_page.getText("words")#list of words on page

        binary_map = image2grid(image_array, tab_cells)
        tab_cells = get_right_cell_format(tab_cells)

        html_table = generate_table_html(binary_map, tab_cells)

        struct_text = insert_pdf_text(html_table, x1, y1, tab2page_scale_x, tab2page_scale_y, scale_x, scale_y, words)

        outfile = open(f"{res_dir}/out-{int(pdf_img_idx)}/tab{idx}.html", "w+")
        outfile.write(struct_text)

def run_pdf_extraction(
        pdf_data: List,
        untar_tabseg_out: str,
        pdf_file_path: str
) -> None:
    """
    Runs the pdf extraction pipeline for each pdf page which contains table

    Parameters
    -----------
    pdf_data
        List of tuples where each tuple contains path to image pdf page and table coordinates for detected tables
    untar_tabseg_out
        Folder name of table segmentation output. Same folder is used to store outputs of pdf_extraction pipeline
    pdf_file_path
        Path of the complete pdf
     """
    for page_data in pdf_data:

        pdf_image_path = page_data[0]
        tab_coords = json.load(open(page_data[1], 'r'))
        pdf_img_idx = pdf_image_path.split(".jpg")[0].split("-")[-1]
        #idx starts at 0, idx 10 is page 11 in original pdf

        extract_tabtext_from_cellcords_proj(untar_tabseg_out, pdf_file_path,
                                            pdf_image_path, tab_coords, pdf_img_idx)

@cli.command()
@click.option('--pageseg_out', required=True, help="Output of page segmentation")
@click.option('--tabseg_out', required=True, help="Output of table segmentation")
@click.option('--pdf_file_path', required=True, help="Path to pdf file")
def pdf_extraction(pageseg_out, tabseg_out, pdf_file_path):

    print(pageseg_out, tabseg_out, pdf_file_path)

    pdf_file_number = pdf_file_path.split(".pdf")[0].split("/")[-1]

    untar_pageseg_output = "tar_page_segout_" + pdf_file_number
    untar_tabseg_output = "tar_tab_segout_" + pdf_file_number

    open_tar_file(tabseg_out, ("tar_tab_segout_" + pdf_file_number))
    open_tar_file(pageseg_out, ("tar_page_segout_" + pdf_file_number))

    dirnames_with_tab = get_folder_names(untar_pageseg_output)
    pdf_data = get_data(dirnames_with_tab, untar_pageseg_output)

    run_pdf_extraction(pdf_data, untar_tabseg_output, pdf_file_path)

if __name__=='__main__':
    cli()