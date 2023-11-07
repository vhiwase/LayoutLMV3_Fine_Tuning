import os
import pathlib
from typing import Tuple
from string import punctuation
import filetype
import fitz

try:
    basedir = pathlib.os.path.abspath(pathlib.os.path.dirname(__file__))
except NameError:
    basedir = pathlib.os.path.abspath(pathlib.os.path.dirname("."))

from utils import to_pathlib


def convert_pdf2img(
    input_file: str,
    pages: Tuple = None,
    create_seperate_folder: bool = False,
    count: int = 0,
    output_path: str = None,
):
    """
    Converts pdf to image and generates a file by page.

    Parameters
    ----------
    input_file : str
        Input file path.
    pages : Tuple, optional
        file pages. The default is None.
    create_seperate_folder : bool, optional
        If True, creaing folder of filename inside output path. The default is False.
    count : int, optional
        If seperate_folder is False, differenciate file with count initials.
        The default is 0.
    output_path : str, optional
        Output folder path where images are saved. The default is None.

    Returns
    -------
    output_files : list
        List of image paths.
    count : int
        Name initails if used.

    """
    # Open the document
    pdf_in = fitz.open(input_file)
    output_files = []
    count += 1
    # Iterate throughout the pages
    for pg in range(pdf_in.pageCount):
        if str(pages) != str(None):
            if str(pg) not in str(pages):
                continue
        # Select a page
        page = pdf_in[pg]
        rotate = int(0)
        # PDF Page is converted into a whole picture 1056*816 and then for
        # each picture a screenshot is taken.
        # zoom = 1.33333333 -----> Image size = 1056*816
        # zoom = 2 ---> 2 * Default Resolution (text is clear, image text is
        # hard to read)    = filesize small / Image size = 1584*1224
        # zoom = 4 ---> 4 * Default Resolution (text is clear, image text is
        # barely readable) = filesize large
        # zoom = 8 ---> 8 * Default Resolution (text is clear, image text is
        # readable) = filesize large
        zoom_x = 2
        zoom_y = 2
        # The zoom factor is equal to 2 in order to make text clear
        # Pre-rotate is to rotate if needed.
        mat = fitz.Matrix(zoom_x, zoom_y).preRotate(rotate)
        pix = page.getPixmap(matrix=mat, alpha=False)
        input_file_pathlib = pathlib.Path(input_file)
        if isinstance(input_file_pathlib, str):
            file_path = pathlib.Path(file_path)
        kind = filetype.guess(input_file_pathlib.as_posix())
        extension = kind.extension
        extension = ".{}".format(extension)
        file_name = input_file_pathlib.name.replace(extension, '')
        file_name = file_name.replace(' ', '_')
        if output_path is None:
            output_path = to_pathlib(
                [input_file_pathlib.parent.parent.as_posix(), "Images"]
            )
        else:
            output_path = pathlib.Path(output_path)
        kind = filetype.guess(input_file_pathlib.as_posix())
        if kind.MIME.split("/") and kind.MIME.split("/")[0] == "image":
            extension = kind.extension
        else:
            extension = "png"
        for symbol in punctuation:
            if symbol in file_name:
                if file_name.startswith(symbol):
                    file_name = file_name.replace(symbol, '_')
                    file_name = 'prefix' + file_name
                else:
                    file_name = file_name.replace(symbol, '_')                    
        if create_seperate_folder:
            output_folder = output_path / file_name
            os.makedirs(output_folder, exist_ok=True)
            output_file_pathlib = output_folder / "page_{}.{}".format(pg + 1, extension)
        else:
            output_folder = output_path
            os.makedirs(output_folder, exist_ok=True)
            output_file_pathlib = output_folder / "image_{}{}.{}".format(
                count, pg + 1, extension
            )
        output_file = output_file_pathlib.as_posix()
        pix.writePNG(output_file)
        output_files.append(output_file)
    pdf_in.close()
    return output_files, count


def convert_pdf2imgs(input_folder_path: str):
    """
    A function which converts all pdf present in input folder path to
    it's images.

    Parameters
    ----------
    input_folder_path : str
        Input path where PDF or images are present.

    Returns
    -------
    None.

    """
    count = 0
    input_folder_path_pathlib = pathlib.Path(input_folder_path)
    total_file_names = os.listdir(input_folder_path_pathlib)
    for file_name in total_file_names:
        input_file = input_folder_path_pathlib / file_name
        try:
            output_files, count = convert_pdf2img(
                input_file, pages=None, create_seperate_folder=True, count=count
            )
        except ValueError:
            print("Value Error")
            continue
        except RuntimeError:
            print("Runtime Error")
    print("Completed")


if __name__ == "__main__":
    input_folder_path = "/mnt/c/Users/Celebal/Downloads/Forgery Dataset/Forged/pdfs"
    convert_pdf2imgs(input_folder_path)
