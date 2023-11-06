import fitz
from pathlib import Path
import os

try:
    FILE_PATH = Path(__file__)
except NameError:
    FILE_PATH = Path('.')
BASE_PATH = FILE_PATH.parent

def convert_pdf_to_images(pdf_path, output_file_image_folderpath):
    dpi = 300
    zoom = dpi/72
    magnify = fitz.Matrix(zoom, zoom)
    count = 0
    doc = fitz.open(pdf_path)
    for page in doc:
        count+=1
        pix = page.get_pixmap(matrix=magnify)
        pix.save(output_file_image_folderpath/f"page_{count}.png")


if __name__ == '__main__':
    input_path = BASE_PATH /'input'
    file_name = sorted(os.listdir(input_path))[0]
    pdf_path = input_path / file_name
    pdf_path = pdf_path.absolute().as_posix()

    output_path = BASE_PATH /'output'
    os.makedirs(output_path, exist_ok=True)
    output_file_image_folderpath = output_path / Path(file_name).stem
    os.makedirs(output_file_image_folderpath, exist_ok=True)

    convert_pdf_to_images(pdf_path, output_file_image_folderpath)
