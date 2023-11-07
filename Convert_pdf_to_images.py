import fitz
from pathlib import Path
import hashlib
import os


try:
    FILE_PATH = Path(__file__)
except NameError:
    FILE_PATH = Path('.')
BASE_PATH = FILE_PATH.parent

def convert_pdf_to_images(pdf_path, output_path):
    dpi = 300
    zoom = dpi/72
    magnify = fitz.Matrix(zoom, zoom)
    count = 0
    doc = fitz.open(pdf_path)
    for page in doc:
        count+=1
        pix = page.get_pixmap(matrix=magnify)
        pdf_name = Path(pdf_path).stem
        pix.save((output_path/f'{pdf_name}.png').as_posix())
        break
    
def remove_duplicates(directory):
    hashes = set()
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        digest = hashlib.sha1(open(path,'rb').read()).digest()
        if digest not in hashes:
            hashes.add(digest)
        else:
            os.remove(path)


if __name__ == '__main__':
    input_path = BASE_PATH /'input'
    output_path = BASE_PATH /'images'
    os.makedirs(output_path, exist_ok=True)
    for file_name in sorted(os.listdir(input_path)):
        pdf_path = input_path / file_name
        pdf_path = pdf_path.absolute().as_posix()
        convert_pdf_to_images(pdf_path, output_path)
    remove_duplicates(directory=output_path)



