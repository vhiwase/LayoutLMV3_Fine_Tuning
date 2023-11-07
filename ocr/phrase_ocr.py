import dataclasses
import os
import pathlib
import time
from collections import defaultdict
from typing import List, Tuple

import cv2
import easyocr
import filetype
import numpy as np
import pandas as pd
from PIL import Image

try:
    basedir = pathlib.os.path.abspath(pathlib.os.path.dirname(__file__))
except NameError:
    basedir = pathlib.os.path.abspath(pathlib.os.path.dirname("."))
pathlib.sys.path.insert(0, basedir)
backend_parentdir = pathlib.os.path.dirname(basedir)
pathlib.sys.path.insert(0, backend_parentdir)


from ocr.ds import OCR, OCRCoordinates
from pdf_to_img import convert_pdf2img
from bounding_box_re_alignment import aligned_bbox

__all__ = ["PhraseOCR"]

reader = easyocr.Reader(lang_list=["en"], gpu=True)
SPACE_THRESHOLD = 20


class PhraseOCR:
    def __init__(self, image_path=None):
        print("OCR Model loaded successfully :)")
        if image_path:
            self.initialize_image_path(image_path)

    def initialize_image_path(self, image_path):
        phrase_ocr = OCR(image_path=image_path)
        phrase_ocr.pil_images = phrase_ocr.pil_images()
        phrase_ocr.cv2_images = phrase_ocr.cv2_images()
        phrase_ocr.dataframes = phrase_ocr.dataframes()        
        self.phrase_ocr = phrase_ocr
        if isinstance(phrase_ocr.image_path, pathlib.Path):
            phrase_ocr.image_path = phrase_ocr.image_path.as_posix()
        self.phrase_ocr.pil_images.pil_image = self.to_pil(self.phrase_ocr.image_path)
        self.phrase_ocr.cv2_images.cv2_image = self.to_cv2(self.phrase_ocr.image_path)

    def to_pil(self, image_path):
        if isinstance(image_path, str):
            pil_image = Image.open(image_path).convert("RGB")
        elif isinstance(image_path, np.ndarray):
            image = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
        else:
            pil_image = None
        return pil_image

    def to_cv2(self, image_path):
        if isinstance(image_path, str):
            cv2_image = cv2.imread(image_path)
        else:
            cv2_image = None
        return cv2_image

    def get_easy_ocr_dataframe(self, image_path=None, bbox_type="default"):
        if image_path:
            self.initialize_image_path(image_path)
        if image_path is None:
            image_path = self.phrase_ocr.image_path
        results = reader.readtext(image_path)
        top_left_x_list = []
        top_left_y_list = []
        top_right_x_list = []
        top_right_y_list = []
        bottom_right_x_list = []
        bottom_right_y_list = []
        bottom_left_x_list = []
        bottom_left_y_list = []
        texts = []
        confidences = []
        for result in results:
            bounding_box, text, confidence = result
            top_left, top_right, bottom_right, bottom_left = bounding_box
            # top_left
            top_left_x, top_left_y = top_left
            top_left_x_list.append(top_left_x)
            top_left_y_list.append(top_left_y)
            # top_right
            top_right_x, top_right_y = top_right
            top_right_x_list.append(top_right_x)
            top_right_y_list.append(top_right_y)
            # bottom_right
            bottom_right_x, bottom_right_y = bottom_right
            bottom_right_x_list.append(bottom_right_x)
            bottom_right_y_list.append(bottom_right_y)
            # bottom_left
            bottom_left_x, bottom_left_y = bottom_left
            bottom_left_x_list.append(bottom_left_x)
            bottom_left_y_list.append(bottom_left_y)
            # texts
            texts.append(text)
            # confidences
            confidences.append(confidence)
        ocr_df = pd.DataFrame()
        ocr_df["top_left_x"] = top_left_x_list
        ocr_df["top_left_y"] = top_left_y_list
        ocr_df["top_right_x"] = top_right_x_list
        ocr_df["top_right_y"] = top_right_y_list
        ocr_df["bottom_right_x"] = bottom_right_x_list
        ocr_df["bottom_right_y"] = bottom_right_y_list
        ocr_df["bottom_left_x"] = bottom_left_x_list
        ocr_df["bottom_left_y"] = bottom_left_y_list
        ocr_df["text"] = texts
        ocr_df["confidence"] = confidences
        self.phrase_ocr.dataframes.ocr_df = ocr_df
        line_numbers = []
        top_left_x_list = ocr_df["top_left_x"].tolist()
        prev_top_left_x = top_left_x_list and top_left_x_list[0]
        top_left_y_list = ocr_df["top_left_y"].tolist()
        prev_top_left_y = top_left_y_list and top_left_y_list[0]
        count = 1
        line_numbers.append(count)
        for top_left_x, top_left_y in zip(top_left_x_list[1:], top_left_y_list[1:]):
            if prev_top_left_x > top_left_x or abs(top_left_y - prev_top_left_y) > 10:
                count += 1
            line_numbers.append(count)
            prev_top_left_x = top_left_x
            prev_top_left_y = top_left_y
        ocr_df["line_numbers"] = line_numbers
        top_left_x_upper_shift = ocr_df["top_left_x"].loc[1:].tolist()
        top_left_x_upper_shift = top_left_x_upper_shift + [top_left_x_upper_shift[-1]]
        ocr_df["top_left_x_upper_shift"] = top_left_x_upper_shift
        ocr_df["word_space"] = ocr_df["top_left_x_upper_shift"] - ocr_df["top_right_x"]
        phrase_numbers = []
        count = 0
        for line_number in sorted(set(ocr_df["line_numbers"])):
            line_ocr_df = ocr_df[ocr_df["line_numbers"] == line_number]
            word_space = line_ocr_df["word_space"].tolist()
            phrase_number = []
            for val in word_space:
                phrase_number.append(count)
                if val > SPACE_THRESHOLD:
                    count += 1
            count += 1
            phrase_numbers.extend(phrase_number)
        ocr_df["phrase_numbers"] = phrase_numbers
        top_left_x = []
        top_left_y = []
        top_right_x = []
        top_right_y = []
        bottom_left_x = []
        bottom_left_y = []
        bottom_right_x = []
        bottom_right_y = []
        text = []
        confidence = []
        line_numbers = []
        phrase_numbers = []
        for phrase_number in sorted(set(ocr_df["phrase_numbers"])):
            phrase_df = ocr_df[ocr_df["phrase_numbers"] == phrase_number]
            t_lx = phrase_df.loc[phrase_df.index[0], "top_left_x"]
            top_left_x.append(t_lx)
            t_ly = phrase_df.loc[phrase_df.index[0], "top_left_y"]
            top_left_y.append(t_ly)
            t_rx = phrase_df.loc[phrase_df.index[-1], "top_right_x"]
            top_right_x.append(t_rx)
            t_ry = phrase_df.loc[phrase_df.index[-1], "top_right_y"]
            top_right_y.append(t_ry)
            b_lx = phrase_df.loc[phrase_df.index[0], "bottom_left_x"]
            bottom_left_x.append(b_lx)
            b_ly = phrase_df.loc[phrase_df.index[0], "bottom_left_y"]
            bottom_left_y.append(b_ly)
            b_rx = phrase_df.loc[phrase_df.index[-1], "bottom_right_x"]
            bottom_right_x.append(b_rx)
            b_ry = phrase_df.loc[phrase_df.index[-1], "bottom_right_y"]
            bottom_right_y.append(b_ry)
            t = " ".join(phrase_df.text.tolist())
            text.append(t)
            conf = phrase_df.confidence.mean()
            confidence.append(conf)
            l_no = sorted(set(phrase_df["line_numbers"].tolist()))
            line_numbers.append(l_no)
            phrase_numbers.append(phrase_number)
        phrase_df = pd.DataFrame()
        phrase_df["top_left_x"] = top_left_x
        phrase_df["top_left_y"] = top_left_y
        phrase_df["top_right_x"] = top_right_x
        phrase_df["top_right_y"] = top_right_y
        phrase_df["bottom_left_x"] = bottom_left_x
        phrase_df["bottom_left_y"] = bottom_left_y
        phrase_df["bottom_right_x"] = bottom_right_x
        phrase_df["bottom_right_y"] = bottom_right_y
        phrase_df["text"] = text
        phrase_df["confidence"] = confidence
        phrase_df["line_numbers"] = line_numbers
        phrase_df["phrase_numbers"] = phrase_numbers
        if bbox_type == "custom":
            phrase_df = aligned_bbox(input_file=image_path, dataframe=phrase_df)
        self.phrase_ocr.dataframes.phrase_df = phrase_df
        return phrase_df


    def plot_bounding_box(self, image_path=None, phrase_df=None):
        if image_path:
            self.initialize_image_path(image_path)
        if image_path is None:
            image_path = self.phrase_ocr.image_path
        if phrase_df is None:
            _ = self.phrase_ocr.dataframes.phrase_df
        if self.phrase_ocr.cv2_images.cv2_image is None:
            bb_img = self.to_cv2(image_path)
            self.phrase_ocr.cv2_images.cv2_image = bb_img
        else:
            bb_img = self.phrase_ocr.cv2_images.cv2_image
        for text, top_left_x, top_left_y, bottom_right_x, bottom_right_y in zip(
            phrase_df["text"],
            phrase_df["top_left_x"],
            phrase_df["top_left_y"],
            phrase_df["bottom_right_x"],
            phrase_df["bottom_right_y"],
        ):
            top_left_x = int(top_left_x)
            top_left_y = int(top_left_y)
            bottom_right_x = int(bottom_right_x)
            bottom_right_y = int(bottom_right_y)
            cv2.rectangle(
                bb_img,
                (top_left_x, top_left_y),
                (bottom_right_x, bottom_right_y),
                (50, 50, 255),
                1,
            )
        # Convert the color.
        bb_im_pil = self.to_pil(bb_img)
        self.phrase_ocr.pil_images.pil_image = bb_im_pil
        self.phrase_ocr.cv2_images.bounding_box_cv2_image = bb_img
        self.phrase_ocr.pil_images.bounding_box_pil_image = bb_im_pil
        return bb_img, bb_im_pil

    def plot_bounding_box_and_name(self, image_path=None, phrase_df=None):
        if image_path:
            self.initialize_image_path(image_path)
        if image_path is None:
            image_path = self.phrase_ocr.image_path
        if phrase_df is None:
            phrase_df = self.phrase_ocr.dataframes.phrase_df
        if self.phrase_ocr.cv2_images.cv2_image is None:
            bb_txt_img = self.to_cv2(image_path)
            self.phrase_ocr.cv2_images.cv2_image = bb_txt_img
        else:
            bb_txt_img = self.phrase_ocr.cv2_images.cv2_image
        for text, top_left_x, top_left_y, bottom_right_x, bottom_right_y in zip(
            phrase_df["text"],
            phrase_df["top_left_x"],
            phrase_df["top_left_y"],
            phrase_df["bottom_right_x"],
            phrase_df["bottom_right_y"],
        ):
            top_left_x = int(top_left_x)
            top_left_y = int(top_left_y)
            bottom_right_x = int(bottom_right_x)
            bottom_right_y = int(bottom_right_y)
            cv2.rectangle(
                bb_txt_img,
                (top_left_x, top_left_y),
                (bottom_right_x, bottom_right_y),
                (50, 50, 255),
                1,
            )
            cv2.putText(
                bb_txt_img,
                text,
                (top_left_x, top_left_y - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 0, 0),
            )
        # Convert the color.
        bb_txt_im_pil = self.to_pil(bb_txt_img)
        self.phrase_ocr.pil_images.pil_image = bb_txt_img
        self.phrase_ocr.cv2_images.bounding_box_text_cv2_image = bb_txt_img
        self.phrase_ocr.pil_images.bounding_box_text_pil_image = bb_txt_im_pil
        return bb_txt_img, bb_txt_im_pil

    def pdf_phrase_ocr(
        self, source: str or pathlib.Path, working_path: str or pathlib.Path
    ) -> List[dict]:
        """
        A trained document ocr model.

        Parameters
        ----------
        source : str or pathlib.Path
            Input file path.
        working_path : str or pathlib.Path
            Output folder directory path where image pages are saved for
            intermediate processing.

        Returns
        -------
        classification_list : List[dict]
            List of dictionaries.

        """
        if not isinstance(working_path, pathlib.Path):
            working_path = pathlib.Path(working_path)
        if not isinstance(source, pathlib.Path):
            source = pathlib.Path(source)
        kind = filetype.guess(source.as_posix())
        if kind is None:
            raise AttributeError
        mime = kind and kind.mime
        try:
            if mime.startswith("application/pdf"):
                output_files, count = convert_pdf2img(
                    input_file=source.as_posix(),
                    count=-1,
                    output_path=working_path.as_posix(),
                )
            elif mime.startswith("image"):
                output_files = [source.as_posix()]
        except ValueError as e:
            return e
        phrase_ocrs = []
        for page_number, output_file in enumerate(output_files):
            _ = self.get_easy_ocr_dataframe(output_file)
            bb_img, bb_im_pil = self.plot_bounding_box()
            bb_txt_img, bb_txt_im_pil = self.plot_bounding_box_and_name()
            phrase_ocr = self.phrase_ocr
            phrase_ocrs.append(phrase_ocr)
        return phrase_ocrs


if __name__ == "__main__":
    project_parentdir = basedir
    demo_folder = pathlib.Path(project_parentdir) / "images"
    for file in os.listdir(demo_folder):
        sample_image = demo_folder / file
        phrase_ocr_object = PhraseOCR(sample_image)
        ocr_df = phrase_ocr_object.get_easy_ocr_dataframe()
        bb_img, bb_im_pil = phrase_ocr_object.plot_bounding_box()
        bb_txt_img, bb_txt_im_pil = phrase_ocr_object.plot_bounding_box_and_name()

        ocr_df = phrase_ocr_object.phrase_ocr.dataframes.ocr_df
        phrase_df = phrase_ocr_object.phrase_ocr.dataframes.phrase_df
