import numpy as np
import pandas as pd
import PIL
from pydantic import BaseModel
from pydantic.dataclasses import dataclass
import dataclasses
from typing import List


class DataFrame(BaseModel):
    ocr_df: pd.DataFrame = None
    phrase_df: pd.DataFrame = None

    class Config:
        arbitrary_types_allowed = True


class Array(BaseModel):
    cv2_image: np.ndarray = None
    bounding_box_cv2_image: np.ndarray = None
    bounding_box_text_cv2_image: np.ndarray = None

    class Config:
        arbitrary_types_allowed = True


class Image(BaseModel):
    pil_image: PIL.Image.Image = None
    bounding_box_pil_image: PIL.Image.Image = None
    bounding_box_text_pil_image: PIL.Image.Image = None

    class Config:
        arbitrary_types_allowed = True


@dataclasses.dataclass
class OCR:
    image_path: str = None
    dataframes: DataFrame= DataFrame
    cv2_images: Array = Array
    pil_images: Image = Image


@dataclass
class OCRCoordinates:
    top_left_x: float = None
    top_left_y: float = None
    top_right_x: float = None
    top_right_y: float = None
    bottom_right_x: float = None
    bottom_right_y: float = None
    bottom_left_x: float = None
    bottom_left_y: float = None
