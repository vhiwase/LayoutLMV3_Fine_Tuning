# pip install azure-ai-formrecognizer==3.3.0b1 --upgrade

# https://westus.dev.cognitive.microsoft.com/docs/services/form-recognizer-api-2023-07-31/operations/AnalyzeDocument
# https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/concept-contract?view=doc-intel-3.1.0#automated-contract-processing

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from collections import defaultdict
import dataclasses
from pydantic.dataclasses import dataclass
import pandas as pd
from utils import packing
from dotenv import load_dotenv
from pathlib import Path
import os

FILE_PATH = Path(__file__)
BASE_PATH = FILE_PATH.parent

dotenv_path = BASE_PATH /'.env'
dotenv_path = dotenv_path.absolute().as_posix()

load_dotenv(dotenv_path=dotenv_path)

ENDPOINT = os.getenv('ENDPOINT')
KEY = os.getenv('KEY')

def get_form_recognizer_result(byte_data):
    document_analysis_client = DocumentAnalysisClient(
        endpoint=ENDPOINT, credential=AzureKeyCredential(KEY)
    )
    poller = document_analysis_client.begin_analyze_document(
            "prebuilt-read", byte_data
    )
    result = poller.result()
    document_analysis_client.close()
    del document_analysis_client
    del poller
    return result


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
    

def create_read_results_dataframes(result):
    """
    A function which parsers encoded response and create dataframes.

    Parameters
    ----------
    result : azure.ai.formrecognizer._models.AnalyzeResult
        A dataclass of RespJson.

    Returns
    -------
    line_dataframe : pd.DataFrame
        Dataframe containing information of  OCR lines.
    word_dataframe : pd.DataFrame
        Dataframe containing information of OCR words.
    selection_mark_dataframe : pd.DataFrame
        Dataframe containing information of OCR selection marks.

    """
    read_results = result.pages

    # Lines and words
    line_ocr_coordinates_dictionary = defaultdict(list)
    line_ocr_texts = []
    line_pages = []
    line_angles = []
    line_widths = []
    line_heights = []
    line_units = []
    line_numbers = []
    line_number = 0

    word_ocr_coordinates_dictionary = defaultdict(list)
    word_ocr_texts = []
    word_ocr_text_confidences = []
    word_pages = []
    word_angles = []
    word_widths = []
    word_heights = []
    word_units = []
    word_line_numbers = []
    word_numbers = []
    word_number_counter = 0

    # Selection marks
    selection_marks_dictionary = defaultdict(list)
    selection_marks_confidences = []
    selection_marks_state = []
    selection_marks_pages = []
    selection_marks_angles = []
    selection_marks_widths = []
    selection_marks_heights = []
    selection_marks_units = []
    selection_marks_line_numbers = []
    selection_mark_numbers = []
    selection_mark_number_counter = 0

    for read_result in read_results:
        page = read_result.page_number
        angle = read_result.angle
        width = read_result.width
        height = read_result.height
        unit = read_result.unit
        # Lines and words
        lines = read_result.lines
        for line in lines:
            line_number += 1
            line_numbers.append(line_number)
            line_text = line.content
            line_ocr_texts.append(line_text)
            line_pages.append(page)
            line_angles.append(angle)
            line_widths.append(width)
            line_heights.append(height)
            line_units.append(unit)
            # get line coordinates
            line_bounding_box = line.polygon
            line_ocr_coordinates = OCRCoordinates(
                top_left_x=line_bounding_box[0].x,
                top_left_y=line_bounding_box[0].y,
                top_right_x=line_bounding_box[1].x,
                top_right_y=line_bounding_box[1].y,
                bottom_right_x=line_bounding_box[2].x,
                bottom_right_y=line_bounding_box[2].y,
                bottom_left_x=line_bounding_box[3].x,
                bottom_left_y=line_bounding_box[3].y,
            )
            for line_corrd_key, line_corrd_value in dataclasses.asdict(
                line_ocr_coordinates
            ).items():
                line_ocr_coordinates_dictionary[line_corrd_key].append(line_corrd_value)
        words = read_result.words
        for word in words:
            word_line_numbers.append(line_number)
            word_number_counter += 1
            word_numbers.append(word_number_counter)
            word_ocr_texts.append(word.content)
            word_ocr_text_confidences.append(word.confidence)
            word_pages.append(page)
            word_angles.append(angle)
            word_widths.append(width)
            word_heights.append(height)
            word_units.append(unit)
            # get word coordinates
            word_bounding_box = word.polygon
            word_ocr_coordinates = OCRCoordinates(
                top_left_x=word_bounding_box[0].x,
                top_left_y=word_bounding_box[0].y,
                top_right_x=word_bounding_box[1].x,
                top_right_y=word_bounding_box[1].y,
                bottom_right_x=word_bounding_box[2].x,
                bottom_right_y=word_bounding_box[2].y,
                bottom_left_x=word_bounding_box[3].x,
                bottom_left_y=word_bounding_box[3].y,
            )
            for word_corrd_key, word_corrd_value in dataclasses.asdict(
                word_ocr_coordinates
            ).items():
                word_ocr_coordinates_dictionary[word_corrd_key].append(word_corrd_value)
        # Selection marks
        selection_marks = read_result.selection_marks
        for selection_mark in selection_marks:
            selection_marks_line_numbers.append(line_number)
            selection_mark_number_counter += 1
            selection_mark_numbers.append(selection_mark_number_counter)
            selection_marks_confidences.append(selection_mark.confidence)
            selection_marks_state.append(selection_mark.state)
            selection_marks_pages.append(page)
            selection_marks_angles.append(angle)
            selection_marks_widths.append(width)
            selection_marks_heights.append(height)
            selection_marks_units.append(unit)
            # get selection_mark coordinates
            selection_mark_bounding_box = selection_mark.polygon
            selection_mark_coordinates = OCRCoordinates(
                top_left_x=selection_mark_bounding_box[0].x,
                top_left_y=selection_mark_bounding_box[0].y,
                top_right_x=selection_mark_bounding_box[1].x,
                top_right_y=selection_mark_bounding_box[1].y,
                bottom_right_x=selection_mark_bounding_box[2].x,
                bottom_right_y=selection_mark_bounding_box[2].y,
                bottom_left_x=selection_mark_bounding_box[3].x,
                bottom_left_y=selection_mark_bounding_box[3].y,
            )
            for (
                selection_mark_corrd_key,
                selection_mark_corrd_value,
            ) in dataclasses.asdict(selection_mark_coordinates).items():
                selection_marks_dictionary[selection_mark_corrd_key].append(
                    selection_mark_corrd_value
                )

    # Creating line DataFrame
    line_dictionary = dict()
    line_dictionary["text"] = line_ocr_texts
    line_dictionary["line_numbers"] = line_numbers
    line_dictionary.update(line_ocr_coordinates_dictionary)
    line_dataframe = pd.DataFrame(line_dictionary)
    line_dataframe["page"] = line_pages
    line_dataframe["angle"] = line_angles
    line_dataframe["width"] = line_widths
    line_dataframe["height"] = line_heights
    line_dataframe["unit"] = line_units
    line_dataframe = line_dataframe.sort_values(by=['page', 'bottom_left_y', 'bottom_left_x'], ignore_index=True)
    line_dataframe['bottom_left_x_diff'] = line_dataframe['bottom_left_y'].diff()
    line_dataframe['bottom_left_x_diff_bool'] = line_dataframe['bottom_left_x_diff'].apply(lambda x: True if 0<=x<0.015 else False)
    line_numbers = []
    line_number_count = 0
    for true_false in line_dataframe['bottom_left_x_diff_bool']:
        if true_false == False:
            line_number_count += 1
        line_numbers.append(line_number_count)
    line_dataframe['line_numbers'] = line_numbers
    line_dataframe = line_dataframe.sort_values(by=['page', 'line_numbers', 'bottom_left_x'], ignore_index=True)


    # Creating word DataFrame
    word_dictionary = dict()
    word_dictionary["text"] = word_ocr_texts
    word_dictionary["line_numbers"] = word_line_numbers
    word_dictionary["word_numbers"] = word_numbers
    word_dictionary["confidence"] = word_ocr_text_confidences
    word_dictionary.update(word_ocr_coordinates_dictionary)
    word_dataframe = pd.DataFrame(word_dictionary)
    word_dataframe["page"] = word_pages
    word_dataframe["angle"] = word_angles
    word_dataframe["width"] = word_widths
    word_dataframe["height"] = word_heights
    word_dataframe["unit"] = word_units

    # Creating selection mark DataFrame
    selection_mark_dictionary = dict()
    selection_mark_dictionary["state"] = selection_marks_state
    selection_mark_dictionary["confidence"] = selection_marks_confidences
    selection_mark_dictionary["line_number"] = selection_marks_line_numbers
    selection_mark_dictionary["selection_mark_number"] = selection_mark_numbers
    selection_mark_dictionary.update(selection_marks_dictionary)
    selection_mark_dataframe = pd.DataFrame(selection_mark_dictionary)
    selection_mark_dataframe["page"] = selection_marks_pages
    selection_mark_dataframe["angle"] = selection_marks_angles
    selection_mark_dataframe["width"] = selection_marks_widths
    selection_mark_dataframe["height"] = selection_marks_heights
    selection_mark_dataframe["unit"] = selection_marks_units

    return line_dataframe, word_dataframe, selection_mark_dataframe


if __name__ == '__main__':
    file_path = '/path/to/some_pdf_file.pdf'
    with open(file_path, 'rb') as f:
        byte_data = f.read()
    result = get_form_recognizer_result(byte_data)
    line_dataframe, word_dataframe, selection_mark_dataframe = create_read_results_dataframes(result)
    packing(line_dataframe, "line_dataframe.pkl")
    packing(word_dataframe, "word_dataframe.pkl")
    packing(selection_mark_dataframe, "selection_mark_dataframe.pkl")
    
