# pip install azure-ai-formrecognizer==3.3.0b1 --upgrade

# https://westus.dev.cognitive.microsoft.com/docs/services/form-recognizer-api-2023-07-31/operations/AnalyzeDocument
# https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/concept-contract?view=doc-intel-3.1.0#automated-contract-processing

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from collections import defaultdict
import dataclasses
from pydantic.dataclasses import dataclass
import pandas as pd
import numpy as np
from utils import packing, unpacking
import time
from sklearn.cluster import MeanShift, estimate_bandwidth

from dotenv import load_dotenv
from pathlib import Path
import os
try:
    FILE_PATH = Path(__file__)
except NameError:
    FILE_PATH = Path('.')
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
            "prebuilt-document", byte_data # prebuilt-layout
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
    line_numbers_count = 0
    for true_false in line_dataframe['bottom_left_x_diff_bool']:
        if true_false == False:
            line_numbers_count += 1
        line_numbers.append(line_numbers_count)
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
    selection_mark_dictionary["line_numbers"] = selection_marks_line_numbers
    selection_mark_dictionary["selection_mark_number"] = selection_mark_numbers
    selection_mark_dictionary.update(selection_marks_dictionary)
    selection_mark_dataframe = pd.DataFrame(selection_mark_dictionary)
    selection_mark_dataframe["page"] = selection_marks_pages
    selection_mark_dataframe["angle"] = selection_marks_angles
    selection_mark_dataframe["width"] = selection_marks_widths
    selection_mark_dataframe["height"] = selection_marks_heights
    selection_mark_dataframe["unit"] = selection_marks_units

    role_lst = []
    content_lst = []
    page_number_lst = []
    offset_lst = []
    offset_length_lst = []
    top_left_x_lst = []
    top_left_y_lst = []
    top_right_x_lst = []
    top_right_y_lst = []
    bottom_right_x_lst = []
    bottom_right_y_lst = []
    bottom_left_x_lst = []
    bottom_left_y_lst = []

    d = result.to_dict()    
    for i in d['paragraphs']:
        role = i['role']
        content = i['content']
        spans = i['spans']
        bounding_regions = i['bounding_regions']
        for boundary_region, span in zip(bounding_regions, spans):
            polygon = boundary_region['polygon']
            page_number = boundary_region['page_number']
            offset = span['offset']
            offset_length = span['length']
            paragraph_ocr_coordinates = OCRCoordinates(
                top_left_x=polygon[0]['x'],
                top_left_y=polygon[0]['y'],
                top_right_x=polygon[1]['x'],
                top_right_y=polygon[1]['y'],
                bottom_right_x=polygon[2]['x'],
                bottom_right_y=polygon[2]['y'],
                bottom_left_x=polygon[3]['x'],
                bottom_left_y=polygon[3]['y'])
            role_lst.append(role)
            content_lst.append(content)
            page_number_lst.append(page_number)
            offset_lst.append(offset)
            offset_length_lst.append(offset_length)
            top_left_x_lst.append(paragraph_ocr_coordinates.top_left_x)
            top_left_y_lst.append(paragraph_ocr_coordinates.top_left_y)
            top_right_x_lst.append(paragraph_ocr_coordinates.top_right_x)
            top_right_y_lst.append(paragraph_ocr_coordinates.top_right_y)
            bottom_right_x_lst.append(paragraph_ocr_coordinates.bottom_right_x)
            bottom_right_y_lst.append(paragraph_ocr_coordinates.bottom_right_y)
            bottom_left_x_lst.append(paragraph_ocr_coordinates.bottom_left_x)
            bottom_left_y_lst.append(paragraph_ocr_coordinates.bottom_left_y)
    
    paragraph_dataframe = pd.DataFrame()
    paragraph_dataframe['role'] = role_lst
    paragraph_dataframe['content'] = content_lst
    paragraph_dataframe['page_number'] = page_number_lst
    paragraph_dataframe['offset'] = offset_lst
    paragraph_dataframe['offset_length'] = offset_length_lst
    paragraph_dataframe['top_left_x'] = top_left_x_lst
    paragraph_dataframe['top_left_y'] = top_left_y_lst
    paragraph_dataframe['top_right_x'] = top_right_x_lst
    paragraph_dataframe['top_right_y'] = top_right_y_lst
    paragraph_dataframe['bottom_right_x'] = bottom_right_x_lst
    paragraph_dataframe['bottom_right_y'] = bottom_right_y_lst
    paragraph_dataframe['bottom_left_x'] = bottom_left_x_lst
    paragraph_dataframe['bottom_left_y'] = bottom_left_y_lst


    kind_lst = []
    row_index_lst = []
    column_index_lst = []
    row_span_lst = []
    column_span_lst = []
    content_lst = []
    page_number_lst = []
    top_left_x_lst = []
    top_left_y_lst = []
    top_right_x_lst = []
    top_right_y_lst = []
    bottom_right_x_lst = []
    bottom_right_y_lst = []
    bottom_left_x_lst = []
    bottom_left_y_lst = []
    
    d = result.to_dict()    
    for i in d['tables']:
        row_count = i['row_count']
        column_count = i['column_count']
        cells = i['cells']
        for cell in cells:
            kind = cell['kind']
            row_index = cell['row_index']
            column_index = cell['column_index']
            row_span = cell['row_span']
            column_span = cell['column_span']
            content = cell['content']
            bounding_regions = cell['bounding_regions']
            for bounding_region in bounding_regions:
                page_number = bounding_region['page_number']
                page_number_lst.append(page_number)
                polygon = bounding_region['polygon']
                table_ocr_coordinates = OCRCoordinates(
                    top_left_x=polygon[0]['x'],
                    top_left_y=polygon[0]['y'],
                    top_right_x=polygon[1]['x'],
                    top_right_y=polygon[1]['y'],
                    bottom_right_x=polygon[2]['x'],
                    bottom_right_y=polygon[2]['y'],
                    bottom_left_x=polygon[3]['x'],
                    bottom_left_y=polygon[3]['y'])
                
                kind_lst.append(kind)
                row_index_lst.append(row_index)
                column_index_lst.append(column_index)
                row_span_lst.append(row_span)
                column_span_lst.append(column_span)
                content_lst.append(content)
                
                top_left_x_lst.append(table_ocr_coordinates.top_left_x)
                top_left_y_lst.append(table_ocr_coordinates.top_left_y)
                top_right_x_lst.append(table_ocr_coordinates.top_right_x)
                top_right_y_lst.append(table_ocr_coordinates.top_right_y)
                bottom_right_x_lst.append(table_ocr_coordinates.bottom_right_x)
                bottom_right_y_lst.append(table_ocr_coordinates.bottom_right_y)
                bottom_left_x_lst.append(table_ocr_coordinates.bottom_left_x)
                bottom_left_y_lst.append(table_ocr_coordinates.bottom_left_y)
        
        bounding_regions = i['bounding_regions']
        spans = i['spans']
        for span, bounding_regions in zip(spans, bounding_regions):
            # To be used in above for loop or based on use cases to join 
            # or merge two different table bounding box
            pass

        table_dataframe = pd.DataFrame()
        table_dataframe['kind'] = kind_lst
        table_dataframe['row_index'] = row_index_lst
        table_dataframe['column_index'] = column_index_lst
        table_dataframe['row_span'] = row_span_lst
        table_dataframe['column_span'] = column_span_lst
        table_dataframe['content'] = content_lst
        table_dataframe['page_number'] = page_number_lst
        table_dataframe['top_left_x'] = top_left_x_lst
        table_dataframe['top_left_y'] = top_left_y_lst
        table_dataframe['top_right_x'] = top_right_x_lst
        table_dataframe['top_right_y'] = top_right_y_lst
        table_dataframe['bottom_right_x'] = bottom_right_x_lst
        table_dataframe['bottom_right_y'] = bottom_right_y_lst
        table_dataframe['bottom_left_x'] = bottom_left_x_lst
        table_dataframe['bottom_left_y'] = bottom_left_y_lst
    
    def get_values_from_key_value_pair(key_value_pair_dict, pair_type):
        content = key_value_pair_dict['content']
        bounding_regions = key_value_pair_dict['bounding_regions']
        spans = key_value_pair_dict['spans']

        for bounding_region in bounding_regions:
            page_number = bounding_region['page_number']
            page_number_lst.append(page_number)
            polygon = bounding_region['polygon']
            key_value_pair_ocr_coordinates = OCRCoordinates(
                top_left_x=polygon[0]['x'],
                top_left_y=polygon[0]['y'],
                top_right_x=polygon[1]['x'],
                top_right_y=polygon[1]['y'],
                bottom_right_x=polygon[2]['x'],
                bottom_right_y=polygon[2]['y'],
                bottom_left_x=polygon[3]['x'],
                bottom_left_y=polygon[3]['y'])
            pair_types.append(pair_type)
            content_lst.append(content)
            top_left_x_lst.append(key_value_pair_ocr_coordinates.top_left_x)
            top_left_y_lst.append(key_value_pair_ocr_coordinates.top_left_y)
            top_right_x_lst.append(key_value_pair_ocr_coordinates.top_right_x)
            top_right_y_lst.append(key_value_pair_ocr_coordinates.top_right_y)
            bottom_right_x_lst.append(key_value_pair_ocr_coordinates.bottom_right_x)
            bottom_right_y_lst.append(key_value_pair_ocr_coordinates.bottom_right_y)
            bottom_left_x_lst.append(key_value_pair_ocr_coordinates.bottom_left_x)
            bottom_left_y_lst.append(key_value_pair_ocr_coordinates.bottom_left_y)
        key_value_pair_dataframe = pd.DataFrame()
        key_value_pair_dataframe['pair_type'] = pair_types
        key_value_pair_dataframe['content'] = content_lst
        key_value_pair_dataframe['page_number'] = page_number_lst
        key_value_pair_dataframe['top_left_x'] = top_left_x_lst
        key_value_pair_dataframe['top_left_y'] = top_left_y_lst
        key_value_pair_dataframe['top_right_x'] = top_right_x_lst
        key_value_pair_dataframe['top_right_y'] = top_right_y_lst
        key_value_pair_dataframe['bottom_right_x'] = bottom_right_x_lst
        key_value_pair_dataframe['bottom_right_y'] = bottom_right_y_lst
        key_value_pair_dataframe['bottom_left_x'] = bottom_left_x_lst
        key_value_pair_dataframe['bottom_left_y'] = bottom_left_y_lst
    
    d = result.to_dict()
    pair_numbers = []
    pair_types = []
    content_lst = []
    page_number_lst = []
    top_left_x_lst = []
    top_left_y_lst = []
    top_right_x_lst = []
    top_right_y_lst = []
    bottom_right_x_lst = []
    bottom_right_y_lst = []
    bottom_left_x_lst = []
    bottom_left_y_lst = []
    for enum, key_value_pair in enumerate(d['key_value_pairs']):
        pair_type = 'key'
        value = key_value_pair[pair_type]
        key_value_pair_dict = value
        if key_value_pair_dict is None:            
            content = key_value_pair_dict
            bounding_regions = []
            spans = []
            pair_numbers.append(enum)
            page_number_lst.append(page_number)
            pair_types.append(pair_type)
            content_lst.append(content)
            top_left_x_lst.append(None)
            top_left_y_lst.append(None)
            top_right_x_lst.append(None)
            top_right_y_lst.append(None)
            bottom_right_x_lst.append(None)
            bottom_right_y_lst.append(None)
            bottom_left_x_lst.append(None)
            bottom_left_y_lst.append(None)
        else:
            content = key_value_pair_dict and key_value_pair_dict['content']
            bounding_regions = key_value_pair_dict and key_value_pair_dict['bounding_regions']
            spans = key_value_pair_dict and key_value_pair_dict['spans']
        for bounding_region in bounding_regions:
            page_number = bounding_region['page_number']
            page_number_lst.append(page_number)
            polygon = bounding_region['polygon']
            key_value_pair_ocr_coordinates = OCRCoordinates(
                top_left_x=polygon[0]['x'],
                top_left_y=polygon[0]['y'],
                top_right_x=polygon[1]['x'],
                top_right_y=polygon[1]['y'],
                bottom_right_x=polygon[2]['x'],
                bottom_right_y=polygon[2]['y'],
                bottom_left_x=polygon[3]['x'],
                bottom_left_y=polygon[3]['y'])
            pair_numbers.append(enum)
            pair_types.append(pair_type)
            content_lst.append(content)
            top_left_x_lst.append(key_value_pair_ocr_coordinates.top_left_x)
            top_left_y_lst.append(key_value_pair_ocr_coordinates.top_left_y)
            top_right_x_lst.append(key_value_pair_ocr_coordinates.top_right_x)
            top_right_y_lst.append(key_value_pair_ocr_coordinates.top_right_y)
            bottom_right_x_lst.append(key_value_pair_ocr_coordinates.bottom_right_x)
            bottom_right_y_lst.append(key_value_pair_ocr_coordinates.bottom_right_y)
            bottom_left_x_lst.append(key_value_pair_ocr_coordinates.bottom_left_x)
            bottom_left_y_lst.append(key_value_pair_ocr_coordinates.bottom_left_y)
        
        pair_type = 'value'
        value = key_value_pair[pair_type]
        key_value_pair_dict = value
        if key_value_pair_dict is None:            
            content = key_value_pair_dict
            bounding_regions = []
            spans = []
            pair_numbers.append(enum)
            page_number_lst.append(page_number)
            pair_types.append(pair_type)
            content_lst.append(content)
            top_left_x_lst.append(None)
            top_left_y_lst.append(None)
            top_right_x_lst.append(None)
            top_right_y_lst.append(None)
            bottom_right_x_lst.append(None)
            bottom_right_y_lst.append(None)
            bottom_left_x_lst.append(None)
            bottom_left_y_lst.append(None)
        else:
            content = key_value_pair_dict and key_value_pair_dict['content']
            bounding_regions = key_value_pair_dict and key_value_pair_dict['bounding_regions']
            spans = key_value_pair_dict and key_value_pair_dict['spans']
        for bounding_region in bounding_regions:
            page_number = bounding_region['page_number']
            page_number_lst.append(page_number)
            polygon = bounding_region['polygon']
            key_value_pair_ocr_coordinates = OCRCoordinates(
                top_left_x=polygon[0]['x'],
                top_left_y=polygon[0]['y'],
                top_right_x=polygon[1]['x'],
                top_right_y=polygon[1]['y'],
                bottom_right_x=polygon[2]['x'],
                bottom_right_y=polygon[2]['y'],
                bottom_left_x=polygon[3]['x'],
                bottom_left_y=polygon[3]['y'])
            pair_numbers.append(enum)
            pair_types.append(pair_type)
            content_lst.append(content)
            top_left_x_lst.append(key_value_pair_ocr_coordinates.top_left_x)
            top_left_y_lst.append(key_value_pair_ocr_coordinates.top_left_y)
            top_right_x_lst.append(key_value_pair_ocr_coordinates.top_right_x)
            top_right_y_lst.append(key_value_pair_ocr_coordinates.top_right_y)
            bottom_right_x_lst.append(key_value_pair_ocr_coordinates.bottom_right_x)
            bottom_right_y_lst.append(key_value_pair_ocr_coordinates.bottom_right_y)
            bottom_left_x_lst.append(key_value_pair_ocr_coordinates.bottom_left_x)
            bottom_left_y_lst.append(key_value_pair_ocr_coordinates.bottom_left_y)

    key_value_pair_dataframe = pd.DataFrame()
    key_value_pair_dataframe['pair_numbers'] = pair_numbers
    key_value_pair_dataframe['pair_type'] = pair_types
    key_value_pair_dataframe['content'] = content_lst
    key_value_pair_dataframe['page_number'] = page_number_lst
    key_value_pair_dataframe['top_left_x'] = top_left_x_lst
    key_value_pair_dataframe['top_left_y'] = top_left_y_lst
    key_value_pair_dataframe['top_right_x'] = top_right_x_lst
    key_value_pair_dataframe['top_right_y'] = top_right_y_lst
    key_value_pair_dataframe['bottom_right_x'] = bottom_right_x_lst
    key_value_pair_dataframe['bottom_right_y'] = bottom_right_y_lst
    key_value_pair_dataframe['bottom_left_x'] = bottom_left_x_lst
    key_value_pair_dataframe['bottom_left_y'] = bottom_left_y_lst
    
    def calculating_paragraph_and_column_per_page(line_dataframe, page_number):
        """
        *Author: Vaibhav Hiwase
        *Details: Creating paragraph attribute for calculating paragraph number of the text
                  present in given dataframe using clustering on coordiantes.
        """

        
        TOP_LEFT_X, TOP_LEFT_Y, TOP_RIGHT_X, TOP_RIGHT_Y, \
        BOTTOM_RIGHT_X, BOTTOM_RIGHT_Y, BOTTOM_LEFT_X, \
        BOTTOM_LEFT_Y, TEXT = 'top_left_x', 'top_left_y', 'top_right_x', \
        'top_right_y', 'bottom_right_x', 'bottom_right_y', \
        'bottom_left_x', 'bottom_left_y', 'text'
        MIN_LINE_SPACE = 0.09
        line_dataframe = line_dataframe.reset_index(drop=True)        
        # Operation on page
        page_df = line_dataframe[line_dataframe['page']==page_number]
        # Calculating vertical text
        page_df['x_diff'] = page_df.loc[:, TOP_RIGHT_X]-page_df.loc[:, TOP_LEFT_X]
        page_df['y_diff'] = page_df[TOP_RIGHT_Y]-page_df[TOP_LEFT_Y]
        temp_page_df = page_df[page_df['x_diff']==0]    
        v_df = pd.DataFrame(index=temp_page_df[TOP_LEFT_X], columns=[TEXT, 'line_numbers'])
        v_df[TEXT] = temp_page_df[TEXT].tolist()
        v_df['line_numbers'] = temp_page_df['line_numbers'].tolist()    
        my_line_num_text_dict = v_df.T.to_dict()
        page_df.loc[temp_page_df.index, 'vertical_text_lines'] = [my_line_num_text_dict for _ in range(len(temp_page_df))]
        line_dataframe.loc[temp_page_df.index, 'vertical_text_lines'] = [my_line_num_text_dict for _ in range(len(temp_page_df))]    
        dd = pd.DataFrame(index = temp_page_df.index)
        dd[TOP_LEFT_X] = temp_page_df[TOP_RIGHT_X].tolist()
        dd[TOP_LEFT_Y] = temp_page_df[TOP_RIGHT_Y].tolist()    
        dd[TOP_RIGHT_X] = temp_page_df[BOTTOM_RIGHT_X].tolist()
        dd[TOP_RIGHT_Y] = temp_page_df[BOTTOM_RIGHT_Y].tolist()    
        dd[BOTTOM_RIGHT_X] = temp_page_df[BOTTOM_LEFT_X].tolist()
        dd[BOTTOM_RIGHT_Y] = temp_page_df[BOTTOM_LEFT_Y].tolist()    
        dd[BOTTOM_LEFT_X] = temp_page_df[TOP_LEFT_X].tolist()
        dd[BOTTOM_LEFT_Y] = temp_page_df[TOP_LEFT_Y].tolist()
        if not dd.empty:
            dd[TOP_LEFT_X] = min(dd[TOP_LEFT_X])
        page_df.loc[dd.index, [TOP_LEFT_X, TOP_LEFT_Y, TOP_RIGHT_X, TOP_RIGHT_Y,
           BOTTOM_RIGHT_X, BOTTOM_RIGHT_Y, BOTTOM_LEFT_X, BOTTOM_LEFT_Y]] = dd.loc[dd.index, [TOP_LEFT_X, TOP_LEFT_Y, TOP_RIGHT_X, TOP_RIGHT_Y,
           BOTTOM_RIGHT_X, BOTTOM_RIGHT_Y, BOTTOM_LEFT_X, BOTTOM_LEFT_Y]]                                                                                                              
        line_dataframe.loc[dd.index, [TOP_LEFT_X, TOP_LEFT_Y, TOP_RIGHT_X, TOP_RIGHT_Y,
           BOTTOM_RIGHT_X, BOTTOM_RIGHT_Y, BOTTOM_LEFT_X, BOTTOM_LEFT_Y]] = dd.loc[dd.index, [TOP_LEFT_X, TOP_LEFT_Y, TOP_RIGHT_X, TOP_RIGHT_Y,
           BOTTOM_RIGHT_X, BOTTOM_RIGHT_Y, BOTTOM_LEFT_X, BOTTOM_LEFT_Y]]
        # Assigning approprate value for coordinated belonging to same line
        for li in sorted(set(page_df.line_numbers)):
            df_li = page_df[page_df['line_numbers']==li]
            page_df.loc[df_li.index, BOTTOM_RIGHT_Y] = max(df_li[BOTTOM_RIGHT_Y])
            page_df.loc[df_li.index, TOP_LEFT_Y] = min(df_li[TOP_LEFT_Y])
            page_df.loc[df_li.index, BOTTOM_LEFT_Y] = max(df_li[BOTTOM_LEFT_Y])
            page_df.loc[df_li.index, TOP_RIGHT_Y] = min(df_li[TOP_RIGHT_Y])
        # Calculating y-coordinates space above and below line
        page_df['bottom'] = [0] + page_df[BOTTOM_RIGHT_Y].tolist()[:-1]
        page_df['up_space'] = page_df[TOP_LEFT_Y] - page_df['bottom']
        page_df['down_space'] = page_df['up_space'][1:].tolist()+ [0]    
        # Assigning approprate value for coordinated belonging to same line
        for li in sorted(set(page_df.line_numbers)):
            df_li = page_df[page_df['line_numbers']==li]
            page_df.loc[df_li.index, 'up_space'] = max(df_li['up_space'])
            page_df.loc[df_li.index, 'down_space'] = max(df_li['down_space'])        
        # Filter for eliminating large bottom blank space before clustering
        page_df1 = page_df[page_df['up_space'] < 1.8]
        page_df2 = page_df[page_df['up_space'] >= 1.8]    
        if page_df1.empty:
            return line_dataframe    
        # MeanShift Clustering in space between two lines
        X = np.array(page_df1.loc[:, ['up_space']])
        model = MeanShift(n_jobs=-1)    
        # fit model and predict clusters
        yhat = model.fit_predict(X)    
        # Adding -1 cluster number for ignored words below large bottom blank space
        page_df['yhat'] = list(yhat) + [-1 for _ in range(len(page_df2))]    
        # Sorting clustering number bases on upper space of line
        page_df = page_df.sort_values(by=['up_space'])
        # Reordering clustering in ascending order based on height of upper blank space of line
        yhat_ascending_sequence = []
        count = 0
        prev_cluster_no = page_df['yhat'].tolist() and page_df['yhat'].tolist()[0]
        for cluster_no in page_df['yhat']:
            if prev_cluster_no != cluster_no:
                count += 1
            yhat_ascending_sequence.append(count)
            prev_cluster_no = cluster_no    
        page_df['yhat'] = yhat_ascending_sequence
        page_df = page_df.sort_index()    
        # Creating paragraph sequence by combining 0 with non-zerp values and lines whose upper space is less than MIN_LINE_SPACE
        paragraph_seq = []
        count = 0
        prev_line = page_df['line_numbers'].tolist() and page_df['line_numbers'].tolist()[0]
        for y, line, up_space in zip(page_df['yhat'], page_df['line_numbers'], page_df['up_space']):
            if y and line != prev_line:
                if up_space > MIN_LINE_SPACE:
                    count += 1
            prev_line = line
            paragraph_seq.append(count)
        # Adding paragraph number and sorting results
        page_df['paragraph'] = paragraph_seq
        page_df= page_df.sort_values(by=['line_numbers', TOP_LEFT_X])
        # MeanShift Clustering in top left x coordinates
        X = np.array(page_df.loc[:, [TOP_LEFT_X]])
        bandwidth = estimate_bandwidth(X, quantile=0.16, n_samples=500, n_jobs=-1)
        if bandwidth:
            model = MeanShift(bandwidth=bandwidth)
        else:
            model = MeanShift()
        xhat = model.fit_predict(X)
        cluster_centers = model.cluster_centers_
        page_df['xhat'] = xhat     
        # Sorting clustering number bases on Top left x of line
        page_df = page_df.sort_values(by=[TOP_LEFT_X])    
        # Reordering clustering in ascending order based on height of upper blank space of line
        xhat_ascending_sequence = []
        count = 0
        prev_cluster_no = page_df['xhat'].tolist() and page_df['xhat'].tolist()[0]
        for cluster_no in page_df['xhat']:
            if prev_cluster_no != cluster_no:
                count += 1
            xhat_ascending_sequence.append(count)
            prev_cluster_no = cluster_no
        
        page_df['column'] = xhat_ascending_sequence
        page_df = page_df.sort_index()    
        # Assignment of value to line_dataframe
        line_dataframe.loc[page_df.index, 'up_space'] = page_df['up_space']
        line_dataframe.loc[page_df.index, 'down_space'] = page_df['down_space']
        line_dataframe.loc[page_df.index, 'xhat'] = page_df['xhat']
        line_dataframe.loc[page_df.index, 'yhat'] = page_df['yhat']
        line_dataframe.loc[page_df.index, 'paragraph'] = page_df['paragraph']
        line_dataframe.loc[page_df.index, 'column'] = page_df['column']        
        return line_dataframe
    
    def paragraph_extraction(line_dataframe=None):  
        """
        *Author: Vaibhav Hiwase
        *Details: Creating paragraph number in line_dataframe.
        """
        line_dataframe ['vertical_text_lines'] = None
        for page_number in sorted(set(line_dataframe ['page'])):    
            line_dataframe = calculating_paragraph_and_column_per_page(line_dataframe , page_number)
        # Calculating paragraph_number column for complete PDF
        paragraph_number = []
        count = 0
        prev_para_num = line_dataframe['paragraph'].tolist() and line_dataframe['paragraph'].tolist()[0]
        for para_num in line_dataframe['paragraph']:
            if para_num==prev_para_num or pd.isna(para_num):
                pass
            else:
                count += 1
                prev_para_num = para_num        
            paragraph_number.append(count)
        line_dataframe['paragraph_numbers'] = paragraph_number
        return line_dataframe
        
    line_dataframe = paragraph_extraction(line_dataframe)
    
    text_lst = []
    paragraph_number_lst = []
    line_number_lst = []
    page_lst = []
    top_left_x_lst = []
    top_left_y_lst = []
    top_right_x_lst = []
    top_right_y_lst = []
    bottom_right_x_lst = []
    bottom_right_y_lst = []
    bottom_left_x_lst = []
    bottom_left_y_lst = []
    angle_lst = []
    width_lst = []
    height_lst = []
    for paragraph_number in line_dataframe['paragraph_numbers']:
        para_df = line_dataframe[line_dataframe['paragraph_numbers']==paragraph_number]
        top_left_x = min(para_df['top_left_x'])
        top_left_y = min(para_df['top_left_y'])
        top_right_x = max(para_df['top_right_x'])
        top_right_y = min(para_df['top_right_y'])
        bottom_left_x = min(para_df['bottom_left_x'])
        bottom_left_y = max(para_df['bottom_left_y'])
        bottom_right_x = max(para_df['bottom_right_x'])
        bottom_right_y = max(para_df['bottom_right_y'])        
        text = '\n'.join(para_df['text'].tolist())
        line_numbers = f"{min(para_df['line_numbers'])}_{max(para_df['line_numbers'])}"
        page = sorted(set(para_df['page']))[0]
        angle = sorted(set(para_df['angle']))[0]
        width = sorted(set(para_df['width']))[0]
        height = sorted(set(para_df['height']))[0]
        text_lst.append(text)
        paragraph_number_lst.append(paragraph_number)
        line_number_lst.append(line_numbers)
        page_lst.append(page)
        top_left_x_lst.append(top_left_x)
        top_left_y_lst.append(top_left_y)
        top_right_x_lst.append(top_right_x)
        top_right_y_lst.append(top_right_y)
        bottom_right_x_lst.append(bottom_right_x)
        bottom_right_y_lst.append(bottom_right_y)
        bottom_left_x_lst.append(bottom_left_x)
        bottom_left_y_lst.append(bottom_left_y)
        angle_lst.append(angle)
        width_lst.append(width)
        height_lst.append(height)
    
    paragraph_dataframe = pd.DataFrame()
    paragraph_dataframe['text'] = text_lst
    paragraph_dataframe['paragraph_number'] = paragraph_number_lst
    paragraph_dataframe['line_number'] = line_number_lst
    paragraph_dataframe['page'] = page_lst
    paragraph_dataframe['top_left_x'] = top_left_x_lst
    paragraph_dataframe['top_left_y'] = top_left_y_lst
    paragraph_dataframe['top_right_x'] = top_right_x_lst
    paragraph_dataframe['top_right_y'] = top_right_y_lst
    paragraph_dataframe['bottom_right_x'] = bottom_right_x_lst
    paragraph_dataframe['bottom_right_y'] = bottom_right_y_lst
    paragraph_dataframe['bottom_left_x'] = bottom_left_x_lst
    paragraph_dataframe['bottom_left_y'] = bottom_left_y_lst
    paragraph_dataframe['angle'] = angle_lst
    paragraph_dataframe['width'] = width_lst
    paragraph_dataframe['height'] = height_lst
    
    return line_dataframe, word_dataframe, selection_mark_dataframe, paragraph_dataframe, table_dataframe, key_value_pair_dataframe


if __name__ == '__main__':
    folder_name = '../Example 6'
    file_path = f'./{folder_name}/example_pdf.pdf'
    with open(file_path, 'rb') as f:
        byte_data = f.read()
    tic = time.time()
    result = get_form_recognizer_result(byte_data)
    toc = time.time()
    print("Total Time taken:", toc-tic)
    _ = packing(result, f'./{folder_name}/prebuilt-document.pkl')
    
    line_dataframe, word_dataframe, selection_mark_dataframe, paragraph_dataframe, table_dataframe, key_value_pair_dataframe = create_read_results_dataframes(result)
    _ = packing(line_dataframe, f"./{folder_name}/prebuilt-document_line_dataframe.pkl")
    _ = packing(word_dataframe, f"./{folder_name}/prebuilt-document_word_dataframe.pkl")
    _ = packing(selection_mark_dataframe, f"./{folder_name}/prebuilt-document_selection_mark_dataframe.pkl")
    _ = packing(paragraph_dataframe, f"./{folder_name}/prebuilt-document_paragraph_dataframe.pkl")
    _ = packing(table_dataframe, f"./{folder_name}/prebuilt-document_table_dataframe.pkl")
    _ = packing(key_value_pair_dataframe, f"./{folder_name}/prebuilt-key_value_pair_dataframe.pkl")
