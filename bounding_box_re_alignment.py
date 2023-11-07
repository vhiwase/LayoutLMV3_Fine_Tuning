import cv2
import numpy as np
import pandas as pd
from PIL import Image
import string
from itertools import chain
from collections import defaultdict
from sklearn.cluster import MeanShift


def rotate_array_clockwise(img, angle):
    arr = img
    if angle==90:
        arr = arr.transpose()
    elif angle==180:
        arr = np.flip(arr,0)
    elif angle == 270:
        arr = np.flip(arr,1)
        arr = arr.transpose()
    return arr


def analyze_char(ch):
    if ch.islower():
        return "L"
    elif ch.isupper():
        return "U"
    elif ch.isdigit():
        return "D"
    elif ch in string.punctuation:
        return "P"
    return "O"


def identify_text_type(text):
    string_type = ""
    for ch in text:
        string_type += analyze_char(ch)
    return string_type


def delete_white_space(arr, white_pixels):
    count = 0
    for row in arr:
        if set(row) <= set(white_pixels):
            count +=1
        else:
            break
    return count 


def extend_width(img, x_new, y_new, w_new, h_new, white_pixels):
    width_increment = 0
    h_orig,w_orig = img.shape
    try:
        while w_new < w_orig:
            new_img = img[y_new:h_new,x_new:w_new + width_increment]
            col = new_img[:,-1]

            lst = [pix for pix in col if pix in white_pixels]

            if len(col) == len(lst):
                break
            width_increment +=1
    except:
        return 0            
    return width_increment-1


def get_range(pixel):    
    X = np.array(pixel).reshape(-1,1)
    clustering = MeanShift().fit(X)    
    d = defaultdict(list)
    for x,l in zip(X,clustering.labels_):
        d[l].append(x[0]) 
    d = dict(d)
    black_px_range = None
    white_px_range = None
    min_thresh = 255
    max_thresh = 0
    for k,v in d.items():
        if min(v) <= min_thresh:
            black_px_range = (min(v),max(v))
        if max(v) >= max_thresh:
            white_px_range = (min(v),max(v))
        min_thresh = min(v)
        max_thresh = max(v)
    return black_px_range,white_px_range


def convert_dataframe_bbox_image(input_file, dataframe):
    img = cv2.imread(input_file)
    for i in range(len(dataframe)):
        x = dataframe.aligned_x[i]
        y = dataframe.aligned_y[i]
        w = dataframe.aligned_w[i]
        h = dataframe.aligned_h[i]
        x, y, w, h = int(x), int(y), int(w), int(h)
        try:
            img = cv2.rectangle(img, (x, y), (w, h), (0,255,0), 1)
        except:
            pass
    return Image.fromarray(img)


def get_pixel_level_details(input_file, dataframe):
    img = cv2.imread(input_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if 'width' in dataframe.columns:
        w_inch = dataframe['width'][0]
        h_inch = dataframe['height'][0]
        # Dots per inch
        dpi_x = img.shape[1]/w_inch
        dpi_y = img.shape[0]/h_inch
    else:
        dpi_x = 1.
        dpi_y = 1.
    pixel_level_dataframe = pd.DataFrame()
    pixel_level_dataframe['text'] = dataframe['text']
    pixel_level_dataframe['text_type'] = pixel_level_dataframe['text'].apply(identify_text_type)
    pixel_level_dataframe['x'] = dataframe.top_left_x*dpi_x
    pixel_level_dataframe['y'] = dataframe.top_left_y*dpi_y
    pixel_level_dataframe['w'] = dataframe.top_right_x*dpi_x
    pixel_level_dataframe['h'] = dataframe.bottom_right_y*dpi_y
    pixel_level_dataframe['x'] = pixel_level_dataframe['x'].astype(int)
    pixel_level_dataframe['y'] = pixel_level_dataframe['y'].astype(int)
    pixel_level_dataframe['w'] = pixel_level_dataframe['w'].astype(int)
    pixel_level_dataframe['h'] = pixel_level_dataframe['h'].astype(int)
    pixel_list =[]
    light_px_range =[]
    dark_px_range =[]
    for i in range(len(dataframe)):
        x = pixel_level_dataframe['x'][i]
        y = pixel_level_dataframe['y'][i]
        w = pixel_level_dataframe['w'][i]
        h = pixel_level_dataframe['h'][i]       
        im = img[y:h,x:w]
        pixels = list(set(chain(*im)))
        pixels = sorted(pixels)
        # Clustering
        dark,light = get_range(pixels)
        light_pixels = pixels[pixels.index(light[0]):]
        dark_pixels = [pixel for pixel in pixels if pixel not in light_pixels]
        pixel_list.append(list(pixels))
        light_px_range.append(light_pixels)
        dark_px_range.append(dark_pixels)
    pixel_level_dataframe['pixels'] = pixel_list
    pixel_level_dataframe['light_px_range'] = light_px_range
    pixel_level_dataframe['dark_px_range'] = dark_px_range
    pixel_level_dataframe['unit'] = 'pixel'
    return pixel_level_dataframe


# GET BLACK AND WHITE PIXELS WITHOUT CLUSTERING
def get_pixels_type(pixels, thresh):
    default_black = [0]
    default_white = [255]
    # BLACK PIXELS ONLY
    if all(pixel <= thresh for pixel in pixels):
        return pixels,default_white
    # WHITE PIXELS ONLY
    elif all(pixel > thresh for pixel in pixels):
        return default_black,pixels
    # BOTH BLACK AND WHITE PIXELS
    else:
        size = len(pixels)
        if size == 0:
            return default_black,default_white  
        elif size == 1:
            if pixels[0]>thresh:
                return default_black,pixels[0]
            else:
                return pixels[0],default_white
        else:
            black = [pixel for pixel in pixels if pixel <= thresh]
            white = [pixel for pixel in pixels if pixel > thresh]
            if len(black)==0:
                black = default_black
            if len(white)==0:
                white = default_white
            return white, black
        
        
def get_alligned_coord(img, white_pixels):
    # y,x,h,w
    angles = [0,90,180,270]
    alligned_coordinates = []
    for angle in angles:
        rotated_img = rotate_array_clockwise(img, angle)
        change = delete_white_space(rotated_img, white_pixels)
        alligned_coordinates.append(change)
    return alligned_coordinates
        
    
def aligned_coord(input_file, pixel_level_dataframe):
    img = cv2.imread(input_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    a_x = []
    a_y = []
    a_w = []
    a_h = []
    for i in range(len(pixel_level_dataframe)):
        x = pixel_level_dataframe['x'][i]
        y = pixel_level_dataframe['y'][i]
        w = pixel_level_dataframe['w'][i]
        h = pixel_level_dataframe['h'][i]       
        im = img[y:h,x:w]
        white_pixels = pixel_level_dataframe['light_px_range'][i]#[-1:]
        black_pixels = pixel_level_dataframe['dark_px_range'][i]#[:10]
        y_new,x_new,h_new,w_new = get_alligned_coord(im, white_pixels)
        aligned_x = x+x_new
        aligned_y = y+y_new
        aligned_w = w-w_new
        aligned_h = h-h_new
        width_increment = extend_width(img, aligned_x, aligned_y, aligned_w, aligned_h, white_pixels)
        a_x.append(aligned_x)
        a_y.append(aligned_y)
        a_w.append(aligned_w + width_increment)
        a_h.append(aligned_h)
    pixel_level_dataframe['aligned_x'] = a_x   
    pixel_level_dataframe['aligned_y'] = a_y 
    pixel_level_dataframe['aligned_w'] = a_w
    pixel_level_dataframe['aligned_h'] = a_h
    # -1 GIVES ACCURATE BBOX.MAYBE ONE PIXEL VALUE WAS SHIFTED IN +X,+Y IN THE PROCESS
    pixel_level_dataframe['aligned_x'] = pixel_level_dataframe['aligned_x'] - 1
    pixel_level_dataframe['aligned_y'] = pixel_level_dataframe['aligned_y'] - 1
    return pixel_level_dataframe


def aligned_bbox(input_file, dataframe):
    pixel_level_dataframe = get_pixel_level_details(input_file, dataframe)
    aligned_bbox_df = aligned_coord(input_file, pixel_level_dataframe)
    aligned_bbox_df['top_left_x'] = aligned_bbox_df['aligned_x']
    aligned_bbox_df['bottom_left_x'] = aligned_bbox_df['aligned_x']
    aligned_bbox_df['top_left_y'] = aligned_bbox_df['aligned_y']
    aligned_bbox_df['top_right_y'] = aligned_bbox_df['aligned_y']
    aligned_bbox_df['top_right_x'] = aligned_bbox_df['aligned_w']
    aligned_bbox_df['bottom_right_x'] = aligned_bbox_df['aligned_w']
    aligned_bbox_df['bottom_left_y'] = aligned_bbox_df['aligned_h']
    aligned_bbox_df['bottom_right_y'] = aligned_bbox_df['aligned_h']    
    return aligned_bbox_df


if __name__ == '__main__':    
    from ocr.phrase_ocr import PhraseOCR
    image_path = 'C:/Users/VaibhavHiwase/OneDrive - TechnoMile/Documents/Python Scripts/ICI/LayoutLMV3_Fine_Tuning/images/19GB5019R0006_+GMC+LES+Health+Insurance-pages-3.png'
    phrase_ocr_object = PhraseOCR()
    phrase_df = phrase_ocr_object.get_easy_ocr_dataframe(image_path=image_path)
    df = aligned_bbox(input_file=image_path, dataframe=phrase_df)
    convert_dataframe_bbox_image(input_file=image_path, dataframe=df)
    
    
    

























