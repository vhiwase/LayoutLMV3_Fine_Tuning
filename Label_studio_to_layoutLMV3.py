import json
from pathlib import Path
import os

try:
    FILE_PATH = Path(__file__)
except NameError:
    FILE_PATH = Path('.')
BASE_PATH = FILE_PATH.parent


def convert_bounding_box(x, y, width, height):
	"""Converts the given bounding box coordinates to the YOLO format.

	Args:
	x: The x-coordinate of the top-left corner of the bounding box.
	y: The y-coordinate of the top-left corner of the bounding box.
	width: The width of the bounding box.
	height: The height of the bounding box.

	Returns:
	A tuple of four coordinates (x1, y1, x2, y2) in the YOLO format.
	"""

	x1 = x
	y1 = y
	x2 = x + width
	y2 = y + height

	return [x1, y1, x2, y2]



####################################### Loading json data ###################################
with open("C:/Users/VaibhavHiwase/OneDrive - TechnoMile/Documents/Python Scripts/ICI/LayoutLMV3_Fine_Tuning/output/W9127821R0012+SF-33.json") as f:
    data = json.load(f)


output = []

for annoatated_image in data:
    data = {}
    annotation = []
    ann_list = []
    if len(annoatated_image) < 8:
        continue

    for k, v in annoatated_image.items():
        if k == 'ocr':
            v = v.split('8080/')[-1]
            print(f'filename: {v}')
            training_image_folder_path = (BASE_PATH / 'working' / 'W9127821R0012+SF-33').absolute().as_posix()
            data["file_name"] = f"{training_image_folder_path}/{v}"
            output.append(data)
        if k == 'bbox':
            width = v[0]['original_width']
            height = v[0]['original_height']
            data["height"] = height
            data["width"] = width
    for bb, text, label in zip(annoatated_image['bbox'], annoatated_image['transcription'],   annoatated_image['label']):
        ann_dict = {}
        print('text :', text)
        ann_dict["box"] = convert_bounding_box(bb['x'], bb['y'], bb['width'], bb['height'])
        ann_dict["text"] = text
        ann_dict["label"] = label['labels'][-1]
        ann_list.append(ann_dict)
    data["annotations"] = ann_list

print(output)
with open("Training_layoutLMV3.json", "w") as f:
  json.dump(output, f, indent=4)

annot_label_set = set()
for item in output:
    for annot in item['annotations']:
        annot_label_set.add(annot['label'])

annot_label_lst = [(k, v) for k, v in enumerate(sorted(annot_label_set))]
annotation_dict = {}
for item in annot_label_lst:
    k, v = item
    annotation_dict[v] = k
    annotation_dict[k] = v

print(annotation_dict)
inputs_path = BASE_PATH / "inputs" 
os.makedirs(inputs_path, exist_ok=True)

with open((inputs_path/"annotation.json").as_posix(), "w") as f:
  json.dump(annotation_dict, f, indent=4)

for item in output:
    for annot in item['annotations']:
        annot['label'] = annotation_dict[annot['label']]

print(output)
with open((inputs_path/"Training_layoutLMV3.json").as_posix(), "w") as f:
  json.dump(output, f, indent=4)




 
