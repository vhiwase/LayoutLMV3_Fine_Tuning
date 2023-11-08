# pip install datasets[vision]
# pip install git+https://github.com/huggingface/transformers.git
# pip install seqeval
# pip install torch torchvision torchaudio
# pip install transformers[torch]
# pip install accelerate -U

from PIL import ImageFont, ImageDraw, ImageEnhance, Image
from PIL.PngImagePlugin import PngImageFile
import pandas as pd
import cv2
import pathlib
import os
import filetype
import json
from pydantic.dataclasses import dataclass
import datasets
from datasets import Dataset, DatasetDict, Features, Sequence, ClassLabel, Value, Array2D, Array3D
import numpy as np
import pandas as pd

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

try:
    base = pathlib.Path(__file__)
except NameError:
    base = pathlib.Path('.')

root = base.parent
annotation_folderpath = root / 'annotation'
images_path = []
for filename in os.listdir(annotation_folderpath):
    file_path = annotation_folderpath / filename
    kind = filetype.guess(file_path)
    if kind and 'image' in kind.MIME.lower():
        images_path.append(file_path)


fields_path = annotation_folderpath / 'fields.json'
with open(fields_path, 'rb') as f:
    fields_dict = json.loads(f.read())
label2id = dict()
id2label = dict()
for fenum, field in enumerate(fields_dict['fields']):
    label = field['fieldKey']
    label2id[label] = fenum
    id2label[fenum] = label
images = []
ids = []
ner_tags_lst = []
tokens_lst = []
bboxes_lst = []
for enum, image_path in enumerate(images_path):
    filerootname = image_path.stem
    if '_resized' in filerootname:
        continue
    label_json_path = image_path.parent / f'{filerootname}.png.labels.json'
    image_path = image_path.parent / f'{filerootname}.png'
    ocr_json_path = image_path.parent / f'{filerootname}.png.ocr.json'
    with open(label_json_path, 'rb') as f:
        label_dict = json.loads(f.read())
    with open(ocr_json_path, 'rb') as f:
        ocr_dict = json.loads(f.read())
    image = Image.open(image_path)
    new_height = 1000    
    # Calculate the new width while maintaining the aspect ratio
    width, height = image.size
    new_width = int(width * (new_height / height))
    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    resized_image_path = image_path.parent / f'{image_path.stem}_resized.png'
    resized_image.save(resized_image_path)
    resized_image = Image.open(resized_image_path)
    labels = label_dict['labels']
    pages = []
    texts = []
    label_lst = []
    top_left_x_lst = []
    top_left_y_lst = []
    top_right_x_lst = []
    top_right_y_lst = []
    bottom_right_x_lst = []
    bottom_right_y_lst = []
    bottom_left_x_lst = []
    bottom_left_y_lst = []
    for l in labels:
        for item in l['value']:
            label_lst.append(label2id[l['label']])
            texts.append(item['text'])
            pages.append(item['page'])
            bbox = item['boundingBoxes'][0]
            ocr_coordinates = OCRCoordinates(*bbox)
            top_left_x_lst.append(ocr_coordinates.top_left_x)
            top_left_y_lst.append(ocr_coordinates.top_left_y)
            top_right_x_lst.append(ocr_coordinates.top_right_x)
            top_right_y_lst.append(ocr_coordinates.top_right_y)
            bottom_right_x_lst.append(ocr_coordinates.bottom_right_x)
            bottom_right_y_lst.append(ocr_coordinates.bottom_right_y)
            bottom_left_x_lst.append(ocr_coordinates.bottom_left_x)
            bottom_left_y_lst.append(ocr_coordinates.bottom_left_y)
    df = pd.DataFrame()
    df['text'] = texts
    df['label'] = label_lst
    df['page'] = pages
    df['top_left_x'] = top_left_x_lst
    df['top_left_y'] = top_left_y_lst
    df['top_right_x'] = top_right_x_lst
    df['top_right_y'] = top_right_y_lst
    df['bottom_right_x'] = bottom_right_x_lst
    df['bottom_right_y'] = bottom_right_y_lst
    df['bottom_left_x'] = bottom_left_x_lst
    df['bottom_left_y'] = bottom_left_y_lst
    df = df.sort_values(by=['page', 'bottom_left_y', 'bottom_left_x'], ignore_index=True)
    ner_tags = df['label'].tolist()
    df_copy = df.copy()
    width, height = resized_image.size
    df_copy['top_left_x'] = df_copy['top_left_x'] * width
    df_copy['top_left_y'] = df_copy['top_left_y'] * height
    df_copy['bottom_right_x'] = df_copy['bottom_right_x'] * width
    df_copy['bottom_right_y'] = df_copy['bottom_right_y'] * height
    bboxes = (df_copy[['top_left_x', 'top_left_y', 'bottom_right_x', 'bottom_right_y']]).astype(np.int64).values.tolist()
    tokens = df['text'].tolist()
    ids.append(enum)
    tokens_lst.append(tokens)
    bboxes_lst.append(bboxes)
    ner_tags_lst.append(ner_tags)
    images.append(resized_image)

dataframe = pd.DataFrame()
dataframe['id'] = ids
dataframe['tokens'] = [tokens_lst[0][:200]]
dataframe['bboxes'] = [bboxes_lst[0][:200]]
dataframe['ner_tags'] = [ner_tags_lst[0][:200]]
dataframe['image'] = images
train_df = dataframe
test_df = dataframe
#########################################################################################

def plot(test_or_train_df, index, should_unnormalize=False):
    def unnormalize_box(bboxes, width, height):
        unnormalize_lst = []
        for bbox in bboxes:
            unnormalize_bbox = [
                width * (bbox[0]),
                height * (bbox[1]),
                width * (bbox[2]),
                height * (bbox[3])
            ]
            unnormalize_lst.append(unnormalize_bbox)
        return unnormalize_lst
    
    def plot_bbox_image(input_file, dataframe):
        img = np.asarray(image)
        for i in range(len(dataframe)):
            x = dataframe.x[i]
            y = dataframe.y[i]
            w = dataframe.w[i]
            h = dataframe.h[i]
            x, y, w, h = int(x), int(y), int(w), int(h)
            try:
                img = cv2.rectangle(img, (x, y), (w, h), (0,255,0), 1)
            except:
                pass
        return Image.fromarray(img)

    plot_df = test_or_train_df.copy()    

    image_path = None
    if image_path:
        image = Image.open(image_path)
    else:
        image = plot_df.loc[index, 'image']
    width, height = image.size
    if should_unnormalize:
        plot_df['bboxes'] = plot_df['bboxes'].apply(unnormalize_box, width=width, height=height)
    x, y, w, h = list(zip(*plot_df.loc[index, 'bboxes']))
    text = plot_df.loc[index, 'tokens']
    df = pd.DataFrame()
    df['text'] = text
    df['x'] = x
    df['y'] = y
    df['w'] = w
    df['h'] = h
    return plot_bbox_image(input_file=image, dataframe=df)

plot(test_or_train_df=test_df, index=0, should_unnormalize=False)

################ Step 1: Convert dataframe to huggingface dataset ###################################

# we need to define the features ourselves
features = Features({
    'id': Value(dtype='string', id=None),
    'tokens': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
    'bboxes': Sequence(feature=Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None), length=-1, id=None),
    'ner_tags': Sequence(feature=ClassLabel(names=sorted(label2id, key=label2id.get), id=None), length=-1, id=None),
    'image': datasets.Image(decode=True, id=None)
   }
)

train_dict = {}
for column in train_df.columns:
    key = column
    value = train_df[column].tolist()
    train_dict[key]=value
    
test_dict = {}
for column in test_df.columns:
    key = column
    value = test_df[column].tolist()
    test_dict[key]=value
    
train_dataset = Dataset.from_dict(train_dict, features=features)
test_dataset = Dataset.from_dict(test_dict, features=features)

train_test_dataset_dict = {
    "train": train_dataset,
    "test": test_dataset
}

dataset = DatasetDict(train_test_dataset_dict)

############################ Step 2: Annotation data Visualization #################################################

from PIL import ImageDraw, ImageFont

def unnormalize_box(bbox, width, height):
     return [
         (bbox[0]),
         (bbox[1]),
         (bbox[2]),
         (bbox[3]),
     ]

example = dataset["train"][0]
labels_list = dataset["train"].features['ner_tags'].feature.names
words, boxes, ner_tags = example["tokens"], example["bboxes"], example["ner_tags"]

source_img = example["image"].copy()
draw = ImageDraw.Draw(source_img)
width, height = source_img.size
true_boxes = [unnormalize_box(box, width, height) for box in boxes]

font = ImageFont.load_default()
w =[]
t = []
for word, box, ner_tag in zip(words, true_boxes, ner_tags):
    print(word, labels_list[ner_tag])
    w.append(word)
    t.append(labels_list[ner_tag])
    try:
        draw.rectangle(box, outline='green')
    except ValueError:
        continue
    draw.text((box[0] + 10, box[1] - 10), text=labels_list[ner_tag], fill="red", font=font)

source_img 

wt_df = pd.DataFrame()
wt_df['w'] = w
wt_df['t'] = t

######################## Step 3: Data Preparation for training ############################################
from transformers import AutoProcessor

# we'll use the Auto API here - it will load LayoutLMv3Processor behind the scenes,
# based on the checkpoint we provide from the hub
processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
from datasets.features import ClassLabel

features = dataset["train"].features
column_names = dataset["train"].column_names
image_column_name = "image"
text_column_name = "tokens"
boxes_column_name = "bboxes"
label_column_name = "ner_tags"

# In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
# unique labels.
def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list

if isinstance(features[label_column_name].feature, ClassLabel):
    label_list = features[label_column_name].feature.names
    # No need to convert the labels since they are already ints.
    id2label = {k: v for k,v in enumerate(label_list)}
    label2id = {v: k for k,v in enumerate(label_list)}
else:
    label_list = get_label_list(dataset["train"][label_column_name])
    id2label = {k: v for k,v in enumerate(label_list)}
    label2id = {v: k for k,v in enumerate(label_list)}
num_labels = len(label_list)

def prepare_examples(examples):
  images = examples[image_column_name]
  words = examples[text_column_name]
  boxes = examples[boxes_column_name]
  word_labels = examples[label_column_name]

  encoding = processor(images, words, boxes=boxes, word_labels=word_labels,
                       truncation=True, padding="max_length")

  return encoding

from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D

# we need to define custom features for `set_format` (used later on) to work properly
features = Features({
    'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'attention_mask': Sequence(Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    'labels': Sequence(feature=Value(dtype='int64')),
})

train_dataset = dataset["train"].map(
    prepare_examples,
    batched=True,
    remove_columns=column_names,
    features=features,
)
eval_dataset = dataset["test"].map(
    prepare_examples,
    batched=True,
    remove_columns=column_names,
    features=features,
)

example = train_dataset[0]
processor.tokenizer.decode(example["input_ids"])

train_dataset.set_format("torch")

############### Step 4: Evaluation Preparation step ############################

import torch

# example = train_dataset[0]
# for k,v in example.items():
#     print(k,v.shape)

# processor.tokenizer.decode(eval_dataset[0]["input_ids"])

w = []
t = []
for id, label in zip(train_dataset[0]["input_ids"], train_dataset[0]["labels"]):
    word = processor.tokenizer.decode([id])
    w.append(word)
    label_index = label.item()
    t.append(label_index)

wtdf = pd.DataFrame()
wtdf['tokenized_word'] = w
wtdf['label_index'] = t
    

    
from datasets import load_metric

metric = load_metric("seqeval")


import numpy as np

return_entity_level_metrics = False

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

################# Step 5: Model Training ######################################

from transformers import LayoutLMv3ForTokenClassification

model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base",
                                                         id2label=id2label,
                                                         label2id=label2id)
                                                         # max_position_embeddings=514*3,
                                                         # ignore_mismatched_sizes=True)

# model.config.max_position_embeddings=514*3
# model_config = model.config
# model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base",
#                                                          config=model_config)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test",
                                  max_steps=1,
                                  per_device_train_batch_size=2,
                                  per_device_eval_batch_size=2,
                                  learning_rate=1e-5,
                                  evaluation_strategy="steps",
                                  eval_steps=1,
                                  load_best_model_at_end=True,
                                  metric_for_best_model="f1")


from transformers.data.data_collator import default_data_collator
import time

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)


# 3 min
tic = time.time()
trainer.train()
toc = time.time()
print(f"Time taken: {toc-tic}")

# 2.8 min
tic = time.time()
trainer.evaluate()
toc = time.time()
print(f"Time taken: {toc-tic}")


############ Step 6: Inferencing #############################

example = dataset["test"][0]
print(example.keys())

image = example["image"]
words = example["tokens"]
boxes = example["bboxes"]
word_labels = example["ner_tags"]

example = train_dataset[0]
for k,v in example.items():
    print(k,v.shape)

encoding = processor(image, words, boxes=boxes, word_labels=word_labels, return_tensors="pt")
for k,v in encoding.items():
  print(k,v.shape)

print(model.config.max_position_embeddings)

with torch.no_grad():
  outputs = model(**encoding)
  
logits = outputs.logits
logits.shape



predictions = logits.argmax(-1).squeeze().tolist()
print(predictions)

labels = encoding.labels.squeeze().tolist()
print(labels)

# from transformers import EvalPrediction
# p = EvalPrediction(predictions=predictions, label_ids=labels, inputs=None)
# compute_metrics(p)


def unnormalize_box(bbox, width, height):
     return [
         (bbox[0]),
         (bbox[1]),
         (bbox[2]),
         (bbox[3]),
     ]

token_boxes = encoding.bbox.squeeze().tolist()
width, height = image.size

true_predictions = [model.config.id2label[pred] for pred, label in zip(predictions, labels) if label != - 100]
true_labels = [model.config.id2label[label] for prediction, label in zip(predictions, labels) if label != -100]
true_boxes = [unnormalize_box(box, width, height) for box, label in zip(token_boxes, labels) if label != -100]






from PIL import ImageDraw, ImageFont

draw = ImageDraw.Draw(image)

font = ImageFont.load_default()

def iob_to_label(label):
    label = label[2:]
    if not label:
      return 'other'
    return label

label2color = {'toc-description':'blue', 'key':'green', 'header':'orange', 'other':'violet', 'footer': "brown", "toc":"red",
               "toc-page": "black", "toc-header": "pink", "toc-column": "purple", "title": "gray", "toc-section": "lime",
               "notselected": "maroon", "value": "olive", "selected": "fuchsia"}

for prediction, box in zip(true_predictions, true_boxes):
    if prediction!='other':
        predicted_label = iob_to_label(prediction).lower()
    try:
        draw.rectangle(box, outline=label2color[predicted_label])
    except ValueError:
        continue
    draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color[predicted_label], font=font)

image

