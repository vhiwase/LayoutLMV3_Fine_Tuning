# pip install datasets[vision]

from datasets import load_dataset 
from PIL import ImageFont, ImageDraw, ImageEnhance
import pandas as pd

# this dataset uses the new Image feature :)
dataset = load_dataset("nielsr/funsd-layoutlmv3")

dataset["train"].features

"""
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
from datasets import Dataset

df = pd.DataFrame({'a': [0,1,2], 'b': [3,4,5]})

### convert to Huggingface dataset
hg_dataset = Dataset(pa.Table.from_pandas(df))

### Huggingface dataset to pandas
df = pd.DataFrame(hg_dataset)

"""


from datasets import Dataset, Image, DatasetDict, Features, Sequence, ClassLabel, Value, Array2D, Array3D

train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

################ Step 1: Convert dataframe to huggingface dataset ###################################

# we need to define the features ourselves
features = Features({
    'id': Value(dtype='string', id=None),
    'tokens': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
    'bboxes': Sequence(feature=Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None), length=-1, id=None),
    'ner_tags': Sequence(feature=ClassLabel(names=['O', 'B-HEADER', 'I-HEADER', 'B-QUESTION', 'I-QUESTION', 'B-ANSWER', 'I-ANSWER'], id=None), length=-1, id=None),
    'image': Image(decode=True, id=None)
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
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
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
    draw.rectangle(box, outline='green')
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

encoding = processor(image, words, boxes=boxes, word_labels=word_labels, return_tensors="pt")
for k,v in encoding.items():
  print(k,v.shape)
  
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
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
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

label2color = {'question':'blue', 'answer':'green', 'header':'orange', 'other':'violet'}

for prediction, box in zip(true_predictions, true_boxes):
    predicted_label = iob_to_label(prediction).lower()
    draw.rectangle(box, outline=label2color[predicted_label])
    draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color[predicted_label], font=font)

image

