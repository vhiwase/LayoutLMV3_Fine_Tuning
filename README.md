# LayoutLMV3_Fine_Tuning

Please find the model video explanation in the youtube - https://www.youtube.com/watch?v=bBwDTY38X58&list=PLeNIpK8NwtHtxa2wC1OcPb8RmQ9vy-Uav

# How to install
```sh
python -m pip install --upgrade pip
```
```sh
python -m pip install -r requirements/requirements_paddle.txt
python -m pip install -r requirements/requirements.txt
```
or
```sh
python -m pip install paddlepaddle==2.5.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install git+https://github.com/PaddlePaddle/PaddleClas.git
python -m pip install git+https://github.com/vhiwase/PaddleOCR.git
```
```sh
python -m pip install git+https://github.com/huggingface/transformers.git
python -m pip install datasets seqeval
python -m pip install torch torchvision torchaudio
```
#### Install the following in different virtual environment
```sh
python -m pip install -r requirements/requirements_label_studio.txt
```
or
```sh
pip install label-studio
```
### Step1:
```sh
python Convert_pdf_to_images.py
```
### Step2:
```sh
python Create_LMv3_dataset_with_paddleOCR.py
```
### Step3:
```sh
python simple_http_server.py
```
### Step4:
```sh
label-studio -p 8081
```
### Step5:
```sh
python Label_studio_to_layoutLMV3.py
```
