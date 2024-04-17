# Image_Captioning_Using_Transformer

<a href="https://huggingface.co/spaces/Saptarshi003/Image_Captioning_Using_Transformer"><img src="https://img.shields.io/badge/Hugging%20Face%20%F0%9F%A4%97-demo-yellow"></a>

![demo pic](./imgs/demo.png)

## Overview

A Image Captioning system.
 - Uses Transformer architecture.
 - Integrates EfficientNetB0 for image embeddings and employs a vocabulary size of 10,000 with token and image embedding size of 512. 

## Usage

1. Install all the packages

 ```python
 python3 -m pip install -r requirements.txt
 ```

 2. Run the Gradio app

 ```python
 gradio app.py
 ```