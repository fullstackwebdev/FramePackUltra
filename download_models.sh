#!/bin/bash

# Create models directory structure
mkdir -p ./models/hunyuanvideo-community/HunyuanVideo
mkdir -p ./models/lllyasviel/flux_redux_bfl
mkdir -p ./models/lllyasviel/FramePackI2V_HY

# Install huggingface_hub if not already installed
pip install huggingface_hub

# Download HunyuanVideo models
echo "Downloading HunyuanVideo..."
python -c "
from huggingface_hub import snapshot_download
import os

# Download main model with specific subfolders
snapshot_download(
    repo_id='hunyuanvideo-community/HunyuanVideo',
    local_dir='./models/hunyuanvideo-community/HunyuanVideo',
    allow_patterns=[
        'text_encoder/*', 
        'text_encoder_2/*', 
        'tokenizer/*', 
        'tokenizer_2/*', 
        'vae/*'
    ]
)

# Download flux_redux_bfl
snapshot_download(
    repo_id='lllyasviel/flux_redux_bfl',
    local_dir='./models/lllyasviel/flux_redux_bfl',
    allow_patterns=[
        'feature_extractor/*', 
        'image_encoder/*'
    ]
)

# Download FramePackI2V_HY
snapshot_download(
    repo_id='lllyasviel/FramePackI2V_HY',
    local_dir='./models/lllyasviel/FramePackI2V_HY'
)
"

echo "All models downloaded successfully!"