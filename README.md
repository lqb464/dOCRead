# DOCRead: Deep Learning OCR Pipeline

DOCRead is an advanced Optical Character Recognition (OCR) platform built entirely from the ground up, utilizing PyTorch and HuggingFace Transformers.

Instead of wrapping third-party APIs or utilizing off-the-shelf pre-packaged libraries, this project demonstrates the end-to-end Machine Learning Lifecycle for building a robust OCR system capable of recognizing both **Printed** and **Handwritten** text.

## Core Architecture

This project implements two deep learning paradigms for sequence recognition:

### 1. Convolutional Recurrent Neural Network (CRNN) with CTC Loss
We have constructed a custom CRNN architecture using PyTorch. 
- **CNN Encoder**: A simplified VGG/ResNet structure extracts deep visual features from the input image.
- **RNN Decoder**: A Bidirectional LSTM (BiLSTM) processes the visual features as a sequence to capture contextual dependencies (e.g., recognizing that "q" is often followed by "u").
- **CTC Loss**: Connectionist Temporal Classification loss handles the alignment between the variable-length visual sequence and the variable-length text output, allowing us to train the model without bounding box annotations for individual characters.

### 2. Vision-Encoder-Decoder (TrOCR)
To achieve State-of-the-Art (SOTA) performance on handwritten text, we integrate Microsoft's TrOCR.
- A Vision Transformer (ViT) encodes the image into patch embeddings.
- A Text Transformer (RoBERTa) decodes these embeddings into text.
- We provide fine-tuning scripts to adapt this massive model to our custom synthetic datasets.

## Directory Structure

```text
DOCRead/
├── src/
│   ├── data/           
│   ├── models/         
│   ├── utils/          
│   └── train.py        
├── notebooks/          
├── backend/            
├── frontend/           
├── tests/              
├── pyproject.toml      
└── Makefile            
```

## Getting Started

### Data Science Environment (Model Training)

Prerequisites: Python 3.10 or higher. CUDA is strongly recommended for training.

To install dependencies and start the CRNN smoke test:
```bash
make install
make train-crnn
```

The `notebooks/` directory contains step-by-step interactive sessions:
1. `01_Data_Exploration_and_Augmentation.ipynb`: Synthetic data generation for both printed and handwritten text using `trdg`, alongside perspective and blur augmentations.
2. `02_CRNN_CTC_Training.ipynb`: A deep dive into the CRNN architecture and CTC alignment.
3. `03_TrOCR_Finetuning.ipynb`: Scripts for freezing the Vision Encoder and fine-tuning the Text Decoder.

### Application Deployment (FastAPI Inference)

The backend provides a REST API that serves our custom TrOCR model.

```bash
docker compose up --build
```

- Application UI: http://localhost:3000
- REST API Documentation: http://localhost:8002/docs

## Technical Rationale: Why build from scratch?

1. **Domain Adaptation**: Off-the-shelf OCR models often fail on specific fonts, domain-specific vocabularies, or heavy document noise. Building the training pipeline allows for total control over the data generation process, ensuring the model is robust against the exact noise profiles expected in production.
2. **Evaluation Rigor**: Standard classification accuracy is insufficient for OCR. By implementing Character Error Rate (CER) and Word Error Rate (WER) using Levenshtein distance, we can accurately benchmark model performance.
3. **Engineering Depth**: Training a CRNN with CTC Loss is a complex procedure that demonstrates a profound understanding of deep sequence modeling, gradient flow, and tensor manipulations.
