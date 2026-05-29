# NLP Regulatory Analysis

An NLP-assisted framework for regulatory and policy document analysis using OCR, document chunking, and BERT-based text classification.

## Overview

This project was developed to support large-scale analysis of legal, regulatory, and policy documents. The workflow combines document extraction, text preprocessing, structure-based chunking, machine learning classification, and automated inference pipelines.

## Features

* OCR and PDF document extraction
* Legal and policy text preprocessing
* Structure-based document chunking
* Single-label classification
* Multi-label classification
* Automated model inference
* CSV output generation

## Project Structure

```text
data_prepare_method/                Document extraction and preprocessing
dataset/                            Training datasets
stage1_training_3_single_label/     Single-label model training
stage1_training_4_multilabel/       Multi-label model training
stage1_training_5_single_label/     Alternative single-label model training
test_model/                         Model testing and inference
upstream_tasks/                     Data collection and web crawling
docs/                               Project documentation
figures/                            Workflow diagrams and repository structure
```

## Documentation

The `docs/` folder contains:

* Project Documentation
* User Guide
* Project Quick Summary


## Author

Junyi Liu

## Organization

United Nations Economic and Social Commission for Asia and the Pacific (UNESCAP)

## License

This repository is provided for research, educational, and demonstration purposes.
