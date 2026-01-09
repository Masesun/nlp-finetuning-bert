# End-to-End BERT Pipeline for Text Classification with Error Analysis

## Overview
This repository contains a complete, end-to-end Natural Language Processing pipeline for binary text classification using a fine-tuned BERT model.  
The project covers the full workflow, including dataset handling, model training, evaluation, inference, and post-hoc error analysis.

The codebase is structured to reflect good engineering practices in applied machine learning and is intended as a finished, portfolio-ready project rather than a tutorial or template.

---

## Task Description
- **Task type**: Binary text classification  
- **Use case**: Sentiment analysis of movie reviews  
- **Input**: Raw text  
- **Output**: Predicted class label (positive / negative)

---

## Model
- **Base model**: `bert-base-uncased`
- **Architecture**: Transformer encoder with a classification head
- **Framework**: PyTorch
- **Library**: Hugging Face Transformers

The model is initialized from a pre-trained BERT checkpoint and fine-tuned on a downstream supervised classification task.

---

## Dataset
- **Dataset**: IMDB movie reviews
- **Source**: Hugging Face Datasets library
- **Classes**: Positive, Negative

The dataset is loaded programmatically during training and evaluation and is not stored in the repository.  
Only lightweight example inputs and outputs used for inference demonstration are included in version control.

---

## Training Details
- **Training framework**: Hugging Face `Trainer` API
- **Optimizer**: AdamW
- **Learning rate**: 2e-5
- **Batch size**: 16
- **Epochs**: 3

All experiments are run with fixed random seeds to ensure reproducibility.

---

## Evaluation
Model performance is evaluated on a held-out test split using standard classification metrics:
- Accuracy
- F1-score

Final evaluation results are stored in `results/eval_metrics.json`.

---

## Project Structure
```text
data/
├── inference_samples.csv          # Example inputs for inference
├── predictions.csv                # Model predictions
├── predictions_with_analysis.csv  # Predictions enriched with analysis

notebooks/
├── eda.ipynb                      # Exploratory data analysis
├── error_analysis.ipynb           # Qualitative error analysis
├── inference.ipynb                # Inference demonstration notebook

results/
└── eval_metrics.json              # Final evaluation metrics

src/
├── __init__.py
├── dataset.py                     # Dataset loading and tokenization
├── model.py                       # Model initialization
├── train.py                       # Training pipeline
├── evaluation.py                  # Evaluation logic and metrics
├── inference.py                   # Batch inference script

.gitignore
README.md
requirements.txt
```

---

## Installation

Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/nlp-finetuning-bert.git
cd nlp-finetuning-bert
```

Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage

### Training the model
To fine-tune the model on the IMDB dataset:
```bash
python -m src.train
```

### Evaluation
To evaluate the trained model and save metrics:
```bash
python -m src.evaluation
```

### Inference
To run batch inference on example inputs:
```python
python -m src.inference --input data/inference_samples.csv
```
Predictions are saved to data/predictions.csv.
An extended version with additional analysis is saved to data/predictions_with_analysis.csv.
---

## Notes
- No raw training datasets are stored in the repository.
- Trained model weights are not committed to version control.
- All paths are relative and platform-independent.
- The project is fully reproducible using the provided scripts and configuration.
- The codebase is organized to clearly separate training, evaluation, and inference concerns.

---

## Possible Extensions
- Comparison of different transformer architectures such as RoBERTa or DistilBERT
- Implementation of a custom PyTorch training loop
- Hyperparameter tuning and ablation studies
- More advanced error analysis and explainability methods
- Deployment of the inference pipeline as a REST API using FastAPI
