# Transformer-Based Sentiment Analysis with Explainability

An advanced Natural Language Processing (NLP) project that performs sentiment classification using the BERT transformer architecture integrated with Explainable AI (XAI) techniques including Attention Visualization, SHAP, and LIME.

---

# Overview

This project focuses on building an interpretable sentiment analysis framework capable of classifying textual reviews into positive or negative sentiment categories while also explaining the reasoning behind model predictions.

The notebook fine-tunes the `bert-base-uncased` transformer model on the Amazon Polarity dataset using Hugging Face Transformers and PyTorch. In addition to achieving high classification performance, the project integrates multiple explainability techniques to improve model transparency and interpretability.

The system demonstrates how modern transformer architectures can be combined with Explainable AI methods to create reliable and trustworthy NLP systems.

---

# Features

* Fine-tuned BERT (`bert-base-uncased`) for binary sentiment classification
* Attention heatmap visualization for transformer interpretability
* SHAP-based token contribution explanations
* LIME-based local prediction explanations
* Misclassified sample analysis
* GPU-supported training using PyTorch and Hugging Face Trainer API
* Evaluation using Accuracy, Precision, Recall, and F1-score
* Transformer attention extraction and visualization
* Explainable AI integration for sentiment prediction transparency

---

# Technologies Used

| Technology                | Purpose                             |
| ------------------------- | ----------------------------------- |
| Python                    | Core programming language           |
| PyTorch                   | Deep learning framework             |
| Hugging Face Transformers | BERT implementation and fine-tuning |
| Hugging Face Datasets     | Dataset loading and preprocessing   |
| SHAP                      | Explainable AI feature attribution  |
| LIME                      | Local interpretable explanations    |
| Scikit-learn              | Evaluation metrics                  |
| Matplotlib                | Visualization                       |
| NumPy                     | Numerical computation               |
| Jupyter Notebook          | Experimentation environment         |

---

# Dataset

## Amazon Polarity Dataset

The project uses the Amazon Polarity dataset for binary sentiment classification.

### Dataset Configuration

* Training Samples: ~15,000
* Testing Samples: ~3,000
* Labels:

  * `0` → Negative Sentiment
  * `1` → Positive Sentiment

### Dataset Source

Dataset loaded using Hugging Face:

```python
from datasets import load_dataset

dataset = load_dataset("fancyzhx/amazon_polarity")
```

---

# Project Workflow

## 1. Data Loading

The dataset is loaded using Hugging Face Datasets library.

## 2. Data Preprocessing

The preprocessing pipeline includes:

* Tokenization using BERT tokenizer
* Sequence padding and truncation
* Tensor conversion
* Dataset shuffling and splitting

### Tokenization Example

```python
def preprocess_function(examples):
    return tokenizer(
        examples["content"],
        truncation=True,
        padding="max_length",
        max_length=64
    )
```

---

# Model Architecture

## BERT Transformer Model

The project uses:

```python
bert-base-uncased
```

### Configuration

| Parameter               | Value             |
| ----------------------- | ----------------- |
| Model                   | bert-base-uncased |
| Maximum Sequence Length | 64                |
| Batch Size              | 4                 |
| Epochs                  | 2                 |
| Labels                  | 2                 |
| Framework               | PyTorch           |
| Hardware                | CUDA GPU          |

### Model Initialization

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
```

---

# Training Procedure

The model is fine-tuned using Hugging Face `Trainer` API.

## Training Arguments

```python
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
)
```

## Optimizer

* AdamW Optimizer

## Training Process

The training loop includes:

* Forward propagation
* Loss computation
* Backpropagation
* Parameter updates
* Epoch-wise evaluation

---

# Evaluation Metrics

The notebook evaluates the model using:

* Accuracy
* Precision
* Recall
* F1-score

## Metric Implementation

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
```

## Performance Results

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 89.97% |
| Precision | 89.67% |
| Recall    | 90.67% |
| F1-Score  | 90.17% |

---

# Explainability Techniques

One of the major contributions of this project is the integration of Explainable AI methods.

---

## Attention Visualization

Transformer attention weights are extracted to visualize contextual token importance.

### Features

* Layer-wise attention analysis
* Head-wise attention visualization
* Token relationship interpretation
* Heatmap generation

### Attention Model

```python
outputs = model(
    **inputs,
    output_attentions=True,
    attn_implementation="eager"
)
```

### Purpose

Attention heatmaps help identify which words the transformer model focuses on during sentiment prediction.

---

## SHAP Explanations

SHAP (SHapley Additive Explanations) is used for feature attribution analysis.

### SHAP Features

* Token-level contribution analysis
* Global interpretability
* Local interpretability
* Positive and negative contribution visualization

### SHAP Workflow

```python
import shap
```

### Purpose

SHAP explanations quantify how individual tokens influence sentiment predictions.

---

## LIME Explanations

LIME (Local Interpretable Model-Agnostic Explanations) is used for generating local explanations.

### LIME Features

* Local surrogate explanations
* Token importance ranking
* Human-readable explanations
* Prediction interpretability

### LIME Workflow

```python
from lime.lime_text import LimeTextExplainer
```

### Purpose

LIME explains individual predictions by approximating transformer outputs using interpretable surrogate models.

---

# Misclassified Sample Analysis

The notebook additionally analyses incorrectly classified samples.

### Analysis Includes

* Comparison between predicted and actual labels
* Collection of misclassified reviews
* Error analysis
* Contextual ambiguity analysis

This helps identify model weaknesses and improve future training strategies.

---

# Installation

## Clone Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

## Install Dependencies

```bash
pip install transformers datasets torch scikit-learn matplotlib shap lime
```

---

# Running the Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```bash
nlp-assignment-4.ipynb
```

Run all notebook cells sequentially.

---

# Project Structure

```bash
├── nlp-assignment-4.ipynb
├── README.md
├── results/
├── figures/
└── checkpoints/
```

---

# Applications

This project can be applied in:

* Customer review analysis
* Product feedback monitoring
* Recommendation systems
* Social media sentiment analysis
* Opinion mining
* Explainable NLP research
* AI transparency systems

---

# Future Improvements

Potential future enhancements include:

* Multilingual sentiment analysis
* RoBERTa and DistilBERT comparison
* Real-time inference API
* Web deployment using Flask or FastAPI
* Interactive explainability dashboard
* Fairness and bias analysis
* Larger dataset training
* Advanced transformer architectures

---

# Research Contributions

This project contributes:

* A transformer-based explainable sentiment analysis framework
* Integration of multiple explainability methods
* Attention visualization for transformer interpretability
* Reproducible NLP experimentation pipeline
* Detailed analysis of transformer prediction behaviour

---

# References

1. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.

2. Vaswani, A. et al. (2017). Attention Is All You Need.

3. Lundberg, S., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions.

4. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). Why Should I Trust You? Explaining the Predictions of Any Classifier.

---

# Author

**Abdullah Saood**
BS Artificial Intelligence
FAST University Faisalabad

---

# License

This project is int
