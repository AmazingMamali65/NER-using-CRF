# 🏥 Custom Named Entity Recognition (NER) for Medical Texts using CRF

This project builds a **Custom NER (Named Entity Recognition)** model for identifying **diseases** and **treatments** from biomedical/medical text using **Conditional Random Fields (CRF)**. The pipeline is designed to run in **Google Colab** with integration to **Google Drive**, and uses **spaCy** and **sklearn-crfsuite** for token-level classification.

---

## 📁 Project Structure

- `train_sent`, `train_label`: Training sentences and their corresponding token-level labels.
- `test_sent`, `test_label`: Testing sentences and their labels.
- Custom functions for:
  - Sentence & label reconstruction
  - POS tagging using spaCy
  - Feature extraction for each word
  - CRF model training and evaluation
- Dictionary extraction for disease-treatment pairs

---

## 🚀 Methodology

- ✅ Converts token-labeled data into full sentences
- 🧠 POS tagging with context-aware handling using spaCy
- ⚙️ Rich feature extraction per word:
  - Word case, suffixes, digit check, POS tag, etc.
- 🧪 Train/test split with CRF model
- 📈 F1-score based evaluation of predictions
- 🔍 Extraction of disease-treatment relationships from the predictions
- 🧾 Final result exported as a pandas DataFrame

---

## 🚀 Features

- 📚 Reads raw token and label files from training and test sets
- 🧠 Performs context-aware POS tagging using spaCy
- ⚙️ Extracts rich features from each word (including suffixes, case info, and neighboring word context)
- 🧪 Trains a CRF model for sequence tagging
- 🔍 Predicts diseases and treatments from new data
- 🧾 Generates a dictionary mapping diseases to potential treatments
- 📈 Evaluates model using F1-score and tag-level comparison

---

## 📁 Data Format

Each file (`train_sent`, `train_label`, etc.) contains **one token or label per line**.  
Sentences are separated by a blank line (`\n`).  

### 🔖 Label Format (Custom BIO-like scheme):

- `D` → Disease  
- `T` → Treatment  
- `O` → Other (non-entity)

### 📦 Example:
```plaintext

train_sent:
Fever
can
be
reduced
with
paracetamol
.

train_label:
D
O
O
O
O
T
O




