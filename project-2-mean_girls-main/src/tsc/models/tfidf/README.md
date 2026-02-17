# TF-IDF Models

## Overview 

This module provides an end-to-end pipeline for training and evaluating classical machine-learning models using **TF-IDF (Term Frequency-Inverse Document Frequency)** features.  
TF-IDF represents text as sparse, weighted lexical vectors that highlight words that are frequent in a given document but rare across the overall corpus, effectively reducing the impact of common, uninformative terms.  
In our project, this approach offers a strong and lightweight baseline for tweet sentiment classification, allowing linear models to leverage discriminative word usage patterns without relying on semantic word embeddings.

## Train Models

Each script trains a classifier on TF-IDF-encoded tweets and stores the resulting model as a `.pkl` file in the saved models directory.

- Logistic Regression
```bash
poetry run python -m tsc.models.tfidf.train_tfidf_logreg
```
- Random Forest
```bash
poetry run python -m tsc.models.tfidf.train_tfidf_rf
```
- Linear SVM (best performance)
```bash
poetry run python -m tsc.models.tfidf.train_tfidf_logreg
```

## Generate predictions

Use the unified prediction script to run any TF-IDF-based model:
```bash
poetry run python -m tsc.models.tfidf.predict_tfidf --model <model>
```
The `model` value can either be `rf`, `logreg` (by default) or `svm`.


## Best Model Performance

Among the classical models trained with TF-IDF features, the **Support Vector Machine** classifier achieved the strongest results on our dataset, offering the best balance between accuracy and robustness.


## Folder Structure

```text
tfidf/
├── README.md              # (this file)
├── data.py                # Loading utilities (train/test)
├── predict_tfidf.py       # Predict sentiment using any trained TF-IDF model
├── train_tfidf_logreg.py  # Train Logistic Regression (best-performing model)
├── train_tfidf_rf.py      # Train Random Forest
├── train_tfidf_svm.py     # Train Linear SVM
|
└── saved_models/          # Folder containing .pkl models (ignored)
```