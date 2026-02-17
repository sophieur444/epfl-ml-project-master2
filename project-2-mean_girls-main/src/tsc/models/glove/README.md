# GloVe Models

## Overview 

This module provides an end-to-end pipeline for training and evaluating classical machine-learning models using **GloVe (Global Vectors for Word Representation)** word embeddings. 
GloVe converts each word into a dense numerical vector based on global co-occurrence statistics in a large corpus, producing representations that capture lexical similarity, semantic structure, and meaningful relationships between words.  
In our project, these embeddings allow us to turn tweets into rich numerical features, enabling simple models to perform effective sentiment classification even with very short text inputs.


## Preprocessing

### 1. Build the vocabulary

Generate vocabulary files:
```bash
poetry run python -m tsc.preprocess.vocab.build_vocab
poetry run python -m tsc.preprocess.vocab.cut_vocab
poetry run python -m tsc.preprocess.vocab.pickle_vocab
```
Vocabulary artifacts are created in `data/intermediate/` and `data/processed/`.

### 2. Build the co-occurrence matrix

```bash
poetry run python -m tsc.preprocess.cooc
```
This produces cooc.pkl in `data/processed/`.

### 3. Train GloVe embeddings

```bash
poetry run python -m tsc.preprocess.embedding.glove_embedding
```
Embeddings are saved in `data/processed/`.

## Train Models

Each script trains a classifier on GloVe-encoded tweets and stores the resulting model as a `.pkl` file in the saved models directory.

- Logistic Regression
```bash
poetry run python -m tsc.models.glove.train_glove_logreg
```
- Random Forest (best performance)
```bash
poetry run python -m tsc.models.glove.train_glove_rf
```
- Linear SVM 
```bash
poetry run python -m tsc.models.glove.train_glove_logreg
```

## Generate predictions

Use the unified prediction script to run any GloVe-based model:
```bash
poetry run python -m tsc.models.glove.predict_glove --model <model>
```
The `model` value can either be `rf`, `logreg` (by default) or `svm`. 


## Best Model Performance

Among the classical models trained with GloVe embeddings, the **Random Forest** classifier achieved the strongest results on our dataset, offering the best balance between accuracy and robustness.

- Random Forest (best performance)
```bash
poetry run python -m tsc.models.glove.train_glove_rf
```
- Logistic Regression
```bash
poetry run python -m tsc.models.glove.train_glove_logreg
```
- Linear SVM
```bash
poetry run python -m tsc.models.glove.train_glove_logreg
```

## Folder Structure

```text
glove/
├── README.md              # (this file)
├── data.py                # GloVe encoder + loading utilities (train/test)
├── predict_glove.py       # Predict sentiment using any trained GloVe model
├── train_glove_logreg.py  # Train Logistic Regression on GloVe features
├── train_glove_rf.py      # Train Random Forest (best-performing model)
├── train_glove_svm.py     # Train Linear SVM
|
└── saved_models/          # Folder containing .pkl models (ignored)
```