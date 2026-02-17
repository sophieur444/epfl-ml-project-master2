# Twitter Sentiment Classification

Predict whether a tweet originally contained a positive :) or negative :( smiley, using only the remaining text.

This project is part of the EPFL CS-433 Machine Learning course.  
Authors: Lilian Noé, Esther Barriol, Sophie Urrea.

## Objective

This repo implements different Machine Learning models in order to achieve the best classification accuracy. Each model is implemented and documented in the `models/` folder.

## Installation

Clone the repository:

```bash
git clone https://github.com/CS-433/project-2-mean_girls.git
cd project-2-mean_girls
```

Install dependecies using [**Poetry**](docs/Poetry.md) from the **project root**:
```bash
poetry config virtualenvs.in-project true
poetry install
```

## Dataset

Download the dataset from AIcrowd:  
https://www.aicrowd.com/challenges/epfl-ml-text-classification/dataset_files

Extract all files into `data/raw/twitter-datasets/`.

The directory must contain:
- `sample_submission.csv`
- `train_pos_full.txt`
- `train_pos.txt`
- `train_neg_full.txt`
- `train_neg.txt`
- `test_data.txt`

Tweets are preprocessed so that each word is separated by a whitespace and labels have been removed.

## Best Model

The best result was achieved with the **DeBERTA transformer**.  
The according AICrowd submission can be found [**here**](https://www.aicrowd.com/challenges/epfl-ml-text-classification/submissions/305406).

To re-produce the best achieved result, run the main script:
```bash
poetry run python .\run.py
```

It must be acknowledged that the script achieving the best result was run on Google Colab, on an A100 GPU.

For more informations, refer to the dedicated [README](./src/tsc/models/deberta/README.md).

## Training

Train the DeBERTa model:
```bash
poetry run python -m tsc.models.deberta.train_deberta
```
This script saves the best model in `models/deberta/artifacts/best_model/`.

## Prediction

Generate predictions for `test_data.txt`:
```bash
poetry run python -m tsc.models.deberta.predict_deberta
```

This produces a `.csv` file in the `results/` folder, matching the AIcrowd submission format.


## Project Structure

```text
project-2-mean_girls/
├── README.md
├── run.py
├── .gitignore
├── poetry.lock
├── pyproject.toml
│
├── data/
│   ├── intermediate/
│   ├── processed/
│   └── raw/
│
├── docs/
│   └── Poetry.md
│
├── results/
│
├── src/
│   └── tsc/
│       │
│       ├── models/
│       │   ├── deberta/
│       │   │   ├── __init__.py
│       │   │   ├── README.md
│       │   │   ├── deberta.ipynb
│       │   │   ├── predict_deberta.py
│       │   │   ├── train_deberta.py
│       │   │   └── artifacts/
│       │   │
│       │   ├── distilbert/
│       │   │   ├── __init__.py
│       │   │   ├── README.md
│       │   │   ├── distilbert.ipynb
│       │   │   ├── predict_distilbert.py
│       │   │   ├── train_distilbert.py
│       │   │   └── artifacts/
│       │   │
│       │   ├── glove/
│       │   │   ├── __init__.py
│       │   │   ├── README.md
│       │   │   ├── data.py
│       │   │   ├── predict_glove.py
│       │   │   ├── train_glove_logreg.py
│       │   │   ├── train_glove_rf.py
│       │   │   ├── train_glove_svm.py
│       │   │   └── saved_models/
│       │   │
│       │   └── tfidf/
│       │       ├── __init__.py
│       │       ├── README.md
│       │       ├── data.py
│       │       ├── predict_tfidf.py
│       │       ├── train_tfidf_logreg.py
│       │       ├── train_tfidf_rf.py
│       │       ├── train_tfidf_svm.py
│       │       └── saved_models/
│       │
│       │
│       │
│       ├── preprocess/
│       │   ├── __init__.py
│       │   ├── cooc.py
│       │   ├── split_dataset.py
│       │   │
│       │   ├── embedding/
│       │   │   ├── __init__.py
│       │   │   ├── awe.py
│       │   │   └── glove_embedding.py
│       │   │
│       │   └── vocab/
│       │       ├── __init__.py
│       │       ├── build_vocab.py
│       │       ├── cut_vocab.py
│       │       └── pickle_vocab.py
│       │
│       │
│       └── utils/
│           ├── __init__.py
│           ├── helpers.py
│           ├── metrics.py
│           └── paths.py
```

