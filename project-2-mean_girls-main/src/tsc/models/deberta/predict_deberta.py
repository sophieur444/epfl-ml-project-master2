# src/tsc/models/deberta/predict_deberta.py

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from functools import partial

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

from tsc.utils.helpers import create_csv_submission
from tsc.utils.paths import DATA_TWITTER, DEBERTA


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


# Create dataset from text file
def load_test_dataset():
    tweets = []
    with open(DATA_TWITTER / "test_data.txt", "r", encoding="utf-8") as f:
        tweets = [line.rstrip("\n") for line in f if line.strip() != ""]

    indices = np.arange(len(tweets))
    test_dataset = Dataset.from_dict({"text": tweets})
    return test_dataset, indices


# Tokenize dataset using DeBERTa pretrained tokenizer
def tokenize_dataset(test_set, tokenizer, max_length):
    tokenize_fct = partial(tokenizer, truncation=True, max_length=max_length)
    test_tokenized = test_set.map(
        lambda batch: tokenize_fct(batch["text"]), batched=True
    )

    # Transform to PyTorch tensors and remove text column
    test_tokenized = test_tokenized.remove_columns(["text"])
    test_tokenized.set_format("torch")

    return test_tokenized


def predict(model, test_tokenized, batch_size, data_collator):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataloader = DataLoader(
        test_tokenized,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    return all_preds


def main():
    args = parse_args()

    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(DEBERTA / "artifacts" / "best_model")
    model = AutoModelForSequenceClassification.from_pretrained(
        DEBERTA / "artifacts" / "best_model"
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    test_set, ids = load_test_dataset()

    test_tokenized = tokenize_dataset(test_set, tokenizer, args.max_length)

    # Predict and convert 0/1 to -1/1
    preds = predict(model, test_tokenized, args.batch_size, data_collator)
    y_pred = 2 * preds - 1

    create_csv_submission(ids, y_pred, "predictions_deberta")


if __name__ == "__main__":
    main()
