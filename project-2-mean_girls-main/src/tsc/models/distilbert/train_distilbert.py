# src/tsc/models/distilbert/train_distilbert.py

import argparse
import pandas as pd
import numpy as np

from functools import partial
from datasets import Dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from tsc.utils.metrics import accuracy, f1_score
from tsc.utils.paths import DISTILBERT, DATA_INTERMEDIATE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=2)
    return parser.parse_args()


def load_datasets(train_csv, val_csv):
    train_frame = pd.read_csv(train_csv)
    val_frame = pd.read_csv(val_csv)

    train_set = Dataset.from_pandas(train_frame[["text", "label"]])
    val_set = Dataset.from_pandas(val_frame[["text", "label"]])
    return train_set, val_set


# Tokenize datasets using DistilBERT tokenizer
def tokenize_datasets(train_set, val_set, max_length):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)
    tokenize_fct = partial(
        tokenizer, padding="max_length", truncation=True, max_length=max_length
    )
    print("Tokenizing datasets for DistilBERT...\n")

    train_tokenized = train_set.map(
        lambda batch: tokenize_fct(batch["text"]), batched=True
    )
    val_tokenized = val_set.map(lambda batch: tokenize_fct(batch["text"]), batched=True)

    train_tokenized = train_tokenized.remove_columns(["text"])
    val_tokenized = val_tokenized.remove_columns(["text"])
    train_tokenized.set_format("torch")
    val_tokenized.set_format("torch")

    return tokenizer, train_tokenized, val_tokenized


# Compute metrics for Trainer
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy(predictions, labels),
        "f1": f1_score(predictions, labels),
    }


def train(args):
    train_set, val_set = load_datasets(
        DATA_INTERMEDIATE / "tweets_train.csv",
        DATA_INTERMEDIATE / "tweets_val.csv",
    )
    tokenizer, train_tokenized, val_tokenized = tokenize_datasets(
        train_set, val_set, args.max_length
    )

    label2id = {"NEG": 0, "POS": 1}
    id2label = {v: k for k, v in label2id.items()}

    # We use DistilBERT for classification
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=DISTILBERT / "artifacts",
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        logging_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",  # Optimize for accuracy
        greater_is_better=True,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.num_epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting DistilBERT training...\n")
    trainer.train()

    trainer.save_model(DISTILBERT / "best_model" / "artifacts")
    tokenizer.save_pretrained(DISTILBERT / "best_model" / "artifacts")


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
