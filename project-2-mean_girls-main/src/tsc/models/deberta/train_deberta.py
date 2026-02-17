# src/tsc/models/deberta/train_deberta.py

import argparse
import pandas as pd
import numpy as np

from functools import partial
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from tsc.utils.metrics import accuracy, f1_score
from tsc.utils.paths import DATA_INTERMEDIATE, DEBERTA


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=2)
    return parser.parse_args()


# Create datasets from CSV files
def load_datasets(train_csv, val_csv):
    train_frame = pd.read_csv(train_csv)
    val_frame = pd.read_csv(val_csv)

    train_set = Dataset.from_pandas(train_frame[["text", "label"]])
    val_set = Dataset.from_pandas(val_frame[["text", "label"]])
    return train_set, val_set


# Tokenize datasets using DeBERTa tokenizer
def tokenize_datasets(train_set, val_set, max_length):
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/deberta-v3-base", use_fast=True
    )
    tokenize_fct = partial(tokenizer, truncation=True, max_length=max_length)
    print("Tokenizing datasets for DeBERTa...\n \n")

    train_tokenized = train_set.map(
        lambda batch: tokenize_fct(batch["text"]), batched=True
    )
    val_tokenized = val_set.map(lambda batch: tokenize_fct(batch["text"]), batched=True)

    # Transform to PyTorch tensors and remove text column
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
        DATA_INTERMEDIATE / "tweets_train.csv", DATA_INTERMEDIATE / "tweets_val.csv"
    )
    tokenizer, train_tokenized, val_tokenized = tokenize_datasets(
        train_set, val_set, args.max_length
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    label2id = {"NEG": 0, "POS": 1}
    id2label = {v: k for k, v in label2id.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-large",
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=DEBERTA / "artifacts",
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        logging_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",  # Optimize for accuracy
        greater_is_better=True,
        learning_rate=1e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=1,
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=True,
        tf32=True,
        report_to="none",
        optim="adamw_torch_fused",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_accumulation_steps=1,
        group_by_length=False,
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting DeBERTa training...\n \n")
    trainer.train()

    trainer.save_model(DEBERTA / "artifacts" / "best_model")
    tokenizer.save_pretrained(DEBERTA / "artifacts" / "best_model")


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
