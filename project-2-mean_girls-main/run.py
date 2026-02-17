#!/usr/bin/env python

from tsc.preprocess.split_dataset import main as split_dataset_main
from tsc.models.deberta.train_deberta import main as deberta_train_main
from tsc.models.deberta.predict_deberta import main as deberta_predict_main


def main():
    split_dataset_main()

    # This method can crash as it requires a GPU
    deberta_train_main()

    deberta_predict_main()


if __name__ == "__main__":
    main()
