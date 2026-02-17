# src/tsc/utils/helpers.py

"""Some helper functions for project 1."""

import csv
from tsc.utils.paths import RESULTS


def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(RESULTS / name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def write_csv(path: str, data):
    """Write a numpy array to a CSV file with a header.

    Args:
        path (str): Path to the output CSV file.
        data (List[(text, label)]): list of tuples (text, label).
    """
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["text", "label"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for text, label in data:
            writer.writerow({"text": text, "label": label})
