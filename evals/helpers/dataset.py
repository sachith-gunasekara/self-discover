import os

from datasets import Dataset, concatenate_datasets
from pyprojroot import here


def load_checkpoints(save_dir: str) -> Dataset:
    all_datasets = []
    for file in os.listdir(save_dir):
        if file.startswith("checkpoint"):
            ds = Dataset.load_from_disk(os.path.join(save_dir, file))
            all_datasets.append(ds)
    full_dataset = concatenate_datasets(all_datasets)

    return full_dataset
