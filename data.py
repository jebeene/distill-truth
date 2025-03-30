# data/liar_dataset.py
from datasets import load_dataset
from config import LABEL2ID

def load_liar_dataset():
    dataset = load_dataset("liar")
    for split in dataset:
        dataset[split] = dataset[split].filter(lambda x: x["label"] != "")
        dataset[split] = dataset[split].map(lambda x: x)
    return dataset
