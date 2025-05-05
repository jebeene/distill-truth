from datasets import load_dataset

def load_hf_dataset(name: str, split: str = "train"):
    """
    Load a dataset from HuggingFace datasets hub.

    :param name: Dataset name (e.g., "imdb", "ag_news")
    :param split: Split to load (e.g., "train", "test")
    :return: HuggingFace Dataset object
    """
    return load_dataset(name, split=split)
