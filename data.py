from datasets import load_dataset

def load_liar_dataset():
    dataset = load_dataset("liar", trust_remote_code=True)
    return dataset
