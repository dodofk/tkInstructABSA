from datasets import Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import pandas as pd
import torch


def preprocess_label(examples):
    return ",".join([f"{pair['category']}:{pair['polarity']}"for pair in examples])


def load_train_test_dataset():
    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/tk-instruct-3b-def-pos", padding=True, Truncation=True
    )

    with open("instruct-2.txt", "r") as f:
        instruction = f.read()

    train_df = pd.read_csv("Dataset/SemEval14/Train/Restaurants_Train.csv")
    test_df = pd.read_csv("Dataset/SemEval14/Test/Restaurants_Test.csv")

    train_df["text"] = train_df["raw_text"].apply(lambda x: instruction + x + "\nOutput: ")
    train_df["aspect"] = train_df["aspectCategories"].apply(
        lambda x: preprocess_label(eval(x))
    )

    test_df["text"] = test_df["raw_text"].apply(lambda x: instruction + x + "\nOutput: ")
    test_df["aspect"] = test_df["aspectCategories"].apply(
        lambda x: preprocess_label(eval(x))
    )

    train_df = train_df[["text", "aspect"]]
    test_df = test_df[["text", "aspect"]]

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    return train_dataset, test_dataset


class Collator(object):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "allenai/tk-instruct-3b-def-pos", padding=True, Truncation=True
        )

    def __call__(self, batch):
        model_input = self.tokenizer(
            [i["text"] for i in batch],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        labels = self.tokenizer(
            [i["aspect"] for i in batch],
            padding=True,
            truncation=True,
        ).input_ids

        labels =  [
            [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels
        ]

        labels = torch.LongTensor(labels)

        return {
            "labels": labels,
            "aspect": [i["aspect"] for i in batch],
            **model_input,
        }


def load_train_test_dataloader(batch_size: int=8):
    train_dataset, test_dataset = load_train_test_dataset()
    collate_fn = Collator()
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
    )

    test_dataloader =  DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=False,
    )
    return train_dataloader, test_dataloader

    

if __name__ == "__main__":
    train, test = load_train_test_dataloader(batch_size=16)
    # sample
    print(next(iter(train)), )
