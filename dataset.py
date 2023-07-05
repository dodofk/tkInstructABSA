from datasets import Dataset
from transformers import AutoTokenizer
import pandas as pd

aspects = [
    "food",
    "service",
    "price",
    "ambience",
    "anecdotes/miscellaneous",
]


def preprocess_label(examples):
    ret = []
    aspect_polarity = {aspect: "none" for aspect in aspects}
    for pair in examples:
        aspect_polarity[pair["category"]] = pair["polarity"]

    for aspect in aspects:
        ret.append(f"{aspect}:{aspect_polarity[aspect]}")
    return ",".join(ret)


def preprocess_tokenization(example, tokenizer):
    model_inputs = tokenizer(
        example["text"],
        padding=True,
        truncation=True,
    )
    labels = tokenizer(
        example["label"],
        padding=True,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def load_train_test_dataloader():
    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/tk-instruct-3b-def-pos", padding=True, Truncation=True
    )

    with open("instruct-1.txt", "r") as f:
        instruction = f.read()

    train_df = pd.read_csv("Dataset/SemEval14/Train/Restaurants_Train.csv")
    test_df = pd.read_csv("Dataset/SemEval14/Test/Restaurants_Test.csv")

    train_df["text"] = train_df["raw_text"].apply(lambda x: instruction + x + "\nOutput: ")
    train_df["label"] = train_df["aspectCategories"].apply(
        lambda x: preprocess_label(eval(x))
    )

    test_df["text"] = test_df["raw_text"].apply(lambda x: instruction + x + "\nOutput: ")
    test_df["label"] = test_df["aspectCategories"].apply(
        lambda x: preprocess_label(eval(x))
    )

    train_df = train_df[["text", "label"]]
    test_df = test_df[["text", "label"]]

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    train_dataset = train_dataset.map(
        lambda example: preprocess_tokenization(example, tokenizer),
        batched=True,
    )

    test_dataset = test_dataset.map(
        lambda example: preprocess_tokenization(example, tokenizer),
        batched=True,
    )

    return train_dataset, test_dataset


if __name__ == "__main__":
    train, test = load_train_test_dataloader()
    print(train.__getitem__(0))
    print(train.__getitem__(0).keys())
