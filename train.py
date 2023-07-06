import pandas as pd
import numpy as np
import torch
from datasets import load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    Adafactor,
)
from dataset import load_train_test_dataloader
from tqdm import tqdm
from lightning.pytorch import Trainer
from model import InstructABSA


# model_name = "allenai/tk-instruct-3b-def-pos"
model_name = "allenai/tk-instruct-base-def-pos"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_dataloader, test_dataloader = load_train_test_dataloader()


device = "cuda:0"
num_epochs = 5
model.train()

optimizer = Adafactor(
    model.parameters(),
    lr=1e-3,
    scale_parameter=False,
    relative_step=False,
)


print("Training =====")
for epoch in range(num_epochs):
    loss_sum = 0.0
    cnt = 0
    for batch in tqdm(train_dataloader):
        # print(batch)
        optimizer.zero_grad()
        loss = model(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
        ).loss

        loss.backward()
        optimizer.step()

        cnt += len(batch)
        loss_sum += torch.sum(loss).item()
    print(f"Epoch: {epoch + 1}. Loss: {loss_sum/cnt}")


print("Predicting =====")

model.eval()
pred, label = [], []
for batch in tqdm(test_dataloader):
    output = model.generate(
        input_ids=batch["input_ids"],
        max_length=100,
    )
    output = tokenizer.decode(
        output[0],
        skip_special_tokens=False,
    )

    pred.append(output)
    label.append(batch["aspect"])

df = pd.DataFrame()

df["pred"] = pred
df["label"] = label

df.to_csv("./results/pred-ft-full-3b.csv")












