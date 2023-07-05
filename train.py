import pandas as pd
import torch
from datasets import load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from dataset import load_train_test_dataloader
from tqdm import tqdm

model_name = "allenai/tk-instruct-3b-def-pos"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
data_collator = DataCollatorForSeq2Seq(model=model, tokenizer=tokenizer)
train_dataset, test_dataset = load_train_test_dataloader()

training_arg = Seq2SeqTrainingArguments(
    output_dir="./results",
    save_total_limit=1,
    save_strategy="epoch",
    fp16=True,
    use_mps_device=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_arg,
    train_dataset=None,
    eval_dataset=None,
    data_collator=data_collator,
)

trainer.predict(test_dataset)






