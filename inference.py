from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dataset import load_train_test_dataloader, load_train_test_dataset
from tqdm import tqdm
import pandas as pd

device = "cuda:0"
tokenizer = AutoTokenizer.from_pretrained("allenai/tk-instruct-3b-def-pos-neg-expl")
model = AutoModelForSeq2SeqLM.from_pretrained("allenai/tk-instruct-3b-def-pos-neg-expl").to(device)

_, test_dataset = load_train_test_dataset()



pred, label = [], []
for row in tqdm(test_dataset):
    input_ids = tokenizer.encode(
        row["text"],
        return_tensors="pt",
    ).to(device)
    output = model.generate(input_ids, max_length=100)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    pred.append(output)
    label.append(row["aspect"])

df = pd.DataFrame()

df["pred"] = pred
df["label"] = label
df.to_csv("./results/pred-11b.csv")