from datasets import load_dataset
import os

os.makedirs("data/raw", exist_ok=True)
ds = load_dataset("dair-ai/emotion")
for split in ["train", "validation", "test"]:
    ds[split].to_csv(f"data/raw/{split}.csv", index=False)
    print(split, len(ds[split]))