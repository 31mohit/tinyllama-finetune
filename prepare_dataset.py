from datasets import load_dataset
import pandas as pd

print("Loading dataset...")
dataset = load_dataset("yahma/alpaca-cleaned", split="train")

dataset = dataset.select(range(1000))

def format_example(example):
    if example["input"]:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    else:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    return {"text": prompt}

formatted = dataset.map(format_example)

df = formatted.to_pandas()[["text"]]
df.to_csv("dataset.csv", index=False)

print(f"✅ Dataset saved! Total examples: {len(df)}")
print(df.head(2))
