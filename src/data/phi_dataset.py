
# src/data/phi_dataset.py

import json
from pathlib import Path
from typing import Dict

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase


def read_jsonl(path: Path):
    """
    Reads a JSONL file with entries like:
        {
          "input": "...",
          "output": { "redacted_text": "..." }
        }
    and maps to {instruction, output}.
    """
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            it = json.loads(line)
            data.append(
                {
                    "instruction": it["input"],
                    "output": it["output"]["redacted_text"],
                }
            )
    return data


def load_raw_phi_dataset(
    data_path: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """
    Loads the single phi_data.jsonl file and splits into train/val/test.
    """
    all_data = read_jsonl(data_path)
    print(f"Total samples in {data_path.name}: {len(all_data)}")

    ds_full = Dataset.from_list(all_data)

    # 1) train vs holdout (val+test)
    test_ratio = 1.0 - train_ratio
    splits = ds_full.train_test_split(test_size=test_ratio, seed=seed)
    ds_train = splits["train"]
    ds_holdout = splits["test"]

    # 2) split holdout into val + test
    val_frac_within_holdout = val_ratio / (1.0 - train_ratio)
    val_test_splits = ds_holdout.train_test_split(
        test_size=1.0 - val_frac_within_holdout,
        seed=seed,
    )
    ds_val = val_test_splits["train"]
    ds_test = val_test_splits["test"]

    dataset = DatasetDict(
        {
            "train": ds_train,
            "validation": ds_val,
            "test": ds_test,
        }
    )

    print(
        f"Split sizes -> train: {len(dataset['train'])}, "
        f"val: {len(dataset['validation'])}, "
        f"test: {len(dataset['test'])}"
    )
    return dataset


def tokenize_and_mask(
    raw_dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    max_len: int,
) -> DatasetDict:
    """
    Tokenizes the dataset and masks the instruction part with label=-100
    so the loss is only on the output tokens.
    """

    def build_example(ex: Dict):
        prompt = f"### Instruction:\n{ex['instruction'].strip()}\n\n### Output:\n"
        target = ex["output"].strip()
        full = prompt + target

        enc = tokenizer(
            full, truncation=True, padding="max_length", max_length=max_len
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

        labels = enc["input_ids"].copy()
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100
        enc["labels"] = labels
        return enc

    print(f"Tokenizing with max_seq_len={max_len} ...")
    tokenized = raw_dataset.map(build_example, batched=False)
    print("Tokenization + label masking complete")
    return tokenized