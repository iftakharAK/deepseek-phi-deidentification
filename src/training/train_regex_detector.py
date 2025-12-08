

# src/training/train_regex_detector.py

"""
Main entry point for training the regex-based PHI detector with QLoRA.

Run from repo root:
    python -m src.training.train_regex_detector

Or via:
    bash scripts/train_regex_detector.sh
"""

import gc
import os

import torch
from datasets import DatasetDict

from src.config.training_config import TrainingConfig
from src.utils.paths import get_repo_root, get_data_path, get_output_dir
from src.data.phi_dataset import load_raw_phi_dataset, tokenize_and_mask
from src.modeling.lora_model import load_lora_model, load_tokenizer
from src.training.trainer_builder import build_trainer


def main():
    # -------------------------
    # Environment + config
    # -------------------------
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    repo_root = get_repo_root()
    config = TrainingConfig()

    data_path = get_data_path()
    output_dir = get_output_dir(config.run_id)

    print(f"Repo root   : {repo_root}")
    print(f"Data path   : {data_path}")
    print(f"Base model  : {config.base_model_dir}")
    print(f"Output dir  : {output_dir}")

    # -------------------------
    # Load raw data & tokenizer
    # -------------------------
    raw_dataset: DatasetDict = load_raw_phi_dataset(
        data_path=data_path,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )

    tokenizer = load_tokenizer(config)

    # -------------------------
    # Tokenize
    # -------------------------
    encoded_dataset = tokenize_and_mask(
        raw_dataset=raw_dataset,
        tokenizer=tokenizer,
        max_len=config.max_seq_len,
    )

    # -------------------------
    # Load model + Trainer
    # -------------------------
    model = load_lora_model(config)
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        encoded_dataset=encoded_dataset,
        config=config,
        output_dir=output_dir,
    )

    torch.cuda.empty_cache()

    # -------------------------
    # Training (with simple OOM fallback)
    # -------------------------
    def run_training_with_optional_shrink():
        nonlocal encoded_dataset

        try:
            print("Starting training...")
            trainer.train()
        except torch.cuda.OutOfMemoryError:
            print(
                "OOM detected. Retrying with smaller max_seq_len=1280 "
                "(you can also change this in TrainingConfig)."
            )
            gc.collect()
            torch.cuda.empty_cache()

            config.max_seq_len = 1280
            encoded_dataset = tokenize_and_mask(
                raw_dataset=raw_dataset,
                tokenizer=tokenizer,
                max_len=config.max_seq_len,
            )

            # rebuild trainer with new dataset
            new_trainer = build_trainer(
                model=model,
                tokenizer=tokenizer,
                encoded_dataset=encoded_dataset,
                config=config,
                output_dir=output_dir,
            )
            print("Restarting training with smaller sequence length...")
            new_trainer.train()
            return new_trainer

        return trainer

    final_trainer = run_training_with_optional_shrink()

    # -------------------------
    # Save final model
    # -------------------------
    final_dir = output_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Final fine-tuned model saved at: {final_dir}")

    # -------------------------
    # Quick verification samples
    # -------------------------
    model.eval()

    def show_sample(i: int = 0):
        ex = raw_dataset["test"][i]
        text = f"### Instruction:\n{ex['instruction']}\n\n### Output:\n"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,
            )
        print(f"\n--- SAMPLE {i} ---")
        print(tokenizer.decode(out[0], skip_special_tokens=True))

    for i in range(2):
        try:
            show_sample(i)
        except Exception as e:
            print(f"Sample {i} skipped: {e}")


if __name__ == "__main__":
    main()