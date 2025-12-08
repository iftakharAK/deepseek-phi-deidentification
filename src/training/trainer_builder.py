
# src/training/trainer_builder.py

from pathlib import Path

from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from datasets import DatasetDict

from src.config.training_config import TrainingConfig


def build_training_arguments(
    config: TrainingConfig,
    output_dir: Path,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_train,
        per_device_eval_batch_size=config.batch_eval,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_dir=str(output_dir / "logs"),
        logging_steps=config.log_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        evaluation_strategy="steps",
        eval_steps=config.save_steps,
        save_total_limit=3,
        fp16=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        eval_accumulation_steps=4,
    )


def build_trainer(
    model,
    tokenizer,
    encoded_dataset: DatasetDict,
    config: TrainingConfig,
    output_dir: Path,
) -> Trainer:
    training_args = build_training_arguments(config, output_dir)
    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    return trainer