

# src/modeling/lora_model.py

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

from src.config.training_config import TrainingConfig


def load_tokenizer(config: TrainingConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_dir,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
    if tokenizer.unk_token is None:
        tokenizer.add_special_tokens({"unk_token": "<unk>"})
    return tokenizer


def load_lora_model(config: TrainingConfig):
    """
    Load the base model in 4-bit and wrap with LoRA adapters.
    """

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print(" Loading base model on GPU-0 (4-bit NF4)...")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_dir,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False  # required for gradient checkpointing
    print("Base model loaded")

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    lora_cfg = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.target_modules),
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    print("LoRA adapters initialized and gradient checkpointing active")
    return model