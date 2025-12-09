# DeepSeek-PHI-Deidentification  
## Trustworthy and Configurable PHI De-Identification with Prompt-Adaptive Filtering, Explanations & Hallucination Detection

[![HF Model](https://img.shields.io/badge/HuggingFace-Iftakhar/deepseek--phi--adapter-yellow)](https://huggingface.co/Iftakhar/deepseek-phi-adapter)
[![Colab](https://img.shields.io/badge/Open%20in%20Colab-DeepSeek%20PHI-blue?logo=googlecolab)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](#license)

This repository contains the official **code, synthetic evaluation datasets, explanation generators, hallucination detection modules, and reproducibility artifacts** for the research paper:

> **Trustworthy and Configurable PHI De-Identification in Clinical Text via Prompt-Adaptive Filtering, Explanation Generation, and Hallucination Detection**

The proposed system fine-tunes a **DeepSeek-LLM using QLoRA** to detect, redact, and explain Protected Health Information (PHI) in a configurable manner, while incorporating an automated hallucination detection layer to enhance clinical safety.

---

## 🔗 Pretrained Model (Hugging Face)

This project uses a publicly released **LoRA adapter** hosted on Hugging Face:

###  https://huggingface.co/Iftakhar/deepseek-phi-adapter

All demo, evaluation, and reproducibility scripts automatically download this adapter during inference using:

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

MODEL_ID = "Iftakhar/deepseek-phi-adapter"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoPeftModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    trust_remote_code=True
)
