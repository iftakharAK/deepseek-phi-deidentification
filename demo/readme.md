# Demo – DeepSeek PHI De-Identification

This folder contains lightweight demo scripts for running inference and
a small-scale evaluation using the Hugging Face LoRA adapter:

- Model: `Iftakhar/deepseek-phi-adapter`
- Data: `data/phi_data.jsonl`

These scripts are meant for:
- Quick qualitative inspection of model outputs
- Small-scale metric computation (precision/recall/F1, hallucination rate, etc.)
- Reproducible examples for reviewers

---

## Requirements

Install dependencies (if not already installed in your environment):

```bash
pip install torch transformers peft accelerate bitsandbytes scikit-learn nltk
