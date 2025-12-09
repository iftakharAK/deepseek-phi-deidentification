

"""
End-to-end evaluation script for the DeepSeek PHI de-identification model.

- Loads data from data/phi_data.jsonl
- Loads LoRA adapter from Hugging Face: Iftakhar/deepseek-phi-adapter
- Runs inference over (optionally subset of) the dataset
- Computes:
    - token-level PHI tag micro/macro Precision/Recall/F1
    - char-level F1
    - hallucination rate
    - (optionally) BLEU and trust score
"""

import os
import json
import argparse
import random
import re
from collections import defaultdict
from typing import List, Dict

import torch
from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM

from .metrics import (
    compute_token_level_metrics,
    compute_char_level_f1,
    detect_hallucination,
    classify_policy,
    extract_phi_tags,
    safe_div,
)


def read_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            data.append(
                {
                    "instruction": j.get("input", ""),
                    "ground_truth": j.get("output", {}).get("redacted_text", ""),
                }
            )
    return data


def clean_output(txt: str) -> str:
    """
    Remove trailing markers and artifacts. Mirrors the logic you used
    in your previous evaluation script.
    """
    txt = txt.split("<END>")[0]
    txt = re.sub(r"(RecordedVote|<HR>|ý:).*", "", txt, flags=re.DOTALL)
    return txt.strip()


def load_model_and_tokenizer(
    hf_adapter: str,
    device: str = "auto",
):
    """
    Load LoRA adapter model from HuggingFace with 4-bit quantization.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(hf_adapter, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if "<END>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<END>"]})
    end_id = tokenizer.convert_tokens_to_ids("<END>") if "<END>" in tokenizer.get_vocab() else tokenizer.eos_token_id

    model = AutoPeftModelForCausalLM.from_pretrained(
        hf_adapter,
        device_map="auto" if device == "cuda" else {"": device},
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )

    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    return model, tokenizer, end_id, device


def run_inference(
    model,
    tokenizer,
    end_id: int,
    device: str,
    samples: List[Dict],
    max_new_tokens: int = 384,
    temperature: float = 0.0,
    top_p: float = 0.9,
) -> List[str]:
    """
    Run batched (but simple) inference over a list of samples.
    Returns list of model outputs (redacted text).
    """
    preds = []

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=end_id,
    )

    for sample in samples:
        prompt = f"### Instruction:\n{sample['instruction']}\n\n### Output:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            gen = model.generate(**inputs, **gen_kwargs)

        decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
        # The model might echo the prompt; keep only after "### Output:"
        out_text = decoded.split("### Output:", 1)[-1]
        cleaned = clean_output(out_text)
        preds.append(cleaned)

    return preds


def aggregate_policy_metrics(
    samples: List[Dict],
    preds: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Break down metrics by instruction policy type (optional, but useful).
    """
    metrics_by_policy = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "Hall": 0, "Count": 0, "BLEU": [], "Trust": []})

    for sample, pred in zip(samples, preds):
        gt = sample["ground_truth"]
        policy = classify_policy(sample["instruction"])

        pred_tags = set(extract_phi_tags(pred))
        gt_tags = set(extract_phi_tags(gt))

        tp = len(pred_tags & gt_tags)
        fp = len(pred_tags - gt_tags)
        fn = len(gt_tags - pred_tags)

        metrics_by_policy[policy]["TP"] += tp
        metrics_by_policy[policy]["FP"] += fp
        metrics_by_policy[policy]["FN"] += fn
        metrics_by_policy[policy]["Count"] += 1

        # hallucination
        is_hall = detect_hallucination(sample["instruction"], pred)
        metrics_by_policy[policy]["Hall"] += int(is_hall)

        # BLEU / Trust (lightweight)
        try:
            bleu = sentence_bleu([gt.split()], pred.split())
        except Exception:
            bleu = 0.0
        metrics_by_policy[policy]["BLEU"].append(bleu)
        trust = (1 - int(is_hall)) * bleu
        metrics_by_policy[policy]["Trust"].append(trust)

    # Convert to summary metrics
    summary = {}
    for pol, m in metrics_by_policy.items():
        P = safe_div(m["TP"], m["TP"] + m["FP"])
        R = safe_div(m["TP"], m["TP"] + m["FN"])
        F1 = safe_div(2 * P * R, P + R) if (P + R) else 0.0
        HallRate = safe_div(m["Hall"], m["Count"])
        BLEU = sum(m["BLEU"]) / max(1, len(m["BLEU"]))
        Trust = sum(m["Trust"]) / max(1, len(m["Trust"]))
        summary[pol] = {
            "precision": P,
            "recall": R,
            "f1": F1,
            "hallucination_rate": HallRate,
            "bleu": BLEU,
            "trust": Trust,
            "count": m["Count"],
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Full evaluation for DeepSeek PHI adapter.")
    parser.add_argument("--data-path", type=str, default="data/phi_data.jsonl", help="Path to JSONL data file.")
    parser.add_argument("--hf-adapter", type=str, default="Iftakhar/deepseek-phi-adapter", help="HuggingFace adapter repo or local path.")
    parser.add_argument("--device", type=str, default="auto", help="cuda | cpu | auto (default).")
    parser.add_argument("--max-samples", type=int, default=-1, help="Limit number of samples for quick evaluation (-1 = all).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save-predictions", type=str, default="", help="Optional output JSONL path for predictions.")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f" Loading data from: {args.data_path}")
    samples = read_jsonl(args.data_path)
    print(f"   Total samples: {len(samples)}")

    if args.max_samples > 0 and args.max_samples < len(samples):
        samples = random.sample(samples, args.max_samples)
        print(f"   Subsampled to: {len(samples)} (max-samples={args.max_samples})")

    print(f"\n Loading model/adapter from: {args.hf_adapter}")
    model, tokenizer, end_id, device = load_model_and_tokenizer(args.hf_adapter, device=args.device)
    print(f"   Using device: {device}")

    print("\n Running inference...")
    preds = run_inference(model, tokenizer, end_id, device, samples)

    gt_texts = [s["ground_truth"] for s in samples]

    # ------------------------------------------------------------
    # GLOBAL METRICS
    # ------------------------------------------------------------
    print("\nComputing token-level PHI metrics...")
    token_metrics = compute_token_level_metrics(gt_texts, preds)

    print(" Computing char-level F1...")
    char_metrics = compute_char_level_f1(gt_texts, preds)

    print(" Computing hallucination rate...")
    hall_flags = [detect_hallucination(s["instruction"], p) for s, p in zip(samples, preds)]
    hall_rate = sum(hall_flags) / len(hall_flags) if hall_flags else 0.0

    # Per-policy breakdown (optional, but useful)
    print(" Computing policy-wise metrics...")
    policy_summary = aggregate_policy_metrics(samples, preds)

    # ------------------------------------------------------------
    # PRINT SUMMARY
    # ------------------------------------------------------------
    print("\n=================  GLOBAL METRICS =================")
    print(f"Token-Level Micro Precision: {token_metrics['micro_precision']:.3f}")
    print(f"Token-Level Micro Recall   : {token_metrics['micro_recall']:.3f}")
    print(f"Token-Level Micro F1       : {token_metrics['micro_f1']:.3f}")
    print(f"Token-Level Macro Precision: {token_metrics['macro_precision']:.3f}")
    print(f"Token-Level Macro Recall   : {token_metrics['macro_recall']:.3f}")
    print(f"Token-Level Macro F1       : {token_metrics['macro_f1']:.3f}")

    print("\nCharacter-Level Metrics:")
    print(f"Char Precision: {char_metrics['char_precision']:.3f}")
    print(f"Char Recall   : {char_metrics['char_recall']:.3f}")
    print(f"Char F1       : {char_metrics['char_f1']:.3f}")

    print(f"\nHallucination Rate (overall): {hall_rate * 100:.2f}%")

    print("\n=================  POLICY BREAKDOWN =================")
    for pol, m in policy_summary.items():
        print(f"\nPolicy: {pol} (n={m['count']})")
        print(f"  Precision         : {m['precision']:.3f}")
        print(f"  Recall            : {m['recall']:.3f}")
        print(f"  F1                : {m['f1']:.3f}")
        print(f"  Hallucination Rate: {m['hallucination_rate'] * 100:.2f}%")
        print(f"  BLEU              : {m['bleu']:.3f}")
        print(f"  Trust             : {m['trust']:.3f}")

    # ------------------------------------------------------------
    # OPTIONAL: SAVE PREDICTIONS
    # ------------------------------------------------------------
    if args.save_predictions:
        out_path = args.save_predictions
        print(f"\n Saving predictions to: {out_path}")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for sample, pred in zip(samples, preds):
                rec = {
                    "instruction": sample["instruction"],
                    "ground_truth": sample["ground_truth"],
                    "prediction": pred,
                    "policy": classify_policy(sample["instruction"]),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("   Done.")


if __name__ == "__main__":
    main()