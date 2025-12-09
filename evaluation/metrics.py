

import re
from typing import List, Tuple, Dict
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support


def safe_div(a: float, b: float) -> float:
    """Safe division with zero-guard."""
    return a / b if b else 0.0


# -------------------------------------------------------------------
# PHI TAG UTILITIES
# -------------------------------------------------------------------

TAG_PATTERN = re.compile(r"<REDACTED:([A-Z_]+)>")


def extract_phi_tags(text: str) -> List[str]:
    """
    Extract PHI tag types from a redacted text.

    Example:
        "foo <REDACTED:DATE> bar <REDACTED:NAME>" -> ["DATE", "NAME"]
    """
    return TAG_PATTERN.findall(text or "")


def compute_token_level_metrics(
    gt_texts: List[str],
    pred_texts: List[str],
) -> Dict[str, float]:
    """
    Compute token-level PHI tag metrics based on presence of tag TYPES.

    We treat each unique tag type per sample as a 'label'. For each sample:
    - ground truth tag set: set(extract_phi_tags(gt_text))
    - predicted tag set:    set(extract_phi_tags(pred_text))

    TP: intersection size, FP: predicted-only, FN: gt-only.

    Returns a dict with micro and macro metrics:
        {
            "micro_precision": ...,
            "micro_recall": ...,
            "micro_f1": ...,
            "macro_precision": ...,
            "macro_recall": ...,
            "macro_f1": ...
        }
    """
    assert len(gt_texts) == len(pred_texts), "Length mismatch between GT and predictions"

    per_sample_stats = []
    total_TP = total_FP = total_FN = 0

    for gt, pred in zip(gt_texts, pred_texts):
        gt_tags = set(extract_phi_tags(gt))
        pred_tags = set(extract_phi_tags(pred))

        tp = len(gt_tags & pred_tags)
        fp = len(pred_tags - gt_tags)
        fn = len(gt_tags - pred_tags)

        total_TP += tp
        total_FP += fp
        total_FN += fn

        per_sample_stats.append((tp, fp, fn))

    # Micro
    micro_P = safe_div(total_TP, total_TP + total_FP)
    micro_R = safe_div(total_TP, total_TP + total_FN)
    micro_F1 = safe_div(2 * micro_P * micro_R, micro_P + micro_R) if (micro_P + micro_R) else 0.0

    # Macro
    macro_Ps, macro_Rs, macro_F1s = [], [], []
    for tp, fp, fn in per_sample_stats:
        P = safe_div(tp, tp + fp)
        R = safe_div(tp, tp + fn)
        F1 = safe_div(2 * P * R, P + R) if (P + R) else 0.0
        macro_Ps.append(P)
        macro_Rs.append(R)
        macro_F1s.append(F1)

    macro_P = sum(macro_Ps) / len(macro_Ps) if macro_Ps else 0.0
    macro_R = sum(macro_Rs) / len(macro_Rs) if macro_Rs else 0.0
    macro_F1 = sum(macro_F1s) / len(macro_F1s) if macro_F1s else 0.0

    return {
        "micro_precision": micro_P,
        "micro_recall": micro_R,
        "micro_f1": micro_F1,
        "macro_precision": macro_P,
        "macro_recall": macro_R,
        "macro_f1": macro_F1,
    }


# -------------------------------------------------------------------
# CHARACTER-LEVEL F1
# -------------------------------------------------------------------

def compute_char_level_f1(
    gt_texts: List[str],
    pred_texts: List[str],
) -> Dict[str, float]:
    """
    Compute micro-averaged character-level Precision/Recall/F1.

    Concatenates all ground-truth and predicted strings and compares
    at the character level.
    """
    assert len(gt_texts) == len(pred_texts), "Length mismatch between GT and predictions"

    y_true = "".join(gt_texts)
    y_pred = "".join(pred_texts)

    P_c, R_c, F_c, _ = precision_recall_fscore_support(
        list(y_true),
        list(y_pred),
        average="micro",
        zero_division=0,
    )
    return {
        "char_precision": float(P_c),
        "char_recall": float(R_c),
        "char_f1": float(F_c),
    }


# -------------------------------------------------------------------
# POLICY CLASSIFICATION (PROMPT-LEVEL)
# -------------------------------------------------------------------

def classify_policy(prompt: str) -> str:
    """
    Heuristic classifier for redaction policy based on natural language
    instructions. Mirrors the logic used in the paper for grouping.
    """
    if not prompt:
        return "General"

    p = prompt.lower()

    if "all phi" in p or "everything" in p or "all protected health information" in p:
        return "Redact All PHI"
    if "do not redact" in p or "keep all" in p:
        return "Do Not Redact"
    if "only" in p:
        if "hospital" in p or "facility" in p:
            return "Facility Only"
        if "vehicle" in p or "vin" in p:
            return "Vehicle Only"
        if "date" in p or "dob" in p:
            return "Date Only"
        if "identifier" in p or "ssn" in p or "id number" in p:
            return "Identifier Only"
        return "Selective"
    if "except" in p or "but not" in p:
        return "Except Some PHI"

    return "General"


# -------------------------------------------------------------------
# HALLUCINATION DETECTION
# -------------------------------------------------------------------

# Very simple regex patterns for common PHI-like entities.
HALL_PATTERNS = [
    re.compile(r"\bssn\b[\s:]*\d{3}[- ]?\d{2}[- ]?\d{4}", re.IGNORECASE),
    re.compile(r"\bmrn\d+\b", re.IGNORECASE),
    re.compile(r"\bvin\d+\b", re.IGNORECASE),
    re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b"),  # naive full names
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),  # emails
]


def detect_hallucination(original_text: str, model_output: str) -> bool:
    """
    Detects whether the model hallucinated PHI-like content that was NOT
    present in the original input.

    Strategy:
    - Convert both original text and model output to lowercase
    - Search for PHI-like patterns (names, ids, emails) in the model output
    - If any such entity is not a substring of the original, mark as hallucination

    NOTE: this is a heuristic, not a perfect detector.
    """
    if not model_output:
        return False

    original = (original_text or "").lower()
    output = model_output

    hallucinated_entities = []

    for pattern in HALL_PATTERNS:
        for match in pattern.findall(output):
            if str(match).lower() not in original:
                hallucinated_entities.append(match)

    return len(hallucinated_entities) > 0