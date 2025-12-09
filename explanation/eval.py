

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def explanation_accuracy(
    gold_path: Path,
    pred_path: Path,
    reason_key: str = "reason",
) -> float:
    """
    Compute simple explanation accuracy between gold and predicted JSONL files.

    We assume that both JSONL files are aligned line-by-line, and each line
    contains at least a `reason` field.

    A prediction is counted as correct if the predicted `reason` string
    exactly matches the gold `reason` string (after stripping whitespace).

    Args:
        gold_path: Path to reference explanations (JSONL).
        pred_path: Path to predicted explanations (JSONL).
        reason_key: Key used for explanation text.

    Returns:
        Accuracy in [0,1].
    """
    gold = _load_jsonl(gold_path)
    pred = _load_jsonl(pred_path)

    if len(gold) != len(pred):
        raise ValueError(
            f"Gold and predicted files have different lengths "
            f"({len(gold)} vs {len(pred)}). They must be aligned."
        )

    correct = 0
    total = len(gold)

    for g, p in zip(gold, pred):
        g_reason = str(g.get(reason_key, "")).strip()
        p_reason = str(p.get(reason_key, "")).strip()
        if g_reason == p_reason:
            correct += 1

    return correct / total if total > 0 else 0.0


def main() -> None:
    """
    Simple CLI to compute explanation accuracy:

        python -m src.explanation.eval \\
            --gold path/to/gold.jsonl \\
            --pred path/to/pred.jsonl
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute simple explanation accuracy between gold and predictions."
    )
    parser.add_argument("--gold", type=str, required=True, help="Gold JSONL file.")
    parser.add_argument("--pred", type=str, required=True, help="Predicted JSONL file.")
    parser.add_argument(
        "--reason-key",
        type=str,
        default="reason",
        help="Key for explanation text in JSON objects.",
    )

    args = parser.parse_args()

    acc = explanation_accuracy(
        gold_path=Path(args.gold),
        pred_path=Path(args.pred),
        reason_key=args.reason_key,
    )
    print(f"Explanation accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()