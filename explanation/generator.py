

import json
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional

from .schema import ExplanationRecord
from .templates import get_explanation_template


def generate_explanations_for_spans(
    spans: Iterable[Dict[str, Any]],
    add_token_in_reason: bool = False,
) -> List[ExplanationRecord]:
    """
    Generate ExplanationRecord objects for a list of PHI spans.

    Each span is expected to be a dict containing at least:
        - "token": text span
        - "label": PHI label (e.g., NAME, DATE, HOSPITAL)

    Any extra fields are ignored here (you can keep them elsewhere).

    Args:
        spans: Iterable of dicts describing PHI detections.
        add_token_in_reason: If True, prepend the token into the reason,
                             e.g., `"John Smith" [...]`.

    Returns:
        A list of ExplanationRecord objects.
    """
    records: List[ExplanationRecord] = []

    for span in spans:
        token = str(span.get("token", "")).strip()
        label = str(span.get("label", "")).strip().upper()

        base_reason = get_explanation_template(label)

        if add_token_in_reason and token:
            reason = f'"{token}" is redacted. {base_reason}'
        else:
            reason = base_reason

        rec = ExplanationRecord(token=token, label=label, reason=reason)
        records.append(rec)

    return records


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield Python dicts from a JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def generate_explanations_from_jsonl(
    input_path: Path,
    output_path: Path,
    token_key: str = "token",
    label_key: str = "label",
    add_token_in_reason: bool = False,
    preserve_input_fields: bool = True,
) -> None:
    """
    Read a JSONL file with PHI spans and write a JSONL file with explanations.

    Expected basic schema per line in input JSONL:
        {
          "token": "...",
          "label": "NAME",
          ... (any other fields)
        }

    Output JSONL will contain at least:
        {
          "token": "...",
          "label": "NAME",
          "reason": "...",
          ...(optionally original fields)
        }

    Args:
        input_path: Path to the input JSONL file.
        output_path: Path to the output JSONL file.
        token_key: Key in input JSON for the token text.
        label_key: Key in input JSON for the PHI label.
        add_token_in_reason: See `generate_explanations_for_spans`.
        preserve_input_fields: If True, merge original fields with explanation.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as fin, \
            output_path.open("w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            token = str(obj.get(token_key, "")).strip()
            label = str(obj.get(label_key, "")).strip().upper()

            base_reason = get_explanation_template(label)
            if add_token_in_reason and token:
                reason = f'"{token}" is redacted. {base_reason}'
            else:
                reason = base_reason

            rec = ExplanationRecord(token=token, label=label, reason=reason)

            if preserve_input_fields:
                merged = {**obj, **rec.to_dict()}
            else:
                merged = rec.to_dict()

            fout.write(json.dumps(merged, ensure_ascii=False) + "\n")


def main(
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    add_token_in_reason: bool = False,
    preserve_input_fields: bool = True,
) -> None:
    """
    Convenient entry point for CLI use.

    You can run:

        python -m src.explanation.generator \\
            --input path/to/phi_spans.jsonl \\
            --output path/to/explanations.jsonl

    If you wire up a `console_scripts` entry point in setup.cfg/pyproject.toml,
    this can also become a simple `generate-explanations` command.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate HIPAA-grounded explanations for PHI spans."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=bool(input_path is None),
        help="Path to the input JSONL file with PHI spans.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=bool(output_path is None),
        help="Path to the output JSONL file with explanations.",
    )
    parser.add_argument(
        "--add-token-in-reason",
        action="store_true",
        help='Include the raw token text inside the explanation string.',
    )
    parser.add_argument(
        "--no-preserve-input",
        action="store_true",
        help="Do not preserve original input fields in the output.",
    )

    args = parser.parse_args([] if input_path else None)

    in_path = Path(input_path or args.input)
    out_path = Path(output_path or args.output)
    add_tok = add_token_in_reason or args.add_token_in_reason
    preserve = preserve_input_fields and (not args.no_preserve_input)

    generate_explanations_from_jsonl(
        input_path=in_path,
        output_path=out_path,
        add_token_in_reason=add_tok,
        preserve_input_fields=preserve,
    )


if __name__ == "__main__":
    main()