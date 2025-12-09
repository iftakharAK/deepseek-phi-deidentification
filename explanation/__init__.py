

"""
Explanation module for PHI de-identification.

This package provides:
- A simple schema for explanation records.
- A template library for HIPAA-grounded explanations.
- Utilities to generate explanations for PHI spans.
- Optional evaluation utilities for explanation accuracy.

Typical usage (Python):

    from explanation.generator import generate_explanations_for_spans
    spans = [
        {"token": "John Smith", "label": "NAME"},
        {"token": "2019-06-20", "label": "DATE"},
    ]
    explanations = generate_explanations_for_spans(spans)

"""

from .schema import ExplanationRecord
from .templates import get_explanation_template
from .generator import (
    generate_explanations_for_spans,
    generate_explanations_from_jsonl,
)

__all__ = [
    "ExplanationRecord",
    "get_explanation_template",
    "generate_explanations_for_spans",
    "generate_explanations_from_jsonl",
]