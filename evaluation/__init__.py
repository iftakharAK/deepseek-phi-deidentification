

"""
Evaluation package for DeepSeek PHI de-identification.

This module exposes reusable metric helpers for:
- token-level PHI tag evaluation
- char-level F1
- hallucination detection wrappers
"""

from .metrics import (
    safe_div,
    extract_phi_tags,
    compute_token_level_metrics,
    compute_char_level_f1,
    detect_hallucination,
    classify_policy,
)