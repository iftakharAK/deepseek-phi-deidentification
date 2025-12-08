#!/usr/bin/env bash
# scripts/train_regex_detector.sh

# Run from repo root:
#   bash scripts/train_regex_detector.sh
#
# Optionally override the base model:
#   BASE_MODEL_DIR="deepseek_phi_masked_20251019-023630/final_model" bash scripts/train_regex_detector.sh

set -e

echo "Using BASE_MODEL_DIR=${BASE_MODEL_DIR:-deepseek-ai/deepseek-llm-7b-base}"

python -m src.training.train_regex_detector
