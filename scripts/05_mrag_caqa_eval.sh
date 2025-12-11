#!/bin/bash
set -e

echo "Running MRAG evaluation on ChroniclingQA (CAQA)..."
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "Running MRAG evaluation on ChroniclingQA (CAQA)..."
python3 -u src/mrag_integration.py --batch-size 128 2>&1 | tee logs/mrag_caqa_eval.log
