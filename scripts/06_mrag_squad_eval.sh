#!/bin/bash
set -e

echo "Running MRAG evaluation on SQuAD (Time Filtered)..."
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "Running MRAG evaluation on SQuAD (Time Filtered)..."
python3 -u src/squad_time_filter_eval.py --batch-size 128 2>&1 | tee logs/mrag_squad.log
