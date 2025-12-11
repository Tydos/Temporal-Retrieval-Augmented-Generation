#!/bin/bash

# Time-Aware RAG: Complete Pipeline Execution
# This script runs the entire time-aware RAG experiment end-to-end

set -e  # Exit on any error

echo "========================================"
echo "TIME-AWARE RAG COMPLETE PIPELINE"
echo "========================================"

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Project root determined as: $PROJECT_ROOT"
echo "Current directory before cd: $(pwd)"
cd "$PROJECT_ROOT"
echo "Current directory after cd: $(pwd)"

# Create necessary directories
mkdir -p data models outputs logs
echo "Created necessary directories"

# Verify we're in the right location
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found. Are we in the right directory?"
    echo "Current directory: $(pwd)"
    echo "Contents: $(ls -la)"
    exit 1
fi

# Step 1: Setup Environment and Dependencies
echo "Step 1/6: Setting up environment and dependencies..."
./scripts/01_setup_environment.sh

# Step 2: Generate Training Data
echo "Step 2/6: Generating temporal questions with T5..."
./scripts/02_generate_questions.sh

# Step 3: Train Time-Aware Contriever
echo "Step 3/6: Training time-aware Contriever model..."
./scripts/03_train_contriever.sh

# Step 4: Evaluate on ChroniclingQA
echo "Step 4/6: Evaluating on ChroniclingQA dataset..."
./scripts/04_evaluate_chroniclingqa.sh

# Step 5: Run MRAG Integration
echo "Step 5/6: Running MRAG integration and CAQA..."
./scripts/05_mrag_caqa_eval.sh

# Step 6: Generate Final Report
echo "Step 6/6: Running MRAG integration and SQUAD..."
./scripts/06_mrag_squad_eval.sh

echo "========================================"
echo "PIPELINE COMPLETED SUCCESSFULLY!"
echo "Results available in: outputs/"
echo "========================================"