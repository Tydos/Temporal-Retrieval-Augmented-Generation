#!/bin/bash

# Step 3: Train Time-Aware Contriever Model

echo "Training time-aware Contriever model..."

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Check if conda environment is available
if command -v conda &> /dev/null && [ -d "/Users/Patron/miniforge3" ]; then
    PYTHON_CMD="/Users/Patron/miniforge3/bin/conda run -p /Users/Patron/miniforge3 --no-capture-output python"
    echo "Using Conda Python: $PYTHON_CMD"
else
    PYTHON_CMD="python3"
    echo "Using System Python: $(which python3)"
fi

# Set environment variables
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export KMP_DUPLICATE_LIB_OK=TRUE  # Fix OpenMP conflict
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Memory optimization

# Check if training data exists
if [ ! -f "data/generated_questions/sample_generated_questions.json" ]; then
    echo "✗ Error: Training data not found!"
    echo "Please run ./scripts/02_generate_questions.sh first"
    exit 1
fi

echo "Starting Contriever fine-tuning..."
echo "This may take several hours depending on your hardware"
echo "Logs will be saved to: logs/contriever_training.log"

# Create model output directory
mkdir -p models/time_aware_contriever

# Set up Weights & Biases logging (optional)
if [ ! -z "$WANDB_API_KEY" ]; then
    export REPORT_TO="wandb"
    echo "Weights & Biases logging enabled"
else
    echo "No WANDB_API_KEY set, skipping wandb logging"
fi

# Run training with monitoring (using simplified version to avoid memory issues)
$PYTHON_CMD -u src/contriever_training.py 2>&1 | tee logs/contriever_training.log &

# Get the PID of the training process
TRAINING_PID=$!

# Monitor training progress
echo "Training started with PID: $TRAINING_PID"
echo "You can monitor progress in real-time with: tail -f logs/contriever_training.log"
echo ""
echo "Training configuration:"
echo "- Base model: facebook/contriever-msmarco"
echo "- Micro batch size: 32"
echo "- Gradient accumulation steps: 8"
echo "- Learning rate: 1e-5"
echo "- Epochs: 14"
echo "- Triplet margin: 1.0"
echo ""

# Wait for training to complete
wait $TRAINING_PID
TRAINING_EXIT_CODE=$?

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
    
    # Check if model was saved (notebook saves to contriever_finetuned_NEW_20k)
    if [ -d "contriever_finetuned_NEW_20k" ]; then
        echo "✓ Model saved to: contriever_finetuned_NEW_20k/"
        
        # Show model info
        echo "Model files:"
        ls -la contriever_finetuned_NEW_20k/
        
    elif [ -d "models/time_aware_contriever" ]; then
        echo "✓ Model saved to: models/time_aware_contriever/"
        ls -la models/time_aware_contriever/
    else
        echo "⚠ Warning: Model directory not found"
    fi
    
else
    echo "✗ Training failed with exit code: $TRAINING_EXIT_CODE"
    echo "Check logs/contriever_training.log for details"
    exit 1
fi

# Optional: Test model loading
echo ""
echo "Testing model loading..."
$PYTHON_CMD -c "
try:
    from transformers import AutoTokenizer, AutoModel
    import os
    
    # Check notebook path first
    model_path = 'contriever_finetuned_NEW_20k'
    if os.path.exists(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        print('✓ Fine-tuned model can be loaded successfully')
        print(f'Model path: {model_path}')
    else:
        print('⚠ Fine-tuned model directory not found')
except Exception as e:
    print(f'⚠ Error loading model: {e}')
"

echo ""
echo "Training completed!"
echo "Fine-tuned model saved to: contriever_finetuned_NEW_20k/"
echo ""
echo "Next step: Run ./scripts/04_evaluate_chroniclingqa.sh"